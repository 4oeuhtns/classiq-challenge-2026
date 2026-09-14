from dataclasses import dataclass, asdict, fields

import classiq
from classiq import *
from classiq.interface.generator.hardware.hardware_data import CustomHardwareSettings
import hashlib
import pathlib
import json

from .compile import build_main, COMPILER_VERSION
from .terms import fingerprint


ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Measurement:
    """Serialized measurement dataclass to cache measurement results"""
    width: int
    depth: int
    cx: int
    qasm: str

def synthesis(main, preferences, constraints):
    model = create_model(main, constraints=constraints)
    return synthesize(model, preferences=preferences)

def post_process(qprog):
    raw_qasm = export(qprog, TargetLanguage.QASM2)
    raw_lines = raw_qasm.splitlines()

    # Locate the two top-level Hadamard preparation calls that belong to the
    # synthesis harness. Removing them produces only a candidate standalone oracle.
    preparation_indices = {
        index
        for index, line in enumerate(raw_lines)
        if line.lstrip().startswith("hadamard_transform_") and "q[" in line
    }
    if len(preparation_indices) != 2:
        raise RuntimeError(
            "Could not safely extract the candidate oracle: expected exactly "
            f"two Hadamard preparation calls, found {len(preparation_indices)}"
        )

    candidate_qasm = "\n".join(
        line
        for index, line in enumerate(raw_lines)
        if index not in preparation_indices
    )
    return quantum_program_from_qasm(candidate_qasm)

def transpilation(candidate, preferences):
    return classiq.transpile(
        candidate,
        preferences=preferences
    )

def _canon(data):
    # dict
    if isinstance(data, dict):
        return {_canon(k): _canon(v) for k,v in data.items()}
    # list, only sort lists of strings (heuristic for determining set)
    if isinstance(data, list):
        li = [_canon(l) for l in data]
        if all(isinstance(x, str) for x in li):
            return sorted(li)
        return li
    return data

def canonical(settings) -> str:
    """classiq preference/constraint into canonical json string"""
    data = json.loads(settings.model_dump_json())
    return json.dumps(_canon(data), sort_keys=True, separators=(',', ':'))

def build_settings(seed):
    syn_pref = Preferences(random_seed=seed)
    syn_const = Constraints(optimization_parameter=OptimizationParameter.DEPTH,max_width=18)

    trans_pref = Preferences(
        transpilation_option=TranspilationOption.AUTO_OPTIMIZE,
        custom_hardware_settings=CustomHardwareSettings(basis_gates=["u3", "cx"]),
        random_seed=seed
    )
    return syn_pref, syn_const, trans_pref

def cache_key(terms, syn_pref, syn_const, trans_pref):
    return hashlib.sha256((
        fingerprint(terms) + ", " +
        canonical(syn_pref) + ", " +
        canonical(syn_const) + ", " +
        canonical(trans_pref) + ", " +
        classiq.__version__ + ", " +
        str(COMPILER_VERSION)
    ).encode()).hexdigest()

def server_version(qasm_str):
    """Return the Classiq version from the QASM header, or None if it isn't there."""
    prefix = "// Classiq version:"
    for line in qasm_str.splitlines()[:10]:  # header only, don't scan the gate list
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None

def measurement(terms, seed):
    """Returns a measurement object AND whether the cache was hit or not."""
    syn_pref, syn_const, trans_pref = build_settings(seed)

    # key to hash measurement
    key = cache_key(terms, syn_pref, syn_const, trans_pref)

    # if exists in cache, return cached data
    path = CACHE_DIR/f"{key}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # A cache file is a wire format: these bytes outlive the code that
            # wrote them. Drop keys this version does not know about, so a file
            # written by a newer checkout is still readable. The other direction
            # -- a file missing a key a newer Measurement needs -- is handled by
            # giving any field added later a default. Neither is a substitute
            # for bumping COMPILER_VERSION when the numbers themselves change.
            known = {f.name for f in fields(Measurement)}
            return Measurement(**{k: v for k, v in data.items() if k in known}), True


    main = build_main(terms)
    qprog = synthesis(main, syn_pref, syn_const)
    candidate = post_process(qprog)
    transpiled = transpilation(candidate, trans_pref)
    metrics = classiq.get_transpiled_circuit_metrics(transpiled)
    submission_qasm = export(
        transpiled,
        TargetLanguage.QASM2,
        transpilation_config=TranspilationConfig(basis_gates=["u3", "cx"]),
    )

    # write to cache
    m = Measurement(metrics.width, metrics.depth, metrics.count_ops.get('cx', 0), submission_qasm)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(m), f)

    return m, False


def save_submission(measurement, path=None):
    """Write a chosen measurement's QASM to disk, for submission.

    Deliberately separate from `measurement()`. Measuring happens thousands of
    times during a search; publishing happens once, for the one design you
    picked. If measure wrote the file itself, every candidate would clobber the
    previous one and the best result would be overwritten by whatever ran next.

    Args:
        measurement: the Measurement whose QASM should be published.
        path: destination; defaults to submission.qasm at the project root.

    Returns:
        The path written to.
    """
    path = ROOT / "submission.qasm" if path is None else pathlib.Path(path)
    path.write_text(measurement.qasm, encoding="utf-8")
    return path
