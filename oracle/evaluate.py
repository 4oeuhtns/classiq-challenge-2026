import re
import time

import classiq
from classiq.interface.exceptions import ClassiqAPIError

from .measure import build_settings, cache_key, measurement, canonical, server_version
from .verify import verify
from .terms import serialize, fingerprint
from .log import append


def required_width(message: str):
    """Pull the qubit count out of a width refusal, or None if it isn't one."""
    match = re.search(r"at least (\d+) qubits", message)
    return int(match.group(1)) if match else None


def evaluate(terms, seed=0, *, with_verify=True, **context):
    """measures and verifies, putting everything into logs and returns run_id.

    A design the server refuses to synthesize is logged like any other run, with
    null metrics plus `error` and `required_width`, and comes back as
    (None, None, run_id). Every other exception propagates -- those are bugs.
    """
    started = time.perf_counter()
    syn_pref, syn_const, trans_pref = build_settings(seed)
    key = cache_key(terms, syn_pref, syn_const, trans_pref)

    # everything knowable before spending a cloud call, so both paths share it
    row = {
        "cover_fingerprint": fingerprint(terms),
        "terms_json": serialize(terms),
        "n_terms": len(terms),
        "cache_key": key,
        "seed": seed,
        "syn_prefs": canonical(syn_pref),
        "syn_constraints": canonical(syn_const),
        "trans_prefs": canonical(trans_pref),
        "classiq_version": classiq.__version__,
        "source_tool": "classiq",
        **context,
    }

    try:
        m, hit = measurement(terms, seed)
    except ClassiqAPIError as e:
        message = str(e).splitlines()[0]
        row["error"] = message
        row["required_width"] = required_width(message)
        row["wall_time"] = time.perf_counter() - started
        return None, None, append(row)

    row.update({
        "cache_hit": hit,
        "width": m.width,
        "real_depth": m.depth,
        "real_cx": m.cx,
        "server_version": server_version(m.qasm),
    })

    v = verify(terms, m.qasm, seed) if with_verify else None
    if v is not None:
        row.update({
            "ok": bool(v.ok),
            "max_error": float(v.max_error),
            "ancilla_error": float(v.ancilla_error),
            "norm_error": float(v.normalization_error),
        })

    row["wall_time"] = time.perf_counter() - started
    return m, v, append(row)
