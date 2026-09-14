"""Score and verify the hand-built oracle.

    python experiments/check_handbuilt.py            # score + local check, no cloud
    python experiments/check_handbuilt.py --verify   # also run the official verifier
    python experiments/check_handbuilt.py --verify --log

The scoring half needs only qiskit. `qasm_metrics` in oracle/verify.py is pure
regex over the QASM, so the depth and CX it reports are the real submission
numbers, computed locally and deterministically -- no synthesis call, no seed
variance, nothing to pay for.

Only correctness needs Classiq, because the official check simulates the
circuit on three random input superpositions. `verify` takes `terms` solely to
build the expected mask via `cover_mask(terms)`, so BASELINE_TERMS is the right
thing to pass: the design under test is the QASM, and the terms only say what
the answer should look like.

A free local pre-flight runs first either way -- it catches a broken circuit
before you spend a cloud call on it.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qiskit import transpile, qasm2
from handbuilt import build, check

ROOT = pathlib.Path(__file__).resolve().parent.parent
QASM_PATH = ROOT / "experiments" / "handbuilt.qasm"


def best_qasm(qc, levels=(2, 3), seeds=range(8)):
    """Transpile under several settings and keep the shallowest.

    The transpiler is the last optimisation pass, and it is free to run, so
    there is no reason to accept its first answer.
    """
    best = None
    for level in levels:
        for seed in seeds:
            t = transpile(qc, basis_gates=["u3", "cx"],
                          optimization_level=level, seed_transpiler=seed)
            src = qasm2.dumps(t)
            if best is None or t.depth() < best[0]:
                best = (t.depth(), t.count_ops().get("cx", 0), src, level, seed)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run the official verifier (needs Classiq)")
    ap.add_argument("--log", action="store_true", help="append a row to logs/experiments.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qc = build()
    err, leak = check(qc)
    print(f"local statevector check : phase error {err:.1e}   ancilla leak {leak:.1e}")
    if err > 1e-9 or leak > 1e-9:
        print("  FAILED locally -- not worth a cloud call")
        return 1

    depth, cx, src, level, seed = best_qasm(qc)
    QASM_PATH.write_text(src)
    print(f"best transpile          : optimization_level={level}, seed={seed}")

    try:
        from oracle.verify import qasm_metrics      # pure regex, but the module imports classiq
        width, off_depth, off_cx = qasm_metrics(src)
        source = "official qasm_metrics"
    except ModuleNotFoundError:
        # No SDK here. qasm_metrics layers gates exactly as qiskit .depth() does,
        # and the two were checked to agree to the gate on this circuit, so this
        # is the same number by a different route -- not an approximation.
        width, off_depth, off_cx = 18, depth, cx
        source = "qiskit (no SDK; == qasm_metrics)"
    print(f"{source:23s} : width {width}  depth {off_depth}  cx {off_cx}")
    print(f"baseline for comparison : width 18  depth 5267  cx 3512")
    print(f"wrote {QASM_PATH.relative_to(ROOT)}")

    if not args.verify:
        print("\n(pass --verify to run the official statevector check)")
        return 0

    from oracle.baseline import BASELINE_TERMS
    from oracle.verify import verify
    started = time.perf_counter()
    v = verify(BASELINE_TERMS, src, args.seed)
    print(f"\nofficial verify         : ok={v.ok}  max_error={v.max_error:.2e}  "
          f"ancilla_error={v.ancilla_error:.2e}  ({time.perf_counter()-started:.1f}s)")

    if args.log:
        import classiq
        from oracle.log import append
        from oracle.terms import fingerprint, serialize
        run_id = append({
            "cover_fingerprint": fingerprint(BASELINE_TERMS),
            "terms_json": serialize(BASELINE_TERMS),
            "n_terms": len(BASELINE_TERMS),
            "strategy_tags": ["handbuilt", "rank10", "margolus"],
            "width": width, "real_depth": off_depth, "real_cx": off_cx,
            "ok": bool(v.ok), "max_error": float(v.max_error),
            "ancilla_error": float(v.ancilla_error),
            "norm_error": float(v.normalization_error),
            "seed": args.seed,
            "classiq_version": classiq.__version__,
            "source_tool": "handbuilt-qiskit",
            "wall_time": time.perf_counter() - started,
        })
        print(f"logged as {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
