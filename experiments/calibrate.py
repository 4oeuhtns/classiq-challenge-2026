"""Phase 1.2 -- measure the calibration set for real. The overnight batch.

Not a test. Tests assert and must pass; this asks the cloud a few hundred
questions and records the answers, and there is no pass or fail. That is why
it lives in experiments/ rather than testing/.

It writes nothing itself. Every result reaches disk through ``evaluate``,
which appends one row per design to logs/experiments.jsonl and caches the
QASM. The proxy fitting reads the log, not this script's output.

Two properties make it safe to start and walk away, both provided by
``runner.run_batch``: it resumes for free after a crash, and ``--hours`` stops
it cleanly rather than running into your morning. A third comes from
``recipes.generate``, which cycles the recipes round-robin -- so a batch that
dies at 3 a.m. is a *smaller* calibration set, not a biased one.

    python experiments/calibrate.py --n 300 --hours 5
    python experiments/calibrate.py --dry            # no cloud calls
"""

import argparse
import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from experiments.recipes import SECONDS_FIXED, SECONDS_PER_TERM, generate
from experiments.runner import hhmm, run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="designs to generate")
    ap.add_argument("--seed", type=int, default=0, help="generator seed (not the synthesis seed)")
    ap.add_argument("--hours", type=float, default=5.0, help="stop cleanly after this long")
    ap.add_argument("--verify", action="store_true",
                    help="also simulate each circuit; roughly doubles the runtime")
    ap.add_argument("--dry", action="store_true", help="generate and report, make no cloud calls")
    args = ap.parse_args()

    designs = generate(args.n, seed=args.seed)

    terms_total = sum(len(t) for t, _ in designs)
    estimate = len(designs) * SECONDS_FIXED + terms_total * SECONDS_PER_TERM
    print(f"upper-bound estimate if nothing is cached: {hhmm(estimate)}"
          f"   deadline: {args.hours} h\n")

    if args.dry:
        print(f"--dry: {len(designs)} designs generated, stopping before any cloud call.")
        return 0

    run_batch(designs, hours=args.hours, with_verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
