"""Tests for the verifier itself -- the only check that compares the CIRCUIT
against the DESIGN.

`matches_target` checks the design and costs microseconds. `test_terms.py`
checks the masks and costs a second. Neither can see a compiler bug: if
`predicate` builds the wrong condition, `render` still reports a perfect match
while the circuit computes something else. This file is what catches that.

It is also the expensive one -- each case is three ExecutionSession round trips,
about 25 s -- so it is not part of the fast suite. Run it when `predicate`,
`emit`, or `verify` itself changes: per code path, not per candidate.

    python testing/test_verify.py --quick     3 cases,  ~2 min
    python testing/test_verify.py             10 cases, ~5 min

How the cases work: `verify` compares a fixed QASM (the baseline circuit)
against the mask implied by whatever terms it is given. Feeding it the WRONG
terms must fail, and mutating the QASM must fail. A design that differs from the
baseline by a single pixel has to be caught -- that is the real bar.

`max_error` is a MAX over 4096 amplitudes, so any disagreement saturates it at
2/sqrt(4096) = 0.03125. It tells you *whether* something is wrong, never *how*
wrong. For distance, use `(cover_mask(terms) ^ TARGET).sum()`, which is free.
"""

import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from oracle.baseline import BASELINE_TERMS
from oracle.measure import measurement
from oracle.terms import Rect
from oracle.verify import verify

QUICK = "--quick" in sys.argv
SATURATED = 0.03125          # 2/sqrt(4096) -- one flipped sign maxes the metric
CLEAN = 1e-10

PASSED, FAILED = 0, []


def check(name, condition):
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


m, hit = measurement(BASELINE_TERMS, seed=42)
print(f"baseline: width={m.width} depth={m.depth} cx={m.cx}  (cache_hit={hit})\n")

swapped = list(BASELINE_TERMS)
swapped[0], swapped[1] = swapped[1], swapped[0]

lines = m.qasm.splitlines()
no_u3 = "\n".join(line for line in lines if not line.startswith("u3"))
no_cx = "\n".join(line for line in lines if not line.startswith("cx"))

# (label, terms, qasm, should_pass)
CASES = [
    ("baseline against its own circuit", BASELINE_TERMS, m.qasm, True),
    ("one term missing", BASELINE_TERMS[:-1], m.qasm, False),
    ("one extra 1-pixel rect", list(BASELINE_TERMS) + [Rect(0, 0, 0, 0)], m.qasm, False),
    ("one extra 5x5 rect", list(BASELINE_TERMS) + [Rect(0, 4, 0, 4)], m.qasm, False),
    ("empty design", [], m.qasm, False),
    ("QASM with every u3 stripped", BASELINE_TERMS, no_u3, False),
    ("QASM with every cx stripped", BASELINE_TERMS, no_cx, False),
    ("terms reordered -- must still PASS", swapped, m.qasm, True),
    ("design XORed with itself -- empty mask", list(BASELINE_TERMS) + swapped, m.qasm, False),
    ("duplicate rect pair cancels -- must still PASS",
     [Rect(0, 0, 0, 0)] + list(BASELINE_TERMS) + [Rect(0, 0, 0, 0)], m.qasm, True),
]
if QUICK:
    CASES = [CASES[0], CASES[2], CASES[9]]

for label, terms, qasm, should_pass in CASES:
    v = verify(terms=terms, qasm_str=qasm, seed=0)
    verdict = "PASS" if should_pass else "fail"
    print(f"  [{verdict}] {label:<46} ok={str(v.ok):<5} max_error={v.max_error:.4e}")

    check(f"{label}: ok is {should_pass}", bool(v.ok) is should_pass)
    if should_pass:
        check(f"{label}: max_error is numerical noise", v.max_error < CLEAN)
    else:
        # Any real disagreement saturates. A value strictly between noise and
        # saturation would mean the metric is not behaving as a maximum.
        check(f"{label}: max_error saturates at 2/64",
              abs(v.max_error - SATURATED) < 1e-9)
    check(f"{label}: ancillas returned to |0>", v.ancilla_error < CLEAN)
    check(f"{label}: statevector normalized", v.normalization_error < CLEAN)

# verify recomputes width/depth/cx from the QASM by its own DAG walk. Agreement
# with Classiq's reported metrics is an independent confirmation of both.
v = verify(terms=BASELINE_TERMS, qasm_str=m.qasm, seed=0)
check("verify's width agrees with the measurement", v.width == m.width)
check("verify's depth agrees with the measurement", v.depth == m.depth)
check("verify's cx count agrees with the measurement", v.cx_count == m.cx)

# Reordering terms is a Phase 4 move; it must never change correctness.
check("reorder_terms is a correctness-preserving move",
      bool(verify(terms=swapped, qasm_str=m.qasm, seed=0).ok))

print(f"\n{PASSED} passed, {len(FAILED)} failed"
      f"{'  (--quick: 3 of 10 cases)' if QUICK else ''}")
if FAILED:
    print("failures:", ", ".join(FAILED))
sys.exit(1 if FAILED else 0)
