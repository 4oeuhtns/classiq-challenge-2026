"""Tests for the classiq-aware layer. Needs the SDK, but makes no cloud call.

`create_model` runs the generative expansion locally, so it catches anything the
Qmod language refuses -- missing parentheses, name collisions, a predicate that
quietly stopped being a quantum expression. It is free and takes seconds.

What it does NOT check is whether the circuit computes the right pixels, or
whether it fits in 18 qubits. Those need synthesis and the official simulator:
see test_verify.py.

    python testing/test_compile.py
"""

import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from classiq import (Const, Output, QNum, allocate, create_model,
                     hadamard_transform, qfunc, qperm)
from classiq.qmod.symbolic_expr import SymbolicExpr

from oracle.compile import emit, predicate
from oracle.mask import COORD_BITS
from oracle.terms import And, Cube, HalfPlane, Not, Or, Parity, Rect, Xor

PASSED, FAILED = 0, []


def check(name, condition):
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


def expand(terms):
    """Run the generative expansion and return the serialized model."""
    @qperm
    def oracle(x: Const[QNum], y: Const[QNum]) -> None:
        for index, term in enumerate(terms):
            emit(term, x, y, index)

    @qfunc
    def main(x: Output[QNum[COORD_BITS]], y: Output[QNum[COORD_BITS]]) -> None:
        allocate(x)
        allocate(y)
        hadamard_transform(x)
        hadamard_transform(y)
        oracle(x, y)

    return create_model(main)


def expands(name, terms):
    try:
        expand(terms)
        check(name, True)
    except Exception as exc:                                  # noqa: BLE001
        check(f"{name} -- {type(exc).__name__}: {str(exc).splitlines()[0][:70]}", False)


R, R2 = Rect(8, 20, 8, 20), Rect(14, 26, 14, 26)

EVERY_TERM = [
    ("Rect", Rect(5, 40, 12, 33)),
    ("Rect, single row", Rect(36, 44, 12, 12)),
    ("Cube", Cube.fix(x={5: 1}, y={4: 0})),
    ("Parity", Parity.fix(x=[0, 1], y=[3])),
    ("HalfPlane", HalfPlane(1, 1, 63)),
    ("HalfPlane, negative coefficients", HalfPlane(1, -1, 10)),
    ("And", And((R, R2))),
    ("Or", Or((R, R2))),
    ("Xor", Xor((R, R2))),
    ("Not", Not(R)),
    ("And containing a Parity", And((R, Parity.fix(y=[2])))),
    ("nested three deep", And((Not(R), Or((R2, Parity.fix(x=[0])))))),
]

# ---------------------------------------------------------------------------
# predicate must return a QUANTUM expression.
#
# This is the cheap guard against a whole family of silent failures. `Not` once
# read `term.child == 0`, comparing a dataclass to an integer: that returns the
# Python bool False, `control(False, ...)` is legal and emits nothing, and the
# term compiled to an empty circuit with no error anywhere. Only the type
# betrayed it.
# ---------------------------------------------------------------------------

def predicate_type(term):
    captured = {}

    @qfunc
    def main(x: Output[QNum[COORD_BITS]], y: Output[QNum[COORD_BITS]]) -> None:
        allocate(x)
        allocate(y)
        captured["value"] = predicate(term, x, y)

    create_model(main)
    return captured["value"]

for label, term in EVERY_TERM:
    check(f"predicate({label}) returns a quantum expression",
          isinstance(predicate_type(term), SymbolicExpr))

# The degenerate cases are the documented exceptions: with nothing to constrain,
# the honest answer is the operator's identity element, as a plain Python bool.
check("predicate(empty Cube) is True (AND identity)", predicate_type(Cube.fix()) is True)
check("predicate(empty And) is True (AND identity)", predicate_type(And(())) is True)
check("predicate(empty Xor) is False (XOR identity)", predicate_type(Xor(())) is False)
check("predicate(empty Or) is False (OR identity)", predicate_type(Or(())) is False)

try:
    predicate(object(), None, None)
    check("predicate raises on an unknown term type", False)
except TypeError:
    check("predicate raises on an unknown term type", True)

# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------

for label, term in EVERY_TERM:
    expands(f"expands: {label}", [term])

expands("expands: whole-design mix", [t for _, t in EVERY_TERM])
expands("expands: empty design", [])

# A QArray name must be unique within one oracle, so emit builds it from the
# term's index. Before that fix, a second Parity crashed the expansion.
expands("expands: TWO Parity terms (unique QArray names)",
        [Parity.fix(x=[0]), Parity.fix(y=[1])])
expands("expands: three Parity terms", [Parity.fix(x=[0]), Parity.fix(x=[1]), Parity.fix(y=[2])])
expands("expands: Parity with an empty axis", [Parity.fix(x=[0, 1])])
expands("expands: Parity with no bits at all", [Parity.fix()])

# ---------------------------------------------------------------------------
# Parity must actually take the cheap path: Z gates, no control, no phase.
# If this regresses, parity silently costs a Toffoli tree instead of k gates.
# ---------------------------------------------------------------------------

parity_model = expand([Parity.fix(x=[0, 1], y=[3])])
check("standalone Parity emits Z gates", '"Z"' in parity_model)
check("standalone Parity binds both axes to bit arrays",
      '"xb0"' in parity_model and '"yb0"' in parity_model)

check("a Rect emits no Z gates (it takes the control path)",
      '"Z"' not in expand([R]))

# Nested, the shortcut is unavailable -- there is no value to combine, so the
# parity pays full price through predicate instead.
check("nested Parity emits no Z gates (falls back to predicate)",
      '"Z"' not in expand([And((R, Parity.fix(y=[2])))]))

print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("failures:", ", ".join(FAILED))
sys.exit(1 if FAILED else 0)
