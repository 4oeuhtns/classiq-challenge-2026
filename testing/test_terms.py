"""Offline tests for the term/mask layer. numpy only -- no classiq, no network.

Runs in milliseconds, so run it after every edit to terms.py.

Every render check compares against a *brute-force* definition written a
different way (per-pixel loops, popcount, binary strings) rather than a copy of
the implementation. A test that restates the code cannot catch a wrong idea.

    python testing/test_terms.py
"""

import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from oracle.baseline import BASELINE_TERMS
from oracle.log import LOG_PATH
from oracle.mask import COORD_BITS, GRID_SIZE, TARGET
from oracle.terms import (And, Cube, HalfPlane, Not, Or, Parity, Rect, Xor,
                          cover_mask, fingerprint, from_data, matches_target,
                          serialize)

PASSED, FAILED = 0, []


def check(name, condition):
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


def same(name, got, want):
    check(name, np.array_equal(got, want))


# --------------------------------------------------------------------------
# brute-force references -- deliberately written differently from terms.py
# --------------------------------------------------------------------------

def brute(predicate):
    """Build a mask by asking a plain Python function about every pixel."""
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            mask[y][x] = predicate(x, y)
    return mask


def bit_of(value, index):
    """Read a bit out of the binary *string*, not by shifting."""
    return int(format(value, f"0{COORD_BITS}b")[::-1][index])


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

same("Rect.render",
     Rect(5, 40, 12, 33).render(),
     brute(lambda x, y: 5 <= x <= 40 and 12 <= y <= 33))

same("Rect.render, single row",
     Rect(36, 44, 12, 12).render(),
     brute(lambda x, y: 36 <= x <= 44 and y == 12))

same("Cube.render, one bit",
     Cube.fix(y={4: 1}).render(),
     brute(lambda x, y: bit_of(y, 4) == 1))

same("Cube.render, several bits, both axes",
     Cube.fix(x={5: 1, 2: 0}, y={0: 1}).render(),
     brute(lambda x, y: bit_of(x, 5) == 1 and bit_of(x, 2) == 0 and bit_of(y, 0) == 1))

# parity is a *count* question, so the reference counts.
same("Parity.render, one axis",
     Parity.fix(x=[0, 1]).render(),
     brute(lambda x, y: (bin(x & 0b000011).count("1")) % 2 == 1))

same("Parity.render, both axes",
     Parity.fix(x=[5, 3, 0], y=[1]).render(),
     brute(lambda x, y: (bin(x & 0b101001).count("1") + bin(y & 0b000010).count("1")) % 2 == 1))

check("Parity covers exactly half the grid",
      all(Parity.fix(x=xb, y=yb).render().sum() == GRID_SIZE ** 2 // 2
          for xb, yb in [((0,), ()), ((), (2,)), ((0, 1), (0, 1)), ((5, 3, 0), (1,))]))

for a, b, c in [(1, 1, 63), (1, -1, 10), (-1, 1, 10), (3, 5, 100), (0, 1, 31), (-1, -1, -40)]:
    same(f"HalfPlane.render a={a} b={b} c={c}",
         HalfPlane(a, b, c).render(),
         brute(lambda x, y, a=a, b=b, c=c: a * x + b * y <= c))

# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

R1, R2 = Rect(0, 31, 0, 31), Rect(16, 47, 16, 47)

same("And.render", And((R1, R2)).render(), R1.render() & R2.render())
same("Or.render", Or((R1, R2)).render(), R1.render() | R2.render())
same("Xor.render", Xor((R1, R2)).render(), R1.render() ^ R2.render())
same("Not.render", Not(R1).render(), ~R1.render())
same("nested containers",
     And((Not(R1), Or((R2, Parity.fix(x=[0]))))).render(),
     (~R1.render()) & (R2.render() | Parity.fix(x=[0]).render()))

# Identity elements. These differ per operator and getting one wrong inverts a
# mask silently -- Parity seeded with ones once produced a plausible wrong plaid.
check("empty And covers everything (AND identity is True)", And(()).render().all())
check("empty Or covers nothing (OR identity is False)", not Or(()).render().any())
check("empty Xor covers nothing (XOR identity is False)", not Xor(()).render().any())
check("empty Cube covers everything", Cube.fix().render().all())
check("empty Parity covers nothing", not Parity.fix().render().any())

# --------------------------------------------------------------------------
# cover_mask: terms XOR, they do not union
# --------------------------------------------------------------------------

check("stamping the same region twice cancels it",
      not cover_mask([R1, R1]).any())
same("overlapping terms XOR rather than union",
     cover_mask([R1, R2]), R1.render() ^ R2.render())
check("empty design covers nothing", not cover_mask([]).any())
check("matches_target(BASELINE_TERMS)", matches_target(BASELINE_TERMS))
check("baseline reproduces TARGET exactly",
      np.array_equal(cover_mask(BASELINE_TERMS), TARGET))

# --------------------------------------------------------------------------
# canonical construction -- two spellings of one design must be one design,
# or the cache pays twice for the same circuit
# --------------------------------------------------------------------------

check("Cube.fix sorts its bits", Cube.fix(x={5: 1, 4: 0}) == Cube.fix(x={4: 0, 5: 1}))
check("Parity.fix sorts its bits", Parity.fix(x=[1, 0]) == Parity.fix(x=[0, 1]))
check("Cube is hashable", hash(Cube.fix(x={5: 1})) == hash(Cube.fix(x={5: 1})))
check("Parity is hashable", hash(Parity.fix(x=[0, 1])) == hash(Parity.fix(x=[1, 0])))
check("canonical spellings share a fingerprint",
      fingerprint([Parity.fix(x=[1, 0])]) == fingerprint([Parity.fix(x=[0, 1])]))

# --------------------------------------------------------------------------
# serialization -- the cache key rides on this
# --------------------------------------------------------------------------

check("list and tuple designs serialize alike",
      serialize([R1, R2]) == serialize((R1, R2)))
check("term ORDER is significant (it changes depth, so it must change the key)",
      serialize([R1, R2]) != serialize([R2, R1]))
check("nested terms keep their type tag",
      '["Rect"' in serialize([And((R1,))]).split('"children"')[1])


class _Lookalike(Rect):
    """Same fields as Rect. Without per-level tags these serialize identically."""


check("different child types do not collide",
      serialize([And((Rect(0, 1, 0, 1),))]) != serialize([And((_Lookalike(0, 1, 0, 1),))]))
check("...and neither do their fingerprints",
      fingerprint([And((Rect(0, 1, 0, 1),))]) != fingerprint([And((_Lookalike(0, 1, 0, 1),))]))
check("deep nesting survives",
      serialize([And((And((R1,)), Cube.fix(y={4: 1})))]).count('"Rect"') == 1)

# --------------------------------------------------------------------------
# from_data -- rebuilding a design from its log row
#
# Phase 4 lineage depends on this: `parent_run_id` only means something if the
# parent design can be recovered from the log. Every case goes through JSON
# rather than calling to_data directly, because JSON is where tuples become
# lists -- and a term holding a list is unhashable, compares unequal to the
# original, and silently changes its own fingerprint. A round-trip test that
# skips the JSON step cannot see any of that.
# --------------------------------------------------------------------------

R3 = Rect(8, 20, 8, 20)

ROUND_TRIP = [
    ("Rect", R3),
    ("Cube", Cube.fix(x={5: 1, 2: 0}, y={0: 1})),
    ("Parity", Parity.fix(x=[0, 1], y=[3])),
    ("HalfPlane", HalfPlane(1, -1, 10)),
    ("And", And((R3, Parity.fix(y=[2])))),
    ("Or", Or((R3, R1))),
    ("Xor", Xor((R3, Rect(0, 1, 0, 1)))),
    ("Not", Not(R3)),
    ("nested three deep", And((Not(R3), Or((Rect(1, 2, 1, 2), Parity.fix(x=[0])))))),
    # Empty sequence fields serialize to [], which is where the first version
    # of from_data crashed indexing obj[0]. Parity.fix(x=...) leaves y_bits
    # empty, so this is the common case, not a corner.
    ("Cube with one empty axis", Cube.fix(x={5: 1})),
    ("Cube with no bits", Cube.fix()),
    ("Parity with one empty axis", Parity.fix(x=[0, 1])),
    ("Parity with no bits", Parity.fix()),
    ("And with no children", And(())),
]

for label, term in ROUND_TRIP:
    back = from_data(json.loads(serialize(term)))
    check(f"from_data rebuilds {label}", back == term)
    check(f"rebuilt {label} is still hashable and equal", hash(back) == hash(term))
    same(f"rebuilt {label} renders identically", back.render(), term.render())

check("a whole design round-trips to the same fingerprint",
      fingerprint(from_data(json.loads(serialize(list(BASELINE_TERMS)))))
      == fingerprint(BASELINE_TERMS))

try:
    from_data(["Rhombus", {"a": 1}])
    check("from_data rejects an unknown term type", False)
except (KeyError, TypeError, ValueError):
    check("from_data rejects an unknown term type", True)

# The real integration check: every design ever measured must be recoverable
# from its own log row. Skipped rather than failed when the log is empty, so a
# fresh clone still passes.
_logged = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]     if LOG_PATH.exists() else []
if _logged:
    _rebuilt = sum(fingerprint(from_data(json.loads(r["terms_json"]))) == r["cover_fingerprint"]
                   for r in _logged)
    check(f"all {len(_logged)} logged designs rebuild with a matching fingerprint",
          _rebuilt == len(_logged))

# Python randomizes string hashing per process, so a key that is stable inside
# one kernel can still differ between runs. Only a second process can see that.
_snippet = (
    "import sys; sys.path.insert(0, r'%s');"
    "from oracle.terms import fingerprint;"
    "from oracle.baseline import BASELINE_TERMS;"
    "print(fingerprint(BASELINE_TERMS))" % ROOT
)
_other = subprocess.run([sys.executable, "-c", _snippet], capture_output=True, text=True)
check("fingerprint is stable across processes",
      _other.returncode == 0 and _other.stdout.strip() == fingerprint(BASELINE_TERMS))

# --------------------------------------------------------------------------

print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("failures:", ", ".join(FAILED))
sys.exit(1 if FAILED else 0)
