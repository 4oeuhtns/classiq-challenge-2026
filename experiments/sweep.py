"""Phase 1.4 -- a designed experiment to price each construct directly.

The calibration set answers "do these features rank designs correctly?" It
cannot answer "what does one control cost?", because almost every design in
it is Rect-dominated at exactly four controls per term. Cost-per-control and
cost-per-term are confounded there: nothing varies one while holding the
other still.

So this file does the opposite. Every group below changes **one thing** and
holds everything else fixed, which means the answer comes out by subtraction
-- no regression, no fitting, no held-out set needed. Roughly 30 designs.

These are deliberately **not valid covers**. A single Rect does not draw the
logo, and it does not need to: depth does not care what a circuit draws. They
are tagged ``sweep`` so that ``fit_proxy`` can keep them out of the
calibration set, which must stay exactly what the search will produce.

    python experiments/sweep.py
    python experiments/sweep.py --dry        # list the designs, no cloud calls
"""

import argparse
import collections
import pathlib
import sys

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

from experiments.runner import run_batch
from oracle.mask import COORD_BITS, GRID_SIZE
from oracle.proxy import features
from oracle.terms import And, Cube, HalfPlane, Not, Or, Parity, Rect, Xor

R = Rect(5, 40, 12, 33)          # 4 constraining bounds -- the reference term
R2 = Rect(14, 26, 14, 26)
R3 = Rect(30, 50, 30, 50)
R4 = Rect(1, 10, 40, 60)
R5 = Rect(20, 40, 2, 10)
ALL_BITS = list(range(COORD_BITS))

C1 = Cube.fix(x={5: 1})                       # 1 fixed bit
C2 = Cube.fix(x={5: 1, 4: 0})                 # 2 bits, x axis
C2B = Cube.fix(y={5: 0, 4: 1})                # 2 bits, y axis
C3 = Cube.fix(x={5: 1, 4: 0}, y={5: 1})       # 3 bits -- the standalone ceiling
P1 = Parity.fix(x=[0])                        # 1 bit
HP_FLAT = HalfPlane(1, 0, 20)                 # x <= 20, no adder
HP_FLAT2 = HalfPlane(0, 1, 30)                # y <= 30, no adder
HP_DIAG = HalfPlane(1, 1, 60)
HP_DIAG2 = HalfPlane(1, -1, 10)
HP_DIAG3 = HalfPlane(-1, 1, 10)


def blocks(k):
    """k distinct 5x5 rects, every one with all four bounds constraining.

    Not k copies of one rect: identical terms cancel in pairs under XOR, and
    a compiler is free to collapse duplicates. Same cost each, different
    terms.
    """
    return [Rect(1 + 7 * (i % 8), 5 + 7 * (i % 8),
                 1 + 7 * (i // 8), 5 + 7 * (i // 8)) for i in range(k)]


def cube(k):
    """A Cube fixing k bits, spread across both axes from the top down."""
    x = {b: 1 for b in range(COORD_BITS - 1, COORD_BITS - 1 - (k + 1) // 2, -1)}
    y = {b: 0 for b in range(COORD_BITS - 1, COORD_BITS - 1 - k // 2, -1)}
    return Cube.fix(x=x, y=y)


SWEEPS = [
    # What does one more term cost? Controls per term held at 4 throughout.
    ("per_term", [(f"{k} rects", blocks(k)) for k in (1, 2, 4, 8, 16, 32)]),

    # What does one more control cost? One term throughout, varying how many
    # bounds actually constrain. The 0-control row is the loose end from
    # not_pair, where a whole-grid Rect scored 0 controls but was not free.
    ("per_control", [
        ("0 ctrl", [Rect(0, GRID_SIZE - 1, 0, GRID_SIZE - 1)]),
        ("1 ctrl", [Rect(0, 40, 0, GRID_SIZE - 1)]),
        ("2 ctrl", [Rect(5, 40, 0, GRID_SIZE - 1)]),
        ("3 ctrl", [Rect(5, 40, 12, GRID_SIZE - 1)]),
        ("4 ctrl", [Rect(5, 40, 12, 33)]),
    ]),

    # Fan-in past 4, which Rects cannot reach. Deltas that stay flat mean cost
    # is linear in controls; deltas that shrink mean the AND tree's height is
    # what matters, and P3 should be summing log2 rather than counts.
    # Also maps the width cliff: 8 bits was refused at 38 qubits, 1 bit is
    # cheap, and everything between is currently guesswork.
    ("cube_bits", [(f"{k} bits", [cube(k)]) for k in (1, 2, 3, 4, 5, 6)]),

    # Same two children, different wrapper. Every difference here is the
    # wrapper's own price.
    ("container", [
        ("bare R", [R]),
        ("R, R2 top-level", [R, R2]),
        ("And(R, R2)", [And((R, R2))]),
        ("Xor(R, R2)", [Xor((R, R2))]),
        ("Or(R, R2)", [Or((R, R2))]),          # expected to refuse; confirms it
        ("Not(R)", [Not(R)]),
    ]),

    # Same leaves, deeper tree. Nesting costs width because Classiq evaluates
    # sub-conditions in parallel -- this says what it costs in depth, and
    # where it stops fitting at all.
    ("nesting", [
        ("depth 2", [And((R, R2))]),
        ("depth 3", [And((R, And((R2, R3))))]),
        ("depth 4", [And((R, And((R2, And((R3, R4))))))]),
    ]),

    # Does a bound's COST depend on its VALUE? `x <= 31` is not a comparison
    # at all -- it is `bit5 == 0`, one wire read. `x <= 40` needs a real
    # comparator. If that is right, a bound landing on a power-of-two boundary
    # should cost what a Cube bit costs (~2), not what a Rect bound costs
    # (~100), and "does this bound constrain" is the wrong question to ask.
    ("threshold_le", [(f"x <= {c}", [Rect(0, c, 0, GRID_SIZE - 1)])
                      for c in (0, 1, 3, 7, 15, 31, 5, 20, 40, 47)]),
    ("threshold_ge", [(f"x >= {c}", [Rect(c, GRID_SIZE - 1, 0, GRID_SIZE - 1)])
                      for c in (8, 16, 32, 5, 21, 40)]),

    # The same idea taken to its conclusion: a Rect whose bounds are ALL
    # aligned is a Cube written differently. If the compiler sees that, these
    # should match the Cube rows above -- including refusing at 4 bits.
    ("aligned_box", [
        ("32<=x<=47  (2 bits)", [Rect(32, 47, 0, GRID_SIZE - 1)]),
        ("Cube same 2 bits", [Cube.fix(x={5: 1, 4: 0})]),
        ("32<=x<=47, 16<=y<=31  (4 bits)", [Rect(32, 47, 16, 31)]),
    ]),

    # ---------------------------------------------------------------------
    # Width limits. These groups are not cost sweeps -- every entry asks
    # "does this synthesize at all?", and the deltas between rows are
    # meaningless. They exist because `fits()` in oracle/proxy.py currently
    # generalizes from four measured refusals, and a rule that wrongly says
    # "does not fit" deletes designs from Phase 4's search space without ever
    # telling you what they would have scored. That is the expensive error.
    # ---------------------------------------------------------------------

    # Is the Cube bit limit per TERM or per EXPRESSION? A standalone 4-bit
    # Cube is refused at 20 qubits. Two 2-bit Cubes ANDed is four bits in
    # total but only two in any one term. Whichever way this lands, it decides
    # whether `fits` should count bits per term or per top-level expression.
    ("limits_cube", [
        ("And(Cube2, Cube2)  4 bits total", [And((C2, C2B))]),
        ("And(R, Cube3)", [And((R, C3))]),
        ("And(R, Cube1)", [And((R, C1))]),
    ]),

    # Does fan-in cost width separately from nesting depth? All of these are
    # depth 2. Phase 3's hand-built AND tree is a wide tree if these fit and a
    # chain if they do not.
    ("limits_arity", [
        ("And x2  (measured: 398)", [And((R, R2))]),
        ("And x3", [And((R, R2, R3))]),
        ("And x4", [And((R, R2, R3, R4))]),
        ("And x5", [And((R, R2, R3, R4, R5))]),
    ]),

    # The only depth-3 shape ever measured is And(R, And(R2, R3)), refused.
    # `fits` currently rejects ALL depth-3 trees on that one data point.
    # And(Not(R), R2) is the case that matters most: if it fits, the rule is
    # too strict and is silently discarding designs.
    ("limits_depth3", [
        ("And(Not(R), R2)", [And((Not(R), R2))]),
        ("And(R, Not(R2))", [And((R, Not(R2)))]),
        ("Not(And(R, R2))", [Not(And((R, R2)))]),
        ("Xor(R, And(R2, R3))", [Xor((R, And((R2, R3))))]),
    ]),

    # Not has only ever been applied to a Rect (987). Does it work on the
    # cheap terms, and does it stay a ~2.5x multiplier or become a flat cost?
    ("limits_not", [
        ("Not(Cube1)", [Not(C1)]),
        ("Not(Parity1)", [Not(P1)]),
        ("Not(HalfPlane)", [Not(HP_FLAT)]),
    ]),

    # Xor costs 2.4x on Rects. With almost-free children, is the multiplier
    # still 2.4, or is the container cost fixed regardless of what is inside?
    # Also the only test of a nested Parity outside an And.
    ("limits_xor", [
        ("Xor(Cube1, Cube2)", [Xor((C1, C2B))]),
        ("Xor(R, Cube1)", [Xor((R, C1))]),
        ("Xor(R, Parity1)", [Xor((R, P1))]),
    ]),

    # Or(Rect, Cube) was refused at 21 -- only three qubits over budget. With
    # two genuinely cheap children it might fit, which would put a construct
    # currently written off as unusable back on the table.
    ("limits_or", [
        ("Or(Cube1, Cube2)", [Or((C1, C2B))]),
        ("Or(Parity1, Cube1)", [Or((P1, C1))]),
    ]),

    # Four half-planes ANDed need 38 qubits (the diamond). Two has never been
    # tried, and two is what an octagon corner actually needs. HP_FLAT costs
    # no adder, so it is the cheapest possible version of the question.
    ("limits_halfplane", [
        ("And(flat, flat)   no adders", [And((HP_FLAT, HP_FLAT2))]),
        ("And(diag, diag)   2 adders", [And((HP_DIAG, HP_DIAG2))]),
        ("And(diag x3)", [And((HP_DIAG, HP_DIAG2, HP_DIAG3))]),
        ("And(R, diag)      clipped diagonal", [And((R, HP_DIAG))]),
    ]),

    # Gaps left by the groups above, each one sitting between a measured
    # "fits" and a measured "refuses". Every entry decides one clause of
    # fits() that is currently inference rather than evidence.
    #
    # NOTE on the Xor-of-And cases: the first attempt used Rect(14,26,14,26)
    # and Rect(30,50,30,50), which are DISJOINT -- so And(R2,R3) folded to
    # False, Xor(R, False) folded to R, and the 398 that came back was just a
    # bare Rect. A degenerate sub-expression silently answers a different
    # question. R and R2 overlap (R2 sits inside R), so And((R, R2)) is a real
    # non-empty region.
    ("limits_gaps", [
        # rule: "And(Rect, Cube[k])" -- k=1 fits (404), k=3 refuses. Where?
        ("And(R, Cube2)", [And((R, C2))]),

        # rule: "And with a container child refuses" -- does Xor share it?
        ("Xor(R3, And(R, R2))", [Xor((R3, And((R, R2))))]),
        ("Xor(Not(R), R2)", [Xor((Not(R), R2))]),

        # Not(And(R,R2)) fits at 889. Does Not tolerate any container?
        ("Not(Xor(R, R2))", [Not(Xor((R, R2)))]),

        # rule: ">=3 children refuses" -- measured for And only.
        ("Xor x3", [Xor((R, R2, R3))]),

        # rule: "4 Cube bits in one expression refuses" -- measured under And.
        # Is the budget the expression's, or the And's?
        ("Xor(Cube2, Cube2)  4 bits", [Xor((C2, C2B))]),

        # And(flat,flat) fits at 228, And(diag,diag) refuses. One adder?
        ("And(flat, diag)   1 adder", [And((HP_FLAT, HP_DIAG))]),
    ]),

    # The asymmetry that P2 exists to capture, at both extremes. A standalone
    # Parity is phases multiplying, so bits should be free; nested it is a
    # value that must be computed, so bits should cost.
    ("parity", [
        ("1 bit, top", [Parity.fix(x=[0])]),
        ("12 bits, top", [Parity.fix(x=ALL_BITS, y=ALL_BITS)]),
        ("1 bit, nested", [And((R, Parity.fix(x=[0])))]),
        ("2 bits, nested", [And((R, Parity.fix(x=[0, 1])))]),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    designs, index = [], []
    for group, entries in SWEEPS:
        for label, terms in entries:
            designs.append((terms, ["sweep", f"{group}:{label}"]))
            index.append((group, label))

    if args.dry:
        for (group, label), (terms, _) in zip(index, designs):
            print(f"{group:<14}{label:<32}n={len(terms):<3}"
                  f"controls={sum(features(terms)['controls'])}")
        print(f"\n--dry: {len(designs)} designs, stopping before any cloud call.")
        return 0

    # skip_measured=False so a re-run still returns the whole table from cache
    results = run_batch(designs, hours=args.hours, skip_measured=False)

    by = collections.defaultdict(list)
    for (group, label), (terms, _, m) in zip(index, results):
        by[group].append((label, terms, m))

    print("\n" + "=" * 74)
    print("Each group varies ONE thing. Read the deltas, not the depths.")
    print("=" * 74)
    for group, _ in SWEEPS:
        print(f"\n{group}")
        previous = None
        for label, terms, m in by[group]:
            controls = sum(features(terms)["controls"])
            if m is None:
                print(f"  {label:<32} n={len(terms):<3} ctrls={controls:<4} refused")
                previous = None
                continue
            delta = f"{m.depth - previous:+6d}" if previous is not None else "      "
            print(f"  {label:<32} n={len(terms):<3} ctrls={controls:<4} "
                  f"depth={m.depth:<7} cx={m.cx:<7} {delta}")
            previous = m.depth
    return 0


if __name__ == "__main__":
    sys.exit(main())
