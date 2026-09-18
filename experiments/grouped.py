"""Optimise the y/x trade by grouping levels, instead of picking one extreme.

shells.py chains the whole chain: the y-masks become nested thresholds (cheap, one
flag walk) but the x-masks become thin annular *shells* (expensive).  handbuilt.py
does the opposite: y-masks are disjoint bands (expensive, rebuilt each time) but
x-masks are the full nested sets (cheap).  Neither is obviously right, and they are
the two endpoints of one family.

Partition a chain's levels into consecutive groups.  Within a group {a..b}, using
P_j = P_{a-1} XOR (S_a XOR ... XOR S_j):

    sum_{j=a..b} B_j P_j  =  [a <= lev <= b] * P_{a-1}  XOR  sum_{i=a..b} [i <= lev <= b] * S_i

The `[i <= lev <= b]` masks are nested in i, so one flag walks them, XOR-ing in one
band per step -- shells.py's trick, but confined to the group.  A singleton group is
just the direct term B_a * P_a, which is handbuilt.py's form.  Searching all 2^(n-1)
groupings per chain therefore searches between the two designs and past both.
"""

from functools import lru_cache
from itertools import product

from qiskit import QuantumCircuit

from handbuilt import MCZ, X, Y, A, cx_mcx, cx_mcz
from shells import _cover, MCX_exact, chains, check, score

MCZ_COST = lambda n: cx_mcz(n + 1)


@lru_cache(maxsize=None)
def ycost(mask):
    return sum(cx_mcx(len(c)) for c in _cover(mask, Y, cx_mcx, "mcx", True))


@lru_cache(maxsize=None)
def xcost(mask):
    return sum(MCZ_COST(len(c)) for c in _cover(mask, X, MCZ_COST, "mcz", True))


def chain_data():
    """Per chain: bands B_j (y-masks), full sets P_j, shells S_j -- all 1-indexed."""
    chs, groups = chains()
    out = []
    for ch in chs:
        P, B, S, prev = [0], [0], [0], 0
        for r in ch:
            P.append(r)
            S.append(r ^ prev)
            prev = r
            m = 0
            for y in groups[r]:
                m |= 1 << y
            B.append(m)
        out.append((P, B, S))
    return out


def groupings(n):
    """All partitions of 1..n into consecutive groups, as lists of (a, b)."""
    for cuts in product([0, 1], repeat=n - 1):
        out, a = [], 1
        for i, c in enumerate(cuts, start=1):
            if c:
                out.append((a, i))
                a = i + 1
        out.append((a, n))
        yield out


def group_plan(P, B, S, a, b):
    """Steps for one group: (y-delta to XOR into the flag, x-mask to apply | None).

    The trailing entry with x-mask None is the flag clear.
    """
    if a == b:                                   # direct term B_a * P_a
        return [(B[a], P[a]), (B[a], None)]
    span = 0                                     # [a <= lev <= b]
    for i in range(a, b + 1):
        span |= B[i]
    steps, live = [], 0
    for i in range(b, a - 1, -1):                # walk inward: [b..b], [b-1..b], ...
        steps.append((live ^ (live | B[i]), S[i]))
        live |= B[i]
    if a > 1:                                    # the P_{a-1} term rides on the span
        steps.append((0, P[a - 1]))
    steps.append((live, None))                   # clear
    assert live == span
    return steps


def plan_cost(steps):
    return sum(ycost(d) for d, _ in steps) + sum(xcost(m) for _, m in steps if m)


def best_plan():
    """Cheapest grouping per chain, by the cube-aware model."""
    plans, total = [], 0
    for P, B, S in chain_data():
        n = len(P) - 1
        best = None
        for g in groupings(n):
            steps = [s for a, b in g for s in group_plan(P, B, S, a, b)]
            c = plan_cost(steps)
            if best is None or c < best[0]:
                best = (c, g, steps)
        plans.append(best)
        total += best[0]
    return plans, total


def build(plans, order=(1, 0)):
    qc = QuantumCircuit(18)
    flag, scr = A[0], A[1:]
    for idx in order:
        if idx >= len(plans):
            continue
        for delta, xmask in plans[idx][2]:
            for c in _cover(delta, Y, cx_mcx, "mcx", True):
                MCX_exact(qc, c, flag, scr)
            if xmask:
                for c in _cover(xmask, X, MCZ_COST, "mcz", True):
                    MCZ(qc, [(flag, 1)] + c, scr)
    return qc


if __name__ == "__main__":
    plans, total = best_plan()
    for i, (c, g, _) in enumerate(plans):
        print(f"chain {i}: best grouping {g}  model cost {c}")
    print(f"model total {total}   (shells.py all-chained, handbuilt all-singleton)")
    print()
    for order in ((0, 1), (1, 0)):
        qc = build(plans, order)
        err, leak = check(qc)
        if max(err, leak) > 1e-9:
            print(f"order {order}: CORRECTNESS FAILED {err:.1e} {leak:.1e}")
            continue
        d, cx = score(qc, seeds=range(12))
        print(f"grouped, order {order} : depth {d}   cx {cx}")
    print()
    print("shells.py   : depth 1738  cx 1171")
    print("handbuilt   : depth 1923  cx 1251")
    print("leader      : depth 129   cx 614")
