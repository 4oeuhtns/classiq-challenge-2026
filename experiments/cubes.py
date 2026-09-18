"""Exact minimum-cost XOR-of-cubes cover for a 6-variable mask.

`best_blocks` only considers *dyadic* blocks -- high bits fixed, low bits free.
That is a bad fit for the nested-shell decomposition, whose shells are thin
mirror-symmetric annuli: {32,48} differs in one bit and is a single cube costing
one 5-control gate, but as dyadic blocks it is two singletons costing two
6-control gates.

A cube is any (fixed-bit set, values) pair, so there are 3^6 = 729 of them.  The
minimum-cost subset XOR-ing to the target is found exactly for up to four cubes by
meet-in-the-middle: tabulate the XOR of every subset of size <= 2 with its best
cost, then pair the table against itself.
"""

from functools import lru_cache
from itertools import combinations


@lru_cache(maxsize=None)
def all_cubes():
    """Every cube as (bitmask over the 64 values, fixed-bit set, fixed values)."""
    out = []
    for fixed in range(64):
        free = [b for b in range(6) if not (fixed >> b) & 1]
        bits = [b for b in range(6) if (fixed >> b) & 1]
        for vals in range(1 << len(bits)):
            values = 0
            for i, b in enumerate(bits):
                if (vals >> i) & 1:
                    values |= 1 << b
            m = 0
            for v in range(64):
                if v & fixed == values:
                    m |= 1 << v
            out.append((m, fixed, values))
    return out


@lru_cache(maxsize=None)
def _halves(cost_key):
    """{xor value: (cost, cubes)} over all subsets of at most two cubes."""
    cost = _COSTS[cost_key]
    cubes = all_cubes()
    tab = {0: (0, ())}
    for m, fixed, values in cubes:
        c = cost(bin(fixed).count("1"))
        if m not in tab or c < tab[m][0]:
            tab[m] = (c, ((fixed, values),))
    for (m1, f1, v1), (m2, f2, v2) in combinations(cubes, 2):
        m = m1 ^ m2
        c = cost(bin(f1).count("1")) + cost(bin(f2).count("1"))
        if m not in tab or c < tab[m][0]:
            tab[m] = (c, ((f1, v1), (f2, v2)))
    return tab


_COSTS = {}


def best_cubes(mask, cost, key):
    """Cheapest XOR-of-cubes cover of `mask` using at most four cubes.

    `key` names the cost function so the meet-in-the-middle table can be cached.
    """
    _COSTS[key] = cost
    tab = _halves(key)
    best = None
    for m, (c, cs) in tab.items():
        other = tab.get(mask ^ m)
        if other is None:
            continue
        total = c + other[0]
        if best is None or total < best[0]:
            best = (total, cs + other[1])
    return best


def cube_controls(fixed, values, reg):
    """Control list for a cube, in the (qubit, required value) form MCZ/MCX take."""
    return [(reg[b], (values >> b) & 1) for b in range(6) if (fixed >> b) & 1]
