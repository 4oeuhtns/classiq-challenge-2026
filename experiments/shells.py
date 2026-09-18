"""Nested-shell oracle: decode y's level once, then gate shells by thresholds on it.

The target has only 10 distinct nonzero rows, and they form two chains under
inclusion (disk-2's five, and the stem/disk-1/arm five).  So for each chain

    row(y) = P_{level(y)}   with   P_1 subset P_2 subset ... subset P_5

and writing S_j = P_j \\ P_{j-1} for the shells, the whole target is

    f(x,y) = XOR_j  [level(y) >= j] * S_j(x)

The point is the y side.  The current design computes ten independent y-masks and
throws each away after one use -- 62% of the circuit.  Here `[level >= j]` is a
*nested* family, so one flag ancilla walks down the chain, XOR-ing in one band per
step, and is never rebuilt.

Correctness note: the flag is NOT uncomputed by a literal inverse here, it is
chained.  So its writes must be exact Toffolis, not relative-phase ones -- an rccx
leaves a control-dependent phase that only cancels against its own inverse.  That is
the silent-wrong-answer trap in handbuilt.py's docstring, and it is why MCX_exact
below spends 6 CX on the final write instead of 3.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from handbuilt import (TARGET, X, Y, A, best_blocks, blk, cx_mcx, cx_mcz,
                       and_tree, MCZ)
from cubes import best_cubes, cube_controls


# ---------------------------------------------------------------- structure ---
def row_masks():
    """The distinct nonzero rows, as (x-bitmask, set of y values)."""
    groups = {}
    for y in range(64):
        r = 0
        for x in range(64):
            if TARGET[y][x]:
                r |= 1 << x
        if r:
            groups.setdefault(r, []).append(y)
    return groups


def chains():
    """Split the distinct rows into chains ordered by inclusion.

    Split geometrically, by which half of the grid the row lives in.  Greedy
    first-fit gets this wrong: the full-width arm row [2,61] is a superset of
    disk 2's [32,48] as well as of the stem, so it lands on whichever chain is
    tested first and drags a large scattered shell with it (chains of 6 and 4,
    and 1632 model CX against 1573 for the geometric split).
    """
    groups = row_masks()
    rows = sorted(groups, key=lambda r: bin(r).count("1"))
    lower = [r for r in rows if max(groups[r]) <= 27]      # disk 2
    upper = [r for r in rows if max(groups[r]) > 27]       # stem / disk 1 / arm
    for ch in (lower, upper):
        for j in range(1, len(ch)):
            assert ch[j - 1] & ch[j] == ch[j - 1], "chain is not nested"
    return [lower, upper], groups


def decompose():
    """(y-threshold mask, x-shell mask) terms. Verified against TARGET by caller."""
    chs, groups = chains()
    terms = []
    for ch in chs:
        prev_x = 0
        for j, r in enumerate(ch):
            shell = r ^ prev_x                 # nested, so this is r \ prev
            prev_x = r
            ymask = 0
            for rr in ch[j:]:                  # level >= j  <=>  row is at or above
                for y in groups[rr]:
                    ymask |= 1 << y
            terms.append((ymask, shell))
    return terms


def verify_decomposition(terms):
    got = np.zeros((64, 64), dtype=int)
    for ymask, xmask in terms:
        for y in range(64):
            if (ymask >> y) & 1:
                for x in range(64):
                    got[y][x] ^= (xmask >> x) & 1
    return bool((got == TARGET.astype(int)).all())


# ------------------------------------------------------------------ circuit ---
def MCX_exact(qc, controls, target, scratch):
    """MCX whose net effect is exactly a permutation -- no relative phase left.

    The AND tree is still relative-phase (it is undone by its own inverse), but the
    final write onto `target` is a true Toffoli because nothing inverts it later.
    """
    qs = [q for q, _ in controls]
    zeros = [q for q, v in controls if v == 0]
    for q in zeros:
        qc.x(q)
    if len(qs) == 0:
        qc.x(target)
    elif len(qs) == 1:
        qc.cx(qs[0], target)
    elif len(qs) == 2:
        qc.ccx(qs[0], qs[1], target)
    else:
        sub = QuantumCircuit(18)
        top = and_tree(sub, qs[:-1], scratch)
        qc.compose(sub, inplace=True)
        qc.ccx(qs[-1], top, target)
        qc.compose(sub.inverse(), inplace=True)
    for q in zeros:
        qc.x(q)


def _cover(mask, reg, cost, key, use_cubes):
    """Control lists covering `mask`, by cubes or dyadic blocks -- whichever is cheaper.

    Cubes fit the shells much better (the mirror-symmetric annuli differ in one bit,
    so {32,48} is a single 5-control cube rather than two 6-control singletons), but
    the meet-in-the-middle search only covers up to four cubes, so large contiguous
    masks still fall back to dyadic blocks.
    """
    dyadic = [blk(s, sz, reg) for s, sz in best_blocks(mask, cost)]
    if not use_cubes:
        return dyadic
    dyadic_cost = sum(cost(len(c)) for c in dyadic)
    found = best_cubes(mask, cost, key)
    if found and found[0] < dyadic_cost:
        return [cube_controls(f, v, reg) for f, v in found[1]]
    return dyadic


def build(terms=None, use_cubes=True, chain_order=(1, 0), inward=True):
    """One flag ancilla walks each chain; shells are applied as it goes.

    Defaults are the measured best of the four walk variants (1738/1171); the others
    land at 1745-1749. Clearing the flag between the two chains also beats carrying it
    across (1738 vs 1791) -- the mask separating the two chains is large and scattered,
    so the transition costs more than a fresh start.
    """
    terms = terms if terms is not None else decompose()
    qc = QuantumCircuit(18)
    flag, scr = A[0], A[1:]
    mcz_cost = lambda n: cx_mcz(n + 1)

    # group terms back into chains by their y-masks being nested
    chs, _ = chains()
    per_chain, k = [], 0
    for ch in chs:
        per_chain.append(terms[k:k + len(ch)])
        k += len(ch)

    for idx in chain_order:
        ch_terms = per_chain[idx]
        live = 0                                # y-mask currently held in `flag`
        for ymask, xmask in (reversed(ch_terms) if inward else ch_terms):
            for ctrls in _cover(live ^ ymask, Y, cx_mcx, "mcx", use_cubes):
                MCX_exact(qc, ctrls, flag, scr)
            live = ymask
            for ctrls in _cover(xmask, X, mcz_cost, "mcz", use_cubes):
                MCZ(qc, [(flag, 1)] + ctrls, scr)
        for ctrls in _cover(live, Y, cx_mcx, "mcx", use_cubes):
            MCX_exact(qc, ctrls, flag, scr)     # clear the flag for the next chain
    return qc


def check(qc):
    full = QuantumCircuit(18)
    for q in X + Y:
        full.h(q)
    full.compose(qc, inplace=True)
    amp = np.asarray(Statevector(full).data).reshape(64, 64, 64)
    want = np.where(TARGET, -1.0, 1.0) / 64.0
    overlap = np.vdot(want, amp[0])
    if abs(overlap) < 1e-12:
        return float("inf"), float(np.abs(amp[1:]).max())
    phase = overlap / abs(overlap)
    return float(np.abs(amp[0] - phase * want).max()), float(np.abs(amp[1:]).max())


def score(qc, seeds=range(8)):
    best = None
    for s in seeds:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3,
                      seed_transpiler=s)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or (d, c) < best:
            best = (d, c)
    return best


if __name__ == "__main__":
    chs, groups = chains()
    print(f"distinct nonzero rows: {sum(len(c) for c in chs)} "
          f"in {len(chs)} chains, sizes {[len(c) for c in chs]}")
    terms = decompose()
    print(f"terms: {len(terms)}   decomposition reproduces TARGET: "
          f"{verify_decomposition(terms)}")
    print()
    ym = sum(2 * sum(cx_mcx(6 - (sz.bit_length() - 1)) for s, sz in
                     best_blocks(t[0], cx_mcx)) for t in terms)
    print("model CX, y-side if each threshold were built independently:", ym)

    qc = build(terms)
    err, leak = check(qc)
    print(f"phase error {err:.1e}   ancilla leak {leak:.1e}")
    if max(err, leak) > 1e-9:
        print("CORRECTNESS FAILED")
    else:
        d, cx = score(qc)
        print(f"nested-shell oracle : depth {d}   cx {cx}")
        print(f"handbuilt (current) : depth 1923  cx 1251")
        print(f"leader              : depth 129   cx 614")
