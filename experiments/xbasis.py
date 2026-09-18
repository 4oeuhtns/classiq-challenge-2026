"""Relabel the X register by a GF(2)-linear map to make the shells cheap.

The nested-shell decomposition's x-masks are thin mirror-symmetric pairs about the
disk centres: {32,48}, {33,47}, {36,37,43,44}, ... A pair is a single cheap cube
exactly when its two elements differ in one bit, and which bits they differ in is
something a linear relabeling can change: L(u) XOR L(v) = L(u XOR v), so choosing L
to send the pair differences to low-weight vectors turns scattered singletons into
one-dimensional cubes.

The earlier basis search (findings §6.1, 1.00x on Y and 1.08x on X) ran against the
*original* piece masks and asked whether dyadic-block cost improved. These are
different masks and a different cost, so it is a different question.

The relabeling costs one CNOT network before the x work and its inverse after --
a few CX against a ~689 CX x-side. It commutes with the entire y side, which never
touches X.
"""

import random

from qiskit import QuantumCircuit

from handbuilt import X, cx_mcz
from cubes import all_cubes

MCZ_COST = lambda n: cx_mcz(n + 1)


# ------------------------------------------------------------ linear algebra ---
def invertible(M):
    A, rank = list(M), 0
    for b in range(6):
        p = next((i for i in range(rank, 6) if (A[i] >> b) & 1), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        for i in range(6):
            if i != rank and (A[i] >> b) & 1:
                A[i] ^= A[rank]
        rank += 1
    return rank == 6


def apply_vec(M, v):
    """Row i of M selects which input bits XOR into output bit i."""
    w = 0
    for i in range(6):
        if bin(M[i] & v).count("1") & 1:
            w |= 1 << i
    return w


def apply_mask(M, mask):
    out = 0
    m = mask
    while m:
        b = (m & -m).bit_length() - 1
        out |= 1 << apply_vec(M, b)
        m &= m - 1
    return out


def linear_circuit(M, reg):
    """CX network transforming the register from v to M v. Verified by caller."""
    A = list(M)
    ops = []
    for b in range(6):                      # reduce A to the identity, recording ops
        p = next((i for i in range(b, 6) if (A[i] >> b) & 1), None)
        if p is None:
            raise ValueError("singular")
        if p != b:                          # swap rows via three row-additions
            A[b] ^= A[p]
            ops.append((p, b))
            A[p] ^= A[b]
            ops.append((b, p))
            A[b] ^= A[p]
            ops.append((p, b))
        for i in range(6):
            if i != b and (A[i] >> b) & 1:
                A[i] ^= A[b]
                ops.append((b, i))
    qc = QuantumCircuit(18)
    for src, dst in reversed(ops):          # undo the elimination on the register
        qc.cx(reg[src], reg[dst])
    return qc


# ------------------------------------------------------------------- search ---
def pair_table():
    """{mask: cost} for every mask expressible as at most two cubes."""
    tab = {0: 0}
    cubes = all_cubes()
    for m, fixed, _ in cubes:
        c = MCZ_COST(bin(fixed).count("1"))
        if m not in tab or c < tab[m]:
            tab[m] = c
    for i in range(len(cubes)):
        m1, f1, _ = cubes[i]
        c1 = MCZ_COST(bin(f1).count("1"))
        for j in range(i + 1, len(cubes)):
            m2, f2, _ = cubes[j]
            m = m1 ^ m2
            c = c1 + MCZ_COST(bin(f2).count("1"))
            if m not in tab or c < tab[m]:
                tab[m] = c
    return tab


def search(shells, tab, dyadic_cost, restarts=80, steps=600, seed=11):
    """Hill-climb over invertible relabelings, scoring by the <=2-cube table.

    Scoring with the pair table is an O(1) lookup, which is what makes the search
    affordable; the winner is re-costed with the full <=4-cube cover afterwards.
    """
    rng = random.Random(seed)
    ident = [1 << i for i in range(6)]
    BIG = 10 ** 6

    def score(M):
        t = 0
        for s in shells:
            sm = apply_mask(M, s)
            t += min(tab.get(sm, BIG), dyadic_cost(sm))
        return t

    best = (score(ident), ident)
    for r in range(restarts):
        if r == 0:
            M = list(ident)
        else:
            while True:
                M = [rng.randrange(64) for _ in range(6)]
                if invertible(M):
                    break
        cur = score(M)
        for _ in range(steps):
            i, j = rng.randrange(6), rng.randrange(6)
            if i == j:
                continue
            N = list(M)
            N[i] ^= N[j]
            if not invertible(N):
                continue
            c = score(N)
            if c <= cur:
                M, cur = N, c
            if cur < best[0]:
                best = (cur, list(M))
    return best
