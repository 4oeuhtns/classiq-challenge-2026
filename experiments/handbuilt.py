"""A gate-level phase oracle, built by hand instead of through Classiq predicates.

    depth 1906, cx 1251  (Qiskit, u3/cx basis, verified exactly)
    vs the 18-rect baseline's 5267 / 3512.

Architecture: ten rank-1 pieces. For each, [y in Y] is computed into one
ancilla, then the x pattern is applied as multi-controlled Z gates using that
ancilla as an extra control, then the predicate is uncomputed. The ancilla
decouples the axes -- a 12-control gate becomes a 6-control plus a 7-control --
and amortises the y side over every x block in the piece.

Three things that worked, largest first:
  1. the ancilla decoupling above
  2. Margolus (relative-phase) Toffolis inside every compute/uncompute pair:
     3 CX instead of 6. Valid ONLY because the ancilla is used solely as a
     control in between and the uncompute is the literal inverse. Uncomputing
     by repeating the block doubles the spurious phase instead of cancelling
     it -- a silent wrong answer that only a statevector check catches.
  3. balanced AND trees instead of v-chains: same gates, depth log(k) not k.
     Worth 2311 -> 1989 on its own.
  4. choosing the rank-10 basis by search rather than taking the natural one.

Measured dead ends, all verified correct and all worse:
  disjoint quadtree cubes            175 cubes   9181 CX est
  optimal XOR of aligned rects (DP)   70 rects   3232 CX est
  greedy ESOP over all 3^12 cubes     61 cubes   2701 CX est
  cubes instead of dyadic blocks                 1989 -> 2261 depth
  chaining the ancilla between pieces            1918 -> 2011 depth
  split scratch pools (parallel x/y)             1918 -> 2477 depth
  arity penalties, per-piece axis swap, merging pieces: no change

Why cubes lose although the model says they are 12% cheaper: consecutive
dyadic blocks share control prefixes, so the transpiler cancels gates between
them. Scattered cube controls share nothing.

Why the parallel split loses: two concurrent pipelines need two targets plus
scratch each, which caps a block at 3 fixed bits. A 3-fixed-bit block spans 8
cells, and the patterns here do not decompose into 8-cell blocks exactly.
With 6 ancillas a single 6-control AND tree already wants 5 scratch, so there
is no room for a second pipeline. That is the hard limit of this design.
"""

import numpy as np
from functools import lru_cache
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

NB, X, Y, A = 6, list(range(0,6)), list(range(6,12)), list(range(12,18))

def logo_pixel(x, y):
    return ((2 <= x <= 26 and 29 <= y <= 53) or (26 <= x <= 49 and 39 <= y <= 43)
        or (x-55)**2 + (y-41)**2 <= 42 or (x-40)**2 + (y-19)**2 <= 72)
TARGET = np.array([[logo_pixel(x,y) for x in range(64)] for y in range(64)])

def cx_mcz(k): return 0 if k <= 1 else (1 if k == 2 else 6*(k-2) + 1)
def cx_mcx(k): return 0 if k == 0 else (1 if k == 1 else (3 if k == 2 else 6*k - 9))

def mask_of(ivs):
    m = 0
    for lo, hi in ivs: m |= ((1 << (hi-lo+1)) - 1) << lo
    return m

def blk(start, size, reg):
    nf = NB - size.bit_length() + 1
    return [(reg[NB-1-i], (start >> (NB-1-i)) & 1) for i in range(nf)]

def best_blocks(mask, cost):
    @lru_cache(maxsize=None)
    def dp(level, index, flipped):
        size = 64 >> level; start = index*size
        bits = (mask >> start) & ((1 << size) - 1)
        if bits == (((1 << size) - 1) if flipped else 0): return (0, ())
        if level == NB: return (cost(NB), ((start, 1),))
        la, lb = dp(level+1, index*2, flipped), dp(level+1, index*2+1, flipped)
        keep = (la[0]+lb[0], la[1]+lb[1])
        fa, fb = dp(level+1, index*2, 1-flipped), dp(level+1, index*2+1, 1-flipped)
        flip = (cost(level)+fa[0]+fb[0], ((start, size),)+fa[1]+fb[1])
        return min(keep, flip, key=lambda t: t[0])
    return dp(0, 0, 0)[1]

def and_tree(qc, qs, scratch):
    cur, free = list(qs), list(scratch)
    while len(cur) > 1:
        nxt, i = [], 0
        while i+1 < len(cur):
            t = free.pop(0); qc.rccx(cur[i], cur[i+1], t); nxt.append(t); i += 2
        if i < len(cur): nxt.append(cur[i])
        cur = nxt
    return cur[0]

def MCZ(qc, controls, scratch):
    qs = [q for q,_ in controls]; zeros = [q for q,v in controls if v == 0]
    for q in zeros: qc.x(q)
    if len(qs) == 1: qc.z(qs[0])
    elif len(qs) == 2: qc.cz(qs[0], qs[1])
    else:
        sub = QuantumCircuit(18); top = and_tree(sub, qs[:-1], scratch)
        qc.compose(sub, inplace=True); qc.cz(top, qs[-1]); qc.compose(sub.inverse(), inplace=True)
    for q in zeros: qc.x(q)

def MCX(qc, controls, target, scratch):
    qs = [q for q,_ in controls]; zeros = [q for q,v in controls if v == 0]
    for q in zeros: qc.x(q)
    if len(qs) == 0: qc.x(target)
    elif len(qs) == 1: qc.cx(qs[0], target)
    elif len(qs) == 2: qc.rccx(qs[0], qs[1], target)
    else:
        sub = QuantumCircuit(18); top = and_tree(sub, qs[:-1], scratch)
        qc.compose(sub, inplace=True); qc.rccx(qs[-1], top, target); qc.compose(sub.inverse(), inplace=True)
    for q in zeros: qc.x(q)

def check(qc):
    full = QuantumCircuit(18)
    for q in X+Y: full.h(q)
    full.compose(qc, inplace=True)
    amp = np.asarray(Statevector(full).data).reshape(64,64,64)
    want = np.where(TARGET, -1.0, 1.0)/64.0
    return float(np.abs(amp[0]-want).max()), float(np.abs(amp[1:]).max())

def score(qc, tries=((2,0),(3,0),(3,1),(3,2))):
    best = None
    for lvl, seed in tries:
        t = transpile(qc, basis_gates=["u3","cx"], optimization_level=lvl, seed_transpiler=seed)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or d < best[0]: best = (d, c)
    return best


PIECES = [
    (140668768878592, 2304717109306851328),
    (70437463654400, 1154047404513689600),
    (17042430230528, 2306968908986318848),
    (67112960, 35115652612096),
    (16744448, 281466386776064),
    (50356224, 140720308486144),
    (4063232, 281479271677952),
    (18014397972611072, 134217724),
    (140771848093696, 279223176896970752),
    (134219776, 8521215115264),
]

def build(pieces=PIECES):
    qc = QuantumCircuit(18); ay, scr = A[0], A[1:]
    for ym, xm in pieces:
        if not xm or not ym: continue
        c = QuantumCircuit(18)
        for s, sz in best_blocks(ym, cx_mcx): MCX(c, blk(s, sz, Y), ay, scr)
        qc.compose(c, inplace=True)
        for s, sz in best_blocks(xm, lambda k: cx_mcz(k+1)):
            MCZ(qc, [(ay, 1)] + blk(s, sz, X), scr)
        qc.compose(c.inverse(), inplace=True)
    return qc


if __name__ == "__main__":
    qc = build()
    err, leak = check(qc)
    d, cx = score(qc, tries=[(3, s) for s in range(8)] + [(2, 0)])
    print(f"phase error {err:.1e}   ancilla leak {leak:.1e}")
    print(f"depth {d}   cx {cx}   (gate depth {qc.depth()}, {qc.count_ops().get('rccx',0)} rccx)")
