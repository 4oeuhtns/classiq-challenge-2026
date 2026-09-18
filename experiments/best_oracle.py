"""Current best phase oracle: nested shells + cube covers + a relabelled X register.

Three stacked ideas, each measured against the one before:

  handbuilt.py, dyadic blocks                        1923 / 1251
  + exact cube covers instead of dyadic blocks       1871 / 1248
  + nested-shell chaining of the y flag              1738 / 1171
  + GF(2)-linear relabelling of X                    1569 / 1078

The relabelling is the x-side analogue of the chaining. The shells are thin
mirror-symmetric pairs about the disk centres, and a pair is one cheap cube exactly
when its elements differ in a single bit. Since L(u) XOR L(v) = L(u XOR v), a linear
relabelling chooses which bits the pairs differ in -- turning scattered singletons
into one-dimensional cubes. It costs one CNOT network plus its inverse (14 CX) and
commutes with the whole y side, which never touches X.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from handbuilt import TARGET, MCZ, X, Y, A, cx_mcx, cx_mcz
from shells import _cover, MCX_exact
from grouped import chain_data, group_plan, groupings, ycost
from xbasis import linear_circuit, apply_mask
from cubes import best_cubes

MCZ_COST = lambda n: cx_mcz(n + 1)

# found by hill-climbing over invertible 6x6 GF(2) matrices, scoring the masks with
# an exact <=2-cube table (xbasis.search). X: model cost 689 -> 496, the big win.
# Y: 646 -> 604 against 16 CX to apply, so nearly a wash -- worth 8 depth, kept.
XMAP = [33, 38, 12, 40, 48, 32]
YMAP = [1, 46, 16, 8, 20, 32]
CHAIN_ORDER = (1, 0)


def xcost_relabelled(mask, M=XMAP):
    tm = apply_mask(M, mask) if M else mask
    return sum(MCZ_COST(len(c)) for c in _cover(tm, X, MCZ_COST, "mcz", True))


def best_plan(M=XMAP):
    """Cheapest level grouping per chain, costed in the relabelled basis."""
    plans = []
    for P, B, S in chain_data():
        n = len(P) - 1
        best = None
        for g in groupings(n):
            steps = [s for a, b in g for s in group_plan(P, B, S, a, b)]
            c = (sum(ycost(d) for d, _ in steps)
                 + sum(xcost_relabelled(m, M) for _, m in steps if m))
            if best is None or c < best[0]:
                best = (c, g, steps)
        plans.append(best)
    return plans


def build(M=XMAP, ymap=YMAP, order=CHAIN_ORDER, plans=None):
    plans = plans if plans is not None else best_plan(M)
    qc = QuantumCircuit(18)
    flag, scr = A[0], A[1:]
    rx = linear_circuit(M, X) if M else None
    ry = linear_circuit(ymap, Y) if ymap else None
    for rel in (rx, ry):                              # disjoint registers, so parallel
        if rel:
            qc.compose(rel, inplace=True)
    for idx in order:
        for delta, xmask in plans[idx][2]:
            td = apply_mask(ymap, delta) if ymap else delta
            for c in _cover(td, Y, cx_mcx, "mcx", True):
                MCX_exact(qc, c, flag, scr)
            if xmask:
                tm = apply_mask(M, xmask) if M else xmask
                for c in _cover(tm, X, MCZ_COST, "mcz", True):
                    MCZ(qc, [(flag, 1)] + c, scr)
    for rel in (ry, rx):
        if rel:
            qc.compose(rel.inverse(), inplace=True)
    return qc


def check(qc):
    """Exact statevector check, global phase corrected as oracle/verify.py does."""
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


def score(qc, seeds=range(16)):
    best = None
    for s in seeds:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3,
                      seed_transpiler=s)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or (d, c) < best:
            best = (d, c)
    return best


if __name__ == "__main__":
    plans = best_plan()
    print("best grouping per chain, costed in the relabelled basis:")
    for i, (c, g, _) in enumerate(plans):
        print(f"   chain {i}: {g}   model cost {c}")
    qc = build(plans=plans)
    err, leak = check(qc)
    print(f"\nphase error {err:.2e}   ancilla leak {leak:.2e}")
    assert max(err, leak) < 1e-9, "CORRECTNESS FAILED"
    d, cx = score(qc)
    print(f"best oracle : depth {d}   cx {cx}")
    print(f"handbuilt   : depth 1923  cx 1251   ->  {1923/d:.2f}x depth, {1251/cx:.2f}x cx")
    print(f"leader      : depth 129   cx 614")
