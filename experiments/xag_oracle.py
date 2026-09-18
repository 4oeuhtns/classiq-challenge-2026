"""Build the full phase oracle from SAT-synthesised XAG chains, verify it, measure it.

Per rank-1 piece i the oracle applies (-1)^(a_i(y) * b_i(x)):
  1. compute the y-flag a_i(y) into one ancilla, using the y-chain
  2. fold the x-chain's output XOR onto a pivot and CZ it against the flag
  3. uncompute the y-flag

The point of the XAG form is ancilla pressure, not gate count.  An AND-tree MCZ
needs k-2 scratch qubits held live across the whole gate; an XAG chain needs one
ancilla per AND gate and nothing else, because the XOR parts are linear and fold
in place onto a pivot qubit.
"""

import json
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from handbuilt import PIECES, TARGET, X, Y, A
from xag_circuit import fold, unfold, emit_chain


def piece_circuit(qc, ychain, xchain, flag, scratch):
    """(-1)^(a(y) * b(x)) for one rank-1 piece, leaving every ancilla clean."""
    gy = len(ychain[0])
    gx = len(xchain[0])
    if gy > len(scratch) or gx > len(scratch):
        raise ValueError(f"chain needs {max(gy, gx)} scratch qubits, only "
                         f"{len(scratch)} available -- would need recomputation "
                         f"(pebbling) to fit")

    # --- 1. y-flag into `flag` -------------------------------------------------
    yfwd = QuantumCircuit(qc.num_qubits)
    ysig = emit_chain(yfwd, ychain, Y, scratch[:gy])
    qc.compose(yfwd, inplace=True)
    for j in ychain[2]:                       # flag ^= XOR of the output subset
        if j == 0:
            qc.x(flag)
        else:
            qc.cx(ysig[j], flag)
    qc.compose(yfwd.inverse(), inplace=True)  # literal inverse: rccx phases cancel

    # --- 2. phase, conditioned on the flag ------------------------------------
    xfwd = QuantumCircuit(qc.num_qubits)
    xsig = emit_chain(xfwd, xchain, X, scratch[:gx])
    qc.compose(xfwd, inplace=True)
    pivot, rest, negate = fold(qc, xsig, xchain[2])
    if pivot is None:                         # b(x) is the constant 1
        qc.z(flag)
    else:
        qc.cz(flag, pivot)
    unfold(qc, xsig, pivot, rest, negate)
    qc.compose(xfwd.inverse(), inplace=True)

    # --- 3. uncompute the y-flag ----------------------------------------------
    qc.compose(yfwd, inplace=True)
    for j in ychain[2]:
        if j == 0:
            qc.x(flag)
        else:
            qc.cx(ysig[j], flag)
    qc.compose(yfwd.inverse(), inplace=True)


def build(chains, n_q=18):
    qc = QuantumCircuit(n_q)
    flag, scratch = A[0], A[1:]
    for i in range(len(PIECES)):
        piece_circuit(qc, chains[f"y{i}"]["chain"], chains[f"x{i}"]["chain"],
                      flag, scratch)
    return qc


def check(qc):
    """Exact statevector check, global phase corrected as the real grader does."""
    full = QuantumCircuit(qc.num_qubits)
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


def score(qc, seeds=range(6)):
    best = None
    for s in seeds:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3,
                      seed_transpiler=s)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or (d, c) < best:
            best = (d, c)
    return best


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mc_chains.json"
    chains = json.load(open(path))
    tot_and = sum(v["and"] for v in chains.values())
    print(f"loaded {len(chains)} chains, {tot_and} AND gates total")

    qc = build(chains)
    err, leak = check(qc)
    print(f"phase error {err:.1e}   ancilla leak {leak:.1e}")
    if max(err, leak) > 1e-9:
        print("CORRECTNESS FAILED -- numbers below are meaningless")
        sys.exit(1)
    d, cx = score(qc)
    print(f"XAG oracle : depth {d}   cx {cx}")
    print(f"handbuilt  : depth 1923  cx 1251")
    print(f"leader     : depth 129   cx 614")
