"""Feed the existing best circuit (1906/1251) into PyZX's automated,
correctness-guaranteed rewriting (full_reduce), and see whether a global
rewrite pass -- one that doesn't respect our hand-built "piece" boundaries at
all -- finds anything our piece-by-piece design missed.

Correctness is never at risk here: every ZX-calculus rewrite rule is a
mathematical identity, so `full_reduce` cannot change what the circuit
computes, only its gate-level form. Verified independently anyway, via this
project's own statevector check, not just trusted.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import Statevector
import pyzx

from handbuilt import build, check, score, PIECES, X, Y, TARGET


def check_up_to_global_phase(qc):
    """Same statevector check as handbuilt.py's check(), but correcting for
    an overall global phase first -- exactly how the real oracle/verify.py
    does it (shared_global_phase = overlap / abs(overlap)), since a shared
    global phase is explicitly permitted by the challenge rules. Without
    this correction, a circuit that's exactly correct up to global phase
    reads as a false failure."""
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

CLIFFORD_T_BASIS = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z"]


def to_pyzx(qc):
    basis_qc = transpile(qc, basis_gates=CLIFFORD_T_BASIS, optimization_level=1)
    qasm = qasm2_dumps(basis_qc)
    return pyzx.Circuit.from_qasm(qasm), basis_qc


def from_pyzx(pyzx_circuit):
    qasm = pyzx_circuit.to_qasm()
    return QuantumCircuit.from_qasm_str(qasm)


if __name__ == "__main__":
    qc = build()
    err, leak = check(qc)
    d0, cx0 = score(qc, tries=[(3, s) for s in range(8)] + [(2, 0)])
    print(f"original: phase err {err:.1e} leak {leak:.1e}  depth={d0} cx={cx0}")

    zx_circ, basis_qc = to_pyzx(qc)
    print(f"clifford+T basis: depth={basis_qc.depth()} "
          f"gates={sum(basis_qc.count_ops().values())}")

    graph = zx_circ.to_graph()
    pyzx.full_reduce(graph)
    graph.normalize()
    reduced = pyzx.extract_circuit(graph)

    result_qc = from_pyzx(reduced)
    print(f"after full_reduce + extract: {sum(result_qc.count_ops().values())} gates "
          f"in {result_qc.num_qubits} qubits")

    err2, leak2 = check_up_to_global_phase(result_qc)
    print(f"re-verified independently (global-phase-corrected): "
          f"phase err {err2:.1e}  leak {leak2:.1e}")
    if max(err2, leak2) > 1e-9:
        print("CORRECTNESS FAILED -- do not trust the numbers below")
    else:
        d1, cx1 = score(result_qc, tries=[(3, s) for s in range(8)] + [(2, 0)])
        print(f"PyZX-optimized: depth={d1}  cx={cx1}  (vs original {d0}/{cx0})")
