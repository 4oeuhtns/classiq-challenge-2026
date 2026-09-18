"""Turn a SAT-synthesised XAG chain into a reversible circuit, and measure it.

The chain from mc_exact.py is a list of AND gates whose two inputs are each an XOR
of earlier signals.  The XOR parts are linear and therefore free of ancilla: pick
one signal of the subset as a pivot and XOR the rest into it in place, use it, then
XOR them back.  Only the AND outputs need ancilla, one each, so a chain with g AND
gates needs g ancilla plus nothing else.

That matters because ancilla pressure -- not gate count -- is what serialises the
current design: the AND-tree approach spends 79% of its gate activity on scratch.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


def fold(qc, sig, subset):
    """XOR a subset of signals onto a pivot qubit, in place.

    Signal 0 is the constant 1, which needs no qubit -- XOR-ing it in is just an X
    on the pivot.  Returns (pivot qubit, undo list) so the caller can fold back.
    """
    real = [j for j in subset if j != 0]
    negate = len(real) != len(subset)
    if not real:                       # the subset is the bare constant 1
        return None, [], negate
    pivot, rest = real[0], real[1:]
    for j in rest:
        qc.cx(sig[j], sig[pivot])
    if negate:
        qc.x(sig[pivot])
    return sig[pivot], rest, negate


def unfold(qc, sig, pivot, rest, negate):
    if pivot is None:
        return
    if negate:
        qc.x(pivot)
    for j in reversed(rest):
        qc.cx(sig[j], pivot)


def emit_chain(qc, chain, inputs, gate_qubits):
    """Compute the chain's AND gates into gate_qubits. Returns the signal->qubit map.

    `inputs` are the n data qubits (signals 1..n); signal 0 is the constant 1 and
    costs no qubit.  Everything except gate_qubits is restored on exit.
    """
    A_sets, B_sets, _ = chain
    if len(gate_qubits) < len(A_sets):
        raise ValueError(f"chain has {len(A_sets)} AND gates but only "
                         f"{len(gate_qubits)} ancilla to hold their outputs -- "
                         f"an XAG chain needs one ancilla per AND gate")
    sig = [None] + list(inputs) + list(gate_qubits)
    n = len(inputs)

    for i, (Ai, Bi) in enumerate(zip(A_sets, B_sets)):
        qa, ra, na = fold(qc, sig, Ai)
        qb, rb, nb = fold(qc, sig, Bi)
        out = sig[n + 1 + i]
        if qa is None:                 # a_i is the constant 1, so t_i = b_i
            qc.cx(qb, out)
        elif qb is None:
            qc.cx(qa, out)
        else:
            qc.rccx(qa, qb, out)
        unfold(qc, sig, qb, rb, nb)
        unfold(qc, sig, qa, ra, na)
    return sig


def mask_phase_circuit(chain, n_and, X=range(6), n_q=18, anc_start=12):
    """Apply (-1)^f on the 6 data qubits, where f is what the chain computes.

    Layout: the AND outputs occupy `n_and` ancilla starting at `anc_start`.  The
    constant-1 signal costs no qubit.  Everything is uncomputed, so all ancilla
    return to |0>.
    """
    inputs = list(X)
    gate_q = list(range(anc_start, anc_start + n_and))
    qc = QuantumCircuit(n_q)

    fwd = QuantumCircuit(n_q)
    sig = emit_chain(fwd, chain, inputs, gate_q)
    qc.compose(fwd, inplace=True)

    # phase on the output XOR: fold the C-subset onto a pivot, Z it, fold back
    pivot, rest, negate = fold(qc, sig, chain[2])
    if pivot is None:                                # f is the constant 1
        qc.global_phase += np.pi
    else:
        qc.z(pivot)
    unfold(qc, sig, pivot, rest, negate)

    qc.compose(fwd.inverse(), inplace=True)          # literal inverse: rccx phases cancel
    return qc


def verify_mask_phase(qc, mask, X=range(6), n_q=18, anc_start=12):
    """Exact check: the circuit must apply -1 exactly on the mask, ancilla clean."""
    full = QuantumCircuit(n_q)
    for q in X:
        full.h(q)
    full.compose(qc, inplace=True)
    amp = np.asarray(Statevector(full).data).reshape(-1, 64)
    want = np.array([-1.0 if (mask >> v) & 1 else 1.0 for v in range(64)]) / 8.0
    overlap = np.vdot(want, amp[0])
    if abs(overlap) < 1e-12:
        return float("inf"), float(np.abs(amp[1:]).max())
    phase = overlap / abs(overlap)
    return float(np.abs(amp[0] - phase * want).max()), float(np.abs(amp[1:]).max())


def score(qc):
    t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3, seed_transpiler=0)
    return t.depth(), t.count_ops().get("cx", 0)
