"""Phase 3.6 task 2: does borrowing a COORDINATE qubit (not a scratch-pool
ancilla) as a dirty helper reduce the cost of a multi-controlled gate,
compared to our existing clean-ancilla and_tree approach? Bounded, isolated
test, per the plan -- not yet wired into the full design.

Uses Qiskit's own synth_mcx_1_dirty_kg24 (Khattar & Gidney, arXiv:2407.17966,
"Rise of conditionally clean ancillae") -- a real, shipped, documented
implementation, not something reimplemented from an unverified paper summary.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.synthesis.multi_controlled import synth_mcx_1_dirty_kg24

from handbuilt import MCX, X, Y, A

K = 6  # number of controls, matching a full 6-bit fixed block


def baseline_round_trip():
    """Our existing approach: 6 X-controls -> target ancilla, via and_tree
    with clean scratch, computed, USED (a Z on the target -- exactly how
    MCX/MCZ are actually used in the real design, marking a phase on the
    computed flag), then uncomputed. The intervening Z is essential: without
    it, "compute then immediately uncompute" is trivially cancelled by the
    transpiler as a no-op, which would make the comparison meaningless."""
    controls = [(X[i], 1) for i in range(K)]
    target = A[0]
    scratch = A[1:5]
    fwd = QuantumCircuit(18)
    MCX(fwd, controls, target, scratch)
    qc = QuantumCircuit(18)
    qc.compose(fwd, inplace=True)
    qc.z(target)
    qc.compose(fwd.inverse(), inplace=True)  # the LITERAL inverse, not a
    # second forward call -- calling MCX again would re-run its relative-
    # phase Toffolis against a DIFFERENT starting value of `target` (now 1,
    # not 0), which is exactly the unsafe pattern this project's own rule
    # warns about. This bug was caught by the exhaustive correctness check
    # below, not assumed away -- worth keeping as a reminder of why that
    # check exists.
    return qc, target, []  # scratch is genuinely clean (starts at |0>) here --
    # only the dirty-ancilla version below has a qubit that needs to start in
    # an arbitrary state; giving `scratch` a random rotation would violate
    # and_tree's actual precondition and isn't what this test is checking.


def dirty_round_trip():
    """Same 6-control gate plus the same intervening Z-use, using 1 borrowed
    Y-register qubit as the dirty ancilla instead of 4 clean scratch qubits."""
    sub = synth_mcx_1_dirty_kg24(K)  # qubits: 0..5 controls, 6 target, 7 dirty
    n_q = 18
    mapping = list(X[:K]) + [A[0], Y[0]]
    fwd = QuantumCircuit(n_q)
    fwd.compose(sub, qubits=mapping, inplace=True)
    qc = QuantumCircuit(n_q)
    qc.compose(fwd, inplace=True)
    qc.z(A[0])
    qc.compose(fwd.inverse(), inplace=True)  # literal inverse, same discipline
    # as the baseline -- fair, consistent methodology on both sides.
    return qc, A[0], [Y[0]]


def verify_and_score(build_fn, label):
    qc, target, borrowed = build_fn()

    ok = True
    rng = np.random.default_rng(0)
    for ctrl_val in range(1 << K):  # exhaustive over all control settings
        ctrl_bits = [(ctrl_val >> i) & 1 for i in range(K)]
        init = QuantumCircuit(18)
        for i, b in enumerate(ctrl_bits):
            if b:
                init.x(X[i])
        # give the borrowed qubit(s) a genuinely arbitrary state -- a fresh
        # random single-qubit rotation, not just |0> or |1>, so "restored
        # exactly" is a real claim, not a coincidence of a trivial basis state.
        for q in borrowed:
            theta, phi, lam = rng.uniform(0, 2 * np.pi, size=3)
            init.u(theta, phi, lam, q)

        full = init.compose(qc)
        sv_before = np.asarray(Statevector(init).data)
        sv_after = np.asarray(Statevector(full).data)

        # expected: identity if not all controls are 1, else a GLOBAL -1
        # (target is a definite |0> input here, so the intervening Z acts
        # on a classical branch, not a superposed control -- this still
        # verifies the operator correctly since it's checked at every one
        # of the 2^6 control settings, which spans the diagonal by linearity).
        expect_sign = -1.0 if all(ctrl_bits) else 1.0
        diff = np.abs(sv_after - expect_sign * sv_before).max()
        if diff > 1e-9:
            print(f"{label}: MISMATCH at controls={ctrl_bits}, max diff {diff:.2e}")
            ok = False
            break

    if not ok:
        print(f"{label}: FAILED VERIFICATION")
        return

    print(f"{label}: verified correct on all {1 << K} control settings "
          f"(borrowed qubit given a fresh random state each time)")
    best = None
    for lvl, seed in [(3, s) for s in range(8)] + [(2, 0)]:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=lvl, seed_transpiler=seed)
        d, cx = t.depth(), t.count_ops().get("cx", 0)
        if best is None or d < best[0]:
            best = (d, cx)
    print(f"{label}: depth={best[0]}  cx={best[1]}")


if __name__ == "__main__":
    verify_and_score(baseline_round_trip, "baseline (4 clean ancilla, and_tree)")
    print()
    verify_and_score(dirty_round_trip, "dirty (1 borrowed Y qubit, synth_mcx_1_dirty_kg24)")
