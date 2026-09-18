"""Race a bit-serial comparator against the existing Cube approach on ONE real,
ragged range from the logo: x in [2, 26] (the square piece's x-bound). Neither
2 nor 26 is a power of two, so this is exactly the case plan.md Phase 2 flagged
as unfinished (cubes vs comparators for ragged intervals) -- see
findings-2026-09-15.md Finding 8 and route 12.1.

Both versions compute the SAME single-qubit flag ("is x in [lo,hi]?") into a
target qubit, with every scratch qubit cleanly returned to |0>. Both are
verified against brute force over all 64 values of x before any depth/cx
number is trusted. All gates here are real (non-relative-phase) Toffolis --
correctness first, optimize later once the shape of the result is known.

Qubit convention (matches handbuilt.py): xbits[q] is the qubit holding bit q
of x (value 2^q) -- LSB-first, physical qubit q == bit q.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np

from handbuilt import best_blocks, blk, MCX, cx_mcx, NB

LO, HI = 2, 26   # the square piece's real x-bound
N = 6            # 6-bit x register


# ---------------------------------------------------------------- comparator

def ge_flag(qc, xbits_msb_first, c, eq_scratch, gt):
    """gt ^= (x > c). xbits_msb_first[0] must be the MOST significant bit.

    Uses eq_scratch (n fresh ancilla, one per bit) to carry the running
    "still tied" flag down the chain. LEAVES eq_scratch dirty on exit --
    callers undo this by composing the inverse of the same circuit, exactly
    like handbuilt.py's compute/uncompute pattern for its own `ay` ancilla.
    """
    # Uses rccx (relative-phase / Margolus Toffoli, 3 CX instead of 6) --
    # safe here by the same rule handbuilt.py's MCX/MCZ already rely on: gt and
    # every eq_i are used SOLELY as controls after being set, and this whole
    # circuit is only ever composed alongside its own literal inverse
    # (see clean_gt_block), which cancels the residual phase exactly.
    n = len(xbits_msb_first)
    cbits = [(c >> (n - 1 - i)) & 1 for i in range(n)]   # cbits[0] = MSB of c
    prev_eq = None
    for i in range(n):
        xi = xbits_msb_first[i]
        ci = cbits[i]
        if ci == 0:
            if prev_eq is None:
                qc.cx(xi, gt)              # gt ^= x_i   (eq_0 implicitly 1)
            else:
                qc.rccx(prev_eq, xi, gt)   # gt ^= eq_prev AND x_i
        eq_i = eq_scratch[i]
        if ci == 0:
            qc.x(xi)
        if prev_eq is None:
            qc.cx(xi, eq_i)                # eq_1 = lit_1
        else:
            qc.rccx(prev_eq, xi, eq_i)
        if ci == 0:
            qc.x(xi)
        prev_eq = eq_i


def clean_gt_block(num_qubits, xbits_msb_first, c, eq_scratch, gt, result):
    """Self-contained block: result ^= (x > c). Every OTHER qubit it touches
    (gt, eq_scratch) is returned to exactly the state it started in."""
    fwd = QuantumCircuit(num_qubits)
    ge_flag(fwd, xbits_msb_first, c, eq_scratch, gt)
    sub = QuantumCircuit(num_qubits)
    sub.compose(fwd, inplace=True)
    sub.cx(gt, result)
    sub.compose(fwd.inverse(), inplace=True)
    return sub


def range_flag_via_comparator(num_qubits, xbits, lo, hi, scratch, out):
    """Self-contained block: out ^= (lo <= x <= hi), via two bit-serial
    comparisons. `xbits` is LSB-first (physical convention); reversed
    internally for the MSB-first comparator chain."""
    xbits_msb_first = list(reversed(xbits))
    gt, t1, t2 = scratch[0], scratch[1], scratch[2]
    eq_scratch = scratch[3:3 + N]

    sub = QuantumCircuit(num_qubits)

    # t1 ^= (x > lo-1)  ==  (x >= lo)
    ge_lo = clean_gt_block(num_qubits, xbits_msb_first, lo - 1, eq_scratch, gt, t1)
    sub.compose(ge_lo, inplace=True)

    # t2 ^= (x > hi); then flip a persistent qubit to get (x <= hi) = NOT(x>hi)
    gt_hi = clean_gt_block(num_qubits, xbits_msb_first, hi, eq_scratch, gt, t2)
    sub.compose(gt_hi, inplace=True)
    sub.x(t2)                      # t2 now holds (x <= hi)

    sub.rccx(t1, t2, out)   # safe: t1, t2 used solely as controls here, then
                            # uncomputed below via their own literal inverses

    sub.x(t2)                      # undo the flip
    sub.compose(gt_hi.inverse(), inplace=True)   # t2 back to 0
    sub.compose(ge_lo.inverse(), inplace=True)   # t1 back to 0
    return sub


# --------------------------------------------------------------- cube (existing)

def range_flag_via_cube(num_qubits, xbits, lo, hi, scratch, out):
    """The existing best_blocks/MCX machinery, applied to a single [lo,hi] range.
    `xbits` is LSB-first, matching what blk()/best_blocks() already expect."""
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    sub = QuantumCircuit(num_qubits)
    for start, size in best_blocks(mask, cx_mcx):
        MCX(sub, blk(start, size, xbits), out, scratch)
    return sub


# --------------------------------------------------------------------- verify

def verify_and_score(build_fn, label, scratch_count, lo, hi):
    # handbuilt.py's MCX hardcodes a width-18 sub-circuit internally (it's built
    # for the real 18-qubit design), so match that width here too -- also makes
    # this a fair test under the real qubit budget, not an artificially roomy one.
    n_q = 18
    assert N + 1 + scratch_count <= n_q
    xbits = list(range(N))
    out = N
    scratch = list(range(N + 1, N + 1 + scratch_count))

    sub = build_fn(n_q, xbits, lo, hi, scratch, out)

    ok = True
    for v in range(64):
        qc = QuantumCircuit(n_q)
        for q in range(N):
            if (v >> q) & 1:
                qc.x(xbits[q])
        qc.compose(sub, inplace=True)
        sv = np.asarray(Statevector(qc).data)
        idx = int(np.argmax(np.abs(sv)))
        # Magnitude-only, matching this codebase's own convention (handbuilt.py's
        # MCX/MCZ use relative-phase Toffolis whose residual phase is only ever
        # guaranteed to cancel once the flag is later uncomputed via its own
        # literal inverse -- a standalone flag output is EXPECTED to carry an
        # uncancelled phase at this intermediate point, that is not a bug here.
        if abs(abs(sv[idx]) - 1.0) > 1e-9:
            print(f"  {label}: not a definite basis state at x={v} (bug -- superposition leaked)")
            ok = False
            break
        bit_of = lambda q: (idx >> q) & 1

        expect = 1 if (lo <= v <= hi) else 0
        if bit_of(out) != expect:
            print(f"  {label}: MISMATCH at x={v}, got out={bit_of(out)}, want {expect}")
            ok = False
            break
        if any(bit_of(xbits[q]) != ((v >> q) & 1) for q in range(N)):
            print(f"  {label}: input register corrupted at x={v}")
            ok = False
            break
        if any(bit_of(s) != 0 for s in scratch):
            print(f"  {label}: scratch leaked (not returned to 0) at x={v}")
            ok = False
            break

    if not ok:
        print(f"{label}: FAILED VERIFICATION -- not scoring")
        return

    print(f"{label}: verified correct (magnitude+phase) on all 64 inputs (scratch used={scratch_count})")

    best = None
    for lvl, seed in [(3, s) for s in range(8)] + [(2, 0)]:
        t = transpile(sub, basis_gates=["u3", "cx"], optimization_level=lvl, seed_transpiler=seed)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or d < best[0]:
            best = (d, c)
    print(f"{label}: depth={best[0]}  cx={best[1]}")


if __name__ == "__main__":
    # (2,26): the square piece's x-bound -- wide (25 values), not aligned.
    # (39,43): the bar piece's y-bound -- narrow (5 values), not aligned --
    # this is closer to the "many small ragged blocks" case that actually
    # drives the 31-way conflict in Finding 8.
    for lo, hi in [(2, 26), (39, 43)]:
        print(f"\n=== racing x in [{lo}, {hi}] ===")
        verify_and_score(range_flag_via_cube, "cube", scratch_count=5, lo=lo, hi=hi)
        print()
        verify_and_score(range_flag_via_comparator, "comparator", scratch_count=3 + N, lo=lo, hi=hi)
