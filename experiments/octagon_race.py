"""Genuine attempt at the octagon idea from plan.md Strategy E: a disk is
(exactly) a square-and-diamond union in ROTATED coordinates u=x+y, v=x-y.
The diamond needs real adder arithmetic (not multiplication -- affordable per
plan.md's own cost table). Built and verified incrementally, riskiest piece
(the adder) first, so a correctness bug surfaces before three layers are
built on top of it.

Correctness is checked with a CLASSICAL bit simulator, not Qiskit's
Statevector -- these circuits are purely classical-reversible (x/cx/ccx/rccx
only, no superposition), and some sub-circuits here run at 20+ qubits, where
a dense statevector (2^20+ amplitudes) would be far too slow to brute-force
over thousands of inputs. Simulating the boolean action of each gate directly
is exact for this gate set and orders of magnitude faster.

Uses MORE than the real 6-ancilla budget for this exploratory measurement
(same precedent as comparator_race.py) -- the question right now is whether
the TECHNIQUE is even competitive on depth/CX at all, before any width
optimization pass.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile

GRID = 64
N = 6  # bits per coordinate


def classical_sim(qc, init_bits):
    """Exact boolean simulation of a circuit built only from x/cx/ccx/rccx.
    rccx (relative-phase Toffoli) has the identical bit-level truth table to
    ccx -- it only differs in phase, invisible to this simulator, which is
    exactly why it's safe to substitute later without re-deriving this check.
    """
    bits = list(init_bits)
    index = {q: i for i, q in enumerate(qc.qubits)}
    for instr in qc.data:
        name = instr.operation.name
        qs = [index[q] for q in instr.qubits]
        if name == "x":
            bits[qs[0]] ^= 1
        elif name == "cx":
            if bits[qs[0]]:
                bits[qs[1]] ^= 1
        elif name in ("ccx", "rccx"):
            if bits[qs[0]] and bits[qs[1]]:
                bits[qs[2]] ^= 1
        else:
            raise ValueError(f"classical_sim: unsupported gate {name}")
    return bits


# ------------------------------------------------------------------- adder

def ripple_add(qc, a_bits, b_bits, sum_bits, carry_bits):
    """Out-of-place, non-mutating: sum_bits[i] for i in 0..n-1 is bit i of
    a+b; carry_bits[n-1] is the final carry-out (bit n of a+b). Never writes
    to a_bits or b_bits -- only reads them as controls. carry_bits[0..n-2]
    are pure internal scratch, must be uncomputed by the caller (compose the
    inverse of this same sequence) once the result is consumed.
    """
    n = len(a_bits)
    carry_in = None
    for i in range(n):
        s, c = sum_bits[i], carry_bits[i]
        qc.cx(a_bits[i], s)
        qc.cx(b_bits[i], s)
        if carry_in is not None:
            qc.cx(carry_in, s)
        # rccx safe here by the same rule as elsewhere in this project: a[i],
        # b[i], carry_in are read-only controls throughout, c is written
        # multiple times then only ever read as a control afterward, and the
        # whole adder is always composed alongside its own literal inverse.
        qc.rccx(a_bits[i], b_bits[i], c)
        if carry_in is not None:
            qc.rccx(a_bits[i], carry_in, c)
            qc.rccx(b_bits[i], carry_in, c)
        carry_in = c


def test_adder():
    n = N
    a_q = list(range(n))
    b_q = list(range(n, 2 * n))
    sum_q = list(range(2 * n, 3 * n))
    carry_q = list(range(3 * n, 4 * n))
    n_q = 4 * n

    fwd = QuantumCircuit(n_q)
    ripple_add(fwd, a_q, b_q, sum_q, carry_q)

    ok = True
    for a in range(GRID):
        for b in range(GRID):
            init = [0] * n_q
            for i in range(n):
                init[a_q[i]] = (a >> i) & 1
                init[b_q[i]] = (b >> i) & 1
            out = classical_sim(fwd, init)
            got_sum = sum(out[sum_q[i]] << i for i in range(n)) | (out[carry_q[n - 1]] << n)
            want = a + b
            if got_sum != want:
                print(f"adder: WRONG at a={a} b={b}: got {got_sum} want {want}")
                ok = False
                break
            if any(out[a_q[i]] != ((a >> i) & 1) for i in range(n)):
                print(f"adder: a register corrupted at a={a} b={b}")
                ok = False
                break
            if any(out[b_q[i]] != ((b >> i) & 1) for i in range(n)):
                print(f"adder: b register corrupted at a={a} b={b}")
                ok = False
                break
        if not ok:
            break

    if ok:
        print("adder: VERIFIED correct on all 4096 (a,b) pairs, registers untouched")
        t = transpile(fwd, basis_gates=["u3", "cx"], optimization_level=3)
        print(f"adder alone (forward only): depth={t.depth()}  cx={t.count_ops().get('cx', 0)}")
    else:
        print("adder: FAILED -- stopping here")
    return ok


# --------------------------------------------------------------- comparator
# (generalized version of the bit-serial comparator already verified in
# comparator_race.py -- same rccx safety discipline, arbitrary bit width)

def ge_flag(qc, bits_msb_first, c, eq_scratch, gt):
    n = len(bits_msb_first)
    cbits = [(c >> (n - 1 - i)) & 1 for i in range(n)]
    prev_eq = None
    for i in range(n):
        xi = bits_msb_first[i]
        ci = cbits[i]
        if ci == 0:
            if prev_eq is None:
                qc.cx(xi, gt)
            else:
                qc.rccx(prev_eq, xi, gt)
        eq_i = eq_scratch[i]
        if ci == 0:
            qc.x(xi)
        if prev_eq is None:
            qc.cx(xi, eq_i)
        else:
            qc.rccx(prev_eq, xi, eq_i)
        if ci == 0:
            qc.x(xi)
        prev_eq = eq_i


def le_flag_block(n_q, bits_lsb_first, c, eq_scratch, gt, result):
    """result ^= (value <= c), for an unsigned value given by bits_lsb_first
    (bit 0 = LSB). Self-contained: eq_scratch and gt return to their starting
    state; only `result` is left changed."""
    bits_msb_first = list(reversed(bits_lsb_first))
    fwd = QuantumCircuit(n_q)
    ge_flag(fwd, bits_msb_first, c, eq_scratch, gt)   # gt ^= (value > c)
    sub = QuantumCircuit(n_q)
    sub.compose(fwd, inplace=True)
    sub.x(result)
    sub.cx(gt, result)     # result = NOT(value > c) = (value <= c)
    sub.compose(fwd.inverse(), inplace=True)
    return sub


def test_halfplane():
    """x + y <= c, via: adder (u=x+y, 7 bits) -> comparator on u -> flag.
    Everything except the final flag qubit returns to |0>."""
    n = N
    x_q = list(range(n))
    y_q = list(range(n, 2 * n))
    sum_q = list(range(2 * n, 3 * n))       # u bits 0..5
    carry_q = list(range(3 * n, 4 * n))     # carry_q[5] is u's bit 6
    eq_q = list(range(4 * n, 4 * n + (n + 1)))
    gt_q = 4 * n + (n + 1)
    result_q = gt_q + 1
    n_q = result_q + 1

    u_bits = sum_q + [carry_q[n - 1]]   # 7-bit sum register, LSB first

    C = 60   # threshold for x+y <= 60, an arbitrary mid-range test constant

    add_fwd = QuantumCircuit(n_q)
    ripple_add(add_fwd, x_q, y_q, sum_q, carry_q)

    full = QuantumCircuit(n_q)
    full.compose(add_fwd, inplace=True)
    full.compose(le_flag_block(n_q, u_bits, C, eq_q, gt_q, result_q), inplace=True)
    full.compose(add_fwd.inverse(), inplace=True)

    ok = True
    for x in range(GRID):
        for y in range(GRID):
            init = [0] * n_q
            for i in range(n):
                init[x_q[i]] = (x >> i) & 1
                init[y_q[i]] = (y >> i) & 1
            out = classical_sim(full, init)
            want = 1 if (x + y <= C) else 0
            if out[result_q] != want:
                print(f"halfplane: WRONG at x={x} y={y}: got {out[result_q]} want {want}")
                ok = False
                break
            if any(v for i, v in enumerate(out) if i != result_q and i not in x_q and i not in y_q):
                print(f"halfplane: leak at x={x} y={y}")
                ok = False
                break
            if any(out[x_q[i]] != ((x >> i) & 1) for i in range(n)) or \
               any(out[y_q[i]] != ((y >> i) & 1) for i in range(n)):
                print(f"halfplane: coordinate corrupted at x={x} y={y}")
                ok = False
                break
        if not ok:
            break

    if ok:
        print(f"halfplane (x+y<={C}): VERIFIED correct on all 4096 inputs")
        t = transpile(full, basis_gates=["u3", "cx"], optimization_level=3)
        print(f"halfplane: depth={t.depth()}  cx={t.count_ops().get('cx', 0)}")
    else:
        print("halfplane: FAILED -- stopping here")
    return ok


# ------------------------------------------------------------- full octagon

def clean_range_flag(n_q, bits_lsb_first, lo, hi, eq_scratch, gt, t1, t2, result):
    """result ^= (lo <= value <= hi). Self-contained: eq_scratch, gt, t1, t2
    all return to their starting state."""
    sub = QuantumCircuit(n_q)
    hi_block = le_flag_block(n_q, bits_lsb_first, hi, eq_scratch, gt, t1)
    sub.compose(hi_block, inplace=True)
    lo_block = le_flag_block(n_q, bits_lsb_first, lo - 1, eq_scratch, gt, t2)
    sub.compose(lo_block, inplace=True)
    sub.x(t2)                       # t2 now holds (value >= lo)
    sub.rccx(t1, t2, result)
    sub.x(t2)
    sub.compose(lo_block.inverse(), inplace=True)
    sub.compose(hi_block.inverse(), inplace=True)
    return sub


def diamond_channel_block(n_q, a_bits, b_bits, invert_b, lo, hi,
                           sum_q, carry_q, eq_scratch, gt, t1, t2, flag):
    """flag ^= (a+b in [lo,hi]) if not invert_b, else (a + NOT(b) in [lo,hi])
    -- the second form gives a-b+63 when b is 6 bits, since NOT(b)=63-b.
    Self-contained: everything but `flag` returns to its starting state,
    including a_bits/b_bits (invert_b temporarily flips b_bits, symmetric
    X gates restore it before this block ends)."""
    fwd = QuantumCircuit(n_q)
    if invert_b:
        for q in b_bits:
            fwd.x(q)
    ripple_add(fwd, a_bits, b_bits, sum_q, carry_q)
    if invert_b:
        for q in b_bits:
            fwd.x(q)
    sub = QuantumCircuit(n_q)
    sub.compose(fwd, inplace=True)
    u_bits = sum_q + [carry_q[len(sum_q) - 1]]
    sub.compose(clean_range_flag(n_q, u_bits, lo, hi, eq_scratch, gt, t1, t2, flag), inplace=True)
    sub.compose(fwd.inverse(), inplace=True)
    return sub


def and4_block(f1, f2, f3, f4, temp1, temp2, result, n_q):
    sub = QuantumCircuit(n_q)
    sub.rccx(f1, f2, temp1)
    sub.rccx(temp1, f3, temp2)
    sub.rccx(temp2, f4, result)
    sub.rccx(temp1, f3, temp2)
    sub.rccx(f1, f2, temp1)
    return sub


def build_octagon_circuit(cx_, cy_, a_, b_):
    n = N
    x_q = list(range(n))
    y_q = list(range(n, 2 * n))
    sum_q = list(range(2 * n, 3 * n))
    carry_q = list(range(3 * n, 4 * n))
    eq_q = list(range(4 * n, 4 * n + (n + 1)))
    gt_q = 4 * n + (n + 1)
    t1 = gt_q + 1
    t2 = t1 + 1
    flag_bx, flag_by, flag_u, flag_v = t2 + 1, t2 + 2, t2 + 3, t2 + 4
    temp1, temp2 = t2 + 5, t2 + 6
    result_q = t2 + 7
    n_q = result_q + 1

    lo_u, hi_u = (cx_ + cy_) - b_, (cx_ + cy_) + b_
    lo_v, hi_v = (cx_ - cy_ - b_) + 63, (cx_ - cy_ + b_) + 63

    bx_block = clean_range_flag(n_q, x_q, cx_ - a_, cx_ + a_, eq_q, gt_q, t1, t2, flag_bx)
    by_block = clean_range_flag(n_q, y_q, cy_ - a_, cy_ + a_, eq_q, gt_q, t1, t2, flag_by)
    u_block = diamond_channel_block(n_q, x_q, y_q, False, lo_u, hi_u,
                                     sum_q, carry_q, eq_q, gt_q, t1, t2, flag_u)
    v_block = diamond_channel_block(n_q, x_q, y_q, True, lo_v, hi_v,
                                     sum_q, carry_q, eq_q, gt_q, t1, t2, flag_v)
    and_block = and4_block(flag_bx, flag_by, flag_u, flag_v, temp1, temp2, result_q, n_q)

    full = QuantumCircuit(n_q)
    for b in (bx_block, by_block, u_block, v_block, and_block):
        full.compose(b, inplace=True)
    for b in (v_block, u_block, by_block, bx_block):
        full.compose(b.inverse(), inplace=True)

    return full, n_q, x_q, y_q, result_q


def test_octagon_disk1():
    """Full octagon for disk 1: center (40,19), a=8, b=11 -- verified against
    the EXACT octagon predicate (corrections to the true disk are a separate
    step, only worth doing if this core number looks competitive)."""
    cx_, cy_, a_, b_ = 40, 19, 8, 11
    full, n_q, x_q, y_q, result_q = build_octagon_circuit(cx_, cy_, a_, b_)

    def octagon_pixel(x, y):
        dx, dy = abs(x - cx_), abs(y - cy_)
        return dx <= a_ and dy <= a_ and dx + dy <= b_

    ok = True
    for x in range(GRID):
        for y in range(GRID):
            init = [0] * n_q
            for i in range(N):
                init[x_q[i]] = (x >> i) & 1
                init[y_q[i]] = (y >> i) & 1
            out = classical_sim(full, init)
            want = 1 if octagon_pixel(x, y) else 0
            if out[result_q] != want:
                print(f"octagon: WRONG at x={x} y={y}: got {out[result_q]} want {want}")
                ok = False
                break
            if any(v for i, v in enumerate(out)
                   if i != result_q and i not in x_q and i not in y_q):
                print(f"octagon: leak at x={x} y={y}: {out}")
                ok = False
                break
        if not ok:
            break

    if ok:
        n_px = sum(1 for x in range(GRID) for y in range(GRID) if octagon_pixel(x, y))
        print(f"octagon: VERIFIED correct on all 4096 inputs ({n_px} pixels, using {n_q} qubits)")
        t = transpile(full, basis_gates=["u3", "cx"], optimization_level=3)
        print(f"octagon (disk 1 approximation): depth={t.depth()}  cx={t.count_ops().get('cx', 0)}")
    else:
        print("octagon: FAILED -- stopping here")
    return ok


if __name__ == "__main__":
    if test_adder():
        print()
        if test_halfplane():
            print()
            test_octagon_disk1()
