"""Phase 3.6 task 1: an EXACT (not greedy) minimum-cost ESOP-over-cubes search
for the already-reduced small disk truth tables, to directly test the SSHR
skepticism from findings-2026-09-16.md -- does a smarter (exact, ILP) choice
of cube-cover beat the existing rank-decomposition, on a search space small
enough (8-10 variables) that "exact" is actually tractable?

Formulation: candidate cubes are all points of {0,1,*}^k (3^k of them). Pick a
minimum-WEIGHT subset whose XOR (parity) reproduces the target exactly. This is
a linear system over GF(2) with a sparsity/cost-minimizing objective -- solved
here as a MILP (each GF(2) equation "sum ≡ target (mod 2)" linearized via an
auxiliary integer variable: sum - 2*carry = target).
"""

import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

import sys
sys.path.insert(0, "../experiments")
from handbuilt import cx_mcz


def cube_cost(pattern):
    """Circuit-cost weight for one cube: same cx_mcz(k) formula already used
    for MCZ trees elsewhere in this project, k = number of FIXED bits."""
    k_fixed = sum(1 for b in pattern if b != '*')
    return cx_mcz(k_fixed)


def all_cubes(k):
    """Every point of {0,1,*}^k, as tuples."""
    return list(itertools.product((0, 1, '*'), repeat=k))


def cube_contains(pattern, point_bits):
    return all(p == '*' or p == point_bits[i] for i, p in enumerate(pattern))


def solve_exact_cover(target_fn, k, time_limit=600):
    n_points = 1 << k
    cubes = all_cubes(k)
    n_cubes = len(cubes)
    print(f"k={k}: {n_points} points, {n_cubes} candidate cubes")

    # membership matrix: rows = points, cols = cubes
    M = lil_matrix((n_points, n_cubes), dtype=np.int8)
    targets = np.zeros(n_points, dtype=np.int8)
    for p in range(n_points):
        bits = [(p >> i) & 1 for i in range(k)]
        targets[p] = 1 if target_fn(bits) else 0
        for ci, pat in enumerate(cubes):
            if cube_contains(pat, bits):
                M[p, ci] = 1
    M = M.tocsr()

    weights = np.array([cube_cost(pat) for pat in cubes], dtype=float)

    # variables: n_cubes binary x_c, then n_points integer carry_p
    n_vars = n_cubes + n_points
    c = np.concatenate([weights, np.zeros(n_points)])

    # constraint: M @ x - 2*carry == targets  for each point
    from scipy.sparse import hstack, eye
    A = hstack([M, -2 * eye(n_points, format="csr")], format="csr")
    con = LinearConstraint(A, lb=targets, ub=targets)

    max_carry = M.sum(axis=1).A1 // 2  # a cube-count upper bound per row
    lb = np.concatenate([np.zeros(n_cubes), np.zeros(n_points)])
    ub = np.concatenate([np.ones(n_cubes), max_carry])
    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)

    res = milp(c, constraints=[con], bounds=bounds, integrality=integrality,
               options={"time_limit": time_limit, "disp": True, "mip_rel_gap": 0.0})

    print(f"MILP status: {res.message}  (success={res.success})")
    if res.x is None:
        print("no feasible solution found at all")
        return None
    # report whatever was found, even if not proven optimal within time_limit --
    # the message/status already say clearly whether it's a proven optimum.

    x = res.x[:n_cubes]
    chosen = [cubes[i] for i in range(n_cubes) if x[i] > 0.5]
    print(f"solution: {len(chosen)} cubes, total weight {res.fun:.0f}, "
          f"mip_gap={getattr(res, 'mip_gap', None)}, "
          f"mip_dual_bound={getattr(res, 'mip_dual_bound', None)}")
    return chosen


def build_and_score(chosen_cubes, kx, ky, target_fn):
    """Turn a chosen cube list into an actual phase-mark circuit (reusing
    handbuilt.py's already-verified MCZ, which self-contains its own AND-tree
    compute/uncompute), verify it exactly via the same Hadamard-superposition
    phase check handbuilt.py's own check() uses, then transpile-measure it."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from handbuilt import MCZ, X, Y, A

    x_qubits, y_qubits, scratch = X[:kx], Y[:ky], A[:]
    qc = QuantumCircuit(18)
    for pat in chosen_cubes:
        controls = []
        for i, v in enumerate(pat):
            if v == '*':
                continue
            q = x_qubits[i] if i < kx else y_qubits[i - kx]
            controls.append((q, v))
        if controls:
            MCZ(qc, controls, scratch)

    used = x_qubits + y_qubits
    full = QuantumCircuit(18)
    for q in used:
        full.h(q)
    full.compose(qc, inplace=True)
    amp = np.asarray(Statevector(full).data)
    n_used = 1 << len(used)
    # correctness check: compare the full statevector to the expected one
    # directly (Hadamard on the used qubits gives an equal-magnitude
    # superposition; only the SIGN per branch should differ from all-plus).
    want_vec = np.zeros(1 << 18, dtype=complex)
    for idx in range(n_used):
        bits = [(idx >> i) & 1 for i in range(len(used))]
        sign = -1.0 if target_fn(bits) else 1.0
        full_idx = sum(b << used[i] for i, b in enumerate(bits))
        want_vec[full_idx] = sign
    norm = 1.0 / np.sqrt(1 << len(used))
    want_vec *= norm
    err = float(np.abs(amp - want_vec).max())
    print(f"phase check max error: {err:.2e}  ({'OK' if err < 1e-9 else 'FAILED'})")

    best = None
    for lvl, seed in [(3, s) for s in range(8)] + [(2, 0)]:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=lvl, seed_transpiler=seed)
        d, cx = t.depth(), t.count_ops().get("cx", 0)
        if best is None or d < best[0]:
            best = (d, cx)
    print(f"exact-cover circuit ({len(chosen_cubes)} cubes): depth={best[0]}  cx={best[1]}")
    return err, best


if __name__ == "__main__":
    def disk2_pixel(xl, yl):
        x, y = 49 + xl, 35 + yl
        return (x - 55) ** 2 + (y - 41) ** 2 <= 42

    def disk2_local(bits):
        xl = sum(bits[i] << i for i in range(4))
        yl = sum(bits[4 + i] << i for i in range(4))
        return disk2_pixel(xl, yl)

    chosen = solve_exact_cover(disk2_local, k=8, time_limit=600)
    if chosen:
        for pat in chosen:
            print("  ", pat, "cost", cube_cost(pat))
