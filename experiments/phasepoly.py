"""Phase-polynomial oracle: no AND trees in the phase, ancilla as a coordinate.

Every design so far was compute -> Z-mark -> uncompute, and every one of them
spent three quarters of its gates managing scratch for AND trees.  This drops the
AND trees from the marking entirely.

A diagonal unitary over {u3, cx} *is* a phase polynomial: one p(theta) per nonzero
Walsh coefficient of the phase function, on a parity assembled by CNOTs.  Done
ancilla-free on 12 qubits that is 4096 gates (measured -- the spectrum of pi f is
completely dense), which is why this never looked viable.

The ancilla change the count.  The target has only 11 distinct COLUMNS, so the
x-axis partitions into 10 nonzero classes and

    f(x,y) = XOR_k  [x in class k] * col_k(y)

Hold col_k(y) -- the colour of class k at the current row -- in ancilla q_k, and
the phase pi * sum_k a_k(x) q_k is LINEAR in the ancilla.  That is what matters:
for a phase function linear in the code, the Walsh support collapses onto the
code-parities {0, e_1, ..., e_r} instead of all 2^r, so the gate count falls to

    sum_k |supp(a_k)|  +  |supp(union of classes)|   =   450 + 64  =  514

against a 4096 ancilla-free spectrum and a 524 floor (sum of column supports).
A nonlinear code of the same width measures 1024 -- LINEARITY is the whole trick,
not the compression.

Rank is 10 and there are 6 ancilla, so the columns cannot all be live.  Instead an
ancilla walks a closed TOUR of columns, stepping from col_j to col_k by XOR-ing in
their difference: rebuilding each column costs 1532 cx, the optimal tour costs 640.
K such tours run at once -- stepping a column touches y and scratch, walking its
parities touches x and the accumulator, and those are disjoint, so one chain encodes
while another walks.

Three things this architecture turns on, none of which hold for compute/Z/uncompute:

  * In the phase walk the x register is only ever a CONTROL, never a target, so the
    accumulators never queue behind shared scratch.  One cx per phase gate.

  * Chaining gives the encoder's writes no literal inverse, so the rccx relative
    phase does NOT cancel (the residue depends on the controls and the target both).
    But the encoder reads only y and returns its ancilla to |0>, so the whole debt is
    a function of y alone -- payable by one diagonal on the idle y register instead
    of by exact Toffolis on every write.

  * Accumulators walking side by side must not share a tie-break rule.  With one rule
    they step the same x qubit on the same layer and collide every time, and two
    walks that should have overlapped come out exactly as deep as one after the other
    (measured: 990 vs 931 for one chain vs two).  Hence the `rot` offsets.

Orientation is not free.  The target has 11 distinct rows as well as 11 distinct
columns, so both give a valid rank-10 factorisation, but the ROW form is cheaper on
both counts -- 458 phase gates against 488, and a 582 cx encoder tour against 640.
That is a fact about the logo, not about the method.  TRANSPOSE picks it.

Depth is the only objective, so CX is spent freely wherever it buys parallelism:
covers are costed by piece DEPTH rather than CX, phases are list-scheduled across
several host qubits at once instead of walking a fixed Gray path, and the encoded
register is relabelled to narrow the pieces.

Measured: depth 998, cx 1213 -- against 1561 for the best AND-tree design.

Six ancilla have to serve three competing roles: tour accumulators, AND-tree scratch,
and walk hosts.  That trade is the whole design.  Hosts must be ancilla the encoder
never touches -- borrowing from the scratch pool makes every walk a dependency of the
next encode step, and the overlap that destroys costs more than the hosts save
(measured: 1227 against 1034).  `depthsearch.py` searches the rest.
"""

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from handbuilt import TARGET, X, Y, A, best_blocks, blk, and_tree, cx_mcx
from cubes import best_cubes, cube_controls
from xbasis import linear_circuit, apply_mask, invertible

NQ = 18
TWO_PI = 2.0 * np.pi

# Which register the ancilla encode, and which the phase walk uses as controls.
# TRANSPOSE swaps them: the ancilla then hold row_k(x) and the walk steps y.  The
# target has 11 distinct rows as well as 11 distinct columns, so both orientations
# are valid decompositions of the same rank-10 matrix -- but they are not equally
# cheap, and which is cheaper is a fact about the logo, not about the method.
TRANSPOSE = True
ENC = X if TRANSPOSE else Y            # encoder reads this register
WLK = Y if TRANSPOSE else X            # walks use this one as controls


def _grid():
    return TARGET.T if TRANSPOSE else TARGET


# ---------------------------------------------------------------- structure ---
def column_classes():
    """Distinct nonzero lines: (mask over ENC, the WLK values carrying that mask)."""
    G = _grid()
    by_col = {}
    for u in range(64):
        c = 0
        for v in range(64):
            if G[v][u]:
                c |= 1 << v
        by_col.setdefault(c, []).append(u)
    by_col.pop(0, None)                       # blank lines phase nothing
    return [(c, xs) for c, xs in sorted(by_col.items())]


def wht(v):
    """Natural-order Walsh-Hadamard: out[U] = sum_z (-1)^popcount(U&z) v[z]."""
    v = np.array(v, dtype=float)
    n, h = v.shape[0], 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = v[i:i + h].copy()
            b = v[i + h:i + 2 * h].copy()
            v[i:i + h] = a + b
            v[i + h:i + 2 * h] = a - b
        h *= 2
    return v


def wrap(t):
    t = float(t) % TWO_PI
    return t - TWO_PI if t > np.pi else t


def phase_terms(g):
    """{U: theta} with sum_U theta_U * parity_U(z) = g(z), for a phase function g."""
    coef = wht(g)
    n = len(g)
    return {U: wrap(-2.0 * coef[U] / n) for U in range(1, n)
            if abs(wrap(-2.0 * coef[U] / n)) > 1e-12}


def residue_phase(enc):
    """The relative-phase debt left by the chained encoder, as a function of y.

    Chaining an ancilla through a tour of columns gives its writes no literal
    inverse, so the rccx residue -- which depends on the controls AND the target --
    does not cancel.  But the encoder reads only y and writes only ancilla that the
    tour returns to |0>, so the whole thing is diagonal and the debt is a function
    of y alone.  That makes it payable by one 6-qubit diagonal on the y register,
    which is idle during the walks, instead of by exact Toffolis on every write.
    """
    full = QuantumCircuit(NQ)
    for q in ENC:
        full.h(q)
    full.compose(enc, inplace=True)
    amp = np.asarray(Statevector(full).data).reshape(64, 64, 64)   # [ancilla][y][x]
    ph = np.angle(amp[0][0, :] if TRANSPOSE else amp[0][:, 0])
    return phase_terms(-(ph - ph[0]))


def class_terms(xs):
    """Phase terms for one class: pi * a(x) * q, split into cross and x-only.

    Returns ({S: theta} on parities S(x) + q, {S: theta} on x alone).  The x-only
    half is the same for every class up to sign and is accumulated globally, so it
    is emitted once at the end rather than ten times.
    """
    a = np.zeros(64)
    a[list(xs)] = 1.0
    psi = np.zeros(128)                       # z = x | (q << 6)
    psi[64:] = np.pi * a                      # phase is pi*a(x) when q = 1
    coef = wht(psi)
    cross, xonly = {}, {}
    for U in range(1, 128):
        th = wrap(-coef[U] / 64.0)            # theta_U = -2 * coef_U / 2^7
        if abs(th) < 1e-12:
            continue
        (cross if U >> 6 else xonly)[U & 63] = th
    return cross, xonly


# ------------------------------------------------------------------- walks ---
def path(masks, rot=0):
    """Greedy nearest-neighbour tour of the parities, opening and closing at 0.

    `rot` rotates which register bit the tie-break prefers.  Accumulators walking
    at the same time must not agree on that: with one rule they step the same x
    qubit on the same layer and collide every single time, and two walks that
    should have run side by side end up exactly as deep as one after the other.
    """
    todo, cur, out = set(masks), 0, []
    key = lambda m: sum(((m >> i) & 1) << ((i + rot) % 6) for i in range(6))
    while todo:
        nxt = min(todo, key=lambda m: (bin(m ^ cur).count("1"), key(m)))
        out.append(nxt)
        todo.discard(nxt)
        cur = nxt
    return out


def walk(qc, acc, terms, reg=None, rot=0):
    """Accumulate parities of `reg` into `acc`, phasing as we go, and restore it.

    `acc` may already hold a value (an ancilla holding col_k(y)); the parity it
    carries is then that value XOR the running parity, which is exactly the cross
    term.  It is left as it was found.
    """
    reg = WLK if reg is None else reg
    order = [(j + rot) % 6 for j in range(6)]
    cur = 0
    for S in path(terms, rot):
        for j in order:
            if (cur ^ S) >> j & 1:
                qc.cx(reg[j], acc)
        qc.p(terms[S], acc)
        cur = S
    for j in order:
        if cur >> j & 1:
            qc.cx(reg[j], acc)


def walk_cost(terms):
    cur, n = 0, 0
    for S in path(terms):
        n += bin(cur ^ S).count("1")
        cur = S
    return n + bin(cur).count("1")


# ----------------------------------------------------------------- encoder ---
def piece_depth(k):
    """Transpiled depth of XOR-ing one k-control cube into an ancilla.

    An AND tree over k-1 controls is ceil(log2(k-1)) levels of rccx, and it is paid
    twice (compute, then inverse) around the write.  This is the cost that matters
    now: the covers were being chosen to minimise CX, which prefers many narrow
    pieces, and every piece lands on the same accumulator so they cannot overlap.
    """
    if k <= 1:
        return 1
    if k == 2:
        return 7
    return (2 * ((k - 2).bit_length()) + 1) * 7


COVER_COST = piece_depth      # depth is the objective, so cost pieces by depth
COVER_KEY = "depth"

# Relabelling of the encoded register, found by hill-climbing the tour's piece depth
# from the identity.  It is free for the phase side, which never touches that
# register.  Random restarts far from the identity are all much worse -- the nested
# chain structure that makes the tour cheap is fragile -- so this is a local optimum
# reached by descent, not a global search result.
BEST_L = [43, 6, 12, 40, 16, 32]


def min_scratch(n):
    """Clean ancilla needed to AND n controls: a tree wants n-2, halving wants n//2."""
    return 0 if n <= 2 else n // 2


def and_write(qc, ctrls, target, scratch):
    """XOR AND(ctrls) into target using at most len(scratch) clean ancilla.

    A tree is cheapest but wants n-2 scratch, and scratch is exactly what caps the
    number of chains that can run at once.  When there is not enough room, split the
    controls in half, build one half into a scratch qubit and uncompute it after --
    one extra recomputation buys a 6-control block down from 4 scratch to 3, which
    is the difference between two chains and three.
    """
    n = len(ctrls)
    if n == 0:
        qc.x(target)
    elif n == 1:
        qc.cx(ctrls[0], target)
    elif n == 2:
        qc.rccx(ctrls[0], ctrls[1], target)
    elif len(scratch) >= n - 2:
        sub = QuantumCircuit(NQ)
        top = and_tree(sub, ctrls[:-1], scratch)
        qc.compose(sub, inplace=True)
        qc.rccx(ctrls[-1], top, target)
        qc.compose(sub.inverse(), inplace=True)
    else:
        if not scratch:
            raise ValueError(f"AND of {n} controls needs scratch, none free")
        s, rest, h = scratch[0], scratch[1:], (n + 1) // 2
        and_write(qc, ctrls[:h], s, rest)
        and_write(qc, [s] + ctrls[h:], target, rest)
        and_write(qc, ctrls[:h], s, rest)


def MCX_rel(qc, controls, target, scratch):
    """Relative-phase MCX.  Safe here: everything it brackets is diagonal."""
    zeros = [q for q, v in controls if v == 0]
    for q in zeros:
        qc.x(q)
    and_write(qc, [q for q, _ in controls], target, scratch)
    for q in zeros:
        qc.x(q)


@lru_cache(maxsize=None)
def _cover_cached(mask):
    """Cheapest exact cover of a y-mask: dyadic blocks, or cubes where they beat them.

    Fewer pieces matters twice over -- each piece is an AND tree plus its inverse and
    they all land on the same accumulator, so the piece count sets the encoder depth,
    not just its CX.
    """
    dyadic = [blk(s, sz, ENC) for s, sz in best_blocks(mask, COVER_COST)]
    if mask == 0:
        return []
    if not USE_CUBES:
        return dyadic
    found = best_cubes(mask, COVER_COST, COVER_KEY)
    if found and found[0] < sum(COVER_COST(len(c)) for c in dyadic):
        return [cube_controls(f, v, ENC) for f, v in found[1]]
    return dyadic


def cover(mask):
    return _cover_cached(mask)


@lru_cache(maxsize=None)
def edge_pieces(mask):
    pieces = cover(mask)
    return sum(COVER_COST(len(c)) for c in pieces), len(pieces)


def need_scratch(mask):
    return max((min_scratch(len(c)) for c in cover(mask)), default=0)


def xor_in(qc, mask, target, scratch):
    """XOR the y-mask into `target`, so an ancilla can step from one column to the next."""
    for ctrls in cover(mask):
        MCX_rel(qc, ctrls, target, scratch)


def mask_cost(mask):
    if mask == 0:
        return 0
    return sum(cx_mcx(6 - (sz.bit_length() - 1))
               for _, sz in best_blocks(mask, cx_mcx))


# ------------------------------------------------------------------- tours ---
def tsp(nodes, dist):
    """Cheapest closed tour 0 -> ... -> 0 over `nodes` (held-karp, n <= 10)."""
    m = len(nodes)
    if m == 0:
        return 0, []
    INF = float("inf")
    dp = [[INF] * m for _ in range(1 << m)]
    par = [[None] * m for _ in range(1 << m)]
    for k, v in enumerate(nodes):
        dp[1 << k][k] = dist(0, v)
    for S in range(1 << m):
        for k in range(m):
            if dp[S][k] == INF or not (S >> k) & 1:
                continue
            for j in range(m):
                if (S >> j) & 1:
                    continue
                c = dp[S][k] + dist(nodes[k], nodes[j])
                if c < dp[S | (1 << j)][j]:
                    dp[S | (1 << j)][j] = c
                    par[S | (1 << j)][j] = k
    full = (1 << m) - 1
    best, bk = min((dp[full][k] + dist(nodes[k], 0), k) for k in range(m))
    order, S, k = [], full, bk
    while k is not None:
        order.append(nodes[k])
        nk = par[S][k]
        S ^= 1 << k
        k = nk
    return best, order[::-1]


USE_CUBES = True
PIECE_WEIGHT = 0.0        # depth per extra cover piece, relative to one cx


def edge_cost(mask):
    """Tour metric.  Pieces, not CX, set the encoder's depth: each piece is an AND
    tree plus its inverse landing on the same accumulator, so they cannot overlap."""
    cx, n = edge_pieces(mask)
    return cx + PIECE_WEIGHT * n


def chains(classes, K, rng=None):
    """Split the classes into K closed tours.

    One tour is cheapest in CX but strictly sequential; K of them cost a little more
    (each pays its own edges to and from the empty mask) and run side by side.  The
    single optimal tour is cut into K contiguous arcs, which keeps neighbours that
    were cheap to step between together.
    """
    cols = [c for c, _ in classes]
    dist = lambda a, b: edge_cost(a ^ b)
    if rng is None:
        _, order = tsp(cols, dist)
    else:                                     # randomised restarts of nearest-neighbour
        best = None
        for _ in range(24):
            todo, cur, seq, tot = list(cols), 0, [], 0
            while todo:
                near = sorted(todo, key=lambda m: dist(cur, m))
                nxt = rng.choice(near[:2]) if len(near) > 1 else near[0]
                tot += dist(cur, nxt)
                seq.append(nxt)
                todo.remove(nxt)
                cur = nxt
            tot += dist(cur, 0)
            if best is None or tot < best[0]:
                best = (tot, seq)
        order = best[1]
    by_col = {c: xs for c, xs in classes}
    n = len(order)
    cut = [order[(i * n) // K:((i + 1) * n) // K] for i in range(K)]
    out = []
    for arc in cut:
        if not arc:
            continue
        _, o = tsp(arc, dist)
        out.append([(c, by_col[c]) for c in o])
    return out


def split_terms(terms, n):
    """Cut the Gray tour into n contiguous chunks so each accumulator stays local."""
    tour = path(terms)
    k = len(tour)
    out = [{S: terms[S] for S in tour[(i * k) // n:((i + 1) * k) // n]}
           for i in range(n)]
    return [c for c in out if c]


def _match(need, rng=None):
    """Max bipartite matching hosts -> register bits (augmenting paths; 6x6)."""
    bit_of = {}
    order = list(range(6))
    if rng is not None:
        rng.shuffle(order)
    def try_assign(h, seen):
        for j in order:
            if not (need[h] >> j) & 1 or j in seen:
                continue
            seen.add(j)
            if j not in bit_of or try_assign(bit_of[j], seen):
                bit_of[j] = h
                return True
        return False
    hosts = list(need)
    if rng is not None:
        rng.shuffle(hosts)
    for h in hosts:
        try_assign(h, set())
    return {h: j for j, h in bit_of.items()}


def schedule_ops(terms, nhosts, rng=None, slack=0):
    """Layer-by-layer plan for phasing `terms` on `nhosts` hosts sharing one register.

    Every host carries its own running parity.  Per layer a host either phases (if it
    has reached its reserved term) or toggles one register bit toward it, and no two
    hosts may toggle the same bit in the same layer -- that cap, six controls, is what
    bounds the whole thing, not the host count.  All hosts end back where they began.
    """
    state = [0] * nhosts
    pending = set(terms)
    reserved = [None] * nhosts
    layers = []
    while pending or any(r is not None for r in reserved) or any(state):
        layer, busy = [], set()
        # reserve BEFORE phasing: a host reserved onto the parity it already holds
        # must phase in this same layer, or the layer comes out empty and the loop
        # stops with the work undone.
        for h in range(nhosts):
            if reserved[h] is None and pending:
                if rng is None or slack <= 0:
                    tgt = min(pending, key=lambda t: (bin(t ^ state[h]).count("1"), t))
                else:
                    near = sorted(pending,
                                  key=lambda t: (bin(t ^ state[h]).count("1"), t))
                    tgt = rng.choice(near[:slack])
                reserved[h] = tgt
                pending.discard(tgt)
        for h in range(nhosts):
            if reserved[h] is not None and state[h] == reserved[h]:
                layer.append(("p", h, terms[reserved[h]]))
                reserved[h] = None
                busy.add(h)
        need = {}
        for h in range(nhosts):
            if h in busy:
                continue
            d = (reserved[h] if reserved[h] is not None else 0) ^ state[h]
            if d:
                need[h] = d
        for h, j in _match(need, rng).items():
            layer.append(("cx", h, j))
            state[h] ^= 1 << j
        if not layer:
            break
        layers.append(layer)
    return layers


def fan(qc, src, dsts):
    """Copy `src` into every dst in log depth rather than one cx at a time."""
    for a, b in _fan_pairs(src, list(dsts)):
        qc.cx(a, b)
    return [src] + list(dsts)


def scheduled_walk(qc, acc, terms, spare, reg=None, uncopy=True, rng=None, slack=0):
    """Phase `terms` using acc plus every spare ancilla as co-hosts.

    The spares are |0>, so one cx each makes them carry the same b_k the accumulator
    does; from then on they are interchangeable hosts and the schedule can keep all
    six register controls busy every layer instead of two.
    """
    reg = WLK if reg is None else reg
    if not terms:
        return
    hosts = fan(qc, acc, list(spare))
    for layer in schedule_ops(terms, len(hosts), rng, slack):
        for op in layer:
            if op[0] == "p":
                qc.p(op[2], hosts[op[1]])
            else:
                qc.cx(reg[op[2]], hosts[op[1]])
    if uncopy:
        for a, b in reversed(list(_fan_pairs(acc, list(spare)))):
            qc.cx(a, b)


def _fan_pairs(src, dsts):
    have, pairs = [src], []
    while len(have) < len(dsts) + 1:
        for s in list(have):
            if len(have) == len(dsts) + 1:
                break
            d = dsts[len(have) - 1]
            pairs.append((s, d))
            have.append(d)
    return pairs


def parallel_walk(qc, acc, terms, spare, reg=None, rot0=0):
    """Walk one parity set on several accumulators at once.

    A lone accumulator alternates cx, p, cx, p on the same qubit, so its walk is
    exactly as deep as it is long -- and the scratch ancilla are idle throughout,
    because an AND tree is only needed while a column is being stepped.  Copying the
    accumulator into them costs one cx each way and divides the walk.
    """
    if not terms:
        return
    accs = [acc] + list(spare)
    chunks = split_terms(terms, len(accs))
    accs = accs[:len(chunks)]
    for q in accs[1:]:
        qc.cx(acc, q)
    for i, (q, chunk) in enumerate(zip(accs, chunks)):
        walk(qc, q, chunk, reg, rot=(rot0 + i) % 6)
    for q in accs[1:]:
        qc.cx(acc, q)


def inv_transpose(rows, n):
    """Rows of N^-T as bitmasks, or None if N is singular."""
    A = [[(b >> j) & 1 for j in range(n)] for b in rows]
    I = [[int(i == j) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c]), None)
        if piv is None:
            return None
        A[c], A[piv] = A[piv], A[c]
        I[c], I[piv] = I[piv], I[c]
        for i in range(n):
            if i != c and A[i][c]:
                A[i] = [u ^ v for u, v in zip(A[i], A[c])]
                I[i] = [u ^ v for u, v in zip(I[i], I[c])]
    return [sum(I[k][j] << k for k in range(n)) for j in range(n)]


def recode(classes, N):
    """Re-express the decomposition in a new GF(2) basis.

    f = sum_k a_k b_k is not canonical: for invertible N the pair (N^-T a, N b)
    describes the same function.  The a_k are disjoint indicators, so every new a'_j
    is the indicator of a UNION of classes, and every new b'_j an XOR of masks.  This
    changes both the phase-gate count and the masks the encoder has to build.
    """
    n = len(classes)
    M = inv_transpose(list(N), n)
    if M is None:
        return None
    out = []
    for j in range(n):
        xs, mask = [], 0
        for k in range(n):
            if (N[j] >> k) & 1:
                xs += classes[k][1]
            if (M[j] >> k) & 1:
                mask ^= classes[k][0]
        out.append((mask, sorted(xs)))
    return out


def build(K=1, P=0, nhost=2, sched=True, L=BEST_L, N=None, rng=None, slack=0,
          rand_tour=False):
    """K ancilla each walk a tour of columns; the rest are shared AND-tree scratch.

    Stepping a column into an ancilla touches y and scratch; walking its parities
    touches x and the accumulator.  Those are disjoint, so interleaving the chains
    round-robin lets one chain encode while another walks instead of queueing.
    """
    cls = column_classes()
    if N is not None:
        cls = recode(cls, N)
        if cls is None:
            raise ValueError("singular code basis")
    if L is not None:
        if not invertible(L):
            raise ValueError("singular relabelling")
        cls = [(apply_mask(L, m), xs) for m, xs in cls]
    groups = chains(cls, K, rng if rand_tour else None)
    # Hosts must be ancilla the ENCODER never touches.  Borrowing from the scratch
    # pool instead makes every walk a dependency of the next chain's encode step, and
    # the overlap that buys costs more than the extra hosts save (measured: 1227
    # against 1034).  So carve the pool: accumulators, then scratch, then hosts.
    top = NQ - 12
    accs = A[:K]
    pool = A[top - nhost * K:] if nhost else []
    scratch = [q for q in A[K:] if q not in pool]
    partners = [pool[c::K] for c in range(K)]
    if len(scratch) < max(need_scratch(c) for g in groups for c, _ in g):
        raise ValueError(f"K={K} leaves {len(scratch)} scratch, blocks need more")
    qc = QuantumCircuit(NQ)
    rel = linear_circuit(L, ENC) if L is not None else None
    if rel is not None:
        qc.compose(rel, inplace=True)
    enc = QuantumCircuit(NQ)                 # the encoder steps alone, for the residue
    xonly = {}
    live = [0] * K
    steps = max(len(g) for g in groups)
    for r in range(steps + 1):
        for c, g in enumerate(groups):
            nxt = g[r][0] if r < len(g) else 0        # step to 0 to close the tour
            if nxt != live[c]:
                for target in (qc, enc):
                    xor_in(target, live[c] ^ nxt, accs[c], scratch)
                live[c] = nxt
            if r < len(g):
                cross, xo = class_terms(g[r][1])
                if sched:
                    scheduled_walk(qc, accs[c], cross, partners[c],
                                   rng=rng, slack=slack)
                else:
                    parallel_walk(qc, accs[c], cross, partners[c],
                                  rot0=(c * 3 + r) % 6)
                for S, th in xo.items():
                    xonly[S] = wrap(xonly.get(S, 0.0) + th)
    assert all(v == 0 for v in live), "a tour did not close"

    # both tails are ordinary diagonals and every ancilla is |0> by now, so they
    # split six ways for free
    xonly = {S: t for S, t in xonly.items() if S and abs(t) > 1e-12}
    for i, (acc, chunk) in enumerate(zip(A, split_terms(xonly, 6))):
        walk(qc, acc, chunk, rot=i)
    for i, (acc, chunk) in enumerate(zip(A, split_terms(residue_phase(enc), 6))):
        walk(qc, acc, chunk, reg=ENC, rot=i)
    if rel is not None:
        qc.compose(rel.inverse(), inplace=True)
    return qc


# ------------------------------------------------------------------- score ---
def check(qc):
    full = QuantumCircuit(NQ)
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


def score(qc, seeds=range(8)):
    best = None
    for s in seeds:
        t = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3,
                      seed_transpiler=s)
        d, c = t.depth(), t.count_ops().get("cx", 0)
        if best is None or (d, c) < best:
            best = (d, c)
    return best


if __name__ == "__main__":
    classes = column_classes()
    print(f"nonzero column classes: {len(classes)}")
    rz = 0
    for mask, xs in classes:
        cross, _ = class_terms(xs)
        rz += len(cross)
        print(f"   |class| {len(xs):2d}  col popcount {bin(mask).count('1'):2d}  "
              f"cross terms {len(cross):3d}  walk {walk_cost(cross):3d} cx  "
              f"encode {sum(cx_mcx(len(c)) for c in cover(mask)):3d} cx  "
              f"scratch {need_scratch(mask)}")
    print(f"\nmodel: {rz} cross-term p gates + one x-only diagonal")

    dist = lambda a, b: edge_cost(a ^ b)
    print(f"\nencoder, rebuilt every pass : "
          f"{2*sum(mask_cost(c) for c, _ in classes)} cx")
    for K in (1, 2, 3):
        edges = [a ^ b for g in chains(classes, K)
                 for a, b in zip([0] + [c for c, _ in g],
                                 [c for c, _ in g] + [0])]
        cx = sum(edge_pieces(m)[0] for m in edges)
        pcs = sum(edge_pieces(m)[1] for m in edges)
        print(f"encoder, {K} closed tour(s)    : {cx} cx in {pcs} pieces")

    print()
    for K in (1, 2, 3):
        try:
            qc = build(K)
        except ValueError as e:
            print(f"K={K}: {e}")
            continue
        err, leak = check(qc)
        d, cx = score(qc, range(16))
        ok = "OK " if max(err, leak) < 1e-9 else "FAIL"
        print(f"K={K}: {ok} err {err:.1e} leak {leak:.1e}   depth {d}   cx {cx}")
    print(f"\nbest AND-tree design : depth 1561  cx 1068")
    print(f"leader               : depth 129   cx 614")
