"""Overnight search for the shallowest phase-polynomial oracle.

    python depthsearch.py [hours] [seed]

Depth is the only objective; CX is spent freely wherever it buys parallelism.  State
lives in depthsearch_best.json and the run is resumable, so it can be stopped and
restarted at any point and will pick up from the incumbent.

WHAT IS SEARCHED, and why this shape

`L`, an invertible 6x6 GF(2) relabelling of the ENCODED register, is the main knob.
It changes which sets are cubes, hence the cover of every mask, hence the encoder --
which is where the depth is (781 of the 1105 baseline).  It is free for the phase
side, which never touches that register, so the encoder's tour depth is an EXACT
objective for it, not a proxy.  That makes the L subproblem cheap to score well.

`N`, the 10x10 GF(2) code basis, is perturbed only locally.  Measured: random N and L
far from the identity are all far worse (depth 2063-6356 against 1095), because the
nested chain structure that makes the tour cheap is fragile.  So this is descent with
basin hopping, not a global search -- jumping is actively harmful here.

`K`, the number of tours running at once, is re-tried exactly for every improved L,
since it trades encoder length against encode/walk overlap and the two swap places.

Scoring is two-tier.  Inside the climb, candidates are ranked by the tour's piece
depth, which is exact for the encoder and independent of the phase side.  Anything
that improves it is then BUILT, verified against the target by exact statevector, and
transpiled for a true depth.  Only verified circuits are ever recorded, so a modelling
error can cost time but cannot produce a wrong answer.
"""

import json
import os
import random
import sys
import time

import phasepoly as pp
from xbasis import invertible, apply_mask

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "depthsearch_best.json")
LOG = os.path.join(HERE, "depthsearch.log")

I6 = [1 << i for i in range(6)]
I10 = [1 << i for i in range(10)]
KS = (1, 2, 3)


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ scoring ---
def masks_of(L, N):
    cls = pp.column_classes()
    if N is not None and N != I10:
        cls = pp.recode(cls, N)
        if cls is None:
            return None
    if L is not None:
        cls = [(apply_mask(L, m), xs) for m, xs in cls]
    return cls


def tour_depth(L, N=None):
    """Exact encoder objective, plus what the masks cost in scratch.

    Scratch is the binding resource: an ancilla the encoder needs for an AND tree is
    an ancilla that cannot be a walk host, and hosts are what divide the walk.  With
    K=1 a mask needing 3 scratch leaves 2 hosts; one needing 2 would leave 3.  So a
    relabelling that narrows the widest piece is worth real depth beyond its own tour
    cost, and the objective says so.
    """
    cls = masks_of(L, N)
    if cls is None:
        return None
    worst = max(pp.need_scratch(m) for m, _ in cls)
    if worst > 5:
        return None
    dist = lambda a, b: pp.edge_pieces(a ^ b)[0]
    return pp.tsp([m for m, _ in cls], dist)[0] + 120 * worst


def exact(L, N, K, nhost=2, seeds=range(6)):
    """Build, verify against the target, transpile.  None if it does not verify."""
    try:
        qc = pp.build(K, 0, nhost, True, L, N)
    except (ValueError, IndexError):
        return None
    err, leak = pp.check(qc)
    if max(err, leak) > 1e-9:
        return None
    return pp.score(qc, seeds)


def best_over_K(L, N):
    """Try every chain count against every host split that leaves enough scratch."""
    out = None
    for K in KS:
        for nhost in range(0, 6):
            if K * (1 + nhost) > 6:
                continue
            got = exact(L, N, K, nhost)
            if got and (out is None or got[0] < out[0]):
                out = (got[0], got[1], K, nhost)
    return out


# ------------------------------------------------------------------- climb ---
def climb(L, N, rng, budget):
    """Steepest descent on tour depth over single row-operations of L."""
    cur = tour_depth(L, N)
    if cur is None:
        return None, None
    improved = True
    while improved and time.time() < budget:
        improved = False
        moves = [(i, j) for i in range(6) for j in range(6) if i != j]
        rng.shuffle(moves)
        for i, j in moves:
            cand = list(L)
            cand[i] ^= cand[j]
            if not invertible(cand):
                continue
            got = tour_depth(cand, N)
            if got is not None and got < cur:
                L, cur, improved = cand, got, True
                break
    return L, cur


def perturb(L, N, rng):
    """Basin hop: a few row-ops away from the incumbent, never a random matrix."""
    L = list(L)
    for _ in range(rng.randrange(1, 4)):
        i, j = rng.randrange(6), rng.randrange(6)
        if i != j:
            cand = list(L)
            cand[i] ^= cand[j]
            if invertible(cand):
                L = cand
    N = list(N)
    if rng.random() < 0.25:
        i, j = rng.randrange(10), rng.randrange(10)
        if i != j:
            N[i] ^= N[j]
    return L, N


def load():
    if os.path.exists(CKPT):
        with open(CKPT) as fh:
            d = json.load(fh)
        return d["L"], d["N"], d["K"], d["depth"], d["cx"]
    return list(pp.BEST_L), list(I10), 1, None, None


def save(L, N, K, depth, cx, nhost=2):
    with open(CKPT, "w") as fh:
        json.dump({"L": L, "N": N, "K": K, "nhost": nhost, "depth": depth,
                   "cx": cx, "transpose": pp.TRANSPOSE}, fh, indent=1)


def run(hours=8.0, seed=0):
    rng = random.Random(seed)
    deadline = time.time() + hours * 3600
    L, N, K, depth, cx = load()
    if depth is None:
        depth, cx, K, NH = best_over_K(L, N)
        save(L, N, K, depth, cx, NH)
    say(f"start: depth {depth} cx {cx} K={K}  tour {tour_depth(L, N)}")

    best_tour = tour_depth(L, N)
    bestL, bestN = list(L), list(N)
    hops = 0
    while time.time() < deadline:
        hops += 1
        cL, cN = perturb(bestL, bestN, rng)
        cL, t = climb(cL, cN, rng, min(deadline, time.time() + 600))
        if t is None:
            continue
        if t < best_tour:
            best_tour, bestL, bestN = t, list(cL), list(cN)
            got = best_over_K(cL, cN)
            if got and got[0] < depth:
                depth, cx, K, NH = got
                save(cL, cN, K, depth, cx, NH)
                say(f"NEW BEST depth {depth} cx {cx} K={K} hosts={NH}  "
                    f"tour {t}  (hop {hops})")
            else:
                say(f"tour {t} (better) but depth did not improve; keeping {depth}")
        if hops % 25 == 0:
            say(f"{hops} hops, best depth {depth}, best tour {best_tour}")

    say(f"done after {hops} hops: depth {depth} cx {cx} K={K}")
    say(f"L = {bestL}")
    say(f"N = {bestN}")


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0,
        int(sys.argv[2]) if len(sys.argv) > 2 else 0)
