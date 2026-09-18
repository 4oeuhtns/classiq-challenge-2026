"""SAT synthesis minimising actual circuit cost, not AND count.

Minimising multiplicative complexity alone is the wrong objective here.  On mask x0
the MC-optimal chain uses 6 AND gates but 75 XORs, while mockturtle's 9-AND chain
uses none -- and in the built circuit that is ~336 CX against ~54, because each
chain is applied repeatedly (fold/unfold per gate, then inverted to clean ancilla).

So search on the cost the circuit actually pays.  One forward pass of a chain costs
roughly `3g` CX for the AND gates (rccx = 3 CX each) plus `2 * XOR` CX for folding
each input subset onto a pivot and folding it back:

    cost(g, x) = 3g + 2x

Candidate (g, x) pairs are enumerated in increasing cost and each is tested for
feasibility with a cardinality bound on the total subset size; the first feasible
pair is optimal for this cost model.
"""

import json
import sys
import time

from pysat.card import CardEnc, EncType

from mc_exact import Synth, simulate


def feasible(target, n, g, xor_budget, time_limit=6.0, chunk=20000):
    """Is there a g-AND chain for `target` using at most `xor_budget` XORs?

    A subset of size k costs k-1 XORs and there are 2g+1 subsets, so bounding the
    total subset size by 2g+1+xor_budget bounds the XOR count.
    """
    syn = Synth(n, g)
    lits = ([syn.A(i, j) for i in range(g) for j in range(syn.avail(i))]
            + [syn.B(i, j) for i in range(g) for j in range(syn.avail(i))]
            + [syn.C(j) for j in range(syn.nsig)])
    bound = 2 * g + 1 + xor_budget
    if bound < len(lits):
        for c in CardEnc.atmost(lits=lits, bound=bound, vpool=syn.pool,
                                encoding=EncType.seqcounter).clauses:
            syn.solver.add_clause(c)

    for v in (0, (1 << n) - 1, 1, (1 << n) - 2):
        syn.constrain(v, (target >> v) & 1)
    t0 = time.time()
    while True:
        if time.time() - t0 > time_limit:
            return None
        syn.solver.conf_budget(chunk)
        res = syn.solver.solve_limited()
        if res is None:
            continue
        if res is False:
            return False
        chain = syn.decode()
        diff = simulate(chain, n) ^ target
        if diff == 0:
            return chain
        v = (diff & -diff).bit_length() - 1
        syn.constrain(v, (target >> v) & 1)


def best_chain(target, n=6, gmax=9, xmax=28, time_limit=3.0):
    """Cheapest chain under cost = 3*AND + 2*XOR.

    Feasibility is monotone in both AND and XOR, so for each g the smallest feasible
    XOR budget is found by binary search, and g is abandoned once 3g alone already
    exceeds the best cost seen.  Adding AND gates never requires *more* XORs, so the
    previous g's answer caps the next g's search range.
    """
    best = (None, None, None, None)                  # cost, g, x, chain
    hi = xmax
    for g in range(gmax + 1):
        if best[0] is not None and 3 * g >= best[0]:
            break
        top = feasible(target, n, g, hi, time_limit)
        if not top:
            continue
        lo, cur, found = 0, hi, (hi, top)
        while lo < cur:
            mid = (lo + cur) // 2
            r = feasible(target, n, g, mid, time_limit)
            if r:
                cur, found = mid, (mid, r)
            else:
                lo = mid + 1
        x, chain = found
        hi = x
        cost = 3 * g + 2 * x
        if best[0] is None or cost < best[0]:
            best = (cost, g, x, chain)
    return best


def _work(args):
    name, mask, tl = args
    t0 = time.time()
    cost, g, x, chain = best_chain(mask, time_limit=tl)
    return name, mask, cost, g, x, chain, time.time() - t0


if __name__ == "__main__":
    from multiprocessing import Pool

    sys.path.insert(0, "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/experiments")
    from handbuilt import PIECES

    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    jobs = []
    for i, (ym, xm) in enumerate(PIECES):
        jobs.append((f"y{i}", ym, tl))
        jobs.append((f"x{i}", xm, tl))

    print(f"SAT synthesis on circuit cost 3*AND + 2*XOR, {tl:.0f}s per feasibility test")
    print(f"{'mask':<6} {'px':>4} {'AND':>4} {'XOR':>4} {'cost':>5} {'secs':>7}")
    out, tot_g, tot_x, tot_c = {}, 0, 0, 0
    with Pool(processes=8) as pool:
        for name, mask, cost, g, x, chain, dt in pool.imap_unordered(_work, jobs):
            if chain is None:
                print(f"{name:<6} {bin(mask).count('1'):>4} {'FAIL':>4}", flush=True)
                continue
            assert simulate(chain, 6) == mask, f"{name}: chain does not reproduce the mask"
            out[name] = {"and": g, "xor": x, "cost": cost, "chain": chain}
            tot_g += g
            tot_x += x
            tot_c += cost
            print(f"{name:<6} {bin(mask).count('1'):>4} {g:>4} {x:>4} {cost:>5} {dt:>7.1f}",
                  flush=True)
    json.dump(out, open("mc_chains.json", "w"))
    print(f"\ntotals over {len(out)} masks: AND={tot_g}  XOR={tot_x}  "
          f"forward-pass cost={tot_c} CX")
    print(f"(iterated mockturtle: AND=208, XOR=56; Toffoli ceiling at depth 129 is ~91)")
