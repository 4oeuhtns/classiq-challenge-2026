"""Second pass: minimise XOR count at a fixed AND count.

min_and() minimises AND gates and lets the XOR subsets grow without limit, which is
the wrong objective for this problem -- in the built circuit each chain is applied
four times (fold/unfold inside emit_chain, then the whole pass inverted to clean the
ancilla), so an XOR costs far more than its nominal single CX.

Here the total subset size is bounded by a cardinality constraint and the bound is
lowered until UNSAT.  Since an input of size k costs k-1 XORs and there are 2g+1
subsets, minimising the total size minimises the XOR count directly.
"""

import json
import sys
import time

from pysat.card import CardEnc, EncType

from mc_exact import Synth, simulate, xor_count


def solve_bounded(target, n, g, bound, time_limit=30.0, chunk=20000):
    """CEGAR search for a g-AND chain whose subsets have total size <= bound."""
    syn = Synth(n, g)
    lits = ([syn.A(i, j) for i in range(g) for j in range(syn.avail(i))]
            + [syn.B(i, j) for i in range(g) for j in range(syn.avail(i))]
            + [syn.C(j) for j in range(syn.nsig)])
    card = CardEnc.atmost(lits=lits, bound=bound, vpool=syn.pool,
                          encoding=EncType.seqcounter)
    for c in card.clauses:
        syn.solver.add_clause(c)

    for v in (0, (1 << n) - 1, 1, (1 << n) - 2):
        syn.constrain(v, (target >> v) & 1)
    t0 = time.time()
    while True:
        if time.time() - t0 > time_limit:
            return "unknown", None
        syn.solver.conf_budget(chunk)
        res = syn.solver.solve_limited()
        if res is None:
            continue
        if res is False:
            return "unsat", None
        chain = syn.decode()
        diff = simulate(chain, n) ^ target
        if diff == 0:
            return "sat", chain
        v = (diff & -diff).bit_length() - 1
        syn.constrain(v, (target >> v) & 1)


def refine(target, n, g, start_chain, time_limit=30.0):
    """Lower the total-subset-size bound while a chain still exists."""
    A_sets, B_sets, C_set = start_chain
    best = start_chain
    size = sum(len(s) for s in A_sets) + sum(len(s) for s in B_sets) + len(C_set)
    while size > 2 * g + 1:
        status, chain = solve_bounded(target, n, g, size - 1, time_limit)
        if status != "sat":
            break
        best = chain
        A_sets, B_sets, C_set = chain
        size = sum(len(s) for s in A_sets) + sum(len(s) for s in B_sets) + len(C_set)
    return best


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "mc_chains.json"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    sys.path.insert(0, "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/experiments")
    from handbuilt import PIECES

    chains = json.load(open(src))
    masks = {}
    for i, (ym, xm) in enumerate(PIECES):
        masks[f"y{i}"], masks[f"x{i}"] = ym, xm

    print(f"{'mask':<6} {'AND':>4} {'XOR before':>11} {'XOR after':>10} {'secs':>7}")
    tot_b = tot_a = 0
    out = {}
    for name, rec in chains.items():
        g, chain = rec["and"], [list(map(list, rec["chain"][0])),
                                list(map(list, rec["chain"][1])),
                                list(rec["chain"][2])]
        before = xor_count(chain)
        t0 = time.time()
        better = refine(masks[name], 6, g, chain, tl)
        dt = time.time() - t0
        assert simulate(better, 6) == masks[name], f"{name}: refined chain is wrong"
        after = xor_count(better)
        tot_b += before
        tot_a += after
        out[name] = {"and": g, "xor": after, "chain": better}
        print(f"{name:<6} {g:>4} {before:>11} {after:>10} {dt:>7.1f}", flush=True)
    json.dump(out, open("mc_chains_refined.json", "w"))
    print(f"\nXOR total {tot_b} -> {tot_a}")
