"""Minimum-AND (multiplicative-complexity) XAG synthesis for the 6-variable masks, by SAT.

Every AND gate in a reversible phase oracle costs a Toffoli (3 CX, ~6 layers when
packed); every XOR costs a bare CX (1 CX, 1 layer).  So the quantity that decides
whether a design fits the depth budget is the *multiplicative complexity* -- the
minimum number of AND gates over {AND, XOR, NOT} -- not the gate count that
cut_rewriting minimises.

Encoding (Boyar-Peralta style).  An XAG with g AND gates computing f is:

    signal 0        = constant 1
    signals 1..n    = the inputs
    signal n+1+i    = t_i = a_i AND b_i        for i = 0..g-1
    a_i, b_i        = XOR of some subset of the signals available before gate i
    f               = XOR of some subset of all signals

The subsets are the SAT variables; per-assignment values are Tseitin-encoded.

Two things make this tractable:
  * CEGAR -- constraining only a subset S of the 2**n assignments is a relaxation,
    so a model on S is checked by simulation and a mismatch is added to S.  The
    solver is incremental: structural clauses are asserted once, assignment
    clauses are appended as they are needed.
  * A conflict budget, so a g that is too hard to decide is reported as UNKNOWN
    rather than hanging.  We need small XAGs, not proofs of minimality, so an
    upper bound found at g is still usable; UNSAT at g-1 is what would upgrade it
    to "exact".
"""

import sys
import time
from pysat.solvers import Cadical153
from pysat.formula import IDPool


class Synth:
    """Incremental CEGAR search for an XAG with exactly g AND gates."""

    def __init__(self, n, g):
        self.n, self.g = n, g
        self.pool = IDPool()
        self.nsig = n + 1 + g
        self.solver = Cadical153()
        self._structural()

    # --- variable accessors -------------------------------------------------
    def A(self, i, j): return self.pool.id(("A", i, j))
    def B(self, i, j): return self.pool.id(("B", i, j))
    def C(self, j):    return self.pool.id(("C", j))
    def T(self, i, v): return self.pool.id(("T", i, v))
    def avail(self, i): return self.n + 1 + i

    def _fresh(self, tag):
        return self.pool.id((tag, self.pool.top + 1))

    def _xor_chain(self, lits, add):
        acc = lits[0]
        for lit in lits[1:]:
            z = self._fresh("xor")
            add([[-z, acc, lit], [-z, -acc, -lit], [z, -acc, lit], [z, acc, -lit]])
            acc = z
        return acc

    def _and_lit(self, u, w, add):
        p = self._fresh("and")
        add([[-p, u], [-p, w], [p, -u, -w]])
        return p

    def _selected_xor(self, sel, limit, v, add):
        """XOR over signals j < limit of (sel(j) AND value_j(v))."""
        lits = []
        for j in range(limit):
            if j == 0:                                   # constant 1
                lits.append(sel(j))
            elif j <= self.n:                            # input x_{j-1}
                if (v >> (j - 1)) & 1:
                    lits.append(sel(j))
            else:                                        # earlier gate output
                lits.append(self._and_lit(sel(j), self.T(j - self.n - 1, v), add))
        return self._xor_chain(lits, add)

    # --- clause groups ------------------------------------------------------
    def _structural(self):
        """Clauses independent of which assignments are constrained."""
        cls = []
        add = cls.extend
        for i in range(self.g):
            cls.append([self.A(i, j) for j in range(self.avail(i))])   # non-empty
            cls.append([self.B(i, j) for j in range(self.avail(i))])

            # AND is commutative, so fix A_i <=_lex B_i.
            eq_prefix = None
            for j in range(self.avail(i)):
                a, b = self.A(i, j), self.B(i, j)
                cls.append([-a, b] if eq_prefix is None else [-eq_prefix, -a, b])
                eqj = self.pool.id(("eq", i, j))
                add([[-eqj, -a, b], [-eqj, a, -b], [eqj, a, b], [eqj, -a, -b]])
                if eq_prefix is None:
                    eq_prefix = eqj
                else:
                    nxt = self.pool.id(("eqp", i, j))
                    add([[-nxt, eq_prefix], [-nxt, eqj], [nxt, -eq_prefix, -eqj]])
                    eq_prefix = nxt

            # every gate output must be consumed: an unused gate could be deleted,
            # which would contradict minimality.
            sig = self.n + 1 + i
            cls.append([self.A(k, sig) for k in range(i + 1, self.g)]
                       + [self.B(k, sig) for k in range(i + 1, self.g)]
                       + [self.C(sig)])
        for c in cls:
            self.solver.add_clause(c)

    def constrain(self, v, bit):
        """Require the chain to output `bit` on assignment v."""
        cls = []
        add = cls.extend
        for i in range(self.g):
            a = self._selected_xor(lambda j: self.A(i, j), self.avail(i), v, add)
            b = self._selected_xor(lambda j: self.B(i, j), self.avail(i), v, add)
            t = self.T(i, v)
            add([[-t, a], [-t, b], [t, -a, -b]])
        out = self._selected_xor(self.C, self.nsig, v, add)
        cls.append([out] if bit else [-out])
        for c in cls:
            self.solver.add_clause(c)

    def decode(self):
        model = set(l for l in self.solver.get_model() if l > 0)
        return ([[j for j in range(self.avail(i)) if self.A(i, j) in model]
                 for i in range(self.g)],
                [[j for j in range(self.avail(i)) if self.B(i, j) in model]
                 for i in range(self.g)],
                [j for j in range(self.nsig) if self.C(j) in model])


def simulate(chain, n):
    """Truth table (as an int bitmask over assignments) produced by a decoded chain."""
    A_sets, B_sets, C_set = chain
    sig = [(1 << (1 << n)) - 1]                          # signal 0 = constant 1
    for k in range(n):
        col = 0
        for v in range(1 << n):
            if (v >> k) & 1:
                col |= 1 << v
        sig.append(col)
    for Ai, Bi in zip(A_sets, B_sets):
        a = b = 0
        for j in Ai:
            a ^= sig[j]
        for j in Bi:
            b ^= sig[j]
        sig.append(a & b)
    out = 0
    for j in C_set:
        out ^= sig[j]
    return out


def try_g(target, n, g, time_limit=45.0, chunk=20000):
    """Returns ('sat', chain) / ('unsat', None) / ('unknown', None).

    Solving is chunked by conflict budget purely so the wall-clock limit can be
    enforced between chunks -- an exhausted chunk is not a verdict, it just yields
    control back so elapsed time can be checked.
    """
    syn = Synth(n, g)
    for v in (0, (1 << n) - 1, 1, (1 << n) - 2):
        syn.constrain(v, (target >> v) & 1)
    t0 = time.time()
    while True:
        if time.time() - t0 > time_limit:
            return "unknown", None
        syn.solver.conf_budget(chunk)
        res = syn.solver.solve_limited()
        if res is None:
            continue                                     # chunk exhausted, keep going
        if res is False:
            return "unsat", None
        chain = syn.decode()
        diff = simulate(chain, n) ^ target
        if diff == 0:
            return "sat", chain
        v = (diff & -diff).bit_length() - 1              # a counterexample assignment
        syn.constrain(v, (target >> v) & 1)


def min_and(target, n=6, gmax=14, time_limit=45.0):
    """Smallest g with a witness. Returns (g, chain, exact?) -- exact iff g-1 was UNSAT.

    An 'unknown' at some g does not stop the search: larger g is strictly easier to
    satisfy, so we keep climbing until a witness appears.
    """
    prev_unsat = True                                    # g = -1 is vacuously UNSAT
    for g in range(gmax + 1):
        status, chain = try_g(target, n, g, time_limit)
        if status == "sat":
            return g, chain, prev_unsat
        prev_unsat = (status == "unsat")
    return None, None, False


def xor_count(chain):
    """XOR gates: an input formed from k signals costs k-1 XORs."""
    A_sets, B_sets, C_set = chain
    return (sum(max(0, len(s) - 1) for s in A_sets)
            + sum(max(0, len(s) - 1) for s in B_sets)
            + max(0, len(C_set) - 1))


def to_target(mask):
    return mask                                          # masks are already bitmasks over v


if __name__ == "__main__":
    sys.path.insert(0, "/Users/aoeuhtns/Documents/quantum/classiq-challenge-2026/experiments")
    from handbuilt import PIECES

    import json

    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    print(f"SAT min-AND XAG synthesis, {tl:.0f}s wall-clock limit per g")
    print(f"{'mask':<6} {'px':>4} {'AND':>4} {'exact?':>7} {'XOR':>5} {'secs':>7}")
    tot_and = tot_xor = 0
    results = {}
    for i, (ym, xm) in enumerate(PIECES):
        for tag, m in (("y", ym), ("x", xm)):
            t0 = time.time()
            g, chain, exact = min_and(m, time_limit=tl)
            dt = time.time() - t0
            if chain is None:
                print(f"{tag}{i:<5} {bin(m).count('1'):>4} {'FAIL':>4}", flush=True)
                continue
            assert simulate(chain, 6) == m, f"{tag}{i}: chain does not reproduce the mask"
            xc = xor_count(chain)
            tot_and += g
            tot_xor += xc
            results[f"{tag}{i}"] = {"and": g, "xor": xc, "exact": exact, "chain": chain}
            print(f"{tag}{i:<5} {bin(m).count('1'):>4} {g:>4} "
                  f"{('exact' if exact else '<=  '):>7} {xc:>5} {dt:>7.1f}", flush=True)
    json.dump(results, open("mc_chains.json", "w"))
    print(f"\ntotal over the 20 masks: AND={tot_and}  XOR={tot_xor}")
    print(f"reversible phase oracle: {2*tot_and} Toffolis + {2*tot_xor} bare CX "
          f"= {6*tot_and + 2*tot_xor} CX")
    print(f"(iterated mockturtle got AND=208; ceiling at depth 129 is ~91 Toffolis)")
