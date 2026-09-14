# Classiq Phase Oracle Challenge — Problem, Analysis, Prior Art, and Optimization Plan

*Self-contained. No prior context assumed. Revision 2 — incorporates a survey of existing tools and literature.*

---

# Part I — The problem

## I.1 What is being asked

A 64×64 black-and-white image of the Classiq logo contains **1097 black pixels** out of 4096. Build a quantum circuit that attaches a minus sign to exactly those 1097 pixel addresses and leaves the other 2999 untouched.

Formally, build a unitary $U$ on 12 qubits such that

$$U\,|x\rangle|y\rangle = (-1)^{f(x,y)}\,|x\rangle|y\rangle$$

where $f(x,y) = 1$ on black pixels and $0$ elsewhere. This is called a **phase oracle**.

The image is defined analytically, not as a bitmap:

```python
def logo_pixel(x, y) -> bool:
    return ((2 <= x <= 26 and 29 <= y <= 53)             # square
            or (26 <= x <= 49 and 39 <= y <= 43)         # horizontal bar
            or (x - 55)**2 + (y - 41)**2 <= 42           # right disk
            or (x - 40)**2 + (y - 19)**2 <= 72)          # lower disk
```

## I.2 Why anyone wants this

Phase oracles are the component that tells a quantum algorithm *which answers are the interesting ones*. Grover's search, QAOA, and quantum feature maps all take an oracle as input and then use interference to amplify whatever the oracle marked.

The oracle is a building block, not a complete algorithm. It is handed a superposition by the surrounding algorithm and hands back the same superposition with signs attached.

## I.3 The rules

| Rule | Detail |
|---|---|
| **Width** | 12 to 18 qubits total |
| **Register layout** | one register named `q`; `q[0:6]` is x, `q[6:12]` is y, little-endian |
| **Ancillas** | up to 6 scratch qubits, each starting and ending in $\lvert 0\rangle$ |
| **Coordinates** | x and y must be unchanged by the circuit |
| **Gate set** | the submitted QASM may contain only `u3` and `cx` |
| **Global phase** | a single shared factor $e^{i\phi}$ on all 4096 amplitudes is permitted |

## I.4 How it is scored

**Circuit depth**, with **CX count** as the tiebreak.

Lexicographic, not multi-objective: depth 199 with 5000 CX beats depth 200 with 50 CX. There is no final Pareto trade-off — one number decides it.

Depth is the length of the longest chain of gates that must run in sequence. Gates on disjoint qubits share a time step. **Total gate count is nearly irrelevant except as a tiebreak.** This matters more than it sounds: almost all published work in this area optimizes T-count or qubit count instead (see Part IV.7).

---

# Part II — Background needed to attack it

## II.1 What a phase actually is

A register of 12 qubits is not 12 independent objects. Its state is a single list of **4096 complex numbers**, one per address $(x,y)$. Each entry has a magnitude and a direction:

- **Magnitude** determines the probability of measuring that address.
- **Direction** is the **phase**, invisible to measurement.

Multiplying an entry by $-1$ changes nothing observable immediately. It matters later, when the surrounding algorithm recombines amplitudes and marked entries interfere differently from unmarked ones.

A **global** phase — the same rotation on all 4096 entries — has no observable consequence at all, which is why the rules permit it. Only *relative* phases matter.

## II.2 The reduction that defines the whole problem

Given an ancilla holding the value of $f$:

```
compute  f(x,y) → ancilla
Z(ancilla)                  # this single gate is the entire minus sign
uncompute f(x,y)            # restore ancilla to |0>
```

Therefore

$$\text{depth} \approx 2 \times \text{depth}(\text{compute } f) + 1$$

**This is not a quantum problem. It is: build the shallowest reversible Boolean circuit for a specific 12-input function, using at most 5 scratch bits.** Every technique below attacks that.

## II.3 Why uncomputation is mandatory

A leftover ancilla holds a record of which pixel was examined. That record entangles with the coordinate register and destroys the interference the oracle exists to enable. Ancillas must return to $\lvert 0\rangle$.

Uncomputation works because these gates are self-inverse: computing `anc ^= (A AND B)` twice, with A and B untouched in between, returns the ancilla to zero.

## II.4 Regions combine by XOR, not OR

A sign flip applied twice cancels: $(-1)\times(-1) = +1$.

So a pixel ends up marked when an **odd number** of operations cover it. The combining rule is XOR.

This is a major degree of freedom. You may cover a large cheap region and then cover the unwanted part a second time to remove it. Covers do **not** need to be disjoint. (The reference implementation self-imposes disjointness; the grader only checks the final phase pattern.)

In classical logic synthesis this is an **ESOP** (exclusive sum of products), and mature minimizers exist — see Part IV.

## II.5 The predicate toolbox

A "predicate" is any reversibly computable yes/no test on $(x,y)$. Your circuit is a set of predicates whose XOR equals the target mask. Rectangles are one option among many.

### Free or nearly free

| Predicate | Cost | Region |
|---|---|---|
| Single bit, e.g. `x5 = 1` | one `u3` | half the grid |
| **Parity of any bit subset** | one `u3` per bit, depth 1, **zero CX** | stripes, checkerboards |
| Open controls (control on 0) | X-gates, which merge into neighbouring `u3`s | — |
| Complementing the whole mask | free (global phase) | marks white instead of black |

Parity follows from $(-1)^{a \oplus b} = (-1)^a(-1)^b$: marking "an odd number of these bits are set" costs one single-qubit gate per bit, for any subset. Since it is free, the search should always be allowed to include such terms.

The complement trick follows from $(-1)^{1 \oplus f} = -(-1)^f$: marking white pixels gives the correct oracle up to a global $-1$, which is allowed.

### Cheap

**Cube** — fix some bits, leave others don't-care. Cost ≈ number of fixed bits. Because the fixed bits are the high bits, **a cube's boundaries always land on powers of 2**.

A free high bit makes the pattern *repeat*. Fixing only `y4 = 1` selects `y ∈ [16,31] ∪ [48,63]`: two stripes for one test. Two values differing in exactly one bit therefore form a single cube — e.g. `y ∈ {11, 27}`, since `001011` and `011011` differ only in bit 4.

**Bounding box, then low bits.** Test a cheap aligned box first. Inside it the high bits are constant, so the shape becomes a function of fewer variables — and **the low bits are already the local coordinates**. No subtraction needed. A shape inside `x ∈ [32,47]` has local x given directly by `x3 x2 x1 x0`.

### Moderate — need carry chains, but buy shapes cubes cannot make

| Predicate | Cost | Region |
|---|---|---|
| Comparator `x ≥ c` | ripple chain, or parallel-prefix for log depth | boundary **anywhere** |
| Register comparison `x ≥ y` | a subtraction's borrow-out | diagonal edges, triangles |
| Half-plane `x + y ≤ c` | one adder + one comparison | 45° cuts |
| Diamond | four half-planes AND-ed | rotated square |
| Octagon | diamond AND square | near-circles |

**Key distinction:** a cube has flexible shape but rigid boundaries; a comparator has any boundary but costs a carry chain. Aligned intervals favour cubes, ragged intervals favour comparators. Measure, don't assume.

### Out of reach

**Multipliers and exact circle equations.** Evaluating $(x-a)^2 + (y-b)^2 \le r^2$ reversibly requires holding every intermediate: two 6-bit differences, two 12-bit squares, a 13-bit sum — roughly **49 scratch qubits** against a budget of **6**. The blocker is width, not depth. Adders (~6 Toffolis) are affordable; multipliers are not.

**Absolute value / mirror folding.** Reflection is free — X gates on the low bits — **only when the symmetry centre is the midpoint of an aligned block**. All four shape centres here are at integer coordinates, so reflection is $x \mapsto 2a - x$, needing an adder. Check this condition per shape.

### Combining predicates

| Operation | Cost |
|---|---|
| XOR | **free at the phase level** — just apply both |
| AND | a Toffoli tree; balanced gives logarithmic depth |
| OR | De Morgan; same price as AND |
| Reuse | compute a signal once, use many times, uncompute once |

The last row is *multi-level* logic. Pure ESOP covers are two-level and cannot express it — one reason an optimal ESOP is not an optimal circuit, and the reason XAG-based methods (Part IV.4) beat ESOP at scale.

## II.6 Sub-circuit tricks

**Relative-phase (Margolus) Toffolis.** Every AND here is uncomputed shortly after, so it need not be a clean Toffoli. A cheaper variant producing correct bits but spurious phases works, because the exact inverse cancels them. **~3 CX instead of ~6, on every AND.** Valid only when the ancilla is used solely as a *control* in between, and the uncompute is the literal inverse. Confirm with the official verifier, never by reasoning.

**Balanced AND trees.** `A AND B AND C AND D` as a chain costs depth 4; as a bracket — `(A,B)` and `(C,D)` in parallel, then combine — costs depth 2. Same gate count.

**Diagonal gates commute.** Every gate in a phase oracle is diagonal, so ordering is free. Scheduling for depth is pure packing with no correctness risk.

**Ancilla pressure.** Sharing a predicate saves gates but creates dependencies: two operations on the same ancilla cannot run concurrently. This is compiler register allocation, with the same tension. See IV.6 for the caveat on how far spending width actually helps.

---

# Part III — Analysis of this specific instance

All figures computed directly from the mask definition.

## III.1 The reference implementation

The baseline decomposes the logo into **18 disjoint rectangles** by scanning rows and merging vertically adjacent identical runs. Each becomes one controlled phase gate:

```python
for x_min, x_max, y_min, y_max in ORACLE_RECTANGLES:
    control((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max),
            lambda: phase(pi))
```

Decomposing each rectangle's ranges into aligned cubes costs **494 controls in total**.

The baseline never allocates ancillas explicitly. It sets `max_width=18` and lets the Classiq synthesizer decompose the multi-controlled gates and choose how many scratch qubits to use.

Pipeline: define the mask in Python → build the rectangle cover → emit 18 controlled phase gates → add Hadamards on all 12 coordinate qubits as a **synthesis harness** → synthesize with depth optimization → export QASM → **delete the two Hadamard lines** → transpile to `u3`/`cx` → verify.

*The Hadamards exist only so the compiler sees all 4096 addresses as live; without them the optimizer could observe x = y = 0 and delete most of the circuit. They are removed before submission because the grading algorithm supplies its own superposition.*

## III.2 Cost of intervals

Every range must be decomposed into aligned blocks, and cost is extremely sensitive to boundaries:

| Interval | Aligned blocks |
|---|---|
| `[0, 31]` | 1 |
| `[32, 47]` | 1 |
| `[32, 48]` | 2 |
| `[29, 53]` | 5 |
| `[2, 26]` | 6 |
| `[2, 61]` | 8 |

One extra pixel on `[32,47]` doubles it. Since parity covers let you approximate cheaply and then correct, **boundary alignment and parity are the same tool.**

## III.3 The mask is a six-term XOR

Verified to reproduce the target exactly:

$$f = S \oplus B \oplus D_1 \oplus D_2 \oplus (S \cap B) \oplus (B \cap D_2)$$

| Term | Definition | Pixels |
|---|---|---|
| $S$ — square | `x[2,26] y[29,53]` | 625 |
| $B$ — bar | `x[26,49] y[39,43]` | 120 |
| $D_1$ — lower disk | centre (40,19), $r^2 = 72$ | 225 |
| $D_2$ — right disk | centre (55,41), $r^2 = 42$ | 137 |
| $S \cap B$ | `x[26,26] y[39,43]` | 5 |
| $B \cap D_2$ | `x[49,49] y[39,43]` | 5 |

Six terms instead of eighteen rectangles, with two 5-pixel seam corrections. This reduces the whole problem to **"mark a disk cheaply," twice.**

## III.4 Where the difficulty lives

ANF (Reed–Müller / XOR-of-monomials) term counts:

| Shape | Pixels | ANF terms |
|---|---|---|
| Square | 625 | 256 |
| Bar | 120 | 64 |
| **Disk 1** | 225 | **596** |
| Disk 2 | 137 | 172 |
| Whole logo | 1097 | 886 |

**Disk 1 alone is roughly two-thirds of the total complexity**, despite being one of the smaller shapes. The square is large and nearly trivial. Direct effort almost entirely at the disks.

## III.5 The direct ANF approach is a dead end

Implementing one multi-controlled Z per ANF monomial needs no ancillas and no uncomputation, which sounds attractive. In practice: **886 gates**, with brutal degrees.

| Degree | Count |
|---|---|
| 2–4 | 74 |
| 5–7 | **526** |
| 8–12 | 286 |

Hundreds of 7-controlled gates. Elegant and substantially worse than the naive baseline. Recorded so it is not re-attempted. Note that this is exactly what Tweedledum's `esop_phase_synth` produces by default (IV.1) — useful as a reference point, not as a solution.

## III.6 Disks are nearly octagons

An octagon is $\lvert dx\rvert \le a$ AND $\lvert dy\rvert \le a$ AND $\lvert dx\rvert + \lvert dy\rvert \le b$ — a square intersected with a diamond, buildable from half-planes.

| Disk | Pixels | Best octagon | Mismatch |
|---|---|---|---|
| Disk 1 | 225 | $a=8$, $b=11$ | **12 px** |
| Disk 2 | 137 | $a=6$, $b=8$ | **8 px** |

So `Disk1 = Octagon ⊕ 12 pixels`, exactly; the corrections cluster into a handful of tiny cubes.

The interesting middle ground: exact circles need multipliers (infeasible), half-planes need only adders (expensive but affordable). Whether the octagon route wins must be measured.

## III.7 Symmetry in the disks

Both disk centres are at integer coordinates, so both are mirror-symmetric. The baseline's own output shows the redundancy:

```
x[38,42] y[11,11]   and   x[38,42] y[27,27]     identical x-range
x[36,44] y[12,12]   and   x[36,44] y[26,26]     identical
x[34,46] y[13,14]   and   x[34,46] y[24,25]     identical
```

The x-ranges are expensive (3–5 blocks each) and the y-tests cheap (1 block). So build each x-predicate **once** and fire it on a two-row y-test. Nine rectangles collapse to five for Disk 1; six to three for Disk 2.

Some mirrored pairs collapse further: `11 = 001011` and `27 = 011011` differ in exactly one bit, so `y ∈ {11,27}` is a single cube with a don't-care. Check each pair individually.

Note that reflection itself is *not* free here (II.5). Symmetry buys **predicate sharing**, not a coordinate transform.

---

# Part IV — Prior art and existing tooling

Substantial machinery exists. Most of it was built for a slightly different objective, so it provides excellent starting points rather than a finished answer.

## IV.1 The exact operation has a library function

**Tweedledum** implements **ESOP-phase synthesis** (`esop_phase_synth`): given an n-variable Boolean function, it produces a circuit of multiple-controlled Z gates implementing $U : |\varphi\rangle \mapsto (-1)^{f(\varphi)}|\varphi\rangle$, via the function's PPRM representation. That is the challenge's target unitary, verbatim.

**Qiskit** wraps it: `PhaseOracle` defaults to Tweedledum's `pkrm_synth` with `phase_esop`, and accepts a custom `synthesizer` callable.

Two practical warnings:

- Performance issues are expected from 16 variables upward with the default synthesizer. At 12 variables it runs.
- `qiskit.circuit.classicalfunction` was deprecated in Qiskit 1.4 and removed in 2.0, specifically to drop the Tweedledum dependency, which is no longer compatible with all supported platforms and Python versions. `PhaseOracle` is being reimplemented without it. **Install Tweedledum standalone or pin an older Qiskit; resolve this in Phase 0, not later.**

Because PPRM is the ANF route, expect roughly the 886-term result from III.5. Use it as a free reference point.

## IV.2 ESOP minimization is a solved research area

**EXORCISM-4** is a C program that produces a *minimized* ESOP from a `.pla` truth-table file. It is used as the front end of MustangQ's oracle synthesis flow. `.pla` is trivial to emit from your 4096-row table.

Also available: **ABC** (multi-level synthesis, i.e. automated predicate sharing), **ESPRESSO** (SOP minimization), and the EPFL libraries **mockturtle** and **easy**.

Two papers worth reading before building your own minimizer:

- *Evaluating ESOP Optimization Methods in Quantum Compilation Flows* (Meuli et al., EPFL) — frames synthesis as an odd-even covering problem over minterms, which is exactly the parity-cover formulation.
- *Scaling-up ESOP Synthesis for Quantum Compilation*.

**Scope limit worth knowing:** no exact minimal ESOP method exists for more than seven variables on arbitrary functions. At 12 variables you are firmly in heuristic territory.

## IV.3 One finding that directly upgrades the search plan

From *Scaling-up ESOP Synthesis*: exhaustive search over orderings shows circuit size varies by up to a **factor of two** depending on ordering, and **simulated annealing over orderings reduces circuit size by approximately 30%** compared with ESOP minimization without reordering, yielding circuits close to optimal.

**Consequence: add `reorder_terms` to the Phase 4 move set.** It costs nothing and the reported gain is large.

## IV.4 ESOP is not state of the art — XAG is

**XAG** (Xor-And-Inverter Graph) is the current best-known synthesis technique. ESOP can exploit a function's global structure, but exact minimum-term methods scale poorly. XAG is a multi-level representation over $\{\oplus, \land, 1\}$ that uses auxiliary qubits to store intermediate results, reducing Toffoli count via local optimization, and performs well at scale.

This is the multi-level-beats-two-level argument from II.5, with a mature implementation: **caterpillar**, part of the EPFL logic synthesis libraries alongside tweedledum, mockturtle, easy and angel. See *Xor-And-Inverter Graphs for Quantum Compilation* (npj Quantum Information).

**Consequence: XAG synthesis becomes a first-class Phase 3 strategy, not an afterthought.**

## IV.5 The 6-ancilla constraint is a named research problem

It is the **reversible pebble game**. Caterpillar includes a compilation algorithm performing quantum memory management to trade qubits against T-count: the number of available helper qubits is a *parameter*, and a SAT solver finds a strategy to fit the logic into that budget by uncomputing intermediate results.

Three results relevant to a scarce-ancilla regime:

- A **divide-and-conquer pebbling strategy** achieves up to 90% qubit reduction on EPFL benchmarks, and has been integrated as a synthesizer for Qiskit's `PhaseOracle` and `BooleanExpression`, outperforming Qiskit's default in **both CNOT count and depth**. Directly droppable into the pipeline.
- **ShallowGrow** targets exactly this regime, reporting average circuit-depth reductions of **54.1%** over the state-of-the-art W-cycle on Qiskit-measured circuits, and **43.7%** on EPFL/ISCAS85 networks under scarce ancilla budgets.
- **ROS** (Resource-constrained Oracle Synthesis) and the **LHRS** framework in **RevKit**.

## IV.6 Multi-controlled Toffoli decomposition

The relative-phase trick is Maslov (2016). Since then:

- *Logarithmic Depth Decomposition of Approximate Multi-Controlled Single-Qubit Gates Without Ancilla Qubits* (arXiv 2507.00400) — log-depth multi-controlled NOT using a single ancilla, plus a relative-phase multi-controlled NOT needing none.
- *On Exact Space-Depth Trade-Offs in Multi-Controlled Toffoli Decomposition* (arXiv 2502.01433) — concrete formulae relating Toffoli depth to clean-ancilla count.
- Khattar et al. (2024) on conditionally clean ancillae; Claudon et al. on polylogarithmic-depth CNOT without ancillas (public repo `Polylog_MCX-public`).

**Important caveat that revises earlier advice.** The space-depth paper reports that beyond a certain point, further increases in ancilla count do **not** reduce Toffoli depth when using conditionally clean ancillae. So: still spend width rather than hoarding it, but **derive the ancilla split from the formulae instead of defaulting to the maximum.**

## IV.7 The literature agrees with this plan's central warning

From the Royal Society review of Boolean satisfiability in quantum compilation: the relation between ESOP minimality and actual implementation cost is worth being investigated.

That is the same point as Phase 1. Minimal ESOP is a proxy, and its quality has not been established. Calibrate it yourself.

**More broadly: nearly all the work above optimizes T-count or qubit count, not depth in a `u3`/`cx` basis.** Sometimes those objectives oppose each other. These tools give strong starting points; your own measurement loop decides the winner.

## IV.8 Prior Classiq competitions

The 2022 Classiq Coding Competition included a multi-controlled Toffoli problem with a 20-qubit total limit — 14 control qubits, one target, up to five auxiliary qubits — and published its winning solutions, alongside constraint satisfaction, Hamiltonian simulation and state preparation problems. Different problems, same family; the write-ups are worth reading for technique.

**No published solution to this specific logo-oracle instance was found.** The geometric region-decomposition angle appears to be open.

---

# Part V — Approach

## V.1 Three principles

**1. Measure, don't guess.** Every decision is validated against real transpiled depth. Fast proxies guide search; they never decide the winner. This matters doubly here because the available literature optimizes a different objective (IV.7).

**2. Close the loop before optimizing.** Build the pipeline end to end first, even if every stage is naive. A working slow loop beats a clever open loop.

**3. Isolate the hard part.** Disk 1 is two-thirds of the complexity. Do not polish the square.

## V.2 The sequencing trap

The natural plan — *find the best cover, then optimize the circuit* — does not work.

**A cover's cost is not a property of the cover.** The same regions transpile to very different depths depending on ancilla allocation, Toffoli variant, gate ordering and transpiler settings. Choosing a cover before those are known locks in a decision made with bad information.

Three further reasons an optimal cover is not an optimal circuit:

- ESOP minimizers minimize *term count*; the score is *depth*. Twenty cheap parallel terms beat twelve expensive serial ones.
- Covers are two-level; real circuits are multi-level, where an intermediate signal feeds several others. This is precisely why XAG supersedes ESOP (IV.4).
- Some terms are cheapest as direct multi-controlled gates, others as compute–Z–uncompute. A pure cover cannot express that mixed choice.

The fix: make "term list → real measured depth" a single function call before any optimization, and feed circuit-level costs back into the cover search later (Phase 6).

## V.3 Two tracks, run in parallel

Given the tooling in Part IV, run two tracks rather than one:

- **Track A — tool-driven.** Feed the truth table to existing synthesizers (EXORCISM-4, ABC, caterpillar/XAG, pebbling-based `PhaseOracle` synthesizers). Cheap to run, strong baselines, no design effort.
- **Track B — geometry-driven.** The hand-built region decompositions of Part III, which exploit structure (disks, symmetry, octagons) that generic Boolean tools cannot see.

They are complementary: Track A output seeds Track B's search, and Track B's structural insights can be encoded as hints to Track A. **Neither should be skipped.**

---

# Part VI — The plan

## Phase 0 — Infrastructure · ~15% of budget

**Goal:** turn a list of terms into a verified depth number with one function call.

**0.1 Reproduce the baseline** and record depth and CX count. Scoreboard zero.

Reference, from the challenge notebook on SDK 1.26.0: **width 18, depth 5329, CX 3502**. Reproducing this from your own term list is the correctness test for 0.4 -- a wildly different number means the compiler is wrong, not that the design is clever.

**Done -- 2026-09-07.** `build_main(BASELINE_TERMS)` driven through the full six-station pipeline reproduces the reference **exactly**: width 18, depth 5329, CX 3502, basis `u3`/`cx` only. Verified independently by recomputing depth and gate counts from `submission.qasm` itself rather than trusting the printed metrics. The `predicate`/`emit` split in 0.4 is therefore bit-identical to the notebook's inline oracle -- the architecture cost nothing.

**5329 was the number to beat. It is now 5267 -- 2026-09-13.** Removing the notebook's `min_y == max_y` special case (which compiles a single-row rectangle as `y == c` instead of `(y >= c) & (y <= c)`) gives **depth 5267, CX 3512**, verified `ok=True`. Eight of the eighteen baseline rectangles are single rows, so the case fires on nearly half the design.

The equality test *is* cheaper in gates -- it saved 10 CX -- but it costs 62 depth, because `y == c` is one six-wide AND that decomposes into a deep Toffoli ladder, while two comparators lay out in parallel. **Gate count and depth are different objectives, and here they oppose each other.** The notebook optimized for the one you are not scored on. Do not add the mirrored `min_x == max_x` case; the evidence says it would hurt. Caveat: n=1, one design, one seed -- a data point, not a law. Phase 1's spread is what would generalize it.

Note what reproducing the baseline does *not* prove: it is correctness on one input, checked against a reference number that evaporates the moment a term changes. That is what 0.6 exists for.

**0.2 Target mask and classical checker.**

```python
TARGET = np.zeros((64, 64), bool)      # from logo_pixel()

def render(term) -> np.ndarray:        # term -> 64x64 bool
    ...

def cover_mask(terms):
    m = np.zeros((64, 64), bool)
    for t in terms:
        m ^= render(t)                 # XOR, not OR
    return m

def is_valid(terms):
    return np.array_equal(cover_mask(terms), TARGET)
```

Because the oracle is diagonal and parity-based, correctness is a numpy XOR — **no quantum simulation required**. Microseconds per candidate. This is what makes a large search feasible.

**0.3 Term representation.** Make it general on day one or you will rewrite everything in Phase 3:

```python
Cube(fixed_bits: dict[int, int])    # bit index -> 0/1; unlisted = don't care
Range(axis, lo, hi)                 # comparator pair
Parity(bits: list[int])             # free at the phase level
HalfPlane(a, b, c)                  # a*x + b*y <= c
And(children), Xor(children), Not(child)
```

**0.4 Term → circuit compiler.** Emit a Classiq model or QASM from a term list. Start naive.

**Two levels, not one.** Collapsing this seam forces a rewrite in Phase 3.

```python
predicate(term, ctx)   # -> Qmod boolean expression. WHAT PIXELS. Composable.
emit(term, ctx)        # -> None. HOW TO PAY FOR IT.
                       #    Default: control(predicate(term, ctx), lambda: phase(pi))
```

- **`predicate` is what makes `And` / `Xor` / `Not` possible.** An octagon is a diamond AND a square (II.5), and the diamond is four half-planes ANDed. Composition needs a *value* to combine; gates already emitted cannot be ANDed.
- **`emit` is what makes the free tricks possible.** Parity costs one `u3` per bit and zero CX (II.5) -- it is not a `control` on anything. A function that only returns conditions can never express that.
- A term may supply either or both. `predicate` only → gets the default `emit`. `emit` only → cheap, but **cannot be nested inside `And`/`Xor`/`Not`**. Parity should supply both: the expensive expression for when it is nested, the cheap override for when it stands alone.
- `ctx` carries the register handles -- both `QNum`s and both bit arrays, `bind` once at the top of the oracle. `Cube`'s predicate is then a plain expression over bits, composable like everything else, with no statement-level machinery per term.
- Terms stay pure data. The rule is **the term/mask layer stays numpy-only** -- `mask.py`, `terms.py` and `baseline.py` never import `classiq`, so they stay fast and testable offline with no auth and no network. `compile.py` and `measure.py` are the classiq-aware layer above them. (Revised: the original wording was "`compile.py` is the only module that imports `classiq`", which 0.5 necessarily breaks. The intent was always the offline layer, not the module count.)

**Parity semantics come free.** Each term emits its own `phase(pi)`, so a pixel covered twice gets $(-1)^2 = +1$ -- exactly `cover_mask`'s XOR, gate for gate. The classical checker and the circuit agree by construction rather than by coincidence. This is why the baseline needs disjoint rectangles and this design does not.

**0.3 / 0.4 Done -- 2026-09-13.** Every term type in the spec is implemented in `terms.py` (numpy `render`) and `compile.py` (`predicate`, plus `emit` where a cheaper path exists), and **every code path is verified against the official simulator**, `max_error` ~1e-16 throughout. `Range` was deliberately skipped: `Rect` already is two ranges ANDed, and an atomic four-integer `Rect` is a better search representation than a nested pair.

Measured cost of each construct, seed 42, one term unless stated:

| design | width | depth | CX | |
|---|---|---|---|---|
| `[Rect]` | 18 | 395 | 244 | the yardstick |
| `[Rect, Rect]` | 18 | 792 | 508 | separate terms just add |
| `[Rect, Cube, Parity]` | 18 | 146 | 90 | three terms, cheaper than one mid-grid rect |
| `HalfPlane` | 18 | **265** | 218 | **cheaper than a rectangle** |
| `And(Rect, Parity)` | 18 | 399 | 252 | AND is nearly free |
| `Not(Rect)` | 18 | 839 | 698 | 2.1x `[Rect]` |
| `Xor(Rect, Rect)` | 18 | 1732 | 1872 | 2.2x the flat `[Rect, Rect]`, identical pixels |
| `Or(Rect, Cube)` | -- | -- | -- | **needs 21 qubits; does not fit** |

Four rules follow, three of them prohibitions:

- **Never nest `Xor` at the top level.** `Xor((a,b))` and `[a, b]` cover identical pixels; the nested form costs 2.2x. XOR is free at the design level precisely because each term applies its own phase -- nesting it throws that away and pays to compute both conditions at once.
- **Never use `Not` at the top level.** Complementing every pixel is a global phase, so `[Not(t)]` and `[t]` are the *same oracle* -- 839 versus 395 for nothing. (Nested inside an `And` it is real work and may be worth it.)
- **`Or` is unusable as written.** Spell it `~(~a & ~b)` and measure, or design around it.
- **`And` is the container that earns its place** -- and the only one the octagon route needs.

Both remain in the codebase for *nested* use: `And((Xor((a,b)), c))` has no flat equivalent. The rule is about where they may appear, not whether they exist. The Phase 4 move set must never emit a top-level `Xor` or `Not`, nor an `Or` at all until a spelling is shown to fit.

**Free win available:** a `normalize(terms)` that splices top-level `Xor` children into the design list and strips top-level `Not` wrappers makes any candidate cheaper for nothing. Pair the `Not`-stripping with letting `matches_target` accept the complement of `TARGET` -- same oracle up to global phase -- which also doubles the accepted search space at the cost of one `or`.

**Parity has two independent implementations** and they must be tested separately: standalone it runs `emit` (a `Z` per chosen bit -- k gates, depth 1, zero CX, zero ancilla, because `(-1)^(a XOR b) = (-1)^a (-1)^b` means the phases multiply on their own). Nested inside a container it runs `predicate` and pays full price, because gates already emitted cannot be ANDed with anything. A top-level parity exercises only the first.

Two Qmod facts worth not rediscovering: `bind` **consumes its source** -- `x` is uninitialized until you bind back -- so a `ctx` object cannot hold the `QNum` and its bit array at once; and a `QArray` name must be unique within an oracle, so `emit` takes the term's index and builds `f"xb{i}"`.

**0.5 Measurement wrapper with disk caching.**

```python
@dataclass(frozen=True)
class Measurement:
    width: int
    depth: int
    cx:    int
    qasm:  str          # 0.6 has to verify *something*

@disk_cache(key=fingerprint(terms))
def measure(terms, seed) -> Measurement:
    ...
```

Transpiling is slow and candidates recur constantly. Cache to disk, not memory.

`fingerprint` must be **order-sensitive**. Diagonal gates commute, so reordering terms never changes correctness -- but it changes depth (III.2, up to 2x), and `reorder_terms` is a Phase 4 move whose hit rate VII.2 tracks. Two orderings of the same terms are two different candidates and must not collide in the cache.

One canonical serialization serves both the cache key and the log's `terms_json` column. Write it once, hash it for the key, store it in the row.

**Refinements, after running all six stations by hand in 0.1.**

- **Return an object, not `tuple[int, int]`.** The original sketch predates knowing what the consumers want. 0.6 needs the QASM string -- there is nowhere else for it to get one -- and VII.1's row wants `width`, which is a constraint that can be violated. Use a frozen dataclass; that is already the idiom in `terms.py`.
- **`measure` must not write files.** The notebook ends in `Path("submission.qasm").write_text(...)` against a hardcoded name. Called thousands of times in Phase 4 that is thousands of clobbers of one file, and the best candidate gets overwritten by whatever ran next. `measure` returns the string; a separate `save_submission()` publishes the one you chose. Measuring and publishing are different verbs.
- **Keep the two-Hadamard guard.** `if len(preparation_indices) != 2: raise` reads like notebook noise. It is a tripwire: if synthesis ever inlines or renames those calls, that guard is the only thing standing between you and silently stripping the wrong lines across a thousand candidates, every one of them reporting a plausible depth. Loud failure beats quiet garbage.
- **`fingerprint` belongs in `terms.py`, not `measure.py`.** Serializing dataclasses to text needs no `classiq`, and `log.py` must be able to import it without dragging in the SDK.
- **Decide on purpose where the QASM is stored.** One `submission.qasm` is 151 KB; times a few thousand candidates that is a real number. In the cache, beside the cache keyed by fingerprint, or dropped and re-derived on demand -- all three are defensible, none should happen by accident.
- **`Preferences.random_seed` exists in SDK 1.28** (verified). Before trusting any measurement, run identical terms twice with the same seed and confirm the depth matches. If synthesis is non-deterministic, every A/B comparison in Phase 4 is noise dressed as signal.
- **Time each station before designing the cache.** `synthesize` is a cloud call; whether `classiq.transpile` is one too decides what is actually worth caching. That is a `%%time` away -- measure it rather than guessing.

**Done -- 2026-09-07.** `measure(terms, seed)` in `measure.py`, split one function per station, settings hoisted into `measure` and passed down. Cold call reproduces 18 / 5329 / 3502; warm call returns from disk in ~6.5 ms, and hits after a kernel restart. `serialize`/`fingerprint` live in `terms.py` (canonical JSON, `sort_keys`, order-sensitive, stable across processes). Cache key = fingerprint + both `Preferences` + `Constraints` + `classiq.__version__`, one JSON file per key under `cache/`.

Findings worth keeping:

- **Seed did nothing here.** Seeds 42 and 7 on `BASELINE_TERMS` gave identical width/depth/CX (two distinct cache entries, same numbers). Synthesis is deterministic for this model. Re-test once designs have real structural ambiguity -- 18 disjoint rectangles may simply leave the synthesizer no choices to make.
- **Never put a default-constructed `Preferences()` in a cache key.** Its `random_seed` is re-randomized on every construction, so the key would differ every call and the cache would never hit once.
- **The key records the *client* SDK; the server is the real compiler.** Local SDK is 1.28.0 while the exported QASM header reports 1.29.0 -- synthesis and transpilation run server-side and can be upgraded with no local change. Unfixable in the key (you would need it before doing the work), so parse the header's version line into the `Measurement` and the log row: it turns an unexplainable jump in depths into a dated, explained one.
- **`Random seed: -1` in the submission QASM is a provenance artifact, not a bug.** `quantum_program_from_qasm(qasm: str)` takes only a string, so the re-imported program carries no preferences at all (`model.preferences.random_seed` is `None`). The server's exporter writes `-1` for "none recorded". The string appears nowhere in the local SDK.
- **The cache key describes the terms, not the compiler.** Editing `predicate` or `emit` changes the circuit while leaving `fingerprint(terms)` identical, so a re-measure silently returns the old depth from disk -- the same shape as the SDK-version trap above and the `VERIFY_VERSION` caveat in 0.6. Fixed with a hand-bumped `COMPILER_VERSION` in `compile.py`, folded into the key. Hand-bumped rather than hashing the source, because in Phase 3 `compile.py` changes constantly and you would pay cloud calls for whitespace; log the source hash as a *field* if you want forgetting to be detectable.
- **File I/O is not the bottleneck and never will be.** A 151 KB cache read+parse is 2.7 ms against 10--60 s for one cloud call -- 0.009%. The only real levers are (a) parallelism, since measuring is network-bound and the CPU idles during the wait, and (b) measuring fewer candidates, which is what the Phase 1 proxy is for.

**0.6 Verifier as a callable.** Wrap the official verification step for programmatic use.

**Cost, measured: ~25 s per call** -- three `ExecutionSession` round trips, one per random test state. That is not free, so be deliberate about when it runs:

- `matches_target(terms)` (microseconds) checks the **design**. Runs on every candidate; this is what makes the search feasible.
- `verify(qasm, terms)` (~25 s) checks the **compiler**. Runs when `emit`/`predicate` change -- **per code path, not per candidate**. `predicate` already branches on `min_y == max_y`, and a design with no single-row rectangles never exercises it, so "the compiler is verified" is only true for the paths a verified design actually took. Add a branch, verify something that hits it; spot-check real candidates besides.
- This is consistent with the Phase 5 rule at the bottom of Part V ("verify every candidate officially, never by reasoning") -- each relative-phase trick *is* a new code path.
- The three test states are independent round trips, so concurrency takes ~25 s to ~8 s. Same lever as parallelising `measure`. Later.

**Cache `verify` only once its own logic is stable.** Its key would hold the QASM, terms and seed but not verify's own code, so editing a threshold would silently return the old verdict from disk -- the same shape as the SDK-version trap in 0.5. Add a hand-bumped `VERIFY_VERSION` constant to the key when you do.

**0.7 Experiment log. Done -- 2026-09-13.** `log.py` holds `FIELDS`, `append` and `rows`; `evaluate.py` assembles the row and is the single front door -- `evaluate(terms, seed, *, with_verify=True, **context)` -> `(m, v, run_id)`. `log.py` imports no `classiq`, so the 0.8 toolchain can write to the same file from an environment where the SDK is not installed.

**Refused designs are logged too**, with null metrics plus `error` and `required_width`, returning `(None, None, run_id)`. "This shape does not fit in 18 qubits" is a result, and the search needs it -- otherwise it re-proposes the same impossible family forever at 5-15 s per rejection. `evaluate` catches `ClassiqAPIError` **only**: that is the server considering the design and saying no. A `TypeError` in `predicate` is a bug and must still crash loudly. Filtering a rich log is one line; un-deleting information is impossible.

**0.8 Toolchain setup — new in revision 2.** Resolve these now, because each has a nontrivial failure mode:

- Install **Tweedledum** standalone, or pin a Qiskit version before 2.0. The `classicalfunction` module and its Tweedledum dependency were deprecated in 1.4 and removed in 2.0.
- Emit your mask as a `.pla` file and run **EXORCISM-4** on it.
- Build the EPFL C++ stack (**mockturtle**, **caterpillar**, **tweedledum**) and/or **RevKit**. Budget real time for this.
- Run `esop_phase_synth` / Qiskit `PhaseOracle` on the mask and measure the result. Expect roughly the 886-term ANF outcome; record it as a reference point.

**Exit criteria**
- `measure(baseline_terms)` returns the baseline's real depth.
- `is_valid()` agrees with the official verifier on 10 hand-built cases.
- At least one external synthesizer runs end to end and produces a measured depth.
- Every run lands in the log.

---

## Phase 1 — Calibration · ~10%

**Goal:** determine whether the fast proxy predicts real depth. The step most people skip, and the one that determines whether everything after it works. The literature explicitly flags this gap (IV.7).

**1.1 Generate a calibration set.** 200–300 valid covers spanning a wide range: few large terms, many small terms, mixed predicate types, varying ancilla counts. Include deliberately bad ones; spread is what makes correlation measurable.

**1.2 Transpile all of them for real.** One overnight batch.

**Done -- 2026-09-13.** 253 measured, 48 refused, 3.9 h, via `experiments/calibrate.py`. Depth spans 5412 to 16686. A second batch at `--seed 1` added 36 designs never used during proxy development; those are the held-out set.

**1.3 Measure transpiler variance.** Transpile the *same* cover with ~20 seeds; record the spread.

**Done -- 2026-09-13. The spread is exactly zero.** 20 seeds on `BASELINE_TERMS`, 19 of them genuine cloud calls, all returning **depth 5267, CX 3512** -- identical, not merely close. Twenty distinct cache keys prove the seed really did vary inside the serialized preferences, so this is determinism rather than a plumbing failure.

**Policy: one seed per measurement.** The budget is not divided. This also retroactively certifies the 5329 -> 5267 result: with a zero noise floor, 62 depth is 62 depth.

Scope: proven for one design, 18 rects, comfortably inside the width limit. A design crowding 18 qubits could force stochastic allocation choices. An unexplained metric change on a tight design is the signal to re-test.

**1.4 Build and rank candidate proxies** by Spearman correlation against real depth:

**Done -- 2026-09-13.** `oracle/proxy.py` (features + proxies), `experiments/fit_proxy.py` (scoring + per-recipe rank-error diagnostics).

The ladder as built. Held-out column is 36 seed-1 designs never seen during development:

| Proxy | Idea | iterated (253) | held out (36) |
|---|---|---|---|
| P1 | number of terms | +0.235 | +0.427 |
| P2 | total controls | +0.674 | +0.784 |
| P3 | examined bits instead of control counts | +0.545 | +0.666 |
| P4 | P3 + per-term base, bit-tests, adders | +0.283 | +0.462 |
| **P5** | **P4 + container multipliers** | **+0.954** | **+0.951** |

**P5 held-out 95% bootstrap interval: +0.877 to +0.977.**

The plan's original guesses were wrong in an instructive way. P4 was predicted to be "where it starts working, because ancilla pressure creates serialization" -- but the `per_term` sweep shows top-level terms are *perfectly additive* at ~366 depth each across a 32x range. There is no parallelism for ancilla pressure to destroy, and nothing to schedule.

The one idea that mattered is **compute--use--uncompute**. A condition nested inside `Xor` or `Not` must be built into scratch, combined, then unbuilt; as separate top-level terms it is built, used and discarded. That is worth +0.49 on its own, more than everything else combined.

Two rungs went *backwards*, which is the more useful lesson:

- **P3** replaced P2's control count with examined bits and silently dropped the only container signal P2 had. Alignment is real -- 8x, measured -- but the calibration covers barely vary in it, so a true feature had nothing to bite on. **A feature can be correct and still not improve a correlation, if the data does not vary in it.**
- **P4** added the per-term base of 22, which is measured and real, in a way that rewards packing leaves into containers -- exactly the wrong lesson. **A correct constant in the wrong place makes a proxy worse.** Remember this when Phase 5 feeds real costs back.

Held-out 0.951 against iterated 0.954 means there was no overfitting to undo, because every constant came from the `sweep` experiment rather than from fitting the covers. Hand-tuning those weights against the 253 would have cost that guarantee.

**Known weaknesses, none blocking.** `halfplane` is overpriced by ~33 rank places across 21 designs -- the `c` handling in `_examine` is admittedly crude. `pixels` by ~122, but on n=3. The alignment term rides almost entirely on the sweep, since only 4 held-out designs vary in it. And 25% of the calibration set is container-heavy, an artifact of recipe balancing: if Phase 4's move set rarely builds containers, P2 would be nearly as good.

**Exit criteria -- both met, 2026-09-13.**
- ~~Best proxy achieves Spearman ρ >= 0.8 on held-out covers.~~ **P5 = 0.951 on 36 unseen covers**, CI lower bound 0.877.
- ~~Seed variance known; seeds-per-measurement policy fixed.~~ **Zero variance; one seed per measurement.**

---

### The measured cost model

From `experiments/sweep.py` -- 49 designs, each group varying exactly one thing, so every number comes out by subtraction rather than regression. This table, not the proxy code, is the real output of Phase 1.

| construct | cost | evidence |
|---|---|---|
| a term at all | 22 depth | `Rect(0,63,0,63)`, no constraints, still costs 22 |
| one Rect bound | 25 x examined bits | 15-point threshold sweep, below |
| one Cube bit | ~2 | 1 bit 3, 2 bits 5, 3 bits 14 |
| one nested Parity bit | ~6 | `And(R, Parity)` 404 vs `R` 398 |
| top-level Parity | **0** | 12 bits, 12 Z gates, one layer: depth 1, CX 0 |
| HalfPlane adder | +137 | `x+y<=20` 265 vs `x<=20` 128 |
| `And` | x1.0 | `And(R,R2)` 398 = `R` alone, 398 |
| `Xor` nested | x2.4 | 1897 vs 789 for the same two terms top-level |
| `Not` | x2.5 | 987 vs 398 |
| `Or` | -- | always refused |

**Top-level terms are perfectly additive.** 1/2/4/8/16/32 identical-cost rects give 342/747/1435/3004/5928/11717 -- about 366 each, flat across 32x. They all act on the same 12 data qubits, so nothing overlaps. Depth is a plain sum over terms; there is no schedule to model.

**A bound's cost is the number of bits the comparator must examine**, which depends on the bound's *value*, not merely on whether it constrains:

| bound | binary | free low bits | depth |
|---|---|---|---|
| `x <= 31` | `0b011111` | 5 trailing 1s | **22** |
| `x <= 47` | `0b101111` | 4 | 27 |
| `x <= 7` | `0b000111` | 3 | 48 |
| `x <= 1` | `0b000001` | 1 | 115 |
| `x <= 40` | `0b101000` | 0 | 162 |
| `x <= 20` | `0b010100` | 0 | 185 |
| `x >= 32` | `0b100000` | 5 trailing 0s | **24** |
| `x >= 21` | `0b010101` | 0 | 149 |

Trailing 1s settle a `<=`; trailing 0s settle a `>=`. Both mean the low bits cannot change the answer, so the comparator never looks at them. About **25 depth per examined bit**, an 8x spread end to end. Bits *above* the trailing run do not matter: `x<=1` and `x<=5` differ by one unit of depth despite different upper patterns.

This subsumes the older "does this bound constrain" rule -- `x<=63` is six trailing 1s, zero examined, free.

**Width refusals -- hard limits, not costs.** These are rejections, not expensive designs, and belong in a separate `fits()` predicate rather than in the proxy:

| construct | qubits required (budget 18) |
|---|---|
| `Cube`, 4 / 5 / 6 / 8 fixed bits | 20 / 24 / 28 / 38 |
| nested `Parity`, 2 bits | 25 |
| `Or(Rect, Cube)` | 21 |
| nesting past depth 2 | refused (message not captured) |

`Cube` with <= 3 bits fits; 4 misses by two qubits. But `Rect(32,47,16,31)` -- the same region as a 4-bit Cube, written as aligned bounds -- **costs 47 depth and synthesizes fine.** The Rect formulation goes where the Cube cannot.

---

## Phase 2 — Structural skeleton · ~10%

**Goal:** replace the 18-rectangle cover with the six-term shape decomposition (III.3) and obtain a real number.

**Read this first -- the largest lever found so far, 2026-09-13.**

```
Rect(5, 40, 12, 33)    arbitrary bounds    398 depth
Rect(32, 47, 16, 31)   aligned bounds       47 depth
```

Same shape, 8.5x apart, purely from where the edges land. Alignment is priced **per bound**, not per rect, so snapping a single bound from 6 examined bits to 2 is already worth ~100 depth. It is a dial, not a switch.

Snapping pushes a bound outward, so the overshoot lands in the residual and the patcher pays ~400 depth for a term to clean it up. Break-even is roughly one extra patch term per fully-aligned rect; it is profit when the overshoot lands on pixels something else already covers, or when several rects snap to a shared boundary and their patches merge. **That is an optimization problem, and it is exactly what P5 is for.**

The baseline's 18 rectangles all use arbitrary bounds. Nobody has measured an aligned cover. `experiments/recipes.py::r_aligned` builds them, and the four in the calibration set are priced correctly by P5 (+13.5 mean rank error).

**Tasks**
- **Aligned-bound cover.** Snap the greedy cover's bounds to powers of two at several strengths, measure, compare against 5267. Cheap, untried, and the arithmetic says it could be large.
- Implement the six-term cover; measure; compare to baseline and to the Phase 0 tool outputs.
- Test the complement (mark white pixels). Free to try, permitted by the global-phase rule.
- For the Square and Bar, race cube decomposition against comparator pairs. `[2,26]` needs 6 cubes; a comparator pair is a fixed cost regardless of raggedness.

**Exit criteria**
- A non-baseline circuit that verifies, with a measured depth.
- A written finding on which of {cubes, comparators} wins for ragged intervals, and by how much.

---

## Phase 3 — The disk sub-problem · ~25% — *the main event*

**Goal:** solve the actual hard part in isolation.

Build a mini-harness measuring "cheapest circuit for Disk 1 alone." Small problem, fast iteration, two-thirds of the difficulty.

**Measured before starting, 2026-09-13 -- read this before attempting strategy E.** Handing Classiq one big ANDed expression cannot build an octagon:

| design | qubits required |
|---|---|
| one `HalfPlane` | **18** -- fits exactly, depth 265, verified |
| `And` of 4 half-planes (diamond) | 38 |
| `And` of a `Rect` + 4 half-planes (octagon) | 42 |

Against a budget of 18. **Each nested half-plane costs about 6 ancillas, and you have exactly 6** -- so you can afford precisely one. Classiq's expression compiler evaluates sub-conditions in parallel, so every intermediate is live simultaneously and the width explodes. This is the same effect that makes `Or` unbuildable and nested `Xor` expensive: **nesting costs width, sequencing costs depth.**

Both shapes render correctly in numpy, so the geometry is right and the arithmetic is affordable in isolation -- it is only the simultaneity that kills it. The route to an octagon is therefore to **build the AND tree by hand**: compute one half-plane into an ancilla, fold it in, uncompute it, reuse the space. That is the "compute once, use, uncompute" row of II.5, and it is the actual work of strategy E. Do not budget strategy E as "write four half-planes and AND them."


**Strategies to race head to head**

| # | Strategy | Track | Notes |
|---|---|---|---|
| A | Row staircase | B | baseline behaviour, 9 rectangles — the control |
| B | Mirror-paired rows | B | share each expensive x-predicate across `y ∈ {r, 38−r}`; collapse one-bit-apart pairs into a single cube |
| C | Column staircase | B | x-ranges are expensive and y-ranges cheap; slicing the other way inverts that |
| D | Bounding box + low-bit ESOP | A+B | condition on the aligned box, reducing the disk to an ~8-variable function; feed that truth table to EXORCISM-4 |
| E | Octagon + corrections | B | fits within 12 px (Disk 1) and 8 px (Disk 2); requires half-plane adders |
| F | Comparator rows | B | each row as two comparators rather than a cube pile |
| **G** | **XAG + pebbling** | **A** | **new in revision 2** — synthesize via caterpillar's XAG flow with the ancilla budget as an explicit parameter; also try the divide-and-conquer pebbling synthesizer plugged into Qiskit's `PhaseOracle` |

**Notes**
- **G is expected to be strong** and requires the least design work, since XAG is the current state of the art (IV.4) and its pebbling layer is built precisely for a fixed ancilla budget (IV.5). Do not treat it as a fallback.
- **D has the most headroom among hand-built routes.** A 17×17 box turns a 12-variable problem into ~8 variables, shrinking the search space ~16×. It is also the natural bridge between tracks: the reduced truth table is exactly what an ESOP or XAG tool wants.
- **E is the only strategy needing arithmetic.** Adders are affordable where multipliers were not. Genuinely open whether it wins.
- Re-check the free-mirror condition (II.5) before assuming symmetry helps. Here it does not hold; symmetry buys predicate sharing only.

**Exit criteria**
- All seven strategies measured on the same harness.
- A winner, plus a one-paragraph explanation of *why*. If the reason is unclear, the measurement is probably wrong.

---

## Phase 4 — Global search · ~20%

**Goal:** automated exploration of the cover space, seeded by everything above.

### Structure: MAP-Elites, not a single best

Maintain a grid of cells; store the best candidate in each. Cells defined by structure, not score:

| Feature | Cells |
|---|---|
| term count | 6–10, 11–15, 16–20, 21+ |
| ancillas used | 2, 3, 4, 5 |
| max AND fan-in | 3, 4, 5, 6+ |
| slicing | row / column / mixed |
| uses arithmetic | yes / no |

**Rationale:** the proxy is imperfect. One elite per structural niche prevents proxy error from collapsing the search into a single bad basin, and preserves candidates that score worse on the proxy but transpile better. Transpile the elites, not the proxy leader.

### Move set

Every move must preserve `XOR == TARGET` by construction, so the search never enters invalid territory.

- **`reorder_terms`** — *added in revision 2.* Reordering alone changes circuit size by up to 2×, and annealing over orderings has been reported to cut size ~30% (IV.3). Cheap, high value, sample it often.
- `snap_edge_to_power_of_2` plus the corresponding correction term
- `merge_two_terms` / `split_term`
- `add_cancelling_pair`, then perturb one — how genuinely new structure enters
- `swap_row_for_column_slicing`
- `substitute_cube_pile_for_comparator` and its inverse
- `complement_all`
- `extract_shared_subexpression` — promote a repeated predicate to a named signal
- `inject_free_parity_term` — costs nothing, occasionally simplifies neighbours

### Seeding

Do not start from random. Seed from the Phase 3 winner, the Phase 2 skeleton, and the raw output of the external tools:

| Tool | Role |
|---|---|
| **EXORCISM-4** | minimized ESOP — the parity cover, solved |
| **ABC** | multi-level synthesis — automated predicate sharing |
| **caterpillar** | XAG synthesis with an ancilla-budget parameter |
| **ESPRESSO** | SOP minimization, for comparison |
| **Tweedledum `esop_phase_synth`** | PPRM/ANF reference point |

**Exit criteria**
- No elite improves over a sustained run.
- The improvement-per-hour curve has flattened.

---

## Phase 5 — Circuit-level optimization · ~15%

**Goal:** everything below the cover. These levers are largely independent of the chosen cover, so they compound with Phase 4's result.

| Lever | Expected effect |
|---|---|
| Relative-phase Toffolis in compute/uncompute pairs | ~2× on every AND |
| Log-depth MCX constructions (arXiv 2507.00400, Khattar et al. 2024) | large on wide ANDs |
| Balanced AND trees instead of chains | linear depth → logarithmic |
| Ancilla allocation and scheduling (reversible pebble game) | large; this is register allocation |
| Reorder commuting diagonal gates to pack time slots | moderate |
| Compute shared predicates once, uncompute once | removes whole compute/uncompute pairs |
| Transpiler optimization level × seed sweep | reliable 10–20% |

**Notes — revised in revision 2**
- **Derive the ancilla split from the space-depth formulae** (arXiv 2502.01433) rather than defaulting to the maximum. Beyond a point, more clean ancillas stop reducing Toffoli depth under conditionally-clean techniques. Still spend width rather than hoarding it, but check the formula.
- Relative-phase Toffolis are valid only under the conditions in II.6. Verify every candidate officially, never by reasoning.
- The transpiler sweep is boring and reliably worth 10–20%. Do it early, not last.
- Published constructions target T-count and qubit count. Re-measure everything in `u3`/`cx` depth before adopting it.

---

## Phase 6 — Close the loop · ~5%

**Goal:** feed Phase 5's real costs back into the Phase 1 proxy, then re-run Phase 4.

The first pass through Phase 4 used a proxy calibrated on naive circuits. After Phase 5 the cost landscape has changed shape — sharing is cheaper than assumed, deep AND chains worse. **The optimal cover under the new costs is a different cover.**

Re-calibrate on Phase 5-quality circuits, re-run the search. Expect one meaningful jump, then diminishing returns.

---

## Phase 7 — Freeze and validate · leave buffer

- Run the official verifier on the final QASM from a clean kernel, reading the saved file.
- Run additional random test states beyond the reference notebook's three.
- Confirm: width 12–18, single register `q`, only `u3`/`cx`, ancillas clean, coordinates preserved, phase pattern correct up to one global phase.
- Re-run end to end from scratch; confirm reproducibility.
- Record the exact seed and settings that produced the winner.

**Stop optimizing with time to spare.** A verified good circuit beats an unverified better one. Keep a verified submission saved at all times and replace it only when a new one verifies.

---

# Part VII — Instrumentation

## VII.1 Experiment log schema

One append-only row per measurement.

| Field | Purpose |
|---|---|
| `run_id`, `timestamp` | ordering |
| `cover_fingerprint` | hash, for dedup and cache lookup |
| `terms_json` | full reconstructable definition |
| `n_terms`, `n_ancillas`, `max_fanin`, `total_controls` | proxy features |
| `strategy_tags` | e.g. `["octagon", "mirror-paired"]`, `["xag", "caterpillar"]` |
| `source_tool` | which synthesizer produced it, if any |
| `proxy_depth`, `proxy_cx` | prediction |
| `real_depth`, `real_cx` | ground truth |
| `seed`, `opt_level` | reproducibility |
| `parent_run_id`, `move_applied` | which mutation produced this candidate |
| `wall_time` | cost accounting |
| `cache_hit` | False means this call actually paid for synthesis -- without it, timing statistics average two populations four orders of magnitude apart, and a search silently going in circles looks fast rather than broken |
| `cache_key`, `classiq_version`, `server_version` | the client SDK is not the compiler; the server's version comes from the QASM header |
| `syn_prefs`, `syn_constraints`, `trans_prefs` | full canonical settings, ~2.5 KB/row -- 25 MB at 10k runs, and the only thing that survives new knobs |
| `ok`, `max_error`, `ancilla_error`, `norm_error` | verification, when it was run; null when skipped |
| `error`, `required_width` | synthesis refusals, kept as data |

`parent_run_id` and `move_applied` are what make trend analysis possible. Without them you can accumulate ten thousand runs and learn nothing.

## VII.2 Trend analysis — run weekly

**Which moves actually work?** Group by `move_applied`; compute hit rate and mean improvement over parent. Drop near-zero-hit-rate moves; sample productive ones more often. Watch `reorder_terms` specifically — the literature predicts it should be productive.

**Is the proxy still honest?** Recompute Spearman(`proxy_depth`, `real_depth`) on **recent runs only**. Proxies drift as the search enters new regions. Re-fit when ρ drops.

**Which features predict depth now?** Regress `real_depth` on the proxy features. Largest coefficients indicate what to attack next.

**Track A vs Track B.** Group by `source_tool` and compare best-achieved depth. If one track dominates consistently, reallocate time toward it.

**Where are the returns?** Plot best-depth-so-far against cumulative wall time, per phase. A flattening curve is the signal to move on — this answers "when do I stop" with data rather than intuition.

**What does the frontier look like?** Scatter `real_depth` against `n_terms`, coloured by `strategy_tags`. Clusters reveal which structural families deserve exploration and which are dead.

## VII.3 Stop rules — decide in advance

- **Move on from a phase** when best-depth has not improved for a stretch equal to 25% of that phase's budget.
- **Abandon a strategy** when beaten by 20%+ on a fair harness *and* you understand why.
- **Stop entirely** near the sanity floor: the mask's ANF contains a degree-12 term, so the phase genuinely depends on all 12 bits jointly. Any circuit must build a 12-way dependency — roughly 4 levels of balanced AND, each a Toffoli's depth, then roughly doubled by uncomputation. That places an absolute floor in the low tens. An anchor, not a theorem.

## VII.4 Risks

| Risk | Mitigation |
|---|---|
| Proxy does not correlate with real depth | Phase 1 exit criterion catches it in week one |
| Seed variance swamps real differences | measured in 1.3; use min-over-N-seeds |
| Search collapses into one basin | MAP-Elites preserves structural diversity |
| Relative-phase Toffolis break correctness subtly | verify every candidate officially, never by reasoning |
| **External tools optimize T-count/qubits, not depth** | treat all tool output as seeds; re-measure in `u3`/`cx` depth |
| **Tweedledum packaging breaks on current Python** | resolve in Phase 0.8; pin Qiskit < 2.0 or install standalone |
| **C++ toolchain build consumes more time than expected** | timebox Phase 0.8; Track B proceeds without it |
| Over-fitting to the reference notebook's three test states | extra random states in Phase 7 |
| Running out of time mid-optimization | always keep a verified submission saved |

---

# Appendix A — Glossary

| Term | Meaning |
|---|---|
| **Amplitude** | one of the 4096 complex numbers describing the register's state |
| **Phase** | the direction of an amplitude; invisible to measurement, decisive under interference |
| **Global phase** | one shared rotation of all amplitudes; unobservable, therefore permitted |
| **Phase oracle** | a circuit applying $(-1)^{f(x)}$ without changing $x$ |
| **Ancilla** | scratch qubit; must start and end in $\lvert 0\rangle$ |
| **Uncompute** | running compute gates in reverse to restore an ancilla to zero |
| **Toffoli** | `target ^= (A AND B)`; a doubly-controlled NOT |
| **MCT / MCX** | multi-controlled Toffoli / multi-controlled NOT |
| **Cube** | a predicate fixing some bits and leaving others free; boundaries fall on powers of 2 |
| **ESOP** | exclusive sum of products — AND-terms combined by XOR; the parity cover |
| **ANF / PPRM** | algebraic normal form / positive-polarity Reed–Müller; the XOR-of-monomials expansion |
| **XAG** | Xor-And-Inverter Graph; multi-level logic network over $\{\oplus, \land, 1\}$ |
| **LUT / k-LUT mapping** | look-up table decomposition of a logic network |
| **Reversible pebble game** | the scheduling problem of computing and uncomputing intermediates within a fixed ancilla budget |
| **STG** | single-target gate; a reversible gate flipping one target under a Boolean condition |
| **`.pla`** | plain truth-table file format consumed by ESPRESSO and EXORCISM-4 |
| **Depth** | length of the longest sequential chain of gates; the score |
| **Transpile** | rewrite a circuit into a permitted gate set |
| **MAP-Elites** | a search keeping the best candidate per structural niche rather than one global best |

---

# Appendix B — Tools and references

## Software

| Tool | What it does | Where |
|---|---|---|
| **Tweedledum** | reversible/quantum synthesis in C++; `esop_phase_synth`, `pkrm_synth` | tweedledum.readthedocs.io |
| **Qiskit `PhaseOracle`** | Python wrapper over Tweedledum's phase-ESOP synthesis | docs.quantum.ibm.com |
| **EXORCISM-4** | minimized ESOP from `.pla` | used by MustangQ; part of the ESOP tooling ecosystem |
| **ABC** | multi-level logic synthesis, k-LUT mapping | Berkeley |
| **ESPRESSO** | two-level SOP minimization | classic |
| **mockturtle / easy / angel** | EPFL logic synthesis libraries | github.com/lsils |
| **caterpillar** | XAG-based quantum circuit synthesis with SAT-based pebbling under a qubit budget | EPFL |
| **RevKit** | reversible logic synthesis framework; hosts the LHRS flow | — |
| **Polylog_MCX-public** | polylogarithmic-depth MCX without ancillas | github.com/BaptisteClaudon |

## Papers

- *The EPFL Logic Synthesis Libraries* — arXiv:1805.05121
- *Xor-And-Inverter Graphs for Quantum Compilation* — npj Quantum Information, doi 10.1038/s41534-021-00514-y
- *Boolean satisfiability in quantum compilation* — Phil. Trans. R. Soc. A, doi 10.1098/rsta.2019.0161
- *Evaluating ESOP Optimization Methods in Quantum Compilation Flows* — Meuli et al., EPFL Infoscience
- *Scaling-up ESOP Synthesis for Quantum Compilation*
- *ROS: Resource-constrained Oracle Synthesis for Quantum Computers* — arXiv:2005.00211
- *A Divide-And-Conquer Pebbling Strategy for Oracle Synthesis*
- *Modeling and Resource Optimization for Quantum Oracles* (ShallowGrow) — arXiv:2605.21380
- *The Role of Multiplicative Complexity in Compiling Low T-count Oracle Circuits* — arXiv:1908.01609
- *Logarithmic Depth Decomposition of Approximate Multi-Controlled Single-Qubit Gates Without Ancilla Qubits* — arXiv:2507.00400
- *On Exact Space-Depth Trade-Offs in Multi-Controlled Toffoli Decomposition* — arXiv:2502.01433
- Maslov, *Advantages of using relative-phase Toffoli gates* — Phys. Rev. A 93, 022311 (2016)
- Barenco et al., *Elementary gates for quantum computation* — Phys. Rev. A 52, 3457 (1995)

## Prior competitions

- Classiq Coding Competition 2022 — winners and write-ups, including a multi-controlled Toffoli problem under a 20-qubit budget (14 controls, 1 target, ≤5 auxiliary).
