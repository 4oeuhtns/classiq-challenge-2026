# Classiq Phase Oracle Challenge — Notes

## 1. The foundation

**A quantum state is a table, not a set of dials.**

12 qubits do not give you 12 numbers. They give you a table with **4096 rows**, one row per possible address (x, y). This has to be true even classically: two linked coins can't be described by "coin 1 is 50/50, coin 2 is 50/50", because that description loses the fact that they always match. State lives on the *combinations*.

**Each row holds an arrow, not just a height.**

- Arrow **length** = how loud that address is → its probability if you measure.
- Arrow **direction** = the **phase**.

Two rows can be equally loud but point opposite ways. Measurement can't tell them apart.

**Phase matters because arrows add like waves.** Same direction → they reinforce. Opposite direction → they cancel. Noise-cancelling headphones are exactly this: same loudness, opposite phase, silence.

**Multiplying a row by −1 = turning its arrow half a turn = `phase(pi)`.**

**Global phase is free.** Turn *every* arrow by the same amount and nothing changes, because only relative directions affect interference. This is the `e^{iφ}` the spec allows.

**Superposition setup:** Hadamard on each qubit turns "only row 0 is up" into "all 4096 rows equally up."

**You cannot read the table.** Measuring returns one address at random. That's why quantum algorithms must use interference, not brute-force readout.

---

## 2. The task

Build a **phase oracle**: a circuit that multiplies the amplitude by −1 on exactly the 1097 black pixels of the Classiq logo, and leaves the other 2999 alone.

*Mailroom analogy:* 4096 envelopes on a conveyor belt, all passing at once. The machine reads addresses and stamps only the black ones. It never rewrites an address, never opens an envelope, never leaves scraps behind.

The oracle is a **component**, not a whole algorithm. Grover's, QAOA and quantum feature maps drop it in and then do interference steps that turn the invisible marks into loud, readable answers.

### Rules

| Rule | Meaning |
|---|---|
| Width 12–18 | 12 coordinate qubits + up to 6 ancillas |
| `q[0:6]` = x, `q[6:12]` = y | Fixed little-endian layout, one register named `q` |
| Ancillas return to \|0⟩ | Scratch must be erased |
| x and y unchanged | Coordinates in = coordinates out |
| Only `u3` and `cx` | Final basis after transpilation |
| Global phase allowed | One shared `e^{iφ}` is fine |

### Scoring

**Depth, not gate count.** Tie-break on CX count.

*Kitchen analogy:* gate count = number of knife cuts; depth = minutes until dinner. Many cooks chop in parallel, but nobody makes the sauce before the onions are done. Depth is the longest must-wait-for-me chain.

---

## 3. How a region becomes gates

### Phases live on rows, so a gate can flip one row

A gate can be **conditional**. A multi-controlled phase flip is a bouncer with a checklist: satisfy all conditions, get stamped; otherwise walk through untouched.

- 12 controls, all required → exactly one address out of 4096 fires.
- X gates before and after relabel which bit pattern counts as "all ones", so any address can be targeted. **X gates are depth-1 and essentially free.**

The gate does not *search*. It's one physical operation applied to the whole superposition; it's shaped so only matching rows are affected.

### Ranges are bit-pattern checks, not comparisons

x is 6 bits `b5 b4 b3 b2 b1 b0`.

- `x ≥ 32` → is `b5 = 1`? One qubit.
- `16 ≤ x ≤ 31` → are `b5 b4 = 01`? Two qubits.

You never compare numbers. You look at the **top bits**, which already say which chunk of the number line you're in. Like reading the first digit of a house number to find the block.

**Ranges that don't line up with powers of 2 get chopped into several aligned blocks:**

`x ≥ 26` = `1?????` (32–63) OR `0111??` (28–31) OR `01101?` (26–27). Three checks instead of one.

> A range is cheap when it's one clean block of bit patterns, expensive when it's a pile of them. Moving a boundary by one pixel can double the cost.

### Ancillas: build the AND as a bracket, not a queue

To test `A AND B AND C AND D`:

- **Queue:** four sequential steps. Depth 4.
- **Bracket:** (A,B) and (C,D) in parallel, then combine. Depth 2.

Ancillas hold the round-1 results.

```
anc0 ^= x[5] AND x[4]      # anc0 = 1 exactly when x >= 48   |  these two
anc1 ^= y[5] AND y[4]      # anc1 = 1 exactly when y >= 48   |  run in parallel

flip phase, controlled on anc0 AND anc1

anc1 ^= y[5] AND y[4]      # uncompute
anc0 ^= x[5] AND x[4]
```

- `target ^= (A AND B)` is a **Toffoli**: NOT the target, but only when both controls are 1. Controlled-NOT and XOR are the same operation under two names.
- **Uncompute works because XOR twice is identity.** The controls weren't touched in between, so the second gate writes the same value and returns the ancilla to 0.
- The phase flip is controlled on the **ancillas**, not on x and y. That's the point of scratch: summarize many conditions into few.
- **Why erasure is mandatory:** a leftover ancilla records which pixel was being examined. That record entangles with the coordinates and destroys the interference the oracle exists to enable.
- Because x and y are in superposition, the ancillas are too: `anc0` reads 1 on the rows where x ≥ 48 and 0 elsewhere, simultaneously.
- A Toffoli is not a `cx`. The transpiler expands each one into roughly six CXs plus single-qubit gates. **Think in Toffolis, get scored in CXs.**

---

## 4. What the baseline notebook does

**Step 1 — Describe the logo with math.** `logo_pixel(x, y)` returns True for a square, a bar, and two disks via `(x-a)² + (y-b)² ≤ r`. Plain Python, no quantum.

**Step 2 — Tile it with rectangles.** Find runs of black pixels per row, extend a box upward when the run above is identical. Result: **18 disjoint rectangles**, heights 1 to 10.

Disjointness is enforced with a hard error, because **two stamps cancel**: −1 × −1 = +1 would turn a black pixel white. Like a second coat of paint erasing the first.

**Step 3 — One bouncer per rectangle.**

```python
for x_min, x_max, y_min, y_max in ORACLE_RECTANGLES:
    control((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max),
            lambda: phase(pi))
```

The notebook never allocates ancillas by name. `max_width=18` is *permission*; Classiq's synthesizer decides how many of the 6 to use, the way a C compiler assigns registers.

**Step 4 — Synthesis harness.**

```python
allocate(x); allocate(y)
hadamard_transform(x); hadamard_transform(y)
logo_phase_oracle(x, y)
```

Without the Hadamards the compiler only sees x = y = 0 and could optimize 17 of the 18 rectangles away. The Hadamards force it to see all 4096 addresses live. They are **scaffolding**: present during construction, removed before delivery, because the grader supplies its own superposition.

**Step 5 — Extract and transpile.** Export QASM, delete exactly the two `hadamard_transform_` lines, rebuild, transpile to `u3`/`cx` with all-to-all connectivity, save `submission.qasm`.

**Step 6 — Verify.** Reads the saved file only. Checks width 12–18, single register `q`, only `u3`/`cx`. Then runs three equal-magnitude product-phase states covering all 4096 coordinates and confirms: coordinates preserved, ancillas clean, normalized, and the phase pattern correct up to one shared global phase.

---

## 5. Where the wins are

The baseline is deliberately naive. Its weaknesses:

- The disks become **staircases** — 9 boxes for one disk, 8 for the other, many only 1–2 pixels tall. Each costs its own control.
- Every rectangle is built independently. Nothing is shared.

### Directions

**Parity covers instead of partitions.** Disjointness is the *baseline's* rule, not the challenge's. The grader only checks the final phase pattern. Since two stamps cancel, you can stamp a large rectangle and then stamp a smaller one inside it to un-stamp the excess. Inclusion–exclusion often needs far fewer and far larger regions.

**Predicate sharing.** Compute `x ≥ 32` onto an ancilla once and reuse it across every region that needs it, instead of rebuilding it 18 times.

**Boundary alignment.** Re-cut regions so their edges land on powers of 2 wherever the exact mask allows. One clean block beats three ragged ones.

**Symmetry.** Both disks are mirror-symmetric, so the staircase rows come in matched pairs. A condition on the distance from the centre row can serve two rows at once.

**Parallelism.** Depth, not count, is the score. Independent sub-checks placed in the same column are free relative to sequential ones. Balanced Toffoli trees beat chains.

**Arithmetic circuits (probably not worth it here).** Building `(x-55)² + (y-41)² ≤ 42` directly needs multipliers and their own scratch registers, against a budget of 6 ancillas. These disks are small enough that block checks win. The trade-off would flip for much larger shapes.

**Additive threshold shifting (probably not worth it).** A real quantum adder to move `x ≥ 26` to `x ≥ 32` costs an adder plus its inverse — more depth than the blocks it saves. The free version of this idea is XOR with a constant (X gates), which relabels bit patterns at depth 1 but can only map one clean block onto another.

---

## Vocabulary

| Term | Plain meaning |
|---|---|
| Amplitude | The arrow on one row of the 4096-table |
| Phase | Which way that arrow points |
| Global phase | Turning every arrow equally; invisible and free |
| Superposition | Many rows non-zero at once |
| Entanglement | The table doesn't factor into independent per-qubit descriptions |
| Ancilla | Scratch qubit; must return to \|0⟩ |
| Uncompute | Run the compute gates in reverse to erase scratch |
| Toffoli | `target ^= (A AND B)` |
| Oracle | The circuit that marks the answers |
| Depth | Longest chain of must-wait-for-me steps; the score |
| Transpile | Rewrite the circuit using only allowed gate types |