# Phase 0 — Infrastructure · Study Notes

*Companion to `plan.md` Part VI, Phase 0. The conceptual layer underneath the task list.*

---

## 1. What Phase 0 is

**The workbench, not the work.**

Phase 0 produces zero good circuits. It produces the *ability to judge* circuits — cheaply, repeatably, and without lying to you.

The plan states the exit condition in one line: **turn a list of terms into a verified depth number with one function call.**

Three facts force this:

1. You will try hundreds of designs. You cannot know which wins by thinking.
2. The only real score comes from a slow machine (the transpiler). Seconds to minutes each.
3. **"Is it correct?" and "is it good?" are different questions** — and the first one is free.

Fact 3 is what makes the whole project feasible.

### Kitchen analogy

| Kitchen | Project |
|---|---|
| recipe card | **term list** — the design, as data |
| tasting a spoonful | **`matches_target()`** — numpy XOR |
| oven + scale | **`measure()`** — real transpile, real depth |
| not re-baking a cake you already weighed | **disk cache** |
| lab notebook | **experiment log** |
| a judge saying "yes, edible" | **official verifier** |

---

## 2. The three boxes — never merge them

| | Question | Method | Cost | Exact? |
|---|---|---|---|---|
| `matches_target()` | is the picture right? | numpy XOR | µs | **exact, always** |
| `measure()` | how deep, really? | real transpile | s–min | it *is* the truth |
| proxy *(Phase 1)* | roughly how deep? | a formula | µs | **never exact** |

**Phase 0 builds boxes 1 and 2 only.** Box 3 is Phase 1, and its bar is Spearman **ρ ≥ 0.8** — a *correlation*, not equality. Nothing ever replicates the transpiler exactly, and nothing needs to. A proxy only has to *rank* well enough to steer.

`measure()` does **not** dodge the transpiler. It runs it for real. The savings come from elsewhere:

- wrong designs are thrown out for free before the transpiler ever sees them
- the same design is never transpiled twice (cache)

---

## 3. Why correctness is free

Ask what the circuit is permitted to do:

- x and y come out unchanged
- ancillas come out at zero
- all it does is attach a phase

So one address goes in, and the *same* address comes out with a `+` or a `−` on it. That is the entire menu. Nothing moves, nothing mixes, nothing spreads.

> **The complete behaviour of the circuit is a list of 4096 plus-or-minus signs.**

The word for this is **diagonal**: every amplitude is scaled in place. There is nothing else to simulate.

And a sign is set by one question: **was this pixel stamped an odd number of times?** That is XOR. That is numpy.

```python
mask1 ^ mask2 ^ mask3 == TARGET
```

This is **not an approximation** of the simulation. It *is* the simulation, in its shortest form. Same answer, a million times faster.

**Recipe analogy.** 3 eggs, 200g flour, 100g sugar — whether it makes a cake or a brick is decided by arithmetic on the card, no oven required. How many *minutes* it takes depends on the oven, what else is in it, and how the trays are arranged. Two different questions.

### Only parity matters

Stamped 3× = stamped 1× = **black**.
Stamped 4× = stamped 0× = **white**.

Covers may overlap freely. Disjointness is the *baseline's* self-imposed rule, not the challenge's. Cover a big cheap region, then cover the unwanted part again to remove it.

### Diagonal gates commute

Stamping A then B is identical to stamping B then A — multiplication does not care about order.

- **Correctness:** immune to ordering, seeds, transpiler settings. Exact.
- **Depth:** sensitive to all of it. Must be measured. Noisy.

Reordering terms swings circuit size by up to **2×** (plan §IV.3) and can never break the answer. A free lunch, safe to eat often. This is why `reorder_terms` is in the Phase 4 move set.

---

## 4. Design must be data, not code

The baseline bakes its 18 rectangles *inside* a quantum function. To try a different design you rewrite the function. **A search program cannot rewrite a function — but it can shuffle a list.**

```
  list of terms  ──render()──▶  64×64 mask  ──▶  correct?   (fast, free)
  (plain Python) ──compile()─▶  circuit     ──▶  how deep?  (slow, costly)
```

One recipe card, two things you can do with it. A card can be photocopied, mutated, hashed, mailed, stored. A meal cannot.

This is why plan §0.3 says *make it general on day one or you will rewrite everything in Phase 3.*

Practical consequences:

- terms must be **immutable and hashable** → cache keys (`fingerprint`) and log rows (`terms_json`)
- terms must be **cheap to mutate** → Phase 4's move set

---

## 5. The layers, and their three verbs

A design's correctness is not the circuit's validity. Different layers own different rules.

| Rule | Layer | Verb |
|---|---|---|
| phase pattern matches the logo | design | **checked** — `matches_target()`, numpy |
| width 12–18 (≤ 6 ancillas) | compile | **constrained** — `max_width=18`; synthesis obeys or fails loudly |
| ancillas end at \|0⟩ | compile | **constructed** — always emit the exact inverse |
| only `u3`/`cx`, one register `q` | transpile | **constructed** — you set the basis gates |
| all of the above, for real | verify | **confirmed** — official verifier on the saved file |

Only **one** thing in Phase 0 is genuinely *checked*, and that is the pattern.

### Clean ancillas are constructed, not tested

Only ever emit this shape:

```
compute  →  phase(π)  →  exact inverse of compute
```

`anc ^= (A AND B)` run twice, with A and B untouched in between, writes the same value twice — and XOR twice is identity. Back to zero by algebra, not by luck.

It is like matched brackets: you do not write a bracket-checker, you never type `(` without `)`. Classiq's `within_apply` emits the inverse for you.

### A check can only live where the facts live

**Does a term list determine how many ancillas get used?** No. The same term list compiled as one giant multi-controlled gate, a balanced AND tree, or a chain uses different numbers of ancillas.

So "≤ 6 ancillas" is a property of *design + compile strategy*, not of the design. It cannot be checked at the design layer because the design genuinely does not know yet.

> **You can only check a constraint at the layer where enough information exists to determine it.**

In Phase 4, ancilla count is not a pass/fail gate at all — it is an **axis of the MAP-Elites grid** (2, 3, 4, 5). A dial to explore, not a rule to satisfy.

### Naming

Call it `matches_target()`, not `is_valid()`. Naming a function after the *one* question it answers is how you stop yourself quietly stuffing four more checks into it three weeks later.

---

## 6. Uncomputation costs depth — measured, not ignored

Plan §II.2:

> depth ≈ 2 × depth(compute) + 1

Uncomputation roughly **doubles** everything. This is the dominant cost, and it is captured by `measure()` rather than predicted. In Phase 1 it becomes explicit proxy feature **P5** (count of compute/uncompute pairs). P4/P5 are where the proxy typically starts working — because ancilla pressure is what creates serialization.

---

## 7. Depth is noisy

Same design + different transpiler seed = different number. Do not fight it.

- `measure()` returns the **min over N seeds**. Fair, because at submission time you keep the best you found.
- Measure the wobble in week one (plan §1.3, ~20 seeds on one design).
- **If seed variance is comparable to the gap between designs, the search is chasing noise.** Finding that out early is the point.

---

## 8. The Phase 0 bargain

> Pay once to prove the cheap checker agrees with the official verifier — then use the cheap checker a million times.

That is what this exit criterion means:

> `matches_target()` agrees with the official verifier on **10 hand-built cases.**

Include deliberately broken designs. Agreement on correct cases alone proves nothing.

---

## 9. Exit criteria (from plan.md)

- [ ] `measure(baseline_terms)` returns the baseline's real depth
- [ ] `matches_target()` agrees with the official verifier on 10 hand-built cases
- [ ] at least one external synthesizer runs end to end and produces a measured depth
- [ ] every run lands in the log

---

## 10. Environment notes (this machine)

- Work in **WSL Ubuntu**, conda env **`quantum`**, Python 3.13.14.
- Present: numpy 2.4.6, scipy, pandas, matplotlib, qiskit 2.4.0 (+ aer, runtime).
- **Missing: `classiq`** — install and authenticate once. Blocker for 0.1 and 0.5.
- qiskit is **2.4.0**, i.e. past 2.0, so `classicalfunction` / `PhaseOracle` / tweedledum are **already removed** — plan §IV.1's warning is live. Pin an older qiskit in a *separate* env, or install tweedledum standalone. Do not downgrade the working env.

---

## Vocabulary added this lesson

| Term | Plain meaning |
|---|---|
| Diagonal | every amplitude scaled in place; nothing moves between addresses |
| Term / term list | the design, expressed as data — the recipe card |
| Render | turn one term into a 64×64 boolean mask |
| Cover mask | XOR of all rendered terms |
| Proxy | a fast formula that *predicts* depth; Phase 1, never exact |
| Fingerprint | hash of a term list; the cache key |
| Seed variance | spread in measured depth from re-running the same transpile |
