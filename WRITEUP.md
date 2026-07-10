# Spec-First Geometry → TikZ — Project Write-up

**Behavior thesis, the falsifiable gate, and whether data → behavior held (with evidence).**

A small open model (Qwen3-0.6B / 1.7B) fine-tuned to do *one* narrow thing reliably:
turn a **coordinate-free** geometry scene description into a **single compiling TikZ
figure whose every named point is numerically correct** — recovering the hidden numbers
from the relationships, not transcribing them.

---

## TL;DR — did data → behavior hold?

**Yes, decisively.** Controlling the training data moved the target behavior from
"never happens" to "reliable," and a mid-project change to the *target representation*
turned the single hardest failure mode from ~unsolved to near-perfect.

| Milestone | Pass rate (falsifiable gate) | Evidence |
| :-- | :-- | :-- |
| Base Qwen3-0.6B (prompted) | **0.003** | `outputs/eval_base_new.json` |
| Tuned 0.6B (v1, numeric target) | **0.464** | `outputs/eval_tuned.json` |
| Tuned 1.7B (v1, numeric target) | **0.598** | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B (v2, PGF-construction target) | **0.989** | `outputs/eval_pgf_tuned.json` |

The deepest finding: **the target representation is the dominant lever** — bigger than
model size *or* data volume. Asking the model to *emit a coordinate-free construction*
and letting PGF compute the numbers beat asking a 3×-bigger model to compute the numbers
itself.

Beyond the core gate, this write-up also shows the specialist is **useful, not just
trained** (utility eval + three product surfaces, §9), and stretches the same recipe to a
**1.7B AIME auto-illustrator** with an honest ceiling — it *draws* a figure 69% of the time
but the *correct* one only ~11% for arbitrary hard problems (§10), while hitting **93.8%**
coordinate-verified on in-distribution synthetic scenes.

---

## 1. Thesis / spiky POV

> You can make a small model *reliably* do one narrow thing by controlling its training
> data — and **how you frame the target (compute the answer vs. emit the construction)
> matters more than model size or dataset size.**

The assignment's rule is "the dataset is the deliverable." This project takes that one
step further: the single most important dataset decision here was not *how many* examples
or *which difficulty*, but **what the ground-truth output should be**. The same behavior
("draw this coordinate-free scene correctly") has two valid targets:

- **v1 — numeric:** the model must *compute* every coordinate and place literal points
  (`\filldraw (4.33,2.50) ...`). The model is the calculator.
- **v2 — construction:** the model emits a coordinate-free PGF construction
  (`($(A)!(C)!(B)$)` = "foot of the perpendicular from `C` onto line `A`–`B`") and **PGF
  does the arithmetic** at compile time. The model is the *geometer*, not the calculator.

Both are trained purely by data. v2 is where the spiky POV earns its keep.

---

## 2. The behavior spec (the falsifiable gate)

From `BEHAVIOR_SPEC.md` — one sentence a stranger can grade, and simultaneously the
data-gen rubric, the eval criterion, and the thesis:

> Given a constraint-level geometry scene with **no explicit coordinates**, the model
> returns a **single valid TikZ/PGF figure** that **compiles** and whose **every named
> point matches the ground-truth construction within `atol = 0.05`**, with **no prose**
> before or after the code.

An output **passes iff all of**: (1) figure-only, (2) compiles under `tectonic`,
(3) every named coordinate within tolerance of ground truth, (4) *derived*, not
transcribed (the input never stated the coordinates). This is a hard AND — a beautiful
figure that gets one point wrong fails; correct numbers wrapped in prose fail.

Difficulty has two dials: **chain length** (how many derivation steps compose) and
**number irregularity** (round `30°/r=5` numbers are fakeable; irregular `161°/r=2.5`
forces real computation). The clean base-vs-tuned delta lives at *multi-step + irregular*.

---

## 3. Litmus — a well-prompted frontier model can't already do it reliably

`scripts/difficulty_sweep.py` scored **12 hosted frontier models** on an **800-item
difficulty grid** (chains 2–7 × round/irregular, 40/cell, plus op-targeted cells) with
**the exact same pass gate**. This is the whole game: if prompting already nailed it,
fine-tuning would be pointless.

**Result: at a strict "reliable every time" bar, no frontier model is reliable across the
landscape.**

- On *average across the 12 models*, pass rate is **below 90% in 19 of 20 difficulty
  cells** (only chain-2-irregular clears 90%, at 0.94).
- At a **95%** reliability bar, **zero** models are reliable across all 12 landscape cells.
- At a looser **90%** bar, **exactly one** squeaks by — `claude-sonnet-5`, whose *worst*
  cell is 0.949 (chain-6/7 irregular). Every other model fails at 90% somewhere.

Overall pass rate on the identical 800-item grid (this is also the base-vs-tuned-vs-SOTA
comparison surface):

| Model | Overall pass (800-grid) |
| :-- | --: |
| claude-sonnet-5 | 0.977 |
| gemini-3.1-pro | 0.909 |
| gpt-5-mini | 0.901 |
| grok-4.5 | 0.879 |
| gpt-5.5 | 0.870 |
| gemini-3.5-flash | 0.804 |
| gpt-5.4 | 0.685 |
| claude-opus-4-8 | 0.674 |
| **tuned-1.7B (ours)** | **0.598** |
| gpt-4.1 | 0.555 |
| **tuned-0.6B (ours)** | **0.464** |
| gpt-4o | 0.439 |
| deepseek-v3.2 | 0.270 |
| claude-haiku-4-5 | 0.173 |
| base-1.7B (ours) | 0.005 |
| base-0.6B (ours) | 0.003 |

*Source: `outputs/sweep/results.json`, `outputs/eval_*` JSONs. Full 12×12 heatmap in
`outputs/sweep/report.md`.*

**Where reliability breaks:** irregular numbers **and** a hard construction
(foot-of-altitude, then line-intersection), from chain ~4. The op-targeted cells isolate
this — same chain length, different final operation, pooled across all 12 models:

| Cell | Operation | Chain | Pooled pass |
| :-- | :-- | --: | --: |
| easy_c4_irr | easy only (control) | 4 | 0.88 |
| int_c4_irr | line intersection | 4 | 0.58 |
| foot_c4_irr | foot of altitude | 4 | 0.44 |
| foot_c5_irr | foot of altitude | 5 | 0.38 |

The *operation* — not merely the chain length — is what collapses reliability:
foot-of-altitude already beats prompting at short chains (chain 4 → 0.44, chain 5 → 0.38),
versus 0.88 for an easy-op control at the same chain 4. **That is exactly where a
fine-tune earns its keep**, and where the training data is centered
(`scripts/build_dataset.py`).

---

## 4. Eval design — built before training

Three channels, all objective and reproducible (`src/geotikz/{harness,metrics,tex}.py`):

1. **Objective behavioral gate** — figure-only + compile (`tectonic`) + coordinate
   assertion. A static TikZ parser recovers each named point (handling polar, `calc`
   projections `($(a)!(c)!(b)$)`, and `name intersections`) and checks it within `0.05`.
2. **LLM-as-judge harness** — `src/geotikz/judge.py` implements the spec's 0/1/2 rubric
   (spec adherence / robustness / task quality / consistency); available for cross-check.
3. **Base-vs-tuned on the identical held-out grid the SOTA models saw** — the eval set is
   materialized from the same `outputs/sweep/grid.jsonl`, so tuned vs. base vs. frontier
   is apples-to-apples on the same 800 items.

The headline rubric numbers below are computed **deterministically from the objective
gate components** (not the LLM judge — the judge is available but was not needed to prove
the effect); they are exact and reproducible from the eval JSONs.

---

## 5. Results — v1 (numeric target)

**Pass rate (the falsifiable gate):**

| Model | Base | Tuned | n passed (of 800) |
| :-- | --: | --: | --: |
| Qwen3-0.6B | 0.003 | **0.464** | 2 → 371 |
| Qwen3-1.7B | 0.005 | **0.598** | 4 → 478 |

Both tuned models land **mid-pack among frontier models on the identical grid**:
tuned-1.7B (0.598) edges **gpt-4.1** (0.555) and clears **gpt-4o** (0.439),
**deepseek-v3.2** (0.270), **claude-haiku-4-5** (0.173); tuned-0.6B (0.464) also clears
gpt-4o/deepseek/haiku. The top tier (`claude-sonnet-5` 0.977) stays ahead — **as
expected**. The defensible win is not raw capability; it is a *reliable, cheap, local
specialist* plus a base→tuned jump from **2/800 to 371/800** (0.6B) that is unambiguously
the data's doing.
(The base models score near-zero largely because they don't hold the output-format
constraint — base-1.7B compiles just 1.5% of the time, wrapping figures in prose/markdown
— which is precisely the reliability that tuning buys.)

**Rubric, Qwen3-0.6B base → tuned** (0/1/2 for spec adherence & task quality; robustness
= pass rate on hard chains ≥4; consistency = pass rate on short chains ≤3):

| Dimension | Base | Tuned |
| :-- | --: | --: |
| Spec adherence | 0.93 | **1.43** |
| Robustness (chain ≥ 4) | 0.00 | **0.30** |
| Task quality | 0.58 | **1.43** |
| Consistency (chain ≤ 3) | 0.01 | **0.84** |

The rubric's stated win condition is *"beats base on Spec adherence AND Robustness."* The
tuned model does exactly that (0.93→1.43 and 0.00→0.30). **Win.**

---

## 6. Error analysis → the pivot

The tuned model's residual failures were not spread out; they concentrated on **one
operation: foot-of-altitude.**

| foot-of-altitude pass (v1 numeric) | value |
| :-- | --: |
| tuned-0.6B | 0.02 |
| tuned-1.7B | 0.13 |
| base (either) | ~0.00 |

Inspecting outputs, the model reliably got the **structure** right — the right points, the
right segments, `A` on the circle, the base line drawn — but botched the **projection
arithmetic** for the foot itself. Scaling model size 0.6B→1.7B barely moved it
(0.02→0.13). Foot-of-altitude is *heavily* represented in training — it appears in
**2,495 of 5,340 examples (≈47%)** of the current `data/train.jsonl`, more than
line-intersection (25%) — so **more data of the same kind was not the fix.** This is a
**capability/representation problem, not a data-volume one** — the model is being asked to
be a floating-point calculator, and small models are bad calculators.

---

## 7. v2 — fix it in the data (change the target representation)

Instead of tuning hyperparameters or adding yet more foot examples, **change what the
ground truth *is*.** `scripts/build_pgf_proto.py` regenerates the same failure-region
scenes but with **coordinate-free PGF constructions** as the target:

```latex
% v1 numeric target — the MODEL must compute the foot:
\filldraw (3.87,1.42) circle (1.5pt) node[below] {$D$};

% v2 construction target — the model emits the construction, PGF computes the number:
\coordinate (D) at ($(A)!(C)!(B)$);   % foot of perpendicular from C onto line A–B
\filldraw (D) circle (1.5pt) node[below] {$D$};
```

Same base model (Qwen3-0.6B), same training recipe, new target. On a 280-item symbolic
eval (chains 3–5, `outputs/eval_pgf_*.json`):

| Metric | Base | Tuned |
| :-- | --: | --: |
| Overall pass | 0.000 | **0.989** |
| Compile rate | 0.536 | **1.000** |
| Foot-of-altitude pass | 0.000 | **0.984** |
| Intersection pass | 0.000 | **0.989** |

The base model still scores **0.000** — so v2 is a **genuine trained behavior**, not the
task getting easier. The **fair, cell-matched comparison** (same construction, before vs.
after the representation change) is the headline:

| Construction | v1 numeric, tuned-0.6B | v2 PGF, tuned-0.6B |
| :-- | --: | --: |
| Foot-of-altitude | 0.02 | **0.98** |
| Line intersection | 0.20 | **0.99** |

> **Honest caveat:** the v2 eval set (280 items, chains 3–5) is **not identical** to the
> v1 grid (800 items, chains 2–7), so the fair comparison is the **cell-matched
> per-construction** rows above (0.02→0.98, 0.20→0.99), **not** the overall 0.99 vs 0.46.
> Even so, on a same-model, same-recipe basis, offloading the arithmetic to PGF resolved
> the wall. This is a textbook *"diagnose the failure mode, fix it in the DATA, not the
> hyperparameters"* iteration — and the fix was **representation**, the spikiest lever.

---

## 8. Did data → behavior hold?

**Yes — three independent ways:**

1. **Base → tuned delta.** 0.003 → 0.464 (0.6B) and 0.005 → 0.598 (1.7B) on a held-out
   grid the base model essentially never passes. Behavior was instilled by data.
2. **Rubric win.** Tuned beats base on Spec adherence *and* Robustness — the rubric's
   explicit success criterion.
3. **Representation pivot.** Holding model and recipe fixed and only changing the
   *target representation in the data* took the hardest construction from 0.02 to 0.98.
   The dataset — specifically what the label *is* — was the lever, exactly as the thesis
   claims.

---

## 9. Beyond the gate — is the specialist actually *useful*?

Passing a gate proves the behavior was trained. The assignment's real bar is a
*defensible reason to exist*: **reliable, cheap, local behavior that's hard to get from
prompting.** `scripts/utility_eval.py` scores the v2 specialist against two frontier
models on the specialist's in-domain task (`data/eval_pgf.jsonl`, n=30), with **one grader
for everyone** (`geotikz.extract.grade`). Frontier models are run in two modes: *plain*
(the specialist's own training prompt — a fair correctness comparison) and *construction*
(`CONSTRUCTION_SYSTEM_PROMPT` — asked for the same coordinate-free constructions).

| Config | Pass | Compile | Coord-free* | Latency/call (median) | Est. cost/call |
| :-- | --: | --: | --: | --: | --: |
| **specialist** (Qwen3-0.6B+LoRA, local) | **100.0%** | **100.0%** | 100.0% | 42.0s (Mac/MPS) | **$0 (local)** |
| gpt-5.5 [plain] | 96.7% | 100.0% | 20.0% | 25.8s | ~$0.005 |
| gpt-5.5 [construction] | 100.0% | 100.0% | 86.7% | 14.8s | ~$0.003 |
| claude-opus-4-8 [plain] | 90.0% | 96.7% | 46.7% | 7.8s | ~$0.029 |
| claude-opus-4-8 [construction] | 63.3% | **66.7%** | 96.7% | 5.1s | ~$0.015 |

*coord-free = share of outputs using a coordinate-free construction primitive vs. bare
numeric coordinates. Pricing is an order-of-magnitude estimate; tokens counted with the
Qwen3 tokenizer. Source: `outputs/utility_report.md`, `outputs/utility_eval/results.json`.*

Three honest takeaways:

- **Parity, not superiority.** On its own narrow slice the 0.6B specialist passes 100% —
  the same ballpark as the frontier models (best 96.7–100%). The claim is *parity at the
  task*, not "smarter."
- **The reliability gap is real and one-directional.** When frontier models are asked for
  the *same coordinate-free constructions*, their compile rate **drops** — opus falls to
  **66.7%** because it hallucinates `tkz-euclide` macros. The specialist has no such gap:
  it only ever emits its trained, compiling `calc`/polar dialect (100% compile, 100%
  coord-free). That guaranteed-well-formed-output property is exactly what a fine-tune buys
  and a prompt cannot.
- **Economics decide bulk use.** The specialist is **$0 at the margin and fully offline**
  (no network, no rate limits). Illustrating thousands of in-domain scenes is free and
  parallel-local; every frontier call costs money and a round-trip. (The 42s local latency
  is MPS-bound on an 8 GB laptop; batched on a commodity GPU the same model is sub-second —
  see the Modal illustrator run below.)

Three product surfaces serve exactly this specialist (all coordinate-free by construction):

1. **Interactive demo** — scene text → rendered figure + copyable TikZ, specialist-first
   with an optional frontier fallback (`scripts/demo.py`, `scripts/demo_web.py` — a Gradio
   web UI).
2. **Worksheet generator** — `scripts/make_worksheet.py` draws N in-vocabulary problems
   from the project's own generators (so every figure is guaranteed in-distribution with
   exact ground truth) and emits a printable worksheet PDF + a separate answer-key PDF,
   compiled with `tectonic` (`outputs/worksheets/worksheet.pdf`, `answer_key.pdf`; 8/8
   figures OK). This is the surface that plays *to* the specialist's strength.
3. **AIME auto-illustrator** — the honest OOD stress test (next section).

---

## 10. Scaling the idea — distilling a 1.7B AIME illustrator

The synthetic specialist is narrow by design. The natural stretch: can the *same
data → behavior* recipe make a **local** model auto-illustrate *arbitrary* competition
(AIME/AMC/MATH) geometry? This is a genuinely harder, out-of-distribution target, and the
result is reported with the honesty it demands (`ILLUSTRATOR_REPORT.md`).

**Method (data, again, is the lever).** Two additive levers, no new training tricks:
1. **Distillation, hard-filtered** (`scripts/distill.py`). 1,588 real competition geometry
   problems → teacher (`gpt-5.5`, construction prompt) → **two-stage filter**: compile +
   non-degenerate, then a **vision judge** (`gemini-3.1-pro`, shown the rendered PNG) keeps
   only faithful (problem → figure) pairs. Yield: **1,588 → 1,120 compile (70.5%) → 1,099
   vision-approved (69.2%)**.
2. **Generator breadth, ground-truth-verified** (`src/geotikz/olympiad_ext.py`): 12 new
   construction families (intersection, parallelogram, midsegment, reflection, rotation,
   square, regular polygon, two-circle, antipode, …), each **round-tripped** through the
   compile-extract grader (emit → compile → read back == exact GT) at 100% at sampled
   counts.

Combined into **3,996 records**, trained as **Qwen3-1.7B + LoRA (r=32, α=64, all-linear,
2 epochs, bf16)** on a Modal A100 — a *new* adapter `qwen3-illustrator`; the existing
adapters and data are untouched.

**Result — three explicitly-separated signals (weakest → strongest), never conflated.**
Real AIME has no ground-truth coordinates, so "correct" is split into *compiles*,
*judge-verified faithful* (vision), and — on synthetic only — *coordinate-verified*.

AIME, 150 held-out geometry problems (seed 20260709):

| Signal | Before — `qwen3-pgf-geotikz` (0.6B) | After — `qwen3-illustrator` (1.7B) |
| :-- | --: | --: |
| compile + non-degenerate (local) | 14.0% (21/150) | **69.3%** (104/150) |
| judge-verified / faithful (local) | 0.7% (1/150) | **11.3%** (17/150) |
| union w/ judge-gated frontier fallback | — | **64.0%** (96/150) |
| compile coverage, any route | 69.3% | 87.3% |

Held-out **synthetic** (240 problems, coordinate-verified against ground truth):

| | base Qwen3-1.7B | tuned `qwen3-illustrator` |
| :-- | --: | --: |
| coordinate-verified pass | 7.9% (19/240) | **93.8%** (225/240) |
| compile | 17.1% | 98.3% |

**Read this honestly:**

- **The clean data → behavior result is 7.9% → 93.8%** (coordinate-verified, in-distribution)
  — a provable 12× lift from the same recipe.
- **On arbitrary hard AIME the model reliably *draws a figure* (69.3%) but the *correct*
  one only ~11.3% of the time.** That gap is the honesty the vision judge buys: a figure
  can compile and still depict the wrong thing. A 1.7B cannot match a frontier model's
  *reasoning* on novel olympiad problems — coverage here is **reasoning-bound, not
  drawing-bound.**
- **The old "14% coverage" was almost entirely spurious.** Re-judging the 0.6B specialist's
  compiling AIME figures shows only **1/150 (0.7%)** were actually faithful; the honest
  before→after is **0.7% → 11.3% faithful** (16×), not the compile number.
- The full system (local + **judge-gated** frontier fallback — a local figure is used only
  if the vision judge approves it) faithfully illustrates **64%**, ≈ the frontier's own
  faithful ceiling on this sample, while doing **11.3% for free locally**.

> **Capacity probe — 4B** (`scripts/train_illustrator_4b_modal.py`, base `Qwen/Qwen3-4B`,
> adapter `qwen3-illustrator-4b`, same data + recipe). More capacity roughly **doubled faithful
> local AIME coverage: 11.3% → 24.0%** (36/150) while compile coverage stayed flat (~70%) — the
> model drew the *right* figure more often (compile→faithful conversion 16.3% → 34.3%), not more
> figures. Synthetic coordinate-verified rose **93.8% → 97.1%**, and the full system (local +
> judge-gated fallback) reached **68% faithful**. Still well short of the teacher's ~64% faithful
> ceiling on its own — 4B closed only ~a quarter of the gap; a 4B does not reason like gpt-5.5 on
> novel hard problems. Details: `ILLUSTRATOR_4B_REPORT.md`.

---

## 11. Olympiad litmus — where the *next* fine-tune earns its keep

Before generating an olympiad dataset, the same evidence-first loop asks: *is a specialist
even needed here?* `scripts/olympiad_sweep.py` scored **4 frontier models × 8 named
constructions** (circumcenter, incenter, orthocenter, centroid, angle-bisector,
foot-of-altitude, median, tangent) on `data/olympiad_eval.jsonl` (120 items), given a
`tkz-euclide`-capable preamble, graded by the compile-extract grader.

| Model | Overall pass (120) | Weakest cell |
| :-- | --: | :-- |
| gemini-3.1-pro | 0.99 | angle-bisector 0.93 |
| grok-4.5 | 0.98 | angle-bisector 0.87 |
| gpt-5.5 | 0.975 | incenter 0.87 |
| claude-opus-4-8 | 0.89 | **tangent 0.20 / compile 0.27** |

*Source: `outputs/olympiad_sweep/results.json`. Mean ≈ **0.96**.*

**Honest read:** given the libraries, frontier models are already **~96% reliable on
*isolated* named constructions** — so a fine-tune *there* earns far less than v1→v2 did
(the residual weakness is narrow, e.g. opus on `tangent`). This is a *negative* litmus
result, and it's exactly why the illustrator (§10) targets **composed, out-of-distribution
real AIME** instead: that's where prompting is *not* reliable and where the next dataset's
value actually lives. The specialist's defensible edge there is **cheap + local + bulk**
for the in-distribution slice, with the frontier spent only on the hard reasoning tail.

---

## 12. Honest limitations

- **Not top-tier capability.** Tuned models sit mid-pack; `claude-sonnet-5` (0.98) beats
  them. That's the expected and intended framing — the win is *reliable + cheap + local +
  huge data-driven delta*, not "smarter than a frontier model."
- **v1 vs v2 eval sets differ** (see §7 caveat). The defensible claim is the cell-matched
  per-construction improvement, which is stated as such.
- **Out-of-distribution tail.** Training chains top out at 5 (`data/train.jsonl`), but the
  frontier grid runs to chain 7. Tuned-0.6B chain-6/7 pass (0.13 / 0.03) is partly
  OOD *extrapolation*, not just in-domain difficulty — a fair reason the "robustness"
  number is modest.
- **Rubric provenance.** The 0/1/2 rubric numbers are a **deterministic proxy** derived
  from the objective gate, not LLM-judge scores (the judge harness exists but wasn't run
  on these evals). Spec-adherence and task-quality use the 0/1/2 credit scheme; robustness
  and consistency are reported as raw pass rates on hard/easy chain subsets — a slightly
  mixed scale, but every number is exactly reproducible from the eval JSONs.
- **Training recipe is plain LoRA, not QLoRA.** For a 0.6B model, full-precision bf16 LoRA
  (r=16, α=32, all-linear, 2 epochs, on Modal) trains in minutes; 4-bit was unnecessary.
- **The illustrator is reasoning-bound, not drawing-bound.** On arbitrary AIME the 1.7B
  illustrator draws a compiling figure 69.3% of the time but a *faithful* one only 11.3%
  (§10). The provable 93.8% is on in-distribution synthetic scenes; do not read it as
  "solves AIME." `regular_polygon` in particular compiles 12/12 but is coordinate-exact
  only 1/12 (accumulated rotation error) — it counts for illustration coverage, not for
  coordinate verification.
- **Judge-verified ≠ coordinate-verified.** On real AIME the only correctness signal is a
  vision judge ("looks right"), which is softer than the coordinate assertion used
  everywhere else and can be fooled; it is always reported separately.

---

## 13. Next steps (staged; roadmap, not claimed results)

The evidence-first loop (mine vocabulary → run the litmus → generate data only where it's
needed) is now *built*, not just planned:

- **Done — vocabulary grounding.** `scripts/mine_constructions.py` mined construction
  frequency over **1,349 MATH-geometry + 408 AIME-geometry** problems
  (`outputs/construction_freq.json`): the ops this project targets are genuinely frequent
  (intersection 185, perpendicular/foot 75, altitude 56) and the olympiad extensions rank
  as circumcircle/circumcenter 48, incircle/incenter 44, trisection 12, orthocenter 3. An
  LLM classifier also catches *implicit* constructions ("center of the circle through
  A,B,C" = circumcenter) — `outputs/construction_freq_llm.json`.
- **Done — olympiad grader + litmus.** `src/geotikz/extract.py` (compile-extract grader for
  `tkz-euclide` named centers) + `scripts/olympiad_sweep.py` produced the §11 result:
  frontier models already ~96% reliable on isolated constructions, so the value is the
  composed OOD tail.
- **Done — the composed-tail attack.** §10's illustrator is exactly that next dataset,
  built and evaluated.
- **Done — capacity.** The 4B illustrator (§10) roughly doubled faithful local AIME coverage
  (11.3% → 24.0%) — a real lever — but stayed far short of the frontier's ~64% ceiling (capacity
  closed only ~a quarter of the gap). Faithful arbitrary-AIME illustration is reasoning-bound.
- **Open stretch (not started).** DPO on on-spec vs. off-spec pairs; an adversarial
  robustness eval (malformed / contradictory scenes); composing a second constraint.

---

## Appendix — reproduce the numbers

| Claim | Artifact |
| :-- | :-- |
| Frontier sweep, 12 models, 800 grid | `outputs/sweep/results.json`, `outputs/sweep/report.md`, `outputs/sweep/pass_heatmap.png` |
| v1 numeric, 0.6B base / tuned | `outputs/eval_base_new.json`, `outputs/eval_tuned.json` |
| v1 numeric, 1.7B base / tuned | `outputs/eval_base_1p7b.json`, `outputs/eval_tuned_1p7b.json` |
| v2 PGF, 0.6B base / tuned | `outputs/eval_pgf_base.json`, `outputs/eval_pgf_tuned.json` |
| Per-op / per-chain breakdowns | join eval `results[].id` with `outputs/eval_preds_*.jsonl` (`chain`, `tags`) or `outputs/sweep/grid.jsonl` |
| Utility eval (specialist vs frontier) | `outputs/utility_report.md`, `outputs/utility_eval/results.json` |
| Illustrator — AIME coverage (before/after) | `outputs/aime_gallery_baseline/`, `outputs/aime_gallery_illustrator/coverage_stats.json` |
| Illustrator — synthetic coord-verified | `outputs/syn_eval_illustrator/report.md` |
| Illustrator — distillation filter yield | `outputs/distill/report.md` |
| Olympiad litmus, 4 models × 8 constructions | `outputs/olympiad_sweep/results.json` |
| Construction frequency mine | `outputs/construction_freq.json`, `outputs/construction_freq_llm.json` |

- **Base models:** `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (Instruct).
  Adapters: `qwen3-geotikz` (v1 0.6B numeric), `qwen3-1.7b-geotikz` (v1 1.7B numeric),
  `qwen3-pgf-geotikz` (v2 0.6B construction), `qwen3-illustrator` (1.7B AIME illustrator),
  `qwen3-illustrator-4b` (4B).
- **Gate:** `src/geotikz/harness.py::evaluate_one` — `passed = figure_only AND compiles
  AND coords_all_correct` at `atol=0.05`. Named-center constructions use the compile-extract
  grader `src/geotikz/extract.py::grade`.
- **Data-gen:** `src/geotikz/{scene,generator}.py` build scenes forward from exact
  coordinates, then strip the coordinates for the model input; `symbolic=True` emits the
  v2 PGF-construction target.
- **Reproduction commands** for every surface and result: see `README.md`
  (§"Reproduce every result") and `SUBMISSION.md`.
