# Local AIME Geometry Illustrator — distillation + generator breadth → a 1.7B specialist

**Goal.** Make a *local* small model auto-illustrate arbitrary competition (AMC/AIME/MATH)
geometry problems — raising the local specialist's AIME illustration coverage far above the
prior **14%**, toward the frontier's ~69% — using coordinate-free constructions only.

**Result (headline, AIME, 150 held-out geometry problems, seed 20260709):**

| Signal | Before — `qwen3-pgf-geotikz` (0.6B) | After — `qwen3-illustrator` (1.7B) |
| :-- | --: | --: |
| **compile + non-degenerate** (local) | **14.0%** (21/150) | **69.3%** (104/150) |
| **judge-verified / faithful** (local, vision judge) | **0.7%** (1/150) | **11.3%** (17/150) |
| union w/ judge-gated frontier fallback (faithful) | — | **64.0%** (96/150) |
| compile coverage, any route | 69.3% | 87.3% |

**Held-out synthetic (240, coordinate-verified against ground truth):**

| | base Qwen3-1.7B | tuned `qwen3-illustrator` |
| :-- | --: | --: |
| coordinate-verified pass | **7.9%** | **93.8%** |
| compile | 17.1% | 98.3% |

The local model's compile coverage went **14% → 69.3%** (5×) and its *faithful* coverage
**0.7% → 11.3%** (16×). On in-distribution constructions with ground truth, the same model
went **7.9% → 93.8%** coordinate-verified — a clean *data → behavior* result. The full
system (local + judge-gated frontier fallback) faithfully illustrates **64%**, essentially
the frontier's own faithful ceiling, while doing 11.3% for free locally.

---

## What "coverage / correct" means (three signals, weakest → strongest)

Real AIME problems have **no ground-truth coordinates**, so "correct" cannot be proven the
way it can on synthetic scenes. We report three separate, explicitly-defined signals and
never conflate them:

1. **compile + non-degenerate** — the model emitted a figure that compiles under `tectonic`
   with the tkz-euclide preamble and is not blank / not a solid blob / not an oversized
   canvas (`serve.compile_and_render`). Necessary, *not* sufficient: a figure can compile
   and still depict the wrong thing.
2. **judge-verified (vision)** — additionally, a capable **vision** model
   (`gemini-group/gemini-3.1-pro`), shown the problem **text + the rendered PNG**, confirms
   the figure *faithfully depicts the described configuration* (`src/geotikz/vision_judge.py`).
   This **keeps 3D / combinatorial / region figures** when they are faithful. It is **softer
   than coordinate verification** — a judge can be fooled — so it is reported separately.
3. **coordinate-verified (ground truth)** — for synthetic scenes only, every named point is
   recovered from the compiled figure and checked within `atol=0.05` of the exact Python
   ground truth (`extract.grade`). This is the gold signal; it is **not computable on real
   AIME** (no GT), which is exactly why signal (2) exists. On AIME, the *union* of
   coordinate- and judge-verified therefore equals judge-verified.

---

## Method

### 1. Distillation from a frontier teacher, hard-filtered (primary lever)

`scripts/distill.py` (cache-backed, resumable at every stage):

- **Corpus.** 1,588 real competition geometry problems: `gneubig/aime-1983-2024` geometry
  (251, **the 150-problem eval sample is held out** to prevent leakage) + `EleutherAI/
  hendrycks_math` config `geometry` train+test (1,337 after dedup). Embedded `[asy]…[/asy]`
  diagram code is stripped so the model learns **text → figure**, not asy transliteration.
- **Teacher.** `openai-group/gpt-5.5`, prompted with `CONSTRUCTION_SYSTEM_PROMPT`
  (`build_construction_messages`) — the exact prompt the AIME illustrator's frontier fallback
  uses — so it returns coordinate-free tkz-euclide / `calc` constructions.
- **Hard filter (the craft).**
  1. **compile + non-degenerate** (reuses the eval's degeneracy guards + preamble).
  2. **vision judge** (`gemini-3.1-pro`, second model) on the rendered PNG: keep the
     (problem → figure) pair only if the figure faithfully depicts the problem — **including
     3D and combinatorial figures**, which are kept rather than discarded.

  **Filter yield:** 1,588 teacher outputs → **1,120 compile+non-degenerate (70.5%)** →
  **1,099 vision-judge-approved (69.2%)**. The judge approved 1,099/1,120 = **98%** of
  compiling figures (a strong teacher's compiling figures are almost always faithful) and
  rejected 21 genuine mismatches. All 1,120 were judged in true **vision** mode (the gateway
  supports image inputs; a text-over-source fallback exists if it did not). Kept pairs:
  169 AIME + 930 MATH = **1,099**.

### 2. Generator expansion, ground-truth-verified (secondary lever)

`src/geotikz/olympiad_ext.py` adds **12 new construction families** on top of the existing 8,
covering the frequency-ranked vocabulary in `outputs/construction_freq_llm.json`:
line/segment **intersection** (quad diagonals, cevian ∩ median), **parallel/parallelogram**,
**midpoint/midsegment**, general **foot-of-perpendicular**, **reflection**, **rotation**,
**square**, **regular polygon** (rotation-built), **two-circle** intersection, **antipode/
diameter**. Every derived point is built with a coordinate-free tkz-euclide macro
(`\tkzInterLL`, `\tkzDefPointBy[rotation=…]`, `\tkzInterCC`, `\tkzDefMidPoint`, …) and each
figure **round-trips** through the compile-extract grader (emit → compile → read back == exact
Python GT). Round-trip yield across all 20 families was **100%** at the sampled counts, with
varied natural phrasing.

### 3. Combined dataset

`scripts/build_illustrator_data.py` → **3,996 training records** = 1,099 distilled real pairs
(×2, to emphasize the natural-language lever) + 1,798 GT-verified synthetic (≈90/family) —
plus a **disjoint 240-problem synthetic eval** with ground-truth coordinates. All records use
the construction prompt; the model emits **coordinate-free constructions only**.

### 4. Training

`scripts/train_illustrator_modal.py` — **Qwen3-1.7B** + LoRA (r=32, α=64, all-linear),
2 epochs, bf16, on a Modal A100. New **`RUN_NAME = qwen3-illustrator`** committed to the
`geotikz-outputs` Volume. **Existing adapters (`qwen3-geotikz`, `qwen3-1.7b-geotikz`,
`qwen3-pgf-geotikz`) and all existing data/eval JSONs are untouched.** Final train loss 0.19.

### 5. Evaluation

- `scripts/illustrate_aime.py` (extended, additive): runs the illustrator on Modal
  (`scripts/infer_illustrator_modal.py`, construction prompt), compiles, **vision-judges**,
  and reports both coverage signals with **judge-gated routing** (a local figure is used only
  if the vision judge approves it, else the frontier handles the problem — so an
  unfaithful-but-compiling local figure never pre-empts a faithful frontier one).
- `scripts/eval_syn_illustrator.py`: coordinate-verified pass rate on the 240 held-out
  synthetic problems, **base vs tuned**.

---

## Honest scoping & caveats

- **"Arbitrary" cannot reach 100% locally.** ~31–36% of AIME geometry is 3D / heavily
  combinatorial / not cleanly planar; even the frontier caps at ~64–69% faithful on this
  sample. The local model's **11.3% faithful** is a real 16× lift, but a 1.7B cannot match
  gpt-5.5's *reasoning* on novel hard problems — it reliably draws a plausible figure (69.3%
  compile) but the *right* figure only 11.3% of the time. That gap is the honesty the vision
  judge buys.
- **Judge-verified is softer than coordinate-verified.** It certifies "looks right," not
  "provably right." The provable number (93.8%) exists only on synthetic scenes with GT.
- **The old 14% was almost entirely spurious.** Re-judging the 0.6B specialist's compiling
  figures shows only **1/150 (0.7%)** were actually faithful — the other 20 compiled but did
  not match. The new 11.3% faithful is therefore the more meaningful before→after.
- **Known weakness:** `regular_polygon` compiles 12/12 but is coordinate-exact only 1/12
  (rotation-built many-vertex figures accumulate small errors / labeling-order); it still
  counts for illustration coverage but not coordinate verification.
- **Leakage guard:** the 150 AIME eval problems are excluded from distillation; training AIME
  problems (169) are disjoint from them.

---

## Reproduce / run the illustrator

```bash
# The adapter lives on the Modal Volume `geotikz-outputs` as `qwen3-illustrator`
# (also downloaded to ./outputs/qwen3-illustrator). Base model: Qwen/Qwen3-1.7B.

# Distill (resumable; caches under outputs/distill/):
uv run python scripts/distill.py --workers 28 --judge-workers 16

# Build the combined dataset:
uv run python scripts/build_illustrator_data.py --syn-per-type 90 --distill-repeat 2

# Train on Modal (detached; new adapter, existing ones untouched):
modal run --detach scripts/train_illustrator_modal.py --epochs 2

# AIME coverage (before→after), two signals + judge-gated union:
uv run python scripts/illustrate_aime.py --n 150 --backend modal \
    --specialist-script scripts/infer_illustrator_modal.py \
    --out-dir outputs/aime_gallery_illustrator --max-new-tokens 1536 \
    --fallback-model openai-group/gpt-5.5

# Coordinate-verified synthetic pass rate (base vs tuned):
uv run python scripts/eval_syn_illustrator.py --also-base
```

**Artifacts:** `data/distill_illustrator.jsonl` (1,099 distilled pairs),
`data/illustrator_train_chat.jsonl` (3,996), `data/illustrator_syn_eval.jsonl` (240),
`outputs/distill/report.md` (filter yield), `outputs/aime_gallery_illustrator/`
(gallery + `coverage_stats.json` + `coverage_report.md`),
`outputs/aime_gallery_baseline/` (before-numbers), `outputs/syn_eval_illustrator/report.md`,
adapter at `outputs/qwen3-illustrator/`.

## Verification (it actually ran)

- Training: exit 0, adapter committed to the Volume and downloaded
  (`adapter_model.safetensors` 139 MB) — `modal volume ls geotikz-outputs qwen3-illustrator`.
- Figures compiled: 1,120 distilled figs + 106 specialist + 87 frontier gallery PNGs rendered.
- All eval scripts exited 0; all output JSON/MD files non-empty.
- Spot-checked faithful local figures (e.g. AIME 1993-12) render as real, relevant diagrams.
