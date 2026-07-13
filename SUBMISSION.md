# Submission package — Spec-First Geometry → TikZ

Maps the assignment's **5 required artifacts** to their location and status, then lists the
exact remaining user actions. Every number below is reproducible from the committed
artifacts (see the per-claim table in [`WRITEUP.md`](WRITEUP.md#appendix--reproduce-the-numbers)).

## Status at a glance

| # | Required artifact | Status | Where |
| :-- | :-- | :-- | :-- |
| 1 | **Dataset, published** | ✅ built & verified · ⏳ needs HF token to push | `data/*.jsonl`, card `cards/dataset_card.md`, `scripts/publish_hf.py` |
| 2 | **Model on HF Hub + running demo** | ✅ adapters + demo run locally · ⏳ needs HF token to push | `outputs/qwen3-*`, `scripts/demo.py`, `scripts/demo_web.py`, card `cards/model_card.md` |
| 3 | **Eval harness + results table (base vs tuned)** | ✅ done | `src/geotikz/harness.py`, `outputs/eval_*.json`, tables below |
| 4 | **Brainlift (thesis + did data→behavior hold, w/ evidence)** | ✅ done | [`DEMO_WRITEUP.md`](DEMO_WRITEUP.md), `WRITEUP.md`, `BEHAVIOR_SPEC.md`, `REVIEW_NOTES.md` |
| 5 | **3–5 min demo video** | ⏳ needs recording | script: [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) |

**Legend:** ✅ complete · ⏳ needs a user action (HF token / recording). Nothing is blocked
on further engineering.

---

## 1. Dataset (published)

- **Built & verified.** 8 configs, all splits disjoint (asserted at build). Schema, sizes,
  provenance, licensing: `cards/dataset_card.md`.

  | Config | Rows | Role |
  | :-- | --: | :-- |
  | `train.jsonl` (v1 numeric) | 5,340 | training |
  | `train_pgf.jsonl` (v2 construction) | 2,050 | training |
  | `eval.jsonl` | 800 | held-out — same grid the 12 frontier models saw |
  | `eval_pgf.jsonl` | 280 | held-out (v2 symbolic) |
  | `golden_set.jsonl` | 150 | curated, individually compile+coord-verified showcase |
  | `olympiad_eval.jsonl` | 120 | named-center constructions (olympiad litmus) |
  | `illustrator_train_chat.jsonl` | 3,996 | AIME-illustrator SFT (distill + synthetic) |
  | `illustrator_syn_eval.jsonl` | 240 | held-out synthetic illustrator eval |

- **To publish (needs token):** `uv run python scripts/publish_hf.py --user YOURNAME --push`
  (dry-run without `--push`). Creates `YOURNAME/spec-first-geometry-tikz` and uploads the
  files + card as `README.md`.

## 2. Model on HF Hub + running demo

- **Adapters (LoRA, ready to upload):** `qwen3-pgf-geotikz` (v2 0.6B — the headline
  specialist), `qwen3-geotikz` (v1 0.6B), `qwen3-illustrator` (1.7B AIME illustrator), and
  `qwen3-illustrator-4b` / `qwen3-illustrator-4b-v2` (promoted in the copilot) live under
  `outputs/` or on the Modal Volume — the publish script prints the `modal volume get`
  command to fetch them. Card: `cards/model_card.md`.
- **Running demo (works now, locally):**
  - CLI: `uv run python scripts/demo.py "<scene>"` → rendered PNG + copyable TikZ.
  - Web: `uv run python scripts/demo_web.py` → Gradio UI (textbox → figure + TikZ).
- **To publish (needs token):** same `scripts/publish_hf.py --push` creates
  `YOURNAME/qwen3-geotikz` with the adapters as subfolders + card. (Optional: host the
  Gradio demo as a HF Space — the `demo_web.py` app runs as-is.)

## 3. Eval harness + results table (base vs tuned)

Objective gate, no human/LLM judge required (`src/geotikz/harness.py::evaluate_one`):
**figure-only AND compiles (`tectonic`) AND every named coordinate within `atol=0.05`**.
Named-center constructions use the compile-extract grader (`src/geotikz/extract.py`).

**v1 numeric — pass on the 800-item grid (identical to the frontier sweep grid):**

| Model | Base | Tuned | Δ (passed) |
| :-- | --: | --: | :-- |
| Qwen3-0.6B | 0.003 | **0.464** | 2 → 371 / 800 |
| Qwen3-1.7B | 0.005 | **0.598** | 4 → 478 / 800 |

**Rubric (Qwen3-0.6B base → tuned)** — deterministic from the gate components:

| Dimension | Base | Tuned |
| :-- | --: | --: |
| Spec adherence | 0.93 | **1.43** |
| Robustness (chain ≥ 4) | 0.00 | **0.30** |
| Task quality | 0.58 | **1.43** |
| Consistency (chain ≤ 3) | 0.01 | **0.84** |

Win condition ("beat base on Spec adherence **and** Robustness") → met.

**v2 construction (`qwen3-pgf-geotikz`) — 280-item PGF eval + the cell-matched fix:**

| Metric | Base | Tuned |
| :-- | --: | --: |
| Overall pass | 0.000 | **0.989** (277/280) |
| Compile rate | 0.536 | **1.000** |
| Foot-of-altitude (v1→v2, tuned-0.6B) | 0.02 | **0.98** |
| Line-intersection (v1→v2, tuned-0.6B) | 0.20 | **0.99** |

**Illustrator (`qwen3-illustrator`, 1.7B):**

| Signal | Base | Tuned |
| :-- | --: | --: |
| Synthetic (240), coordinate-verified | 7.9% | **93.8%** |
| AIME (150), compile + non-degenerate | 14.0% | **69.3%** |
| AIME (150), judge-verified faithful | 0.7% | **11.3%** |

Reproduce any row: `outputs/eval_*.json`, `outputs/*/coverage_stats.json`,
`outputs/syn_eval_illustrator/report.md` (per-claim table in `WRITEUP.md`).

## 4. Brainlift

- **Start here:** [`DEMO_WRITEUP.md`](DEMO_WRITEUP.md) — canonical demo write-up (pitch, eval, results, product, terms). Spoken script: [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md).
- **`WRITEUP.md`** — the full evidence-first narrative: thesis + the representation-pivot
  insight, the 12-model litmus (with the 90%/95% reliability nuance), eval-built-before-
  training, v1 → 1.7B → v2 tables, error analysis → the pivot, the "useful" positioning
  (utility eval + product surfaces), the illustrator distillation (honest 69% draws /
  ~11% correct + reasoning-ceiling caveat), and the olympiad litmus.
- **`BEHAVIOR_SPEC.md`** — the one-sentence falsifiable gate (data-gen rubric = eval
  criterion = thesis).
- **`REVIEW_NOTES.md`** — a live-defense cheat sheet.

## 5. Demo video (3–5 min)

- **Script + shot list:** [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md). Shows base-vs-tuned, the
  interactive demo, a generated worksheet, and the AIME illustrator — the base model failing
  where the tuned model succeeds.
- Pre-rendered "money shots" for screen capture already exist:
  `outputs/renders/before_after.png`, `outputs/renders/llm_vs_slm.png`,
  `outputs/sweep/pass_heatmap.png`, `outputs/worksheets/worksheet.pdf`,
  `outputs/aime_gallery_illustrator/index.html`.

---

## Remaining user actions (nothing else is blocked)

1. **HF token → publish** dataset + model:

   ```bash
   export HF_TOKEN=hf_...                                  # WRITE token
   # optional: fetch Modal-only adapters first
   modal volume get geotikz-outputs qwen3-1.7b-geotikz ./outputs/qwen3-1.7b-geotikz
   uv run python scripts/publish_hf.py --user YOURNAME --push
   ```

   Then paste the resulting `https://huggingface.co/...` URLs into the model/dataset cards'
   usage snippet (the `<user>/...` placeholder) if you want them clickable.

2. **Record the 3–5 min video** following `VIDEO_SCRIPT.md`.
