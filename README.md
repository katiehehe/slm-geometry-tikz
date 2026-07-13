# Spec-First Geometry → TikZ

A small open model (Qwen3-0.6B / 1.7B) fine-tuned to do **one** narrow thing reliably:
turn a **coordinate-free** geometry scene description (relationships only, *no explicit
coordinates*) into a **single compiling TikZ/PGF figure whose every named point is
numerically correct**: recovering the hidden numbers from the geometry, not transcribing
them.

> **Thesis:** you can make a small model *reliably* do one thing by controlling its
> training data, and *how you frame the target* (compute the answer vs. emit the
> construction) matters more than model size or dataset size.

**Start here:**

- [`DEMO_WRITEUP.md`](DEMO_WRITEUP.md): canonical demo write-up (pitch, eval, results, product, terms, Q&A).
- [`WRITEUP.md`](WRITEUP.md): the full evidence-first narrative (the brainlift).
- [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md): the one-sentence falsifiable gate.
- [`cards/`](cards/): Hugging Face model + dataset cards.

## The result in one table

Pass = **figure-only AND compiles (`tectonic`) AND every named coordinate within 0.05**.

| Milestone | Pass rate | Evidence |
| :-- | --: | :-- |
| Base Qwen3-0.6B (prompted) | 0.003 | `outputs/eval_base_new.json` |
| Tuned 0.6B, v1 numeric target | 0.464 | `outputs/eval_tuned.json` |
| Tuned 1.7B, v1 numeric target | 0.598 | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B, **v2 construction target** | **0.989** | `outputs/eval_pgf_tuned.json` |

Same base model + recipe for v1 vs v2 (**only the target representation changed**) took
the hardest construction (foot-of-altitude) from **0.02 → 0.98**.

---

## Setup

```bash
# 1. Python deps (uv reads pyproject.toml / uv.lock)
uv sync

# 2. tectonic (LaTeX compiler used by the grader + all rendering)
brew install tectonic          # macOS; see tectonic-typesetting.github.io otherwise

# 3. (optional) frontier-model access for the sweep / utility eval / illustrator fallback
cp .env.example .env            # then fill OPENAI_BASE_URL + OPENAI_API_KEY

# 4. (optional) Modal, for GPU training + batched inference
uv tool install modal && modal setup
```

The trained specialist adapter lives on the Modal Volume `geotikz-outputs`. Download the v2
specialist once (or point `--adapter` at any local adapter dir):

```bash
modal volume get geotikz-outputs qwen3-pgf-geotikz ./outputs/qwen3-pgf-geotikz
```

---

## Run each surface

All product surfaces emit **coordinate-free constructions** and share one serving
layer (`src/geotikz/serve.py`): the local specialist (`Qwen/Qwen3-0.6B` + `qwen3-pgf-geotikz`
LoRA) with its exact training prompt, plus an optional frontier model via the gateway.

**1. Interactive demo:** scene text → rendered figure + copyable TikZ:

```bash
uv run python scripts/demo.py "There is a circle centered at the origin with radius 3. \
Point A on the circle at 40 degrees. Point B at 200 degrees. M is the midpoint of AB."

uv run python scripts/demo_web.py            # Gradio web UI (textbox -> figure + TikZ)
```

**2. Worksheet generator:** printable geometry worksheet + answer key, all figures
guaranteed in-vocabulary and correct-by-construction:

```bash
uv run python scripts/make_worksheet.py --source generator --n 8 --seed 7
# -> outputs/worksheets/worksheet.pdf + answer_key.pdf
uv run python scripts/make_worksheet.py --source olympiad \
    --topics circumcenter incenter centroid median --n 8      # named constructions
```

**3. AIME auto-illustrator:** specialist first, judge-gated frontier fallback, coverage
report + rendered gallery:

```bash
uv run python scripts/illustrate_aime.py --n 150 --backend modal \
    --specialist-script scripts/infer_illustrator_modal.py \
    --out-dir outputs/aime_gallery_illustrator --max-new-tokens 1536 \
    --fallback-model openai-group/gpt-5.5
# -> outputs/aime_gallery_illustrator/index.html + coverage_report.md + coverage_stats.json
```

**4. Geometry Figure Copilot:** an interactive chat platform: a geometry scene **or a
screenshot of a problem** → figure + copyable TikZ, then **edit it conversationally**
(“make it bigger”, “add color”, “move / rename the labels”). Text routes to the local
specialist (optional) with a frontier fallback; **screenshots** route to a frontier
**vision** model; **edits** route to a frontier model. **Every reply states which model
produced it**, and any non-compiling figure gets one automatic self-repair pass.
The shipped product UI is a **custom chat SPA** under [`web/`](web/) (Figure | Interactive | TikZ tabs), served by `src/geotikz/webapp.py` / `scripts/copilot.py`.

```bash
uv run python scripts/copilot.py             # custom SPA: text / screenshot in -> figure + TikZ, then edit
```

See [`EXAMPLES.md`](EXAMPLES.md) for copy‑pasteable prompts the local `qwen3-illustrator-4b`
specialist handles itself (also wired as clickable examples in the app).

---

## The eval gate (built before training)

Objective, no human/LLM judge required. An output **passes iff**: (1) figure-only, (2)
compiles under `tectonic`, (3) every named coordinate within `0.05` of the ground-truth
construction. A static TikZ parser recovers each point (polar, `calc` projections
`($(a)!(c)!(b)$)`, `name intersections`); named-center constructions
(`\tkzCircumCenter`, …) are graded by a compile-extract grader that reads coordinates back
out of TeX (`src/geotikz/extract.py`).

Channels: **figure-only rate · compile rate · coordinate accuracy · pass rate** (the gate) ·
optional render-diff (SSIM/MSE) · optional LLM-judge rubric (`src/geotikz/judge.py`, needs
`OPENAI_API_KEY`). Code: `src/geotikz/{harness,metrics,tex,extract}.py`.

---

## Reproduce every result

Model inference of base+LoRA thrashes on an 8 GB Mac, so predictions are generated on a GPU
(Modal) and **scored locally** (compile + coordinate check is lightweight). The committed
`outputs/eval_preds_*.jsonl` let you re-score everything **locally with just tectonic**.

| Result | Command | Artifact |
| :-- | :-- | :-- |
| v1 numeric, 0.6B base/tuned | `uv run python scripts/score_preds.py --preds outputs/eval_preds_tuned.jsonl --tag tuned --out outputs/eval_tuned.json` | `outputs/eval_tuned.json` |
| v1 numeric, 1.7B base/tuned | `... --preds outputs/eval_preds_tuned_1p7b.jsonl --model Qwen/Qwen3-1.7B --adapter qwen3-1.7b-geotikz --tag tuned-1.7b` | `outputs/eval_tuned_1p7b.json` |
| v2 construction, 0.6B | `... --preds outputs/eval_preds_pgf_tuned.jsonl --tag pgf-tuned --out outputs/eval_pgf_tuned.json` | `outputs/eval_pgf_tuned.json` |
| Frontier sweep (12 models × grid) | `uv run python scripts/difficulty_sweep.py --chains 2 3 4 5 6 7 --k 40 --op-dial --workers 16 --out outputs/sweep` then `scripts/sweep_report.py --dir outputs/sweep --threshold 0.9` | `outputs/sweep/{results.json,report.md,pass_heatmap.png}` |
| Utility eval (specialist vs frontier) | `uv run python scripts/utility_eval.py --n 30 --models openai-group/gpt-5.5 claude-group/claude-opus-4-8` | `outputs/utility_report.md` |
| Olympiad litmus (4 models × 8 constructions) | `uv run python scripts/olympiad_sweep.py --preset frontier --n 15` | `outputs/olympiad_sweep/results.json` |
| Illustrator, synthetic coord-verified | `uv run python scripts/eval_syn_illustrator.py --also-base` | `outputs/syn_eval_illustrator/report.md` |
| Illustrator, AIME coverage | `scripts/illustrate_aime.py …` (above) | `outputs/aime_gallery_illustrator/coverage_stats.json` |
| Dataset composition | `uv run python scripts/analyze_dataset.py` | `outputs/renders/data_composition.png` |

Full-model re-run from scratch (needs a GPU): generate preds with
`modal run scripts/train_modal.py::eval_infer`, download with `modal volume get`, then
`score_preds.py`. The per-claim → artifact map is also in
[`WRITEUP.md`](WRITEUP.md#appendix--reproduce-the-numbers).

---

## How the adapters were trained

Supervised fine-tuning with LoRA (TRL + PEFT, bf16) on a Modal serverless GPU. A 0.6B model
trains in minutes, so 4-bit QLoRA was unnecessary. Inference disables Qwen3's "thinking"
block so output is pure, figure-only TikZ.

```bash
modal run --detach scripts/train_modal.py                 # v2 0.6B (qwen3-pgf-geotikz)
modal volume get geotikz-outputs qwen3-pgf-geotikz ./outputs/qwen3-pgf-geotikz
modal run --detach scripts/train_illustrator_modal.py --epochs 2   # 1.7B illustrator
```

| Adapter | Base | Target | LoRA | Data |
| :-- | :-- | :-- | :-- | --: |
| `qwen3-geotikz` | Qwen3-0.6B | v1 numeric | r=16, α=32 | 5,340 |
| `qwen3-1.7b-geotikz` | Qwen3-1.7B | v1 numeric | r=16, α=32 | 5,340 |
| `qwen3-pgf-geotikz` | Qwen3-0.6B | v2 construction | r=16, α=32 | 2,050 |
| `qwen3-illustrator` | Qwen3-1.7B | illustrator | r=32, α=64 | 3,996 |
| `qwen3-illustrator-4b` *(in progress)* | Qwen3-4B | illustrator | r=32, α=64 | 3,996 |

Data is **self-verifying synthetic**: scenes are built *forward from exact coordinates*
(`src/geotikz/{scene,generator,olympiad_ext}.py`), then coordinates are stripped for the
model input, so labels are correct by construction. `symbolic=True` emits the v2
construction target. The illustrator additionally distills a frontier teacher hard-filtered
by a vision judge (`scripts/distill.py`). A local Apple-Silicon smoke path
(`scripts/{generate,train,run_smoke}.py`, `notebooks/train_colab_unsloth.py`) exists to
prove the loop closes; the real runs use Modal.

---

## Publish to Hugging Face

Cards are ready (`cards/model_card.md`, `cards/dataset_card.md`). Publishing needs your own
**write token**. The script is dry-run by default:

```bash
uv run python scripts/publish_hf.py --user YOURNAME              # preview the plan
export HF_TOKEN=hf_...                                           # WRITE token
uv run python scripts/publish_hf.py --user YOURNAME --push       # create repos + upload
```

## Layout

```
src/geotikz/
  scene.py, generator.py, olympiad_ext.py   # spec-first scene sampler (the IP); dials difficulty
  prompts.py                                  # system prompts (numeric + construction) + chat formatting
  tex.py, metrics.py                          # tectonic compile, render, figure-only + coordinate checks
  harness.py, extract.py                      # the pass gate; compile-extract grader for named centers
  infer.py, serve.py, gateway.py             # local specialist load/gen; serving layer; multi-provider gateway
  judge.py, vision_judge.py                   # optional LLM / vision judges
scripts/
  build_*.py, generate.py                      # dataset builders (v1, v2, golden set, illustrator, olympiad)
  train_modal.py, train_illustrator*_modal.py  # LoRA training on Modal
  evaluate.py, score_preds.py                  # scoring
  difficulty_sweep.py, sweep_report.py, olympiad_sweep.py   # frontier litmus
  utility_eval.py, eval_syn_illustrator.py     # utility + illustrator evals
  demo.py, demo_web.py, copilot.py, make_worksheet.py, illustrate_aime.py   # product surfaces (copilot = chat + edit)
  publish_hf.py                                # Hub publishing (dry-run by default)
```
