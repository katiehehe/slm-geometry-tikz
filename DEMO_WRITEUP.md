# Geometry → TikZ: a one-week specialist, end to end

**What I built.** A small open model that turns coordinate-free geometry scenes into compiling TikZ/PGF figures, plus a live **Geometry Figure Copilot** (specialist first, frontier fallback; screenshot/PDF in; conversational edits; interactive drag board).

> **Thesis.** You can make a small open model *reliably* do one narrow thing by controlling its training data — and **how you frame the target** (compute the answer vs. emit the construction) matters more than model size or dataset size.

**Live demo:** [Geometry Figure Copilot](https://katie-he--geotikz-copilot-web.modal.run) (auth `demo` / `geotikz-gpu-8t3n`; stop when idle) · Full evidence: [`WRITEUP.md`](WRITEUP.md) · Spec: [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md) · Spoken script: [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md)

---

## Pitch

I wanted a model that turns a **coordinate-free** geometry scene — relationships only, no explicit coordinates — into a **single compiling TikZ/PGF figure whose every named point is numerically correct**. Not “draw something pretty.” Recover the hidden numbers from the geometry.

That passes the assignment’s litmus: a well-prompted base model (and most frontier models) **cannot already do it reliably**. So fine-tuning earns its keep. The win is not “smarter than GPT.” It’s a tiny, cheap, local specialist that holds a falsifiable behavior every time on the **in-domain** slice it was trained for.

**In-domain** means coordinate-free or lightly templated plane-geometry construction scenes that match the specialist’s training vocabulary (triangle centers, feet of altitudes, midpoints, tangents, and similar named constructions). These are scored with the synthetic gate: figure-only TikZ, compiles, and named coordinates within about 0.05 of ground truth. It does **not** mean arbitrary free-form AIME/contest word problems, 3D geometry, or other out-of-vocabulary setups — those are out-of-domain, where compile can still be high but faithfulness is much harder.

---

## Terms

**In-domain** / **out-of-domain** — see the definition in Pitch above. Out-of-domain also covers free-form contest text and faithfulness-to-problem judgment, not just the synthetic compile-and-coords gate.

**Synthetic gate** means I evaluate on made-up construction scenes with a figure-only TikZ output that must compile under `tectonic` and land every named point within about `0.05` of ground truth. Labels are correct by construction because the generator builds each scene forward from exact coordinates, then strips those coordinates for the model input.

**PGF** is the engine under TikZ. “PGF constructions” means emitting `calc` / intersection macros so LaTeX does the arithmetic instead of the small model inventing coordinates — the pivot that took foot-of-altitude from about 2% to about 98%.

**Opus 4.8** in comparisons is Claude Opus 4.8 via gateway id `claude-group/claude-opus-4-8`.

**Faithful** (AIME / real problems) means a vision judge checked that the drawn construction matches the intended problem scene. Compile ≠ faithful: a figure can compile and look geometric while still misrepresenting the problem.

---

## Eval suite

The core pass gate (from `BEHAVIOR_SPEC.md`) is: figure-only ∧ compiles under `tectonic` ∧ every named point within `atol = 0.05` of ground truth ∧ derived, not transcribed.

On top of that I ran:

1. **Synthetic numeric 800-grid** — same held-out scenes for base, tuned specialists, and a 12-model frontier sweep (`outputs/sweep/`).
2. **Synthetic PGF / construction eval** — 280-item symbolic set after the target pivot (`outputs/eval_pgf_tuned.json`).
3. **Utility check** — specialist vs gpt-5.5 and Opus 4.8 on in-domain constructions (`outputs/utility_report.md`, n=30).
4. **AIME / real geometry** — compile vs vision-judge faithful, plus a judge-gated frontier union (`ILLUSTRATOR_*_REPORT.md`).

![12-model pass heatmap](outputs/sweep/pass_heatmap.png)

---

## Results (the numbers worth saying)

**Most impressive arc (same style of gate, specialist side):** base ≈ **0.3%** → tuned 0.6B numeric **46%** → tuned 1.7B numeric **60%** → PGF construction target **99%**. Foot-of-altitude went from **2% → 98%**. On the numeric grid the 1.7B specialist **beats gpt-4o and gpt-4.1**. On the PGF utility set it **matches or beats gpt-5.5 and Opus 4.8** on pass/utility. It does **not** beat frontier on open AIME faithfulness.

| Stage | Pass % | Count | Artifact |
| :-- | --: | --: | :-- |
| Base Qwen3-0.6B | **0.25%** | 2/800 | `outputs/eval_base_new.json` |
| Tuned 0.6B (v1 numeric) | **46.4%** | 371/800 | `outputs/eval_tuned.json` |
| Tuned 1.7B (v1 numeric) | **59.8%** | 478/800 | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B (v2 PGF) | **98.9%** | 277/280 | `outputs/eval_pgf_tuned.json` |

I did **not** distill a teacher for the core specialist. A small geometry engine (`src/geotikz/scene.py`, `generator.py`) builds scenes forward, strips coordinates, and dials difficulty (chain length, irregular numbers, hard ops). Training is LoRA on Qwen3 (Modal); thinking mode must be off or the figure-only gate fails.

![Base fails, tuned passes](outputs/renders/before_after.png)

### Frontier on the numeric 800-grid

At a strict “reliable every time” bar, **no** frontier model is reliable across the landscape. Hard cells are specific constructions plus ugly numbers (foot-of-altitude at chain 4–5 with irregular numbers pools around 38–44% across models).

| Model | Overall pass % |
| :-- | --: |
| claude-sonnet-5 | **97.7%** |
| gpt-5.5 | 87.0% |
| claude-opus-4-8 | 67.4% |
| **tuned-1.7B (ours)** | **59.8%** |
| gpt-4.1 | 55.5% |
| **tuned-0.6B (ours)** | **46.4%** |
| gpt-4o | 43.9% |
| base-0.6B | 0.25% |

*Full table: `outputs/sweep/pass_rates.csv`.*

**Beat / don’t-beat (numeric grid):** tuned-0.6B beats gpt-4o (and weaker); tuned-1.7B also beats gpt-4.1. Neither beats Opus, gpt-5.4, gpt-5.5, or sonnet on that grid.

### Pivot — construction target (PGF does the arithmetic)

Failures piled onto projection arithmetic: foot-of-altitude was **2%** (0.6B) / **13%** (1.7B) under the numeric target. Scaling barely moved it. Same scenes, new label: emit a coordinate-free PGF construction and let LaTeX compute at compile time (`\coordinate (D) at ($(A)!(C)!(B)$);`). Foot **2% → 98%**; intersection **20% → 99%**. Base still scores **0%** on the PGF eval, so this is a trained behavior, not “the task got easier.”

### Utility vs gpt-5.5 / Opus 4.8 (in-domain)

| Config | Pass % | Compile % | Est. cost |
| :-- | --: | --: | :-- |
| **specialist (ours)** | **100%** | **100%** | **$0** |
| gpt-5.5 [construction] | **100%** | 100% | ~$0.003 |
| gpt-5.5 [plain] | 96.7% | 100% | ~$0.005 |
| claude-opus-4-8 [plain] | 90.0% | 96.7% | ~$0.029 |
| claude-opus-4-8 [construction] | 63.3% | **66.7%** | ~$0.015 |

The specialist matches gpt-5.5 construction on pass, beats both models’ plain modes, and beats Opus hard when Opus tries constructions (compile collapses on hallucinated macros). Wins on cost and dialect; this laptop MPS run is not a latency win.

### AIME / out-of-domain

Synthetic in-domain was essentially solved. On held-out AIME the narrow PGF specialist was format-locked (~14% compile, **0.7%** faithful). Distilling a frontier teacher with a vision judge and broader synthetic mix produced an illustrator: compile ~**69%**, faithful **11.3%** (1.7B) / **24%** (4B). Union with judge-gated frontier is ~**64–68%**. Local specialist does **not** beat frontier on AIME faithfulness. Compile coverage is competitive; “compiled” is not “correct.”

Illustrator 4B v2 (more paraphrases + harder families) hits **98.1%** on an expanded synthetic gate and **99.1%** on paraphrase — promoted in the live app as `qwen3-illustrator-4b-v2`.

---

## Product / demo

The specialist became a **custom chat SPA** on Modal (`web/` + `src/geotikz/webapp.py`): text or screenshot/PDF in → figure + TikZ, then conversational edits. Right pane: **Figure | Interactive | TikZ**. Routing is specialist first for in-vocab constructions, frontier fallback for OOD / too many derived points / many-vertex polygons, vision for uploads, and one automatic repair pass on non-compiling figures. Every reply attributes which model produced it. The interactive board keeps free points draggable and re-solves derived points from inferred constraints.

**Live:** https://katie-he--geotikz-copilot-web.modal.run *(auth: `demo` / `geotikz-gpu-8t3n`)*

Also shipped: CLI/Gradio demo and a worksheet generator (in-vocabulary problems, PDF + answer key).

![Custom copilot SPA](outputs/renders/copilot_web_screenshot.png)

---

## What I’d do better

I’d switch to the construction (PGF) target earlier — that was the real unlock, more than 3× model size. I’d optimize more for **faithfulness** on real problems, not just compile rate, and distill only diagrams that pass a vision check. Next work I’d actually prioritize: DPO on on-spec vs off-spec pairs, adversarial robustness, more polygon training, and climbing AIME faithful with data rather than hyperparameter churn.

---

## Short Q&A

**Why not just use GPT?** For in-domain constructions the specialist is free at inference, always emits a well-formed dialect, and on the utility set it matches gpt-5.5 and beats Opus on construction-mode compile. Frontier is the fallback for out-of-distribution AIME and free-form contest text.

**What does “faithful” mean?** Compile means the TikZ ran. Faithful means the picture matches the problem statement. Local 4B is about **24%** faithful on AIME; frontier union is about **64–68%**.

**What is an adapter?** A small LoRA on top of Qwen3. I don’t train a whole model from scratch; I teach the base this one skill: geometry text → TikZ.

**In-domain vs OOD in one line?** In-domain = coordinate-free / lightly templated construction scenes in the training vocab, scored by the synthetic gate; OOD = free-form AIME/contest text, 3D, or other out-of-vocab setups (compile can still be high; faithfulness is harder).

---

## Assignment checklist map

| Required artifact | Where |
| :-- | :-- |
| Behavior Spec | [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md) |
| Dataset | `data/*.jsonl`, [`cards/dataset_card.md`](cards/dataset_card.md) |
| Model + demo | adapters under `outputs/qwen3-*`, live URL above, [`cards/model_card.md`](cards/model_card.md) |
| Eval harness + tables | `src/geotikz/harness.py`, tables above, [`WRITEUP.md`](WRITEUP.md) |
| Brainlift | this doc + [`WRITEUP.md`](WRITEUP.md) |
| 3–5 min demo video | [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) |
| Eval review speaking script | [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md) |

*Numbers are from committed eval JSONs / reports cited above. Reproduce commands: [`README.md`](README.md).*
