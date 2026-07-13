# Geometry → TikZ: a one-week specialist, end to end

This is the readable write-up of what I built and what the numbers actually say. For the short speaking script, see [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md). Full evidence lives in [`WRITEUP.md`](WRITEUP.md), and the behavior contract is in [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md). The live product is the [Geometry Figure Copilot](https://katie-he--geotikz-copilot-web.modal.run) (auth `demo` / `geotikz-gpu-8t3n`; stop Modal when idle).

## Pitch

I wanted a model that turns a coordinate-free geometry scene — relationships only, no explicit coordinates — into a single compiling TikZ/PGF figure whose every named point is numerically correct. Not “draw something pretty.” Recover the hidden numbers from the geometry.

That passes the assignment’s litmus: a well-prompted base model, and most frontier models, cannot already do it reliably. So fine-tuning earns its keep. The win is not “smarter than GPT.” It’s a tiny, cheap, local specialist that holds a falsifiable behavior every time on the in-domain slice it was trained for.

**In-domain** means coordinate-free or lightly templated plane-geometry construction scenes that match the specialist’s training vocabulary: triangle centers, feet of altitudes, midpoints, tangents, and similar named constructions. These are scored with the **synthetic gate** — figure-only TikZ, compiles under tectonic, and named coordinates within about 0.05 of ground truth. It does not mean arbitrary free-form AIME or contest word problems, 3D geometry, or other out-of-vocabulary setups. Those are out-of-domain, where compile can still be high but faithfulness is much harder.

The thesis is simple. You can make a small open model reliably do one narrow thing by controlling its training data — and how you frame the target, compute the answer versus emit the construction, matters more than model size or dataset size.

## Setup

The synthetic gate uses made-up construction scenes. Labels are correct by construction because a small geometry engine in `src/geotikz/scene.py` and `generator.py` builds each scene forward from exact coordinates, then strips those coordinates for the model input. Training is LoRA on Qwen3 on Modal; thinking mode must be off or the figure-only gate fails. I did not distill a teacher for the core specialist.

**PGF** is the engine under TikZ. “PGF constructions” means emitting calc and intersection macros so LaTeX does the arithmetic instead of the small model inventing coordinates. That pivot took foot-of-altitude from about 2 percent to about 98 percent.

**Opus 4.8** in comparisons is Claude Opus 4.8 via gateway id `claude-group/claude-opus-4-8`.

**Faithful**, for AIME and real problems, means a vision judge checked that the drawn construction matches the intended problem scene. Compile is not the same as faithful: a figure can compile and look geometric while still misrepresenting the problem.

On top of the core gate from `BEHAVIOR_SPEC.md`, I ran four evaluations: a synthetic numeric 800-grid with a twelve-model frontier sweep in `outputs/sweep/`; a 280-item PGF construction eval in `outputs/eval_pgf_tuned.json`; a thirty-example utility check versus gpt-5.5 and Opus 4.8 in `outputs/utility_report.md`; and AIME plus real geometry separating compile from vision-judge faithful in the `ILLUSTRATOR_*_REPORT.md` files.

The twelve-model pass heatmap below shows how hard the numeric grid still is even for frontier models.

![12-model pass heatmap](outputs/sweep/pass_heatmap.png)

## Results

The specialist arc on the synthetic gate is the clearest story. The same style of gate moves from near-zero on the base model, through mid-pack numeric fine-tunes, to near-perfect once the target becomes PGF constructions. Foot-of-altitude went from 2 percent to 98 percent under that pivot; intersection went from 20 percent to 99 percent. Base still scores 0 percent on the PGF eval, so this is a trained behavior, not “the task got easier.”

| Model / target | Pass rate | Counts | Source |
| --- | ---: | --- | --- |
| Base Qwen3-0.6B (numeric) | 0.25% | 2 / 800 | `outputs/eval_base_new.json` |
| Tuned 0.6B (v1 numeric) | 46.4% | 371 / 800 | `outputs/eval_tuned.json` |
| Tuned 1.7B (v1 numeric) | 59.8% | 478 / 800 | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B (v2 PGF) | 98.9% | 277 / 280 | `outputs/eval_pgf_tuned.json` |

On the numeric 800-grid frontier sweep, no model is reliable every time across the whole landscape. Hard cells are specific constructions plus ugly numbers; foot-of-altitude at chain length four to five with irregular numbers pools around 38 to 44 percent across models. Relative to that grid, the tuned specialists beat some frontier models and do not beat others.

| Model | Pass rate | Versus specialist |
| --- | ---: | --- |
| claude-sonnet-5 | 97.7% | Specialist does not beat |
| gpt-5.5 | 87.0% | Specialist does not beat |
| Claude Opus 4.8 | 67.4% | Specialist does not beat |
| Tuned 1.7B (ours) | 59.8% | Beats gpt-4.1 and gpt-4o |
| gpt-4.1 | 55.5% | Beaten by tuned 1.7B |
| Tuned 0.6B (ours) | 46.4% | Beats gpt-4o only |
| gpt-4o | 43.9% | Beaten by both tuned specialists |
| Base 0.6B | 0.25% | — |

The full frontier table is in `outputs/sweep/pass_rates.csv`. Failures under the numeric target piled onto projection arithmetic: foot-of-altitude was 2 percent on 0.6B and 13 percent on 1.7B, and scaling barely moved it. Same scenes with a coordinate-free PGF label — for example `\coordinate (D) at ($(A)!(C)!(B)$);` — unlocked the jump above.

On the thirty-example in-domain utility check, the specialist hits 100 percent pass and 100 percent compile at essentially zero inference cost. It matches gpt-5.5 construction mode on pass, beats both models’ plain modes, and beats Opus hard when Opus tries constructions and compile collapses on hallucinated macros. It wins on cost and dialect; this laptop MPS run is not a latency win.

Synthetic in-domain was essentially solved. On held-out AIME the narrow PGF specialist was format-locked. Distilling a frontier teacher with a vision judge and a broader synthetic mix produced an illustrator that raises compile coverage a lot while faithfulness stays much lower. The local specialist does not beat frontier on AIME faithfulness; compile coverage is competitive, but compiled is not correct.

| System | Compile | Faithful |
| --- | ---: | ---: |
| Narrow PGF specialist | ~14% | 0.7% |
| Illustrator 1.7B | ~69% | 11.3% |
| Illustrator 4B | — | 24% |
| Judge-gated frontier union | — | ~64–68% |

Illustrator 4B v2, with more paraphrases and harder families, hits 98.1 percent on an expanded synthetic gate and 99.1 percent on paraphrase. That adapter is promoted in the live app as `qwen3-illustrator-4b-v2`.

The before-and-after render below is the same style of scene failing on base and passing after the tune.

![Base fails, tuned passes](outputs/renders/before_after.png)

## Product

The specialist became a custom chat SPA on Modal in `web/` and `src/geotikz/webapp.py`. Text, screenshot, or PDF goes in; a figure and TikZ come out; then conversational edits. Routing prefers the specialist for in-vocab constructions and falls back to frontier for out-of-domain scenes, with vision for uploads and one automatic repair pass on non-compiling figures. Every reply attributes which model produced it. I also shipped a CLI, Gradio demo, and worksheet generator for in-vocabulary problems.

![Custom copilot SPA](outputs/renders/copilot_web_screenshot.png)

## What I’d do better

I’d switch to the construction PGF target earlier — that was the real unlock, more than three times the model size. I’d optimize more for faithfulness on real problems, not just compile rate, and distill only diagrams that pass a vision check. Next I’d prioritize DPO on on-spec versus off-spec pairs, adversarial robustness, more polygon training, and climbing AIME faithful with data rather than hyperparameter churn.

## Assignment checklist

The Behavior Spec is [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md). The dataset is under `data/*.jsonl` with [`cards/dataset_card.md`](cards/dataset_card.md). The model and demo are the adapters under `outputs/qwen3-*`, the live URL above, and [`cards/model_card.md`](cards/model_card.md). The eval harness and tables are in `src/geotikz/harness.py`, this write-up, and [`WRITEUP.md`](WRITEUP.md). The brainlift is this doc plus [`WRITEUP.md`](WRITEUP.md). The three-to-five-minute demo video plan is [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md). The eval review speaking script is [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md).

Numbers are from committed eval JSONs and reports cited above. Reproduce commands are in [`README.md`](README.md).
