# Geometry → TikZ: a one-week specialist, end to end

This is the readable write-up of what I built and what the numbers actually say. For the short speaking script, see [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md). Full evidence lives in [`WRITEUP.md`](WRITEUP.md), and the behavior contract is in [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md). The live product is the [Geometry Figure Copilot](https://katie-he--geotikz-copilot-web.modal.run) (auth `demo` / `geotikz-gpu-8t3n`; stop Modal when idle).

## Pitch

I wanted a model that turns a coordinate-free geometry scene — relationships only, no explicit coordinates — into a single compiling TikZ/PGF figure whose every named point is numerically correct. Not “draw something pretty.” Recover the hidden numbers from the geometry.

That passes the assignment’s litmus: a well-prompted base model, and most frontier models, cannot already do it reliably. So fine-tuning earns its keep. The win is not “smarter than GPT.” It’s a tiny, cheap, local specialist that holds a falsifiable behavior every time on the in-domain slice it was trained for.

In-domain means coordinate-free or lightly templated plane-geometry construction scenes that match the specialist’s training vocabulary: triangle centers, feet of altitudes, midpoints, tangents, and similar named constructions. These are scored with the synthetic gate — figure-only TikZ, compiles, and named coordinates within about 0.05 of ground truth. It does not mean arbitrary free-form AIME or contest word problems, 3D geometry, or other out-of-vocabulary setups. Those are out-of-domain, where compile can still be high but faithfulness is much harder.

The thesis is simple. You can make a small open model reliably do one narrow thing by controlling its training data — and how you frame the target, compute the answer versus emit the construction, matters more than model size or dataset size.

## Terms

Out-of-domain also covers free-form contest text and faithfulness-to-problem judgment, not just the synthetic compile-and-coords gate.

The synthetic gate means I evaluate on made-up construction scenes with a figure-only TikZ output that must compile under tectonic and land every named point within about 0.05 of ground truth. Labels are correct by construction because the generator builds each scene forward from exact coordinates, then strips those coordinates for the model input.

PGF is the engine under TikZ. “PGF constructions” means emitting calc and intersection macros so LaTeX does the arithmetic instead of the small model inventing coordinates. That pivot took foot-of-altitude from about 2 percent to about 98 percent.

Opus 4.8 in comparisons is Claude Opus 4.8 via gateway id `claude-group/claude-opus-4-8`.

Faithful, for AIME and real problems, means a vision judge checked that the drawn construction matches the intended problem scene. Compile is not the same as faithful: a figure can compile and look geometric while still misrepresenting the problem.

## Eval suite

The core pass gate from `BEHAVIOR_SPEC.md` is figure-only, compiles under tectonic, every named point within absolute tolerance 0.05 of ground truth, and derived rather than transcribed.

On top of that I ran four evaluations. First was a synthetic numeric 800-grid with the same held-out scenes for base, tuned specialists, and a twelve-model frontier sweep in `outputs/sweep/`. Second was a synthetic PGF construction eval, a 280-item symbolic set after the target pivot, in `outputs/eval_pgf_tuned.json`. Third was a utility check of the specialist versus gpt-5.5 and Opus 4.8 on thirty in-domain constructions, reported in `outputs/utility_report.md`. Fourth was AIME and real geometry, separating compile from vision-judge faithful, plus a judge-gated frontier union in the `ILLUSTRATOR_*_REPORT.md` files.

The twelve-model pass heatmap below shows how hard the numeric grid still is even for frontier models.

![12-model pass heatmap](outputs/sweep/pass_heatmap.png)

## Results

The most impressive arc, same style of gate on the specialist side, goes like this. Base was about 0.3 percent. The tuned 0.6B numeric model reached 46 percent. The tuned 1.7B numeric model reached 60 percent. The PGF construction target reached 99 percent. Foot-of-altitude went from 2 percent to 98 percent. On the numeric grid the 1.7B specialist beats gpt-4o and gpt-4.1. On the PGF utility set it matches or beats gpt-5.5 and Opus 4.8 on pass and utility. It does not beat frontier on open AIME faithfulness.

In exact counts: base Qwen3-0.6B passed 2 of 800, or 0.25 percent, in `outputs/eval_base_new.json`. Tuned 0.6B on the v1 numeric target passed 371 of 800, or 46.4 percent, in `outputs/eval_tuned.json`. Tuned 1.7B on the same numeric target passed 478 of 800, or 59.8 percent, in `outputs/eval_tuned_1p7b.json`. Tuned 0.6B on the v2 PGF target passed 277 of 280, or 98.9 percent, in `outputs/eval_pgf_tuned.json`.

I did not distill a teacher for the core specialist. A small geometry engine in `src/geotikz/scene.py` and `generator.py` builds scenes forward, strips coordinates, and dials difficulty through chain length, irregular numbers, and hard ops. Training is LoRA on Qwen3 on Modal. Thinking mode must be off or the figure-only gate fails.

The before-and-after render below is the same style of scene failing on base and passing after the tune.

![Base fails, tuned passes](outputs/renders/before_after.png)

On the numeric 800-grid frontier sweep, no model is reliable every time across the whole landscape if you take that bar seriously. Hard cells are specific constructions plus ugly numbers; foot-of-altitude at chain length four to five with irregular numbers pools around 38 to 44 percent across models. Overall pass rates were 97.7 percent for claude-sonnet-5, 87.0 percent for gpt-5.5, 67.4 percent for Claude Opus 4.8, 59.8 percent for our tuned 1.7B, 55.5 percent for gpt-4.1, 46.4 percent for our tuned 0.6B, 43.9 percent for gpt-4o, and 0.25 percent for base 0.6B. The full table is in `outputs/sweep/pass_rates.csv`.

So the beat and don’t-beat line on that numeric grid is: tuned 0.6B beats gpt-4o and weaker models; tuned 1.7B also beats gpt-4.1. Neither beats Opus, gpt-5.4, gpt-5.5, or sonnet on that grid.

The real unlock was changing the target. Failures piled onto projection arithmetic: foot-of-altitude was 2 percent on 0.6B and 13 percent on 1.7B under the numeric target. Scaling barely moved it. Same scenes, new label: emit a coordinate-free PGF construction and let LaTeX compute at compile time, for example `\coordinate (D) at ($(A)!(C)!(B)$);`. Foot went from 2 percent to 98 percent; intersection went from 20 percent to 99 percent. Base still scores 0 percent on the PGF eval, so this is a trained behavior, not “the task got easier.”

On the thirty-example in-domain utility check, the specialist hits 100 percent pass and 100 percent compile at essentially zero inference cost. gpt-5.5 in construction mode also hits 100 percent pass and compile at about three-tenths of a cent. gpt-5.5 in plain mode is 96.7 percent pass and 100 percent compile at about half a cent. Claude Opus 4.8 in plain mode is 90.0 percent pass and 96.7 percent compile at about three cents. Opus in construction mode falls to 63.3 percent pass and 66.7 percent compile at about one and a half cents, because compile collapses on hallucinated macros. The specialist matches gpt-5.5 construction on pass, beats both models’ plain modes, and beats Opus hard when Opus tries constructions. It wins on cost and dialect; this laptop MPS run is not a latency win.

Synthetic in-domain was essentially solved. On held-out AIME the narrow PGF specialist was format-locked at about 14 percent compile and 0.7 percent faithful. Distilling a frontier teacher with a vision judge and a broader synthetic mix produced an illustrator with about 69 percent compile and 11.3 percent faithful at 1.7B, or 24 percent faithful at 4B. Union with judge-gated frontier is about 64 to 68 percent. The local specialist does not beat frontier on AIME faithfulness. Compile coverage is competitive; compiled is not correct.

Illustrator 4B v2, with more paraphrases and harder families, hits 98.1 percent on an expanded synthetic gate and 99.1 percent on paraphrase. That adapter is promoted in the live app as `qwen3-illustrator-4b-v2`.

## Product

The specialist became a custom chat SPA on Modal in `web/` and `src/geotikz/webapp.py`. Text or screenshot or PDF goes in; a figure and TikZ come out; then conversational edits. The right pane is Figure, Interactive, and TikZ. Routing is specialist first for in-vocab constructions, frontier fallback for out-of-domain scenes, too many derived points, or many-vertex polygons, vision for uploads, and one automatic repair pass on non-compiling figures. Every reply attributes which model produced it. The interactive board keeps free points draggable and re-solves derived points from inferred constraints.

The live URL is https://katie-he--geotikz-copilot-web.modal.run, with auth `demo` / `geotikz-gpu-8t3n`. I also shipped a CLI and Gradio demo and a worksheet generator for in-vocabulary problems with PDF and answer key.

The screenshot below is the custom copilot SPA.

![Custom copilot SPA](outputs/renders/copilot_web_screenshot.png)

## What I’d do better

I’d switch to the construction PGF target earlier — that was the real unlock, more than three times the model size. I’d optimize more for faithfulness on real problems, not just compile rate, and distill only diagrams that pass a vision check. Next work I’d actually prioritize is DPO on on-spec versus off-spec pairs, adversarial robustness, more polygon training, and climbing AIME faithful with data rather than hyperparameter churn.

## Short Q&A

Why not just use GPT? For in-domain constructions the specialist is free at inference, always emits a well-formed dialect, and on the utility set it matches gpt-5.5 and beats Opus on construction-mode compile. Frontier is the fallback for out-of-distribution AIME and free-form contest text.

What does “faithful” mean? Compile means the TikZ ran. Faithful means the picture matches the problem statement. Local 4B is about 24 percent faithful on AIME; frontier union is about 64 to 68 percent.

What is an adapter? A small LoRA on top of Qwen3. I don’t train a whole model from scratch; I teach the base this one skill: geometry text to TikZ.

In-domain versus out-of-domain in one line? In-domain is coordinate-free or lightly templated construction scenes in the training vocab, scored by the synthetic gate. Out-of-domain is free-form AIME or contest text, 3D, or other out-of-vocab setups, where compile can still be high and faithfulness is harder.

## Assignment checklist

The Behavior Spec is [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md). The dataset is under `data/*.jsonl` with [`cards/dataset_card.md`](cards/dataset_card.md). The model and demo are the adapters under `outputs/qwen3-*`, the live URL above, and [`cards/model_card.md`](cards/model_card.md). The eval harness and tables are in `src/geotikz/harness.py`, this write-up, and [`WRITEUP.md`](WRITEUP.md). The brainlift is this doc plus [`WRITEUP.md`](WRITEUP.md). The three-to-five-minute demo video plan is [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md). The eval review speaking script is [`EVAL_REVIEW_PREP.md`](EVAL_REVIEW_PREP.md).

Numbers are from committed eval JSONs and reports cited above. Reproduce commands are in [`README.md`](README.md).
