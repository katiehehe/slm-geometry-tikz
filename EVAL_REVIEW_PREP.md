# Speaking script — Geometry → TikZ eval review

This is the script to read aloud. The longer essay with images and full evidence is [`DEMO_WRITEUP.md`](DEMO_WRITEUP.md). While speaking, keep open that write-up, the heatmap at `outputs/sweep/pass_heatmap.png`, and the [live copilot](https://katie-he--geotikz-copilot-web.modal.run) with auth `demo` / `geotikz-gpu-8t3n`. Stop Modal when idle. Aim for about five to seven minutes, and skip the demo section if you are not showing the site.

## Opening

Hi — I’m Katie, and I built a small specialist model that turns a geometry description into a compiling TikZ figure. The idea isn’t “beat GPT at everything.” It’s that if you control the data and how you represent the target, a small model can do one narrow job really reliably.

**In-domain** means coordinate-free or lightly templated plane-geometry construction scenes in the specialist’s training vocabulary — triangle centers, feet of altitudes, midpoints, tangents, and similar named constructions. Those scenes are scored by a hard **synthetic gate**: figure-only TikZ, compiles with tectonic, and every named point within about 0.05 of ground truth. Out-of-domain covers free-form AIME or contest text, 3D, and other out-of-vocab setups, where compile can still be high but faithfulness is much harder.

**PGF** constructions mean the model emits calc and intersection macros so LaTeX does the arithmetic instead of inventing coordinates. **Opus 4.8** in my comparisons is Claude Opus 4.8 via gateway id `claude-group/claude-opus-4-8`. Faithful means a vision judge says the picture matches the problem; compile alone is not enough.

## Results

The specialist arc on the synthetic gate is the story I want remembered first.

| Model / target | Pass rate |
| --- | ---: |
| Base Qwen3-0.6B (numeric) | 0.25% |
| Tuned 0.6B (numeric) | 46.4% |
| Tuned 1.7B (numeric) | 59.8% |
| Tuned 0.6B (PGF) | 98.9% |

Foot-of-altitude went from about 2 percent to 98 percent under the PGF pivot. Base still scores zero on that PGF eval, so this is a trained behavior, not the task getting easier.

On the numeric 800-grid, the tuned specialists beat some frontier models and do not beat others.

| Model | Pass rate | Versus specialist |
| --- | ---: | --- |
| claude-sonnet-5 | 97.7% | Specialist does not beat |
| gpt-5.5 | 87.0% | Specialist does not beat |
| Claude Opus 4.8 | 67.4% | Specialist does not beat |
| Tuned 1.7B (ours) | 59.8% | Beats gpt-4.1 and gpt-4o |
| gpt-4.1 | 55.5% | Beaten by tuned 1.7B |
| Tuned 0.6B (ours) | 46.4% | Beats gpt-4o only |
| gpt-4o | 43.9% | Beaten by both tuned specialists |

On the thirty-example in-domain utility check, the specialist hits 100 percent pass at zero dollars local, matches gpt-5.5 construction-mode pass, and beats Opus when Opus tries constructions. On AIME I need to be honest: compile rates are much higher than faithfulness, and I do not beat frontier there.

| System | Compile | Faithful |
| --- | ---: | ---: |
| Narrow PGF specialist | ~14% | 0.7% |
| Illustrator 1.7B | ~69% | 11.3% |
| Illustrator 4B | — | 24% |
| Judge-gated frontier union | — | ~64–68% |

So the line to remember: on in-domain constructions I beat gpt-4o and gpt-4.1 on the numeric grid, and I match or beat gpt-5.5 and Opus on the PGF utility set; I don’t beat frontier on open AIME faithfulness.

## Demo

If I open the copilot, you can see the product surface. I paste a geometry scene, and the badge tells you whether my specialist or a frontier model drew it. On the right, Figure is the rendered diagram, Interactive lets you drag points, and TikZ shows the code. Routing prefers the specialist for in-vocab constructions and falls back to frontier for out-of-domain scenes. The point of the badge is honesty — I’m not pretending every figure came from the small model.

## What I’d do better next time

If I did this again, I’d switch to the construction target earlier, because that was the real unlock — more than three times the model size. I’d also optimize more for faithfulness on real problems, not just compile rate, so I’d distill only diagrams that pass a vision check. Next I’d prioritize DPO on on-spec versus off-spec pairs, adversarial robustness, more polygon training, and climbing AIME faithful with data rather than hyperparameter churn.
