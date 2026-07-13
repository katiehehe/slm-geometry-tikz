# Speaking script — Geometry → TikZ eval review

This is the script to read aloud. The longer essay with images and full evidence is [`DEMO_WRITEUP.md`](DEMO_WRITEUP.md). While speaking, keep open that write-up, the heatmap at `outputs/sweep/pass_heatmap.png`, and the [live copilot](https://katie-he--geotikz-copilot-web.modal.run) with auth `demo` / `geotikz-gpu-8t3n`. Stop Modal when idle. Aim for about five to seven minutes, and skip the demo section if you are not showing the site.

## Opening

Hi — I’m Katie, and I built a small specialist model that turns a geometry description into a compiling TikZ figure. The idea isn’t “beat GPT at everything.” It’s that if you control the data and how you represent the target, a small model can do one narrow job really reliably.

## What I evaluated

Before I cared about demos, I defined a hard synthetic gate. An output only counts as a pass if it’s figure-only TikZ, it compiles with tectonic, and every named point is within about 0.05 of ground truth. The eval scenes are made-up construction scenes with coordinates stripped from the prompt, so the model has to recover the geometry rather than copy numbers.

In-domain means coordinate-free or lightly templated plane-geometry construction scenes that match the specialist’s training vocabulary — triangle centers, feet of altitudes, midpoints, tangents, and similar named constructions — scored by that synthetic gate. It does not mean arbitrary free-form AIME or contest word problems, 3D geometry, or other out-of-vocabulary setups. Those are out-of-domain, where compile can still be high but faithfulness is much harder.

On top of the synthetic gate, I ran a difficulty sweep across twelve frontier models on the same eight-hundred-item grid — things like foot-of-altitude and irregular numbers. Then I evaluated my own models on that gate, and later on real AIME geometry, where I separate “it compiled” from “it’s actually faithful to the problem.”

PGF is the engine under TikZ. When I say PGF constructions, I mean the model emits calc and intersection macros so LaTeX does the arithmetic instead of inventing coordinates. Opus 4.8 in my comparisons is Claude Opus 4.8 via the gateway id `claude-group/claude-opus-4-8`.

## Results

On the synthetic numeric gate, base Qwen3-0.6B was basically unusable — about 0.3 percent pass, two out of eight hundred. After fine-tuning on a numeric TikZ target, I got to 46 percent. Scaling to 1.7B got me to 60 percent.

That mid-pack number matters for the frontier comparison. At 60 percent, the 1.7B specialist beats gpt-4.1 at 55.5 percent and gpt-4o at 43.9 percent. It does not beat Opus 4.8 at 67 percent, gpt-5.5 at 87 percent, or sonnet at 97.7 percent. The 0.6B numeric model only clears gpt-4o and weaker.

The real jump was changing the target. Instead of making the model compute coordinates, I trained it to emit PGF constructions and let LaTeX do the arithmetic. Same style of small model, and pass rate went to 99 percent on the PGF eval. Foot-of-altitude went from about 2 percent to 98 percent. Base still scores zero on that PGF eval, so this is a trained behavior, not the task getting easier.

On a thirty-example in-domain utility check, the specialist hits 100 percent pass at zero dollars local. It matches gpt-5.5 on construction-mode pass and beats Opus 4.8, especially when Opus tries constructions and compile drops to about 67 percent.

For AIME, I need to be honest. Compile rates are much higher than faithfulness. Local faithful coverage is 11 percent at 1.7B and 24 percent at 4B. I do not beat frontier there; with a judge-gated fallback the union is about 64 to 68 percent. So the line I want you to remember is this: on in-domain constructions I beat gpt-4o and gpt-4.1 on the numeric grid, and I match or beat gpt-5.5 and Opus on the PGF utility set; I don’t beat frontier on open AIME faithfulness.

## Demo

If I open the copilot, you can see the product surface. I paste a geometry scene, and the badge tells you whether my specialist or a frontier model drew it. On the right, Figure is the rendered diagram, Interactive lets you drag points, and TikZ shows the code. Routing prefers the specialist for in-vocab constructions and falls back to frontier for out-of-domain scenes. The point of the badge is honesty — I’m not pretending every figure came from the small model.

## What I’d do better next time

If I did this again, I’d switch to the construction target earlier, because that was the real unlock — more than three times the model size. I’d also optimize more for faithfulness on real problems, not just compile rate, so I’d distill only diagrams that pass a vision check. Next I’d prioritize DPO on on-spec versus off-spec pairs, adversarial robustness, more polygon training, and climbing AIME faithful with data rather than hyperparameter churn.

## Q&A answers you can read

If they ask why not just use GPT: for in-domain constructions my specialist is free at inference, always emits a well-formed dialect, and on the utility set it matches gpt-5.5 and beats Opus on construction-mode compile. Frontier is the fallback for out-of-distribution AIME and free-form contest text.

If they ask what “faithful” means: compile means the TikZ ran. Faithful means the picture actually matches the problem statement — right configuration, not just some triangle with a circle. Local 4B is 24 percent faithful; frontier union is about 64 to 68 percent.

If they ask what “in-domain” means: in-domain means coordinate-free construction scenes in the specialist’s training vocabulary, scored by the synthetic gate — not free-form AIME or contest problems, 3D, or other out-of-vocab setups.

If they ask what an adapter is: it’s a small LoRA on top of Qwen3. I don’t train a whole model from scratch; I teach the base model this one skill: geometry text to TikZ.
