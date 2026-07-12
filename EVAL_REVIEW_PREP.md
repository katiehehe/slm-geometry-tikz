# Speaking script — Geometry → TikZ eval review

**Open:** this file · `outputs/sweep/pass_heatmap.png` · [https://katie-he--geotikz-copilot-web.modal.run](https://katie-he--geotikz-copilot-web.modal.run) (`demo` / `geotikz-gpu-8t3n`)  
**Deploy if needed:** `modal deploy scripts/copilot_modal.py` · cold ~55s · **stop Modal after**

Aim ~5–7 minutes; skip Demo if you’re not showing the site.

---

## Opening

Hi — I’m Katie, and I built a small specialist model that turns a geometry description into a compiling TikZ figure. The idea isn’t “beat GPT at everything.” It’s: if you control the data and how you represent the target, a small model can do one narrow job really reliably.

---

## What I evaluated

Before I cared about demos, I defined a hard gate. An output only counts as a pass if it’s figure-only TikZ, it compiles with tectonic, and every named point is within a small tolerance of ground truth. That’s the core of my eval suite.

On top of that, I ran a difficulty sweep across twelve frontier models on the same 800-item grid — things like foot-of-altitude and irregular numbers. Then I evaluated my own models on that synthetic gate, and later on real AIME geometry, where I separate “it compiled” from “it’s actually faithful to the problem.”

---

## Results

On the synthetic numeric gate, base Qwen3-0.6B was basically unusable — **0.25%** pass, two out of eight hundred. After fine-tuning on a numeric TikZ target, I got to **46.4%**. Scaling to 1.7B got me to **59.8%**.

That mid-pack number matters for the frontier comparison: at **59.8%**, the 1.7B specialist **beats gpt-4.1 at 55.5% and gpt-4o at 43.9%**. It does **not** beat Claude Opus 4.8 (“opus” below — gateway id `claude-group/claude-opus-4-8`) at 67%, gpt-5.5 at 87%, or sonnet at **97.7%**. The 0.6B numeric model only clears gpt-4o and weaker.

The real jump was changing the target. PGF is the engine under TikZ; “PGF constructions” means emitting calc/intersections so LaTeX does the arithmetic instead of the small model inventing coordinates. Same style of small model, and pass rate went to **98.9%** on the PGF eval — foot-of-altitude from about **2% to 98%**. On a thirty-example utility check, the specialist hits **100%** pass at **$0** local; when opus tries the same constructions, compile drops to **66.7%**.

For AIME, I need to be honest. Compile rates are much higher than faithfulness. Local faithful coverage is **11.3%** at 1.7B and **24%** at 4B — I do **not** beat frontier there; with a judge-gated fallback the union is about **64–68%**. So: **on in-domain constructions I beat gpt-4o and gpt-4.1, and I match or beat gpt-5.5 and opus on the PGF utility set; I don’t beat frontier on open AIME faithfulness.**

---

## Demo

If I open the copilot, you can see the product surface. I paste a geometry scene, and the badge tells you whether my specialist or a frontier model drew it. On the right, Figure is the rendered diagram, Interactive lets you drag points, and TikZ shows the code. The point of the badge is honesty — I’m not pretending every figure came from the small model.

---

## What I’d do better next time

If I did this again, I’d switch to the construction target earlier, because that was the real unlock. I’d also optimize more for faithfulness on real problems, not just compile rate — so distill only diagrams that pass a vision check. And I’d ship the write-up and demo video earlier, instead of leaving packaging for the end.

---

## Q&A

**If they ask “why not just use GPT?”**  
Because for in-domain constructions my specialist is free at inference, always emits a well-formed dialect, and on the utility set it matches gpt-5.5 and beats opus on construction-mode compile. Frontier is the fallback for out-of-distribution AIME.

**If they ask what “faithful” means**  
Compile means the TikZ ran. Faithful means the picture actually matches the problem statement — right configuration, not just “some triangle with a circle.” Local 4B is **24%** faithful; frontier union is ~**68%**.

**If they ask what an adapter is**  
It’s a small LoRA on top of Qwen3. I don’t train a whole model from scratch; I teach the base model this one skill: geometry text to TikZ.
