# Live Review — Cheat Sheet

**Project:** Spec-First Geometry → TikZ. A tiny open model (Qwen3-0.6B) fine-tuned to do
*one* narrow thing reliably: turn a **coordinate-free** geometry scene into a **single
compiling TikZ figure whose every named point is numerically correct** — recovering the
hidden numbers from the relationships, not transcribing them.

---

## 30-second opener (say this first)

> "My behavior is: given a geometry scene described only by *relationships* — no
> coordinates — the model emits a single TikZ figure that compiles and whose every named
> point is correct within 0.05, with no prose. I picked it because it passes the litmus
> test: I ran **12 frontier models** over an **800-scene difficulty grid** with my exact
> pass gate, and at a 'reliable every time' bar **none of them are reliable** across the
> board. So this is a behavior worth *training*, not prompting. My thesis is that the
> dataset is the deliverable — and I'll show that the single biggest lever wasn't model
> size or data volume, it was **how I framed the target in the data**."

---

## Pre-flight — open these before the call

1. `outputs/renders/before_after.png` — **money shot #1**: Ground Truth | Base (FAIL) | Tuned (PASS)
2. `outputs/renders/llm_vs_slm.png` — **money shot #2**: a frontier LLM (GPT-4o) FAILS, the tuned SLM PASSES
3. `data/train.jsonl` (or use the pretty sample below) — the data
3. `outputs/sweep/pass_heatmap.png` — litmus: 12 frontier models × difficulty
4. `outputs/sweep/op_effect.png` — *where* prompting breaks (foot-of-altitude)
5. `WRITEUP.md` — full results tables (backup for hard questions)
6. this file

Quick open on macOS:
```bash
open outputs/renders/before_after.png outputs/sweep/pass_heatmap.png \
     outputs/sweep/op_effect.png
```

Regenerate the before/after live if asked (uses cached preds, ~15s, no GPU):
```bash
uv run python scripts/render_before_after.py
```

---

## 1. A sample of the data

**SHOW:** one row of `data/train.jsonl`, then one row of `data/train_pgf.jsonl` (the v2 pivot).

Pretty-print a clean example live:
```bash
python3 -c "import json;r=json.loads(open('data/train_pgf.jsonl').readline());print(r['description']);print();print(r['tikz'])"
```

**v1 example (numeric target — the MODEL computes every coordinate):**
- Input (scene, no coordinates): *"Circle at origin r=4.5. A on circle at 74°. B at 97°. C is reflection of A across the y-axis. D is where line B-origin meets the vertical line through C."*
- Label (TikZ): literal points `\filldraw (1.24,4.33) ... {$A$};` … `(-1.24,10.08) {$D$}`
- Metadata: `chain: 4`, `irregular: true`, `tags: [point_on_circle, point_on_circle, reflect_y, intersection]`

**v2 example (construction target — the model emits the construction, PGF computes numbers):**
```latex
\coordinate (A) at (161:4.5);
\coordinate (B) at ($(A)!0.5!(O)$);          % midpoint
\coordinate (D) at ($(B)!(C)!(A)$);          % foot of perpendicular from C onto line B–A
```

**SAY:** "Every example carries the scene, the ground-truth TikZ, the exact points, and
difficulty metadata (chain length, irregular numbers, and which constructions it uses).
That metadata is what lets me aim the data and slice the eval."

---

## 2. How I collected & organized the data

**SAY — collection (the spiky part):**
> "I did **not** distill from a teacher LLM. I built a small geometry engine
> (`src/geotikz/scene.py`, `generator.py`) that constructs each scene *forward from exact
> coordinates*, then **strips the coordinates** to make the model's input. So the label is
> **correct by construction** — no teacher hallucination, and I can dial difficulty
> precisely (chain length, round vs irregular numbers, which construction). It's
> self-verifying synthetic data: the same engine that makes the label is what the grader
> checks against."

**SAY — organization:**
> "The mixture is **aimed at the failure region** the frontier sweep found — irregular
> numbers + hard constructions (foot-of-altitude, line-intersection) at chain 4–5 — with a
> tail of easy/round/short examples for robustness (`scripts/build_dataset.py`, the
> `TRAIN_MIX` table). Training is built **disjoint from eval** — there's an assertion that
> zero scenes are shared. Each example is written twice: raw JSONL (for grading) and
> chat-formatted JSONL (system+user+assistant, for SFT)."

**SHOW:** `scripts/build_dataset.py` (the `TRAIN_MIX` list + the `disjoint OK` assertion),
and `outputs/renders/data_composition.png` (the category breakdown below).

**Randomness vs. design (say this):** one seeded RNG (`random.Random(7)`) → fully
reproducible. Randomness fills the *content* (radius, angles, which op, which points), a
rejection filter drops degenerate figures at the source, and results are deduped + disjoint
from eval. The *difficulty* is NOT random — the mixture **pins** it. Random draws:

| Draw | Pool | Controls |
| :-- | :-- | :-- |
| radius | round `{3,4,5,6}` / irregular `{2.5,3.5,4.5,5.5}` | scale + regularity |
| angle | 15 round vs 15 irregular values | position + regularity |
| operation | uniform over 6 ops (4 easy if constrained) | which construction |
| operand points | uniform over existing points | which points combine |

**Categories — 4 axes (computed from the 5,340 rows, `scripts/analyze_dataset.py`):**

- **Op (easy/hard):** point-on-circle 100%, midpoint 52%, **foot-altitude 47% (hard)**,
  reflect-y 46%, reflect-x 42%, **intersection 25% (hard)**. → **70% contain a hard op.**
- **Chain length:** c2 3% · c3 19% · **c4 50%** · c5 28%  (78% at the hard 4–5 range).
- **Number regularity:** **83% irregular**, 17% round.
- **Designed role:** core failure region (foot @ c4/c5 irregular) + secondary hard op
  (intersection) + easy/round/short robustness tail.

Regenerate: `uv run python scripts/analyze_dataset.py`

---

## 3. How much data

| File | Examples | What it is |
| :-- | --: | :-- |
| `data/train.jsonl` | **5,340** | v1 numeric-target training set |
| `data/train_pgf.jsonl` | **2,050** | v2 construction-target training set |
| `data/eval.jsonl` | **800** | held-out eval — *identical grid the 12 frontier models saw* |
| `data/eval_pgf.jsonl` | **280** | v2 symbolic held-out eval |
| `data/olympiad_eval.jsonl` | **120** | next-step OOD probe (real competition geometry) |
| `data/smoke.jsonl` | **50** | Day-2 smoke set (prove the loop closes) |

**SAY:** "~5.3k training examples for v1, ~2k for the v2 pivot. Volume was never the
bottleneck — the assignment's point is that quality and *framing* beat volume, and I'll
show exactly that. Held-out eval is the same 800 items the frontier models were scored on,
so base-vs-tuned-vs-frontier is apples-to-apples."

---

## 4. Training plan

**SAY:**
> "Supervised fine-tuning with LoRA on **Qwen3-0.6B** (and 1.7B to test the size axis),
> run on **Modal serverless GPU** (A10G) so it survives my laptop closing — a 0.6B model
> trains in minutes. Recipe: LoRA r=16, α=32, all-linear, 2 epochs, bf16, lr 2e-4
> (`scripts/train_modal.py`). For a 0.6B model full-precision LoRA is fine — 4-bit QLoRA
> was unnecessary. Inference disables Qwen3's 'thinking' block so the output is pure,
> figure-only TikZ."

**The plan had two versions — this is the story:**
- **v1 — numeric target:** model must *compute* every coordinate. "The model is the calculator."
- **v2 — construction target:** model emits a coordinate-free PGF construction and **PGF does the arithmetic** at compile time. "The model is the geometer, not the calculator."

**SAY:** "Same base model, same recipe — I only changed **what the label is**. That's the
whole thesis: the dataset (specifically the target representation) is the lever."

**SHOW:** `scripts/train_modal.py` (LoRA config + Modal function).

---

## 5. How I evaluate whether it worked

**SAY — eval was built before training, and it's fully objective:**
> "My gate needs no human and no LLM judge. An output **passes iff**: (1) figure-only,
> (2) compiles under `tectonic`, and (3) every named point is within 0.05 of ground truth.
> I wrote a static TikZ parser that recovers each point (polar, `calc` projections,
> `name intersections`) and checks it. Base vs tuned is scored on the **identical 800-item
> held-out grid** the frontier models saw."

**Headline results (all reproducible from `outputs/eval_*.json`):**

| Model | Base pass | Tuned pass | Delta |
| :-- | --: | --: | --: |
| Qwen3-0.6B (v1 numeric) | 0.003 (2/800) | **0.464** (371/800) | huge |
| Qwen3-1.7B (v1 numeric) | 0.005 (4/800) | **0.598** (478/800) | huge |
| Qwen3-0.6B (v2 construction) | 0.000 | **0.989** (277/280) | the pivot |

**The killer slide — the foot-of-altitude fix (diagnose → fix in DATA, not hyperparams):**

| Construction | v1 numeric, tuned-0.6B | v2 PGF, tuned-0.6B |
| :-- | --: | --: |
| Foot-of-altitude | 0.02 | **0.98** |
| Line intersection | 0.20 | **0.99** |

**SAY:**
> "v1 got the tuned 0.6B from **2/800 to 371/800** — unambiguously the data's doing; the
> base model essentially never passes. Then error analysis showed the residual failures
> concentrated on **one operation, foot-of-altitude** (0.02), and scaling to 1.7B barely
> helped (0.13) — so it's a *representation* problem, not a volume problem: I was asking a
> small model to be a floating-point calculator. I fixed it **in the data** by changing the
> target to a construction and letting PGF do the arithmetic: **0.02 → 0.98**, same model,
> same recipe. Base still scores 0.000 on v2, so it's a genuine trained behavior, not the
> task getting easier."

**Also mention (breadth of eval):**
- **Rubric win:** tuned beats base on Spec adherence (0.93→1.43) AND Robustness (0.00→0.30) — the assignment's explicit success criterion. LLM-judge harness (`judge.py`) exists as a cross-check.
- **Downstream utility eval:** on its in-domain slice the specialist passes 100% vs gpt-4o 0%, at **$0 / fully local**. And an honest OOD story: on real AIME geometry the specialist alone covers ~12.5% and a frontier fallback lifts it to ~62.5% — the intended "cheap local specialist + frontier for the hard tail" division of labor.

---

## LLM fails, SLM wins — the concrete example (`outputs/renders/llm_vs_slm.png`)

Same scene, same gate, same 800-item grid. **Scene (eval id 653, chain 4):** circle r=3.5;
A at 23°; B = reflect A across x-axis; C = reflect B across y-axis; **D = where line
B–origin meets the vertical line through C.**

- **Ground truth:** D = (-3.22, **1.37**)
- **GPT-4o (frontier LLM): FAIL** — D = (-3.22, **0.00**), off by 1.37. It drew a clean,
  elaborate figure (coordinate axes, dashed construction lines, correct A/B/C with symbolic
  `cos/sin`) but **botched the final intersection**: it stopped the line at the x-axis and
  put D where the vertical line crosses y=0, instead of extending line B–origin to meet
  x = -3.22 (which lands at y = +1.37).
- **Tuned SLM Qwen3-0.6B (local): PASS** — D = (-3.22, 1.37), exact.

**What to say:** "This is the whole thesis in one image. GPT-4o *looks* more sophisticated —
it even drew the axes — but it got the actual geometry wrong on a 4-step construction. My
0.6B local model, trained only on this behavior, computed the intersection exactly. It's not
smarter than GPT-4o in general; it's more *reliable* on the one thing I trained it for — and
it runs free and offline." (Reproduce live: `uv run python scripts/render_llm_vs_slm.py`.)

Backup numbers: on the identical 800-item grid, tuned-0.6B (0.464) and tuned-1.7B (0.598)
both beat **gpt-4o (0.439)**, **deepseek-v3.2 (0.270)** and **claude-haiku-4-5 (0.173)**;
tuned-1.7B also edges **gpt-4.1 (0.555)**.

---

## Bonus — olympiad direction & next steps (if asked "what's next?")

**Framing:** "v1→v2 proved the method on synthetic mid-difficulty. The next arc pushes toward
**real competition (olympiad) geometry**, and I'm doing it evidence-first — same loop that
worked before: mine the vocabulary → run the litmus → generate data only where it's needed."

1. **Vocabulary grounding (done).** Mined construction frequency over **1,349 MATH-geometry +
   408 AIME-geometry** problems (`outputs/construction_freq.json`). Confirms the ops I already
   target are frequent (intersection 185, perpendicular/foot 75, altitude 56) and points at the
   olympiad extensions: **circumcircle/circumcenter 48, incircle/incenter 44, trisect 12,
   orthocenter 3**. An LLM classifier also catches *implicit* constructions ("center of the
   circle through A,B,C" = circumcenter) — `outputs/construction_freq_llm.json`.
2. **Olympiad-capable grader (done).** `src/geotikz/extract.py` — a *compile-extract* grader:
   lets `tkz-euclide` place points (`\tkzCircumCenter`, `\tkzInCenter`, …) and reads their
   coordinates back out of TeX, since those can't be statically parsed. Needed for the
   circumcenter/incircle/trisection direction.
3. **Olympiad litmus (done).** `scripts/olympiad_sweep.py` scored **4 frontier models × 8
   constructions** (circumcenter, incenter, orthocenter, centroid, angle-bisector,
   foot-altitude, median, tangent) on `data/olympiad_eval.jsonl`.

   | Model | Overall pass (120) | Notable weak cell |
   | :-- | --: | :-- |
   | gemini-3.1-pro | 0.99 | — |
   | grok-4.5 | 0.98 | angle-bisector 0.87 |
   | gpt-5.5 | 0.975 | incenter 0.87 |
   | claude-opus-4-8 | 0.89 | **tangent 0.20 / compile 0.27** |

   **Honest read:** individually, named-center constructions are *mostly* solved by prompting
   (so a fine-tune there earns less than v1 did) — the residual frontier weakness is narrow
   (e.g. opus on tangent). The real unsolved region is **composed / out-of-distribution real
   AIME**, which is where the next dataset value actually is.
4. **AIME auto-illustrator (prototype).** On real AIME geometry, the specialist alone covers
   **~12.5%** (it's a narrow synthetic specialist — expected); a frontier fallback lifts total
   coverage to **~62.5%**, with ~37.5% still unillustratable (3D solids, word problems). This is
   the intended **division of labor**: free local specialist for the in-distribution slice,
   frontier spent only on the hard tail. Gallery: `outputs/aime_gallery/index.html`.
5. **Utility / economics (done).** In-domain, the specialist passes **100%** vs gpt-4o **0%** on
   that slice, at **$0 and fully offline** (`outputs/utility_report.md`). The defensible value
   isn't raw smarts — it's a reliable, free, local specialist for bulk in-distribution work.

**One-liner:** "Frontier models are already reliable on *isolated* olympiad constructions, so
the next fine-tune's value is **cheap/local/bulk + the composed OOD tail** — and I've already
built the grader and the litmus to target it."

---

## The one-line thesis (repeat at the end)

> "You can make a small model *reliably* do one narrow thing by controlling its training
> data — and **how you frame the target (compute the answer vs. emit the construction)
> matters more than model size or dataset size.** Base 0.003 → tuned 0.98 on the hardest
> construction proves it."

---

## Anticipated questions

- **"Isn't 0.46 low?"** → That's v1 on the full chain-2-to-7 grid *including* out-of-distribution chain 6–7 the model never trained on. On its trained distribution and with the v2 representation it's **0.99**. The point is the *delta from 0.003* and the representation finding, not the absolute on OOD extrapolation.
- **"Why not just prompt a frontier model?"** → The heatmap: at a 95% bar, 0 of 12 frontier models are reliable across the grid; foot-of-altitude at chain 4 pools to 0.44 vs 0.88 for an easy control. Prompting doesn't guarantee reliability — that's what fine-tuning bought.
- **"Synthetic data — is that legit vs distillation?"** → It's *stronger* here: labels are correct by construction (no teacher errors), difficulty is exactly controllable, and it's the same engine the grader uses, so there's no label noise.
- **"Did you tune hyperparameters to fix the failure?"** → No — explicitly not. Same model, same recipe; I only changed the target representation in the data. That's the assignment's "fix it in the data" done literally.
- **"Base vs tuned — is it a fair comparison?"** → Identical held-out 800-item grid, same gate, same prompt. The base 0.6B mostly fails because it won't hold the figure-only format (compiles 1.5% for 1.7B), which is exactly the reliability tuning buys.

---

## Live commands (optional — pre-rendered PNGs are safer on stage)

```bash
# data sample
python3 -c "import json;r=json.loads(open('data/train_pgf.jsonl').readline());print(r['description']);print();print(r['tikz'])"

# counts
wc -l data/*.jsonl

# aggregate eval numbers
python3 -c "import json;print(json.load(open('outputs/eval_pgf_tuned.json'))['summary'])"

# render fresh ground-truth figures
uv run python scripts/render_examples.py --n 6
```
