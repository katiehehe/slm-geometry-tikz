# Demo video — shot list & script (target 4:00, hard cap 5:00)

The required video shows the model doing the thing the base model **fails** to do reliably.
This plan does that four ways: base-vs-tuned, the live demo, a worksheet, and the AIME
illustrator. Times are cumulative. Spoken lines are a script; trim to taste.

## Record these assets first (all already exist — no GPU, no spend)

```bash
open outputs/renders/before_after.png        # GT | Base (FAIL) | Tuned (PASS)
open outputs/renders/llm_vs_slm.png          # GPT-4o FAILS | tuned 0.6B PASSES (same scene)
open outputs/sweep/pass_heatmap.png          # 12 frontier models x difficulty
open outputs/worksheets/worksheet.pdf outputs/worksheets/answer_key.pdf
open outputs/aime_gallery_illustrator/index.html   # illustrator contact sheet
```

Optional live regen (uses cached preds, ~15s, no GPU):
`uv run python scripts/render_before_after.py` · `uv run python scripts/render_llm_vs_slm.py`

---

## 0:00–0:30 — Hook + the litmus (why this is worth training)

**On screen:** `outputs/sweep/pass_heatmap.png`.

> "Task: turn a geometry scene described *only by relationships* — no coordinates — into a
> TikZ figure that compiles and whose every point is correct. I ran **12 frontier models**
> over an **800-scene difficulty grid** with a strict pass gate. At a 'reliable every time'
> bar, **zero** are reliable across the board — below 90% in 19 of 20 difficulty cells. So
> this is a behavior worth *training*, not prompting."

## 0:30–1:20 — The behavior + base vs tuned

**On screen:** `outputs/renders/before_after.png` (Ground Truth | Base FAIL | Tuned PASS).

> "Here's one scene. The **base** Qwen3-0.6B fails — it won't even hold the figure-only
> format. The **tuned** 0.6B nails it. Across the held-out grid the base passes **2 out of
> 800**; tuning takes that to **371 out of 800** — and the 1.7B to **478**. Same grid the
> frontier models saw, so it's apples-to-apples: our tuned 1.7B beats prompted gpt-4.1,
> gpt-4o, deepseek, and haiku. It's not smarter than a frontier model — it's *reliable* at
> the one thing I trained it for."

## 1:20–2:05 — The spiky insight: fix it in the DATA (representation pivot)

**On screen:** split — a v1 numeric label vs. a v2 construction label (from `WRITEUP.md` §7),
then the fix table.

> "Error analysis showed the failures piled up on **one** operation — foot-of-altitude:
> 0.02 pass, and scaling to 1.7B barely helped. So it's not a data-volume problem, it's a
> *representation* problem — I was asking a small model to be a floating-point calculator.
> The fix was in the **data**, not the hyperparameters: change the label from *compute the
> number* to *emit the construction* and let PGF do the arithmetic. Same model, same recipe.
> Foot-of-altitude went **0.02 → 0.98**; overall **0.46 → 0.99**. The base still scores
> zero, so it's a genuine trained behavior."

## 2:05–2:45 — Live demo

**On screen:** terminal + browser.

```bash
uv run python scripts/demo_web.py     # or: scripts/demo.py "<scene>"
```

> "Here's the specialist as a product. I paste a scene…" *(use the foot-of-perpendicular
> example)* "…and it returns a compiling, coordinate-free construction and the rendered
> figure — locally, free, offline." *(Cut to `llm_vs_slm.png`:)* "Same scene, GPT-4o drew a
> prettier figure but got the final intersection **wrong**; the 0.6B local model got it
> **exact**."

## 2:45–3:20 — Worksheet (the specialist is *useful*)

**On screen:** `outputs/worksheets/worksheet.pdf` + `answer_key.pdf`.

```bash
uv run python scripts/make_worksheet.py --source generator --n 8 --seed 7
```

> "Because every figure is coordinate-free and correct-by-construction, I can generate a
> **printable geometry worksheet** with an answer key in one command — 8 problems, 8 correct
> figures, compiled with tectonic. On its in-domain slice the specialist passes **100%** vs
> frontier models at **$0** and fully offline."

## 3:20–4:15 — AIME illustrator (the honest stretch)

**On screen:** `outputs/aime_gallery_illustrator/index.html` (scroll the contact sheet).

> "Can the same recipe scale to *real* competition geometry? I distilled a frontier teacher,
> hard-filtered with a vision judge, and trained a **1.7B illustrator**. On held-out
> **synthetic** scenes with ground truth it's **7.9% → 93.8%** coordinate-verified — a clean
> data→behavior win. On **arbitrary AIME**, honestly: it reliably *draws* a figure **69%** of
> the time, but the *correct* one only about **11%** — that gap is reasoning, not drawing,
> and a 1.7B can't out-reason a frontier model on novel problems. So the system routes: free
> local specialist for what it knows, frontier for the hard tail — together **64%** faithful."

## 4:15–4:40 — Close

**On screen:** `WRITEUP.md` TL;DR table.

> "Did data → behavior hold? Yes, three ways: the base→tuned delta, the rubric win, and the
> representation pivot — **0.003 → 0.98** on the hardest construction by changing only what
> the label *is*. The dataset was the deliverable."

---

### Cutaways / B-roll (optional)

- `outputs/sweep/op_effect.png` — *where* prompting breaks (foot-of-altitude vs. control).
- A single row of `data/train_pgf.jsonl` — the coordinate-free construction label.
- `outputs/renders/data_composition.png` — the data aimed at the failure region.

### Do / don't

- **Do** show a base failure and a tuned success on the *same* scene (before_after / llm_vs_slm).
- **Do** state the numbers as reliability + cost, not "beats GPT."
- **Don't** overclaim the illustrator — say "draws 69%, correct ~11%" explicitly.
