# Geometry → TikZ: a one-week specialist, end to end

**What I built.** A small open model that turns coordinate-free geometry scenes into compiling TikZ/PGF figures, plus a live **Geometry Figure Copilot** (specialist + frontier fallback, screenshot/PDF in, conversational edits, interactive drag board).[^pgf-opus]

[^pgf-opus]: **PGF** is the engine under TikZ; “PGF constructions” means emitting `calc`/intersection macros so LaTeX does the arithmetic instead of the small model inventing coordinates (the pivot that took foot-of-altitude from ~2% → ~98%). **Opus** in comparisons is Claude Opus 4.8 via gateway id `claude-group/claude-opus-4-8`.

> **Thesis.** You can make a small open model *reliably* do one narrow thing by controlling its training data — and **how you frame the target** (compute the answer vs. emit the construction) matters more than model size or dataset size.

**Live demo:** [Geometry Figure Copilot](https://katie-he--geotikz-copilot-web.modal.run) (Modal custom chat SPA; auth `demo` / `geotikz-gpu-8t3n`; stop when idle to save credits — redeploy with `modal deploy scripts/copilot_modal.py` if needed) · Full evidence: [`WRITEUP.md`](WRITEUP.md) · Spec: [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md)

---

## 1. Problem & thesis

I wanted a model that turns a **coordinate-free** geometry scene — relationships only, no explicit coordinates — into a **single compiling TikZ/PGF figure whose every named point is numerically correct**. Not “draw something pretty.” Recover the hidden numbers from the geometry.

That passes the assignment’s litmus: a well-prompted base model (and, it turns out, most frontier models) **cannot already do it reliably**. So fine-tuning earns its keep. The win is not “smarter than GPT” — it’s a tiny, cheap, local specialist that holds a falsifiable behavior every time, on the slice it was trained for.

**Pass gate** (from `BEHAVIOR_SPEC.md`): figure-only ∧ compiles under `tectonic` ∧ every named point within `atol = 0.05` of ground truth ∧ derived, not transcribed.

---

## 2. Phase 1 — Synthetic data → 0.6B

I did **not** distill from a teacher for the core specialist. I built a small geometry engine (`src/geotikz/scene.py`, `generator.py`) that constructs each scene *forward from exact coordinates*, then **strips the coordinates** for the model input. Labels are correct by construction; difficulty is dialable (chain length, round vs irregular numbers, which ops).

| Dataset | Rows | Role |
| :-- | --: | :-- |
| `data/train.jsonl` | 5,340 | v1 numeric-target SFT |
| `data/eval.jsonl` | 800 | held-out — same grid as the frontier sweep |

The mixture is aimed at the hard region: ~83% irregular numbers, ~78% chain 4–5, ~70% contain a hard op (foot-of-altitude or intersection).

![Training data composition](outputs/renders/data_composition.png)

**Training.** LoRA on Qwen3-0.6B (r=16, α=32, all-linear, 2 epochs, bf16) on Modal. A local Colab/Unsloth smoke path exists to prove the loop closes; real runs use Modal so closing the laptop doesn’t kill the job.

**Early eval** on the 800-item grid (same pass gate as the frontier sweep):

| Model | Pass % | Count | Artifact |
| :-- | --: | --: | :-- |
| Base Qwen3-0.6B | **0.25%** | 2/800 | `outputs/eval_base_new.json` |
| Tuned 0.6B (v1 numeric) | **46.4%** | 371/800 | `outputs/eval_tuned.json` |

![Base fails, tuned passes](outputs/renders/before_after.png)

*Ground truth | base FAIL | tuned PASS — same scene.*

> **Trial-and-error — thinking mode.** Qwen3 defaults to “thinking,” emitting a `<think>…</think>` block before the answer. That breaks the figure-only gate and can burn the whole token budget before any TikZ appears. Fix: `enable_thinking=False` at train *and* infer (`src/geotikz/infer.py`). Without that, “the model failed” was often just format pollution.

> **Trial-and-error — Colab → Modal.** Browser-tab training dies on idle disconnect. `scripts/train_modal.py` runs to completion with `--detach`; no tab to keep alive.

---

## 3. Frontier difficulty sweep — where prompting breaks

Before claiming the task was worth training, I scored **12 hosted frontier models** on the same **800-item difficulty grid** (chains 2–7 × round/irregular, plus op-targeted cells) with the **exact same pass gate**.

![12-model pass heatmap](outputs/sweep/pass_heatmap.png)

**Finding:** at a strict “reliable every time” bar, **no** frontier model is reliable across the landscape. Below 90% in 19 of 20 difficulty cells (average across models). At 95%, zero models clear every cell; at 90%, only `claude-sonnet-5` squeaks by.

| Model | Overall pass % | Count |
| :-- | --: | --: |
| claude-sonnet-5 | **97.7%** | 775/793 |
| gemini-3.1-pro | 90.9% | 726/799 |
| gpt-5-mini | 90.1% | 721/800 |
| grok-4.5 | 87.9% | 703/800 |
| gpt-5.5 | 87.0% | 696/800 |
| gemini-3.5-flash | 80.4% | 642/799 |
| gpt-5.4 | 68.5% | 548/800 |
| claude-opus-4-8 | 67.4% | 539/800 |
| **tuned-1.7B (ours)** | **59.8%** | 478/800 |
| gpt-4.1 | 55.5% | 444/800 |
| **tuned-0.6B (ours)** | **46.4%** | 371/800 |
| gpt-4o | 43.9% | 351/800 |
| deepseek-v3.2 | 27.0% | 216/800 |
| claude-haiku-4-5 | 17.2% | 138/800 |
| base-0.6B | 0.25% | 2/800 |

*Source: overall rates from `outputs/sweep/pass_rates.csv` (frontier) + `outputs/eval_*.json` (ours); heatmap/report: `outputs/sweep/pass_heatmap.png`, `outputs/sweep/report.md`.*

### Beat / don’t-beat (numeric 800-grid)

Same gate, same scenes. Plain read of the table above:

| Ours | Beats (lower overall pass %) | Does **not** beat |
| :-- | :-- | :-- |
| **tuned-0.6B (46.4%)** | gpt-4o (43.9%), deepseek-v3.2 (27.0%), claude-haiku-4-5 (17.2%) | gpt-4.1 and every stronger frontier (gpt-5.4 / opus / gpt-5.5 / … / sonnet) |
| **tuned-1.7B (59.8%)** | gpt-4.1 (55.5%), gpt-4o, deepseek, haiku | claude-opus-4-8 (67.4%), gpt-5.4 (68.5%), and all models above that |

> **One sentence:** On the in-domain numeric construction grid I beat gpt-4o (and with 1.7B also gpt-4.1); I do **not** beat opus / gpt-5.5 / sonnet there — and on open AIME faithfulness I don’t beat frontier at all (see §7).

**Where reliability collapses** — not just “longer chains,” but **specific constructions + ugly numbers**:

![Op effect: foot vs control](outputs/sweep/op_effect.png)

| Cell | Operation | Chain | Pooled pass (12 models) |
| :-- | --: | --: | --: |
| easy_c4_irr | easy only (control) | 4 | **88%** |
| int_c4_irr | line intersection | 4 | **58%** |
| foot_c4_irr | foot of altitude | 4 | **44%** |
| foot_c5_irr | foot of altitude | 5 | **38%** |

That is exactly where the training data is centered — and where a fine-tune earns its keep.

---

## 4. Scale-up 1.7B + early product surfaces

Same recipe, bigger base — **specialist arc in percents** (numeric gate, n=800; PGF gate in §5):

| Stage | Pass % | Count | Artifact |
| :-- | --: | --: | :-- |
| Base Qwen3-0.6B | **0.25%** | 2/800 | `outputs/eval_base_new.json` |
| Tuned 0.6B (v1 numeric) | **46.4%** | 371/800 | `outputs/eval_tuned.json` |
| Base Qwen3-1.7B | **0.5%** | 4/800 | `outputs/eval_base_1p7b.json` |
| Tuned 1.7B (v1 numeric) | **59.8%** | 478/800 | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B (v2 PGF) | **98.9%** | 277/280 | `outputs/eval_pgf_tuned.json` |

Tuned-1.7B lands mid-pack on the identical numeric grid: **beats gpt-4.1 (55.5%) and gpt-4o (43.9%)**; **does not beat** opus (67.4%), gpt-5.4 (68.5%), or the top cluster (gpt-5.5 / grok / sonnet at 87–98%). Scaling helped — but not enough on the hard op (see next section).

I also paused to ship **useful** surfaces while the specialist was still numeric-target:

1. **CLI / Gradio demo** — scene text → figure + copyable TikZ (`scripts/demo.py`, `demo_web.py`).
2. **Worksheet generator** — N in-vocabulary problems, printable PDF + answer key (`scripts/make_worksheet.py` → `outputs/worksheets/`).

![Example worksheet figure](outputs/worksheets/figures/fig_01.png)

*In-vocabulary, correct-by-construction — the specialist’s home turf.*

![Same scene: GPT-4o vs tuned SLM](outputs/renders/llm_vs_slm.png)

*Frontier can draw a prettier figure and still get a point wrong; the specialist got this one exact.*

---

## 5. Pivot — construction target (PGF does the arithmetic)

Error analysis was not diffuse. Failures piled onto **one** operation:

| foot-of-altitude pass (v1 numeric) | Pass % |
| :-- | --: |
| tuned-0.6B | **2%** |
| tuned-1.7B | **13%** |

The model got the *structure* right and botched the **projection arithmetic**. Foot appears in ~47% of training — so more data of the same kind was not the fix. Scaling 0.6B→1.7B barely moved it. **Capability/representation problem, not volume.**

**The fix was in the label**, not the hyperparameters. Same scenes, new target: emit a coordinate-free PGF construction and let LaTeX compute the numbers at compile time.

```latex
% v1 — MODEL computes the foot:
\filldraw (3.87,1.42) circle (1.5pt) node[below] {$D$};

% v2 — model emits the construction; PGF computes:
\coordinate (D) at ($(A)!(C)!(B)$);
```

Same base (Qwen3-0.6B), same LoRA recipe, new data (`data/train_pgf.jsonl`, 2,050 rows). On the 280-item symbolic eval:

| Metric | Base | Tuned | Artifact |
| :-- | --: | --: | :-- |
| Overall pass | **0%** | **98.9%** (277/280) | `outputs/eval_pgf_tuned.json` |
| Compile rate | 53.6% | **100%** | |
| Foot-of-altitude | 0% | **98.4%** | |
| Intersection | 0% | **98.9%** | |

**Cell-matched before → after** (fair comparison; eval sets differ in size):

| Construction | v1 numeric tuned-0.6B | v2 PGF tuned-0.6B |
| :-- | --: | --: |
| Foot-of-altitude | **2%** | **98%** |
| Line intersection | **20%** | **99%** |

Base still scores **0%** on the PGF eval — so this is a genuine trained behavior, not “the task got easier.”

### Frontier comparison — PGF / utility (in-domain)

Same held-out construction task (`outputs/utility_report.md`, n=30, seed 20260709). Specialist is Qwen3-0.6B+LoRA local; frontier in **plain** vs **construction** prompt modes:

| Config | Pass % | Compile % | Coord-free % | Median latency | Est. cost/call |
| :-- | --: | --: | --: | --: | --: |
| **specialist (ours)** | **100%** | **100%** | **100%** | 42.0s (local MPS) | **$0** |
| gpt-5.5 [plain] | 96.7% | 100% | 20.0% | 25.8s | ~$0.005 |
| gpt-5.5 [construction] | **100%** | 100% | 86.7% | 14.8s | ~$0.003 |
| claude-opus-4-8 [plain] | 90.0% | 96.7% | 46.7% | 7.8s | ~$0.029 |
| claude-opus-4-8 [construction] | 63.3% | **66.7%** | 96.7% | 5.1s | ~$0.015 |

**Beat / don’t-beat (honest):**
- **Beats / ties on pass:** specialist **100%** matches gpt-5.5 construction and **beats** gpt-5.5 plain (96.7%) and opus plain (90%); **beats** opus construction hard (63.3% pass / 66.7% compile — hallucinated `tkz-euclide` macros).
- **Does not “beat” on latency** in this laptop MPS run (42s vs frontier’s ~5–26s); on a commodity GPU the same adapter is sub-second batched (Modal AIME path).
- **Wins on cost / locality / guaranteed dialect:** $0 offline vs paid API; always emits the trained `calc` dialect.

> **Line you can say:** On in-domain constructions I match or beat gpt-5.5 and opus on pass, and I beat opus on compile reliability when they try constructions; I don’t beat frontier on open AIME faithfulness.


> **Trial-and-error — numeric vs construction.** Asking a 0.6B to be a floating-point calculator was the wrong job. Asking it to be a *geometer* (emit the construction) and offloading arithmetic to PGF was the lever that mattered more than 3× model size.

---

## 6. Website / Geometry Figure Copilot

The specialist became a product: a **custom chat SPA** on Modal (`web/` + `src/geotikz/webapp.py`) — text or screenshot/PDF in → figure + TikZ, then conversational edits (“make it bigger,” “rename the labels”). The right pane has **Figure | Interactive | TikZ** tabs (rendered preview, drag board, copyable source).

**Live:** https://katie-he--geotikz-copilot-web.modal.run *(auth: `demo` / `geotikz-gpu-8t3n`; stop when idle to save credits)*

![Custom copilot SPA](outputs/renders/copilot_web_screenshot.png)

![Deployed copilot home (earlier Gradio surface)](outputs/copilot_deploy_proof/live_home.png)

Routing today:

- **Specialist first** (promoted adapter: `qwen3-illustrator-4b-v2` on Modal GPU) for in-vocab constructions.
- **Frontier fallback** when the request is out of vocab, too many simultaneous derived points, or a many-vertex polygon.
- **Vision path** for screenshots / PDFs; **edit path** for conversational changes.
- Every reply **attributes which model** produced it; non-compiling figures get one automatic repair pass.
- **Clarify questions**, **prompt normalizer** (free-form → specialist template), and **op-vocab routing**.

**Interactive board.** Free/base points are magenta and draggable; derived points are red and locked — they re-solve when bases move via inferred **constraints** (triangle centers, midpoint, foot/projection, reflections/rotation/translation, line intersections, angle-bisector∩side, simple PGF calc). Uninferable constructions stay free-drag. After editing, **Apply board edits** syncs the board back into the figure / TikZ / chat state.

![Routing badges / model attribution](outputs/copilot_deploy_proof/routing_badges.png)

![Interactive board — circumcenter before drag](outputs/copilot_deploy_proof/live_circum_board_before.png)

![Interactive board — circumcenter after drag (re-solved)](outputs/copilot_deploy_proof/live_circum_board_after.png)

![Interactive board — foot before drag](outputs/copilot_deploy_proof/live_foot_board_before.png)

![Interactive board — foot after drag (re-solved)](outputs/copilot_deploy_proof/live_foot_board_after.png)

### Example prompts → figure

**In-domain specialist** (from [`EXAMPLES.md`](EXAMPLES.md)):

> Triangle ABC has vertices A=(2,6), B=(0,0), C=(7,0). Let F be the foot of the altitude from A onto line BC…

![Foot construction (specialist-friendly)](outputs/copilot_deploy_proof/foot_before.png)

**Frontier fallback** — vague / multi-part / OOD scenes, or asking for “all three medians” without defining midpoints (specialist tends to reference undefined points → compile fail → fallback).

**Edit** — after a figure exists: “tidy the labels,” “add color,” “make it bigger” (frontier edit route; proof shots in `outputs/copilot_deploy_proof/live_edit.png`).

> **Trial-and-error — Gradio 6 Chatbot format.** Passing message *tuples* broke under Gradio 6 (tuples removed; Chatbot expects dicts). Fix in `src/geotikz/copilot.py`.

> **Trial-and-error — upload 401.** Auth cookies broke after Modal container recycle → `/gradio_api/upload` 401. Fixed by aligning session/auth with how Modal serves the app (`scripts/copilot_modal.py`).

---

## 7. AIME / real problems — distill an illustrator

Synthetic in-domain was essentially solved. Next stretch: can the *same recipe* auto-illustrate **arbitrary** competition geometry?

The narrow PGF specialist was **format-locked / narrow-vocab** — on held-out AIME it compiled only ~14% and was almost never *faithful* (re-judged: **0.7%**). So I distilled a frontier teacher on real AIME/MATH geometry, hard-filtered with a **vision judge**, and mixed in broader synthetic constructions.

| Stage | Yield |
| :-- | --: |
| Teacher outputs | 1,588 |
| Compile + non-degenerate | 1,120 (70.5%) |
| Vision-approved (faithful) | 1,099 (69.2%) |
| + synthetic families → train | **3,996** records |

Trained `qwen3-illustrator` (Qwen3-1.7B + LoRA r=32/α=64). Honest dual signal on AIME (n=150, seed 20260709):

> **Compile ≠ faithful.** A figure can compile and look geometric while still misrepresenting the problem; **faithful** means a vision judge checked that the drawn construction matches the intended AIME scene — so high compile (~69%) with much lower faithful (~11%) means the model often draws *something valid*, not *the right figure*.

| Signal | 0.6B PGF specialist | 1.7B illustrator |
| :-- | --: | --: |
| Compile + non-degenerate | 14.0% | **69.3%** |
| Judge-verified faithful | 0.7% | **11.3%** |
| Union w/ judge-gated frontier | — | **64.0%** |

Held-out **synthetic** (240, coordinate-verified): base 7.9% → tuned **93.8%**. That is the clean data→behavior win. On arbitrary AIME the model reliably *draws* (~69%) but the *correct* figure only ~11% — **reasoning-bound, not drawing-bound**.

**AIME beat / don’t-beat:** local illustrator does **not** beat frontier on faithfulness (1.7B **11.3%** / 4B **24.0%** vs gpt-5.5-anchored union **64–68%**). Compile coverage is competitive locally (~**69–70%** vs teacher filter yield ~**70.5%**), but “compiled” ≠ “faithful.”

**Capacity probe 4B** (same data/recipe): local faithful **11.3% → 24.0%**; synthetic coord-verified **93.8% → 97.1%**; system union **68%**. Capacity doubled faithfulness without changing drawing rate — still far from the teacher’s ~64% local ceiling. Details: [`ILLUSTRATOR_REPORT.md`](ILLUSTRATOR_REPORT.md), [`ILLUSTRATOR_4B_REPORT.md`](ILLUSTRATOR_4B_REPORT.md).

![AIME specialist example](outputs/aime_gallery_illustrator/specialist/2017-I-6.png)

*Local illustrator figure (gallery under `outputs/aime_gallery_illustrator/`). Faithfulness is judged separately by vision — don’t read “compiled” as “correct.”*

---

## 8. v2 fine-tune — paraphrases + harder constructions → promote

v1 already hit paraphrase robustness (~98.7% on loose wordings) but **zeroed** on families needing new vocabulary (nine-point centre, incenter-via-bisectors, square centre, …). So I changed **only the data**:

| Ingredient | Records |
| :-- | --: |
| v1 base (distill + synthetic) | 3,996 |
| Paraphrase augmentation | 3,416 |
| Harder constructions (12 families) | 1,440 |
| **Total** `illustrator_train_chat_v2.jsonl` | **8,852** |

Adapter `qwen3-illustrator-4b-v2` (same LoRA recipe as v1). GT-verified gates:

| Gate | base 4B | v1 | **v2** |
| :-- | --: | --: | --: |
| Synthetic v2 (360) | 7.2% | 84.7% | **98.1%** |
| Paraphrase (228) | 7.5% | 98.7% | **99.1%** |
| Harder 12 families | — | 60.0% | **100%** |

*Source: `outputs/syn_eval_illustrator_4b_v2/report.md`, [`ILLUSTRATOR_4B_V2_REPORT.md`](ILLUSTRATOR_4B_V2_REPORT.md).*

**Ceiling sweep** (`outputs/specialist_ceiling_robust/`): for *in-vocab robust ops*, long chains stay reliable (≥90% through chain 5 on mixed). The real ceiling is **vocabulary / phrasing / simultaneous derived points** — not chain depth. That finding is now wired into the app: route on op-vocab + derived-count, **not** chain length; normalize free-form prompts into the specialist template.

![Ceiling: family pass rates](outputs/specialist_ceiling_robust/family_passrate.png)

![Ceiling: chain heatmap](outputs/specialist_ceiling_robust/chain_heatmap.png)

v2 is **promoted** in the live Modal app (`scripts/copilot_modal.py` loads `qwen3-illustrator-4b-v2` first). Remaining synthetic gap: `regular_polygon` still ~5/12 coordinate-exact (rotation / label-order drift).

---

## 9. What works today / limitations / next

### Works today

| Surface | What it does |
| :-- | :-- |
| **Copilot** | [modal.run](https://katie-he--geotikz-copilot-web.modal.run) (custom SPA: Figure \| Interactive \| TikZ; auth demo / geotikz-gpu-8t3n; may be idle-stopped) — specialist + frontier + edits + screenshots + clarify + interactive board + normalizer routing |
| **Worksheet generator** | Printable PDF + answer key, in-vocab figures |
| **PGF specialist (0.6B)** | ~**98.9%** on in-domain construction eval |
| **Illustrator 4B v2** | **98.1%** GT on expanded synthetic gate; **99.1%** paraphrase gate |

### Baselines at a glance

| Milestone | Pass / faithful | Artifact |
| :-- | --: | --: |
| Base 0.6B | **0.25%** (2/800) | `outputs/eval_base_new.json` |
| Tuned 0.6B v1 numeric | **46.4%** (371/800) | `outputs/eval_tuned.json` |
| Tuned 1.7B v1 numeric | **59.8%** (478/800) | `outputs/eval_tuned_1p7b.json` |
| Tuned 0.6B v2 PGF | **98.9%** (277/280) | `outputs/eval_pgf_tuned.json` |
| Illustrator 1.7B synthetic | **93.8%** coord | `outputs/syn_eval_illustrator/report.md` |
| Illustrator 1.7B AIME faithful | **11.3%** | `outputs/aime_gallery_illustrator/coverage_stats.json` |
| Illustrator 4B AIME faithful | **24.0%** | `outputs/aime_gallery_illustrator_4b/coverage_stats.json` |
| Illustrator 4B + frontier union | **68.0%** faithful | [`ILLUSTRATOR_4B_REPORT.md`](ILLUSTRATOR_4B_REPORT.md) |
| Illustrator 4B v2 synthetic | **98.1%** | `outputs/syn_eval_illustrator_4b_v2/report.md` |
| Frontier sweep (sonnet best) | **97.7%** | `outputs/sweep/pass_rates.csv` |

### Limitations (honest)

- On the numeric 800-grid, **not** top-tier: sonnet **97.7%** still wins; 1.7B specialist **59.8%** beats gpt-4.1/gpt-4o but not opus/gpt-5.5/sonnet.
- On **AIME faithfulness**, local specialist (**24%** at 4B) does **not** beat the frontier teacher ceiling (~**64–68%** with judge-gated fallback).
- `regular_polygon` remains weak on coordinate verification.
- Vision-judge “faithful” is softer than coordinate assertion.

### Next (not claimed done)

DPO on on-spec vs off-spec pairs; adversarial robustness eval; more polygon training; keep climbing the AIME faithful rate with data (not hyperparameter churn).

---

## Assignment checklist map

| Required artifact | Where |
| :-- | :-- |
| Behavior Spec (falsifiable gate) | [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md) |
| Dataset | `data/*.jsonl`, [`cards/dataset_card.md`](cards/dataset_card.md) |
| Model + demo | adapters under `outputs/qwen3-*`, live copilot URL above, [`cards/model_card.md`](cards/model_card.md) |
| Eval harness + base-vs-tuned table | `src/geotikz/harness.py`, tables above, [`WRITEUP.md`](WRITEUP.md) |
| Brainlift (data → behavior?) | this doc + [`WRITEUP.md`](WRITEUP.md) — **yes**, three ways: base→tuned delta, rubric win, representation pivot |
| 3–5 min demo video | script: [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) |

Stretch ladder (DPO / adversarial / composed behavior): open, not started — core arc + product stretch (illustrator, copilot) took priority.

---

*Numbers are from committed eval JSONs / reports cited above. For the full formal write-up and reproduce commands, see [`WRITEUP.md`](WRITEUP.md) and [`README.md`](README.md).*
