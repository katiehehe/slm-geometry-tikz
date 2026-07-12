# Eval review prep — Geometry → TikZ SLM

**When:** live ~5–10 min @ 4 · **Repo:** `/Users/katiehe/dev/projects/slm-geometry-tikz`  
**Live app:** https://katie-he--geotikz-copilot-web.modal.run · auth `demo` / `geotikz-gpu-8t3n`  
**Redeploy only if needed:** `modal deploy scripts/copilot_modal.py` *(burns GPU credits — stop app after)*

---

## Open these tabs (one page)

| # | Open | Why |
| -: | :--- | :--- |
| 1 | This file (`EVAL_REVIEW_PREP.md`) | Script |
| 2 | `BEHAVIOR_SPEC.md` | Pass gate in one sentence |
| 3 | `outputs/sweep/pass_heatmap.png` | 12-model litmus |
| 4 | `outputs/sweep/op_effect.png` | Foot collapses prompting |
| 5 | `outputs/renders/before_after.png` | Base FAIL → tuned PASS |
| 6 | `outputs/renders/llm_vs_slm.png` | GPT-4o wrong, SLM exact |
| 7 | `outputs/eval_pgf_tuned.json` (or table below) | Headline 0.989 |
| 8 | `outputs/syn_eval_illustrator_4b_v2/report.md` | Illustrator v2 gates |
| 9 | `outputs/aime_gallery_illustrator_4b/coverage_stats.json` | Compile ≠ faithful |
| 10 | `outputs/aime_amc_v3/aime_amc_specialist_coverage.md` | Honest routing funnel |
| 11 | Copilot URL (above) | Live demo |
| 12 | `EXAMPLES.md` §B #6 (foot) | Specialist paste prompt |
| 13 | Optional: `outputs/specialist_ceiling_robust/family_passrate.png` | Ceiling = vocab, not chain |

```bash
open BEHAVIOR_SPEC.md \
  outputs/sweep/pass_heatmap.png outputs/sweep/op_effect.png \
  outputs/renders/before_after.png outputs/renders/llm_vs_slm.png \
  outputs/syn_eval_illustrator_4b_v2/report.md \
  outputs/aime_amc_v3/aime_amc_specialist_coverage.md \
  EXAMPLES.md
# then browser → https://katie-he--geotikz-copilot-web.modal.run
```

**Cold start:** first GPU call after idle ~55s; warm ~8–15s. **Stop Modal after** to save $.

---

## 1. 30-second pitch

> Given a **coordinate-free** geometry scene (relationships only), the model emits a **single TikZ/PGF figure** that **compiles** and whose every named point is correct within **0.05** — no prose. I ran **12 frontier models** on the same **800-scene** gate; at a “reliable every time” bar, **none** clear the landscape. Fine-tuning earns its keep. Thesis: **controlling the data — especially how you frame the target (compute numbers vs emit the construction) — matters more than model size.** Same 0.6B + LoRA: numeric foot **0.02 → PGF foot 0.98**.

---

## 2. Eval suite walkthrough

**What to say:** “Eval was written before training. Same gate grades data and the model.”

| Eval | Measures | Pass / signal | Open on screen | Command (if asked) |
| :--- | :--- | :--- | :--- | :--- |
| **Behavior gate** | Figure-only ∧ `tectonic` compile ∧ coords ≤0.05 (derived, not transcribed) | Binary PASS | `BEHAVIOR_SPEC.md`, `src/geotikz/harness.py` | `uv run python scripts/score_preds.py …` |
| **v1 numeric (800)** | Same grid as frontier sweep; base vs tuned 0.6B / 1.7B | Pass rate | `outputs/eval_base_new.json`, `eval_tuned.json`, `eval_tuned_1p7b.json` | score from `outputs/eval_preds_*.jsonl` |
| **Difficulty sweep** | 12 models × chains 2–7 × round/irr + hard-op cells | Reliability collapse (esp. foot) | `outputs/sweep/pass_heatmap.png`, `report.md`, `op_effect.png` | `scripts/difficulty_sweep.py` → `sweep_report.py` |
| **v2 PGF (280)** | Construction target; compile-extract / calc coords | Pass + per-op | `outputs/eval_pgf_base.json`, `eval_pgf_tuned.json` | same score path, PGF preds |
| **Olympiad litmus** | Named centers (circum/in/ortho/…) compile-extract | Frontier mostly strong; residual narrow | `outputs/olympiad_sweep/results.json` | `scripts/olympiad_sweep.py` |
| **Utility (n=30)** | In-domain specialist vs frontier cost/compile/coord-free | Specialist 100% @ $0 | `outputs/utility_report.md` | `scripts/utility_eval.py` |
| **Illustrator synthetic** | GT-verified constructions (held-out) | Coord pass | `outputs/syn_eval_illustrator*/report.md` | `scripts/eval_syn_illustrator.py` |
| **AIME coverage** | Real problems: **compile** vs **vision-judge faithful** | Dual signal — never conflate | `outputs/aime_gallery_*/coverage_stats.json` | `scripts/illustrate_aime.py` |
| **Ceiling robust** | In-vocab chains / families | Ceiling = vocab & simultaneous derived pts, not chain depth | `outputs/specialist_ceiling_robust/` | — |
| **App routing (AIME n=100)** | Specialist try/skip/fallback vs faithful | Honest funnel | `outputs/aime_amc_v3/aime_amc_specialist_coverage.md` | — |

**Talk track (60–90s):**
1. Gate = figure-only + compile + coords (`BEHAVIOR_SPEC.md`).
2. Heatmap: prompting fails on hard ops (`op_effect.png` — foot_c4_irr pooled **0.44** vs easy control **0.88**).
3. Base→tuned delta on same 800; then PGF pivot kills foot failures.
4. Illustrator: synthetic GT ≈ solved; AIME **compile ≫ faithful** (reasoning-bound).
5. Product uses that truth: route on **op-vocab + derived count**, not chain length.

---

## 3. Results table (headline numbers)

### Core specialist (falsifiable gate)

| Milestone | Pass | Artifact |
| :--- | --: | :--- |
| Base Qwen3-0.6B | **0.003** (2/800) | `outputs/eval_base_new.json` |
| Tuned 0.6B v1 numeric | **0.464** (371/800) | `outputs/eval_tuned.json` |
| Base Qwen3-1.7B | **0.005** (4/800) | `outputs/eval_base_1p7b.json` |
| Tuned 1.7B v1 numeric | **0.598** (478/800) | `outputs/eval_tuned_1p7b.json` |
| Base 0.6B PGF | **0.000** (0/280) | `outputs/eval_pgf_base.json` |
| Tuned 0.6B PGF | **0.989** (277/280) | `outputs/eval_pgf_tuned.json` |

**Cell-matched pivot (same recipe, only label changed):** foot **0.02 → 0.98**; intersection **0.20 → 0.99**. Scaling 0.6B→1.7B only moved foot **0.02 → 0.13**.

### Frontier litmus (same 800-grid gate)

| Model | Overall pass ≈ | Source |
| :--- | --: | :--- |
| claude-sonnet-5 | **0.977** | `outputs/sweep/pass_rates.csv` |
| gemini-3.1-pro | 0.909 | same |
| gpt-5-mini | 0.901 | same |
| **tuned-1.7B (ours)** | **0.598** | `eval_tuned_1p7b.json` |
| gpt-4.1 | 0.555 | sweep |
| **tuned-0.6B (ours)** | **0.464** | `eval_tuned.json` |
| gpt-4o | 0.439 | sweep |
| base-0.6B | 0.003 | `eval_base_new.json` |

*At 95% reliability across cells: zero models clear every cell; heatmap = `outputs/sweep/pass_heatmap.png`.*

### Utility (in-domain PGF, n=30)

| Config | Pass | Note | Artifact |
| :--- | --: | :--- | :--- |
| Specialist 0.6B+LoRA | **100%** | $0 local, 100% coord-free | `outputs/utility_report.md` |
| gpt-5.5 construction | 100% | compile OK | same |
| claude-opus construction | **63.3%** | compile **66.7%** (hallucinated tkz macros) | same |

### Illustrator (synthetic GT → AIME dual signal)

| Model | Synthetic GT | AIME compile | AIME faithful | Artifact |
| :--- | --: | --: | --: | :--- |
| PGF specialist 0.6B (baseline AIME) | — | **14.0%** | **0.7%** | `aime_gallery_baseline/coverage_stats.json` |
| Illustrator 1.7B | **93.8%** (225/240) | **69.3%** | **11.3%** | `syn_eval_illustrator/report.md`, `aime_gallery_illustrator/coverage_stats.json` |
| Illustrator 4B | **97.1%** (233/240) | **70.0%** | **24.0%** | `syn_eval_illustrator_4b/report.md`, `aime_gallery_illustrator_4b/coverage_stats.json` |
| Illustrator 4B v2 | **98.1%** (353/360); paraphrase **99.1%**; harder families **100%** | app funnel below | — | `syn_eval_illustrator_4b_v2/report.md` |
| Union + judge-gated frontier (1.7B / 4B) | — | — | **64% / 68%** | coverage_stats |

**Say this line:** “Compile means it drew *a* figure; faithful means a vision judge checked it matches the problem. High compile + low faithful = drawing-bound ≠ reasoning-bound.”

### App routing honesty (promoted `qwen3-illustrator-4b-v2`, AIME n=100)

| Metric | Value | Artifact |
| :--- | :--- | :--- |
| Specialist tried | 79% | `outputs/aime_amc_v3/aime_amc_specialist_coverage.md` |
| Compile/usable among tried | **79.7%** (63/79) | same |
| Faithful among tried | **19.0%** (15/79) | same |
| Demo-friendly faithful example | **2001-II-7** (compile ✓, faithful ✓) | same table |

---

## 4. Live demo plan (2–3 min)

1. **If app is cold / 404:** `modal deploy scripts/copilot_modal.py` — then wait for URL. *Don’t redeploy “just in case.” Credits.*
2. Open https://katie-he--geotikz-copilot-web.modal.run → login **`demo` / `geotikz-gpu-8t3n`**.
3. **Warn viewers:** cold start ~55s; then warm.
4. Paste **specialist example** (`EXAMPLES.md` §B #6 — foot of altitude from A onto BC). Confirm badge: specialist / `qwen3-illustrator-4b-v2`.
5. Show tabs: **Figure** → **Interactive** (drag magenta base; red derived re-solves) → **TikZ**.
6. Optional stretch: paste AIME **2001-II-7** (right triangle / altitude story) — known compile+faithful in coverage table.
7. Point at **model attribution badge** on the reply.
8. **Stop / idle the Modal app after** the review.

---

## 5. What I’d do better next time

*(Reflection, not apology — “if I restarted Monday…”)*

1. **Construction target earlier.** Foot was a representation bug, not a “need more foot data” bug. Would start PGF/labels on day 1–2.
2. **Faithfulness-gated distill from the start.** Don’t optimize AIME compile alone; filter teacher pairs on vision-faithful *before* SFT so local faithful climbs with draw rate.
3. **Publish HF + record video earlier.** Cards/scripts exist (`scripts/publish_hf.py`, `VIDEO_SCRIPT.md`); shipping evidence surfaces shouldn’t wait on the last adapter.
4. **Interactive Apply preserving constructions.** Board → figure sync should keep inferred constraints / PGF structure, not flatten to free coordinates when possible.
5. **Vocabulary / polygon data first for ceiling.** Ceiling sweep says reliability holds through chain 5 in-vocab; `regular_polygon` and OOV ops are the real wall — train those before more chain length.
6. **Dual metrics in every report.** Always report compile **and** faithful (or GT) side-by-side so “69% coverage” never reads as “69% correct.”

---

## 6. Likely Q&A

**Q: Why not just use GPT / Claude?**  
A: On the *exact* gate + 800-grid, frontier isn’t reliably perfect (heatmap; foot pools to ~0.44). Also: specialist is **$0 local**, figure-only, and emits well-formed PGF — opus construction compile drops to **66.7%** hallucinating macros. Win = reliability + cost on the slice, not “smarter than GPT.”

**Q: What does “faithful” mean?**  
A: On synthetic: every named point within 0.05 of GT. On real AIME: no GT coords → a **vision judge** (problem text + PNG) checks the figure depicts the intended configuration. Compile ≠ faithful.

**Q: Why Modal?**  
A: Serverless GPU so training/infer survive laptop close; detachable jobs; same volume holds adapters the copilot loads. Local Mac can’t batch 0.6B–4B LoRA eval cleanly.

**Q: What’s an adapter?**  
A: **LoRA** — small trainable matrices on top of a frozen base (Qwen3). Ship/swap adapters (`qwen3-pgf-geotikz`, `qwen3-illustrator-4b-v2`) without retraining the whole model.

**Q: Isn’t 0.46 / 11% faithful low?**  
A: 0.46 is v1 numeric on chains through 7 (incl. OOD). v2 PGF is **0.989**. AIME faithful ~11–24% is honest: drawing is mostly solved (~70% compile); **configuration reasoning** is the bottleneck — capacity helped (11→24%) but didn’t reach the teacher’s ~64% local ceiling.

---

## Pocket numbers (memorize)

- Gate: figure-only · compile · atol **0.05**  
- 0.6B: **0.003 → 0.464 → 0.989** (base → v1 → PGF)  
- Foot: **0.02 → 0.98**  
- 1.7B v1: **0.598**  
- Illustrator syn: **93.8% → 97.1% → 98.1%** (1.7B → 4B → v2)  
- AIME: compile **~69–70%** · faithful **11% → 24%** · union **~64–68%**  
- Demo auth: `demo` / `geotikz-gpu-8t3n` · stop after · redeploy only if down  
