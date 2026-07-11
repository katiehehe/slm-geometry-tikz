# Illustrator 4B v2 — phrasing-robust + harder constructions

A NEW LoRA adapter (`qwen3-illustrator-4b-v2`, base `Qwen/Qwen3-4B`) trained to be
**robust to phrasing** and to **cover harder/broader constructions**, without
touching the deployed v1 adapter (`qwen3-illustrator-4b`). Everything here is
additive; the app is untouched (promotion is a separate step — see below).

## What changed vs v1 (data only; identical LoRA recipe)

The learning recipe is byte-for-byte v1's (r=32, α=64, dropout=0.05, all-linear;
lr=2e-4 cosine, warmup 0.05, 2 epochs, effective batch 16, max_len=2560, bf16,
A100-80GB). The **only** moving parts are the training data and the adapter name.

### Dataset composition — `data/illustrator_train_chat_v2.jsonl` (8,852 records)

| ingredient | records | notes |
|---|---:|---|
| v1 base (distilled AIME/MATH + template synthetic) | 3,996 | unchanged, additive |
| **paraphrase augmentation** (synthetic prompts) | 3,416 | frontier rewrites; TikZ target unchanged |
| **harder/broader constructions** | 1,440 | 12 new families, round-trip validated |
| **total** | **8,852** | |

- **Paraphrases**: `gemini-3.1-pro` rewrote every synthetic construction prompt
  k=2 ways (varying order/formality/vocabulary). The assistant TikZ target is
  kept **byte-for-byte unchanged**; each rewrite is validated to preserve every
  number (coords/radii/angles) and every requested point name, so the
  (prompt → figure) pair stays correct-by-construction. Yield: **3,416 / 3,596 = 95%**.
- **Harder constructions** (`src/geotikz/olympiad_hard.py`, 12 families with 2–4
  derived points + light compositions): Euler line, nine-point centre, medial
  triangle, orthic triangle, incircle contact triangle, incenter-via-two-bisectors,
  three medians, parallelogram centre, midpoint-reflection chain, circumcircle
  antipode, square centre, reflection of orthocentre over a side. Every figure is
  built forward from exact coordinates and **round-trip validated** (emit →
  compile → read back == GT); **round-trip yield was 100% (1,440/1,440 train,
  120/120 eval)**.

### Held-out evals (GT-graded, disjoint from training)
- `data/illustrator_syn_eval_v2.jsonl` — **360** = v1 gate (240) + harder (120).
- `data/illustrator_paraphrase_eval.jsonl` — **228** loose rewordings of the v1
  gate problems (unseen wordings of unseen problems; `regular_polygon` dropped
  because its `P0…Pn` phrasing can't be strictly number-validated).

## New adapter

- **Name**: `qwen3-illustrator-4b-v2`
- **Location**: Modal Volume `geotikz-outputs`, path `/qwen3-illustrator-4b-v2/`
  (`adapter_model.safetensors` + config + tokenizer). v1 and the 1.7B adapter
  are untouched.
- Train runtime 6,348s (~106 min), final train loss 0.109, ~14M tokens.

## Results — base vs v1 vs v2 (grader: figure-only ∧ compiles ∧ every named point within 0.05 of GT)

| gate | base 4B | v1 | **v2** |
|---|---|---|---|
| Synthetic gate v2 (coord-verified, 360) | 7.2% | 84.7% | **98.1%** |
| Paraphrase gate (unseen wordings, 228) | 7.5% | 98.7% | **99.1%** |

Decomposed:

| slice | v1 | v2 | Δ |
|---|---|---|---|
| Base 20 construction types (gate) | 233/240 = 97.1% | 233/240 = 97.1% | 0.0 (held, no regression) |
| **Harder 12 construction types (gate)** | 72/120 = 60.0% | **120/120 = 100%** | **+40.0** |
| Paraphrase (phrasing robustness) | 225/228 = 98.7% | 226/228 = 99.1% | +0.4 (held at ceiling) |

**v2 solved every harder family v1 could not** (each 10/10 vs v1):
`bisector_incenter` 0→10, `incircle_contact` 2→10, `midpoint_reflect_chain` 0→10,
`nine_point_center` 0→10, `square_center` 0→10 — while holding the base gate and
the paraphrase gate at ceiling.

### Interpretation
- **Coverage is the big win.** v1 already generalised to compositions of
  primitives it knew (Euler line, medial/orthic triangle, three medians) but
  scored **0** on families needing new point-chains/vocabulary
  (nine-point centre, incenter-via-bisectors, midpoint-reflection, square centre)
  and only 2/10 on the contact triangle. v2 gets all of these to 10/10.
- **Phrasing robustness was already largely present** (v1 hit 98.7% on loose
  wordings — the 55% distilled AIME/MATH share had already taught NL variety).
  v2's paraphrase augmentation's job was therefore to **preserve** that robustness
  while the data mix shifted toward synthetic — which it did (99.1%, no regression).

### Remaining gap
- `regular_polygon` is unchanged at **5/12** for both v1 and v2: v2 got no extra
  polygon signal (most polygon paraphrases were dropped by the strict
  number-preservation check on `P0…Pn`). Fix: add polygon-specific training
  (looser paraphrase validation for `P#` names, or more `regular_polygon`
  round-trip samples with explicit per-vertex construction).

## Recommendation on promoting v2 into the app
**Promote v2 — but as the separate, app-owned step, after a real-AIME spot-check.**

- v2 strictly dominates v1 on the GT-verified gate (+13.4 pts overall, +40 pts on
  harder families) and matches it on phrasing (99.1% vs 98.7%), with **no
  regression** on the base 20 types. It is a clean, drop-in upgrade (same base
  Qwen3-4B, same construction prompt, adapter already on the Volume).
- **One check before flipping the switch**: the distilled real-problem share fell
  from 55% → 25% of the mix, so confirm real-AIME illustration coverage did not
  regress before promotion:
  ```
  uv run python scripts/illustrate_aime.py --n 150 --backend modal \
      --specialist-script scripts/infer_illustrator_4b_modal.py \
      --out-dir outputs/aime_gallery_illustrator_4b_v2 --max-new-tokens 1536 \
      --fallback-model openai-group/gpt-5.5
  ```
  (point `infer_illustrator_4b_modal.py`'s `RUN_NAME`/`--adapter` at
  `qwen3-illustrator-4b-v2` for that run).
- **Promotion mechanism (app agent, not done here)**: `scripts/copilot_modal.py`
  loads the first present adapter in its `ADAPTERS` list. Promotion = make the top
  entry `("qwen3-illustrator-4b-v2", "Qwen/Qwen3-4B", "construction")` and redeploy.
  Left untouched here to avoid conflicting with the web-app workstream.

## Reproduce
```
uv run python scripts/build_illustrator_v2_data.py           # data (resumable, cached)
modal run --detach scripts/train_illustrator_4b_v2_modal.py --epochs 2
uv run python scripts/eval_illustrator_v2.py                 # base vs v1 vs v2
```

## Blockers
None. Real Modal GPU (train ~106 min on A100-80GB; eval ~3×588 gens) + gateway
(~2,040 paraphrase calls) all completed within budget. Only caveat is the
`regular_polygon` gap noted above.
