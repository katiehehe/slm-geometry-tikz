---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
tags:
  - geometry
  - tikz
  - pgf
  - latex
  - text-to-figure
  - synthetic
  - self-verifying
size_categories:
  - 1K<n<10K
configs:
  - config_name: v1_numeric_train
    data_files: train.jsonl
  - config_name: v2_construction_train
    data_files: train_pgf.jsonl
  - config_name: v1_eval
    data_files: eval.jsonl
  - config_name: v2_eval
    data_files: eval_pgf.jsonl
  - config_name: golden_set
    data_files: golden_set.jsonl
  - config_name: olympiad_eval
    data_files: olympiad_eval.jsonl
  - config_name: illustrator_train
    data_files: illustrator_train_chat.jsonl
  - config_name: illustrator_syn_eval
    data_files: illustrator_syn_eval.jsonl
---

# Spec-First Geometry → TikZ — dataset

Coordinate-free geometry scenes paired with a **single TikZ/PGF figure that draws them
correctly**. Each scene is described by *relationships only* (no explicit coordinates); the
label is a figure whose every named point is correct within `atol=0.05` of the ground-truth
construction. The data is **self-verifying synthetic**: scenes are generated *forward from
exact coordinates*, the coordinates are then stripped to form the model input, so every
label is **correct by construction** — no teacher hallucination, and difficulty is exactly
controllable (chain length × number regularity × which construction).

> This is the deliverable of the "dataset is the deliverable" thesis. The single most
> important design decision was not *how many* examples but **what the label is**: a numeric
> target (model computes every coordinate) vs. a construction target (model emits a
> coordinate-free PGF/`tkz-euclide` construction and lets the compiler do the arithmetic).
> Full narrative: `WRITEUP.md` in the source repo.

## Configurations

| Config | Rows | What it is |
| :-- | --: | :-- |
| `v1_numeric_train` | 5,340 | v1 training set — **numeric** target (literal `\filldraw (x,y)` points) |
| `v2_construction_train` | 2,050 | v2 training set — **construction** target (coordinate-free PGF `calc`/polar; PGF computes numbers) |
| `v1_eval` | 800 | held-out v1 eval — **the identical difficulty grid 12 frontier models were scored on** |
| `v2_eval` | 280 | held-out v2 symbolic eval (chains 3–5) |
| `golden_set` | 150 | curated, individually compile+coordinate-verified showcase (v2 construction target); 450/450 candidates verified, 150/150 written rows re-checked |
| `olympiad_eval` | 120 | named-center constructions (circumcenter, incenter, orthocenter, centroid, angle-bisector, foot-of-altitude, median, tangent) for the olympiad litmus |
| `illustrator_train` | 3,996 | AIME-illustrator SFT set (chat format): 1,099 distilled real competition pairs (×2) + 1,798 GT-verified synthetic across 20 construction families |
| `illustrator_syn_eval` | 240 | held-out synthetic illustrator eval, coordinate-verified, disjoint from training |

All splits are **disjoint** (asserted at build time): no eval scene appears in any train set.

## Schema

Raw records (`*.jsonl`), one JSON object per line:

| field | type | meaning |
| :-- | :-- | :-- |
| `id` | int | row index |
| `constraints` | list[str] | the scene as individual relationship statements |
| `description` | str | the constraints joined into one paragraph — **the model input** |
| `tikz` | str | ground-truth figure — **the label** |
| `points` | {name: [x, y]} | ground-truth coordinates used for grading |
| `chain` | int | number of derivation steps that must compose (difficulty axis 1) |
| `irregular` | bool | irregular vs. round numbers (difficulty axis 2) |
| `tags` | list[str] | constructions used (e.g. `foot_altitude`, `intersection`, `reflect_y`) |

Chat-format variants (`*_chat.jsonl`, and `illustrator_train_chat.jsonl`) carry a single
`messages` field (`system`/`user`/`assistant`) ready for SFT.

## Difficulty & composition (why prompting can't guarantee this)

Two dials make the behavior hard enough to be worth *training* rather than prompting:

- **Chain length** — how many derivation steps compose (1 = easy, 4+ breaks base models).
- **Number irregularity** — round `30°/r=5` numbers are fakeable; irregular `161°/r=2.5`
  forces real computation.

`v1_numeric_train` is aimed at the failure region a 12-model frontier sweep found — irregular
numbers + hard constructions (**foot-of-altitude ≈47%**, line-intersection ≈25%) at chain
4–5 — with an easy/round/short tail for robustness. 83% of rows use irregular numbers; 78%
sit at chain 4–5.

## Intended use

- SFT of small models to emit compiling, coordinate-correct, figure-only TikZ.
- A **base-vs-tuned eval harness** (`v1_eval`, `v2_eval`) with an objective gate.
- Studying **representation as a lever**: `v1_numeric_train` vs. `v2_construction_train`
  encode the *same* behavior with different targets; the construction target takes the
  hardest op from 0.02 → 0.98 pass at fixed model + recipe.

## Provenance & licensing

- Synthetic scenes: generated by the project's own seeded engine (`src/geotikz/scene.py`,
  `generator.py`, `olympiad_ext.py`) — labels correct by construction.
- `illustrator_train` additionally includes pairs distilled from a frontier teacher
  (`gpt-5.5`) and hard-filtered (compile + vision-judge). AIME/MATH problem *statements* are
  sourced from public datasets (`gneubig/aime-1983-2024`, `EleutherAI/hendrycks_math`);
  embedded `[asy]` diagram code is stripped. The 150-problem AIME eval sample is held out of
  distillation to prevent leakage.
- Released under **Apache-2.0**. Respect the upstream licenses of the source problem banks
  for the distilled subset.
