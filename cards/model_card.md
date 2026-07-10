---
library_name: peft
license: apache-2.0
base_model:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-1.7B
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - peft
  - geometry
  - tikz
  - pgf
  - latex
  - text-to-figure
  - qwen3
  - synthetic-data
---

# Spec-First Geometry → TikZ — LoRA adapters (Qwen3)

LoRA adapters that fine-tune small **Qwen3** base models to turn a **coordinate-free**
geometry scene description (relationships only, *no explicit coordinates*) into a **single
compiling TikZ/PGF figure whose every named point is numerically correct** — recovering the
hidden numbers from the geometry rather than transcribing them.

The behavior is graded by one falsifiable gate: **figure-only AND compiles under
`tectonic` AND every named coordinate within `atol=0.05` of the ground-truth construction.**

> **Thesis:** you can make a small model *reliably* do one narrow thing by controlling its
> training data — and *how you frame the target* (compute the answer vs. emit the
> construction) matters more than model size or dataset size. Full narrative + evidence:
> `WRITEUP.md` in the source repo.

## Adapters in this repo

| Adapter (subfolder) | Base model | Target representation | Train data | Headline result |
| :-- | :-- | :-- | :-- | :-- |
| `qwen3-pgf-geotikz` | `Qwen/Qwen3-0.6B` | **v2 construction** (coordinate-free PGF; PGF computes the numbers) | 2,050 | **0.989** pass on 280-item PGF eval (base 0.000); foot-of-altitude **0.02 → 0.98** vs v1 |
| `qwen3-1.7b-geotikz` | `Qwen/Qwen3-1.7B` | v1 numeric (model computes coordinates) | 5,340 | **0.598** pass on 800-item grid (base 0.005) — beats gpt-4.1/gpt-4o/deepseek/haiku on the same grid |
| `qwen3-geotikz` | `Qwen/Qwen3-0.6B` | v1 numeric | 5,340 | **0.464** pass on 800-item grid (base 0.003) |
| `qwen3-illustrator` | `Qwen/Qwen3-1.7B` | construction (distilled + generator breadth) | 3,996 | **93.8%** coordinate-verified on 240 synthetic (base 7.9%); AIME **69.3%** compile / **11.3%** judge-faithful |
| `qwen3-illustrator-4b` | `Qwen/Qwen3-4B` | construction (distilled + generator breadth) | 3,996 | **97.1%** coordinate-verified on 240 synthetic (base 9.2%); AIME **70.0%** compile / **24.0%** judge-faithful (2.1× the 1.7B) |

All adapters are LoRA (`all-linear`), bf16, trained with TRL + PEFT on a Modal GPU.
`qwen3-*-geotikz`: r=16, α=32, 2 epochs. `qwen3-illustrator*`: r=32, α=64, 2 epochs.

## How to use

Each adapter is a standard PEFT LoRA on the stated base. Inference disables Qwen3's
"thinking" block so the output is pure, figure-only TikZ.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-0.6B"                 # match the adapter's base model
REPO = "<user>/qwen3-geotikz"            # this repo
ADAPTER = "qwen3-pgf-geotikz"            # subfolder for the v2 specialist

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, REPO, subfolder=ADAPTER).eval()

SYSTEM = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only through "
    "relationships and constraints (no explicit coordinates), you must derive the exact "
    "coordinates yourself and output a single valid TikZ/PGF figure that compiles and "
    "renders the described geometry. Output ONLY the TikZ code, starting with "
    "\\begin{tikzpicture} and ending with \\end{tikzpicture}. No prose, no markdown fences."
)
scene = ("There is a circle centered at the origin with radius 3. Point A lies on the "
         "circle at 40 degrees. Point B lies on the circle at 200 degrees. Point M is the "
         "midpoint of segment AB.")
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Scene:\n{scene}\n\nReturn the TikZ figure."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

Compile the emitted `\begin{tikzpicture}…\end{tikzpicture}` with `tectonic` (wrap in a
`standalone` doc with `tkz-euclide` + `calc`/`intersections`; see `src/geotikz/serve.py`).

## Evaluation

Objective gate, no human/LLM judge required: **figure-only + compile (`tectonic`) +
static-parser coordinate assertion** at `atol=0.05` (`src/geotikz/harness.py`). Named-center
constructions use a compile-extract grader (`src/geotikz/extract.py`). Base vs. tuned is
scored on the **identical held-out grid** the frontier models were scored on.

**v1 numeric target — pass rate on the 800-item difficulty grid:**

| Model | Base | Tuned |
| :-- | --: | --: |
| Qwen3-0.6B | 0.003 (2/800) | **0.464** (371/800) |
| Qwen3-1.7B | 0.005 (4/800) | **0.598** (478/800) |

On that same grid, tuned-1.7B (0.598) beats prompted `gpt-4.1` (0.555), `gpt-4o` (0.439),
`deepseek-v3.2` (0.270), `claude-haiku-4-5` (0.173). Top-tier frontier (e.g.
`claude-sonnet-5` 0.977) stays ahead — as expected; the win is *reliable + cheap + local +
a data-driven jump from 2/800 to 371/800*.

**v2 construction target (`qwen3-pgf-geotikz`) — 280-item PGF eval:**

| Metric | Base | Tuned |
| :-- | --: | --: |
| Overall pass | 0.000 | **0.989** (277/280) |
| Compile rate | 0.536 | **1.000** |
| Foot-of-altitude pass | 0.000 | **0.984** |
| Line-intersection pass | 0.000 | **0.989** |

Same base model + recipe as v1; **only the target representation changed** — the hardest
construction went **0.02 → 0.98**.

**`qwen3-illustrator` — held-out synthetic (240, coordinate-verified) and real AIME (150):**

| Signal | Base 1.7B | Tuned |
| :-- | --: | --: |
| Synthetic, coordinate-verified pass | 7.9% | **93.8%** |
| AIME, compile + non-degenerate (local) | 14.0% | **69.3%** |
| AIME, judge-verified faithful (local) | 0.7% | **11.3%** |

## Intended use & limitations

- **Intended:** bulk, local, offline illustration of *in-distribution* coordinate-free
  geometry scenes (worksheets, figure drafts, the trained construction vocabulary). Guaranteed
  well-formed, compiling, coordinate-free output.
- **Not intended:** general-purpose diagramming or "solve/illustrate any competition
  problem." On arbitrary hard AIME the illustrator reliably *draws* a figure (69%) but the
  *correct* one only ~11% — coverage is **reasoning-bound**, not drawing-bound. Use the
  frontier-fallback routing for the hard tail.
- **Scope:** planar Euclidean constructions; 3D / heavily combinatorial figures are out of
  scope for the local model.
- License: adapters released under **Apache-2.0**, matching the Qwen3 base models.

## Training & data

Synthetic data is generated **forward from exact coordinates** and then the coordinates are
stripped for the model input, so labels are *correct by construction* (no teacher
hallucination) and difficulty is exactly controllable. The illustrator additionally
distills a frontier teacher (`gpt-5.5`) with a two-stage compile + **vision-judge** filter.
Dataset: see the companion **dataset card**. Code: source repo (`src/geotikz/`, `scripts/`).
