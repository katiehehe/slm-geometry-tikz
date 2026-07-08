# Spec-First Geometry → TikZ

A small text model that reliably compiles **coordinate-free** geometry scene
descriptions into correct, renderable TikZ — recovering the hidden numbers by
reasoning about the geometry, not transliterating a spec.

See [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md) for the falsifiable gate (the single
sentence that is simultaneously the data-gen rubric, the eval criterion, and the
thesis).

## Layout

```
src/geotikz/
  scene.py       # Scene: exact coords -> ground-truth TikZ + coord-free constraints
  generator.py   # spec-first sampler (the real IP): builds scenes, dials difficulty
  prompts.py     # system prompt encodes the Behavior Spec; chat formatting for SFT
  tex.py         # compile TikZ via tectonic; render PDF -> grayscale array (pymupdf)
  metrics.py     # figure-only check, SSIM/MSE render-diff, coordinate assertion
  judge.py       # optional LLM-as-judge (needs OPENAI_API_KEY)
  harness.py     # evaluate_one / aggregate -> compile rate, ssim, coord acc, pass rate
  infer.py       # load base/tuned model, generate TikZ (CUDA/MPS/CPU)
scripts/
  generate.py    # write dataset JSONL (+ chat-format for SFT)
  train.py       # LOCAL SMOKE SFT (LoRA, tiny model, MPS/CPU) — proves the loop
  evaluate.py    # run a model over data, score vs the spec, write results JSON
  run_smoke.py   # generate -> train -> eval end to end (Day-2 checkpoint)
notebooks/
  train_colab_unsloth.py   # REAL training: Qwen3 QLoRA on a cloud GPU (Day 3)
```

## Run the Day-2 smoke loop

```bash
uv run python scripts/run_smoke.py            # generate 50 -> train -> eval base vs tuned
```

Or step by step:

```bash
uv run python scripts/generate.py --n 50 --out data/smoke.jsonl
uv run python scripts/train.py    --data data/smoke_chat.jsonl --out outputs/smoke-adapter
uv run python scripts/evaluate.py --data data/smoke.jsonl --tag base  --out outputs/eval_base.json
uv run python scripts/evaluate.py --data data/smoke.jsonl --adapter outputs/smoke-adapter --tag tuned --out outputs/eval_tuned.json
```

## Why local training is only a smoke test

Apple Silicon has no CUDA, so Unsloth/QLoRA can't run here. `scripts/train.py`
fine-tunes a tiny model on MPS/CPU purely to prove the loop closes on junk data.
The **real** run (Day 3) uses `notebooks/train_colab_unsloth.py` — Qwen3-0.6B
QLoRA on a Colab/Modal/RunPod GPU, on filtered data at controlled difficulty.

## Eval channels (all objective, no human judge required)

- **figure-only rate** — output is just the TikZ figure, no prose
- **compile rate** — emitted TikZ compiles under tectonic
- **render-diff** — SSIM/MSE of the rendered figure vs the ground-truth render
- **coordinate accuracy** — parsed named points vs known ground-truth within tolerance
- **pass rate** — the falsifiable gate: figure-only AND compiles AND coords all correct
- **LLM-judge** (optional) — rubric scores if `OPENAI_API_KEY` is set
