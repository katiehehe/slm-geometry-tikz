"""Batched specialist inference on Modal GPU over an ARBITRARY description list.

This is the batch counterpart to the interactive/local path in
``src/geotikz/serve.py``. It reuses the *exact* proven mechanism from
``scripts/train_modal.py::infer_eval`` -- same Modal image/stack, same Volume,
same base model + ``qwen3-pgf-geotikz`` adapter, the same training-time system
prompt, and ``enable_thinking=False`` -- but instead of reading a fixed eval
file it accepts any list of scene descriptions. Used by the AIME auto-illustrator
(and optionally the utility eval) to run hundreds of scenes in ~a minute instead
of tens of minutes locally.

ADDITIVE: this does not touch train_modal.py or any existing data/adapters.

Programmatic use (from another script):

    modal run scripts/infer_modal.py --input in.jsonl --output out.jsonl

where ``in.jsonl`` has one ``{"id": ..., "description": ...}`` per line and
``out.jsonl`` gets ``{"id": ..., "description": ..., "output": ...}`` back.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Same knobs as train_modal.py so remote behaviour matches what was trained.
GPU = "A10G"
MODEL = "Qwen/Qwen3-0.6B"
RUN_NAME = "qwen3-pgf-geotikz"

# The training-time system prompt (identical to prompts.SYSTEM_PROMPT /
# train_modal._SYSTEM_PROMPT). The specialist was trained with THIS prompt, so we
# must serve with it -- NOT the newer CONSTRUCTION_SYSTEM_PROMPT.
_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only "
    "through relationships and constraints (no explicit coordinates), you must "
    "derive the exact coordinates yourself and output a single valid TikZ/PGF "
    "figure that compiles and renders the described geometry. "
    "Output ONLY the TikZ code, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. No prose, no explanations, no markdown fences."
)
_USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."

app = modal.App("geotikz-infer")
outputs_vol = modal.Volume.from_name("geotikz-outputs", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.12.1",
    "transformers>=5.13.0",
    "peft>=0.19.1",
    "accelerate>=1.14.0",
)


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/outputs": outputs_vol})
def infer_descriptions(
    descriptions: list[str],
    adapter_dir: str | None = RUN_NAME,
    max_new_tokens: int = 512,
    batch_size: int = 16,
) -> list[str]:
    """Return one raw TikZ generation per input description (batched, greedy)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"/outputs/{adapter_dir}")
    model.to("cuda").eval()
    tok.padding_side = "left"  # decoder-only: left-pad so generated tokens align

    tag = adapter_dir or "base"
    print(f"[{tag}] generating for {len(descriptions)} scenes (batch={batch_size}) ...")
    out: list[str] = []
    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start : start + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": _SYSTEM_PROMPT},
                 {"role": "user", "content": _USER_TEMPLATE.format(description=d)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for d in batch
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = g[:, inputs["input_ids"].shape[1] :]
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  [{tag}] {min(start + batch_size, len(descriptions))}/{len(descriptions)}")
    return out


@app.local_entrypoint()
def main(input: str, output: str, max_new_tokens: int = 512, batch_size: int = 16) -> None:
    rows = [json.loads(l) for l in Path(input).read_text().splitlines() if l.strip()]
    descs = [r["description"] for r in rows]
    print(f"read {len(descs)} descriptions from {input}")
    outs = infer_descriptions.remote(descs, max_new_tokens=max_new_tokens, batch_size=batch_size)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r.get("id"), "description": r["description"], "output": o}) + "\n")
    print(f"wrote {len(outs)} outputs -> {output}")
