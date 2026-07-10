"""Batched ILLUSTRATOR inference on Modal GPU (Qwen3-4B + qwen3-illustrator-4b).

Drop-in counterpart to scripts/infer_illustrator_modal.py (the 1.7B illustrator)
with the SAME local-entrypoint interface (``--input in.jsonl --output out.jsonl
--max-new-tokens --batch-size --adapter``), so the AIME auto-illustrator and the
synthetic eval can select it with
``--specialist-script scripts/infer_illustrator_4b_modal.py``.

Differences from infer_illustrator_modal.py (all deliberate, all additive):
  * base model  Qwen/Qwen3-4B (the capacity probe),
  * adapter     qwen3-illustrator-4b (the 4B distilled+synthetic run; the 1.7B
                adapter qwen3-illustrator and all others are never touched),
  * GPU         A10G -> A100 (a 4B in bf16 + a batch-16 KV cache wants >24GB).
The CONSTRUCTION system prompt is byte-for-byte identical to the 1.7B script, so
the ONLY moving part between the 1.7B and 4B evals is model capacity.

ADDITIVE: does not modify train/infer scripts, data, or existing adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

GPU = "A100"
MODEL = "Qwen/Qwen3-4B"
RUN_NAME = "qwen3-illustrator-4b"

# Kept verbatim in sync with scripts/infer_illustrator_modal.py::_SYSTEM_PROMPT
# (which mirrors geotikz.prompts.CONSTRUCTION_SYSTEM_PROMPT). The Modal image has
# no local geotikz install, so it is embedded here identically.
_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler for olympiad constructions. You are given "
    "a geometry scene: some base points are given by coordinates, and one or more "
    "further points are described only by their geometric construction (e.g. the "
    "circumcenter, incenter, orthocenter, centroid, the foot of an altitude, where "
    "an angle bisector meets a side, a midpoint, or a point of tangency).\n\n"
    "Output ONE TikZ figure that realizes the scene and defines every requested "
    "named point at its correct location. You may work either way:\n"
    "  - compute the coordinates yourself and place them, e.g. "
    "\\coordinate (O) at (2.5,1.375); , or\n"
    "  - use coordinate-free constructions. The full tkz-euclide package and the "
    "tikz libraries calc, intersections, through, angles, positioning are ALREADY "
    "loaded, so macros like \\tkzDefPoint(0,0){A}, "
    "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}, "
    "\\tkzDefTriangleCenter[in]/[ortho]/[centroid], "
    "\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{F}, "
    "\\tkzDefMidPoint(B,C)\\tkzGetPoint{M}, \\tkzDefLine[bisector](B,A,C), "
    "\\tkzDefTangent[from = P](O,W)\\tkzGetPoints{T1}{T2}, and PGF calc "
    "($(a)!(c)!(b)$) are all available.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "  1. Every requested point MUST be a referenceable named coordinate/node using "
    "the EXACT name requested (case-sensitive), created by any of: "
    "\\coordinate (NAME) at (...);  \\tkzDefPoint(...){NAME}  \\tkzGetPoint{NAME}  "
    "or \\node (NAME) at (...) {}. Do not rename points.\n"
    "  2. Output ONLY the figure, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. Do NOT include \\documentclass, \\usepackage, or "
    "\\begin{document} — only the tikzpicture. No prose, no explanations, no "
    "markdown fences."
)
_USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."

app = modal.App("geotikz-infer-illustrator-4b")
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
    max_new_tokens: int = 640,
    batch_size: int = 16,
) -> list[str]:
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
    tok.padding_side = "left"

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
def main(input: str, output: str, max_new_tokens: int = 640, batch_size: int = 16,
         adapter: str = RUN_NAME) -> None:
    # adapter="none"/"base" -> run the untuned base model (for base-vs-tuned evals).
    adapter_dir = None if adapter.lower() in ("none", "base", "") else adapter
    rows = [json.loads(l) for l in Path(input).read_text().splitlines() if l.strip()]
    descs = [r["description"] for r in rows]
    print(f"read {len(descs)} descriptions from {input} (adapter={adapter_dir})")
    outs = infer_descriptions.remote(descs, adapter_dir=adapter_dir,
                                     max_new_tokens=max_new_tokens, batch_size=batch_size)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r.get("id"), "description": r["description"], "output": o}) + "\n")
    print(f"wrote {len(outs)} outputs -> {output}")
