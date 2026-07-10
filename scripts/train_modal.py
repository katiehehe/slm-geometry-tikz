"""REAL training run on Modal — serverless GPU, no Colab timeouts.

Why this instead of Colab: you launch it as a *script* that runs to completion on
a cloud GPU. With `--detach` it keeps running even if you close your laptop or
your wifi drops. There is no browser tab to keep alive and no idle disconnect.

This mirrors notebooks/train_colab_unsloth.py but uses the repo's own stack
(transformers + TRL + PEFT LoRA, bf16). For a 0.6B model that trains in a few
minutes on a single GPU, so Unsloth/4-bit is unnecessary and just adds install
risk. The trained LoRA adapter is written to a Modal Volume you download after.

One-time setup (local):
  uv tool install modal        # installs the `modal` CLI on your PATH
  modal setup                  # opens a browser once to log in (free account)

Train (runs to completion, detached):
  modal run --detach scripts/train_modal.py

Download the adapter when it's done:
  modal volume get geotikz-outputs qwen3-geotikz ./outputs/qwen3-geotikz

Then evaluate locally (tectonic is already installed on your Mac):
  uv run python scripts/evaluate.py --data data/eval.jsonl --n 20 \
      --model Qwen/Qwen3-0.6B --adapter outputs/qwen3-geotikz \
      --tag tuned --out outputs/eval_tuned.json
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pick your GPU here. "A10G" (24GB) is a fast, reliable default; "T4" (16GB) is
# the cheapest and is plenty for a 0.6B LoRA; "L4"/"A100"/"H100" also work.
GPU = "A10G"

# Base model to fine-tune. Public, ungated — no HF token required.
MODEL = "Qwen/Qwen3-0.6B"  # v2 PGF prototype: fast 0.6B to test the construction target

# Where the adapter lands on the Volume (and thus what you `modal volume get`).
RUN_NAME = "qwen3-pgf-geotikz"

app = modal.App("geotikz-train")

# Persistent Volume: training output survives after the container exits so you
# can download it. `modal volume get geotikz-outputs <RUN_NAME> ./outputs/...`.
outputs_vol = modal.Volume.from_name("geotikz-outputs", create_if_missing=True)

# Pin to the same stack the project already uses locally, so remote behavior
# matches what you've tested. On Modal these resolve to CUDA-enabled wheels.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.12.1",
        "transformers>=5.13.0",
        "trl>=1.7.1",
        "peft>=0.19.1",
        "datasets>=5.0.0",
        "accelerate>=1.14.0",
    )
    # Ship the training data (data/train_chat.jsonl) into the image.
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,  # 1h ceiling; the real run is minutes. Colab has no such luxury.
    volumes={"/outputs": outputs_vol},
)
def train(
    epochs: float = 2.0,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_len: int = 2048,
    data_file: str = "/root/data/train_chat.jsonl",
) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"gpu={GPU}  model={MODEL}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=data_file, split="train")
    print(f"loaded {len(ds)} training examples from {data_file}")

    def to_text(batch):
        # enable_thinking=False keeps training symmetric with inference
        # (src/geotikz/infer.py). For Qwen3 the templated text is identical either
        # way here, but we set it explicitly so intent is unambiguous.
        return {
            "text": [
                tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                for m in batch["messages"]
            ]
        }

    ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    out_dir = f"/outputs/{RUN_NAME}"
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_strategy="no",
        report_to=[],
        max_length=max_len,
        dataset_text_field="text",
        bf16=True,
    )

    trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    outputs_vol.commit()  # persist to the Volume so you can download it

    print(f"saved adapter -> {out_dir} (on Volume 'geotikz-outputs')")
    return out_dir


@app.local_entrypoint()
def main(epochs: float = 2.0) -> None:
    remote_path = train.remote(epochs=epochs, data_file="/root/data/train_pgf_chat.jsonl")
    print("\n" + "=" * 70)
    print("DONE. Download the adapter with:")
    print(f"  modal volume get geotikz-outputs {RUN_NAME} ./outputs/{RUN_NAME}")
    print("Then evaluate locally:")
    print(
        "  uv run python scripts/evaluate.py --data data/eval.jsonl --n 20 "
        f"--model {MODEL} --adapter outputs/{RUN_NAME} --tag tuned "
        "--out outputs/eval_tuned.json"
    )
    print("=" * 70)
    print(f"(adapter is at {remote_path} on the 'geotikz-outputs' Volume)")


# --- Eval inference on the GPU (an 8GB Mac thrashes on base+LoRA inference) ---
# Generates model outputs for every eval example on the cloud GPU and writes them
# to the Volume. Scoring (TikZ compile + coord check) is done LOCALLY afterwards,
# since that's lightweight and needs no model.

_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only "
    "through relationships and constraints (no explicit coordinates), you must "
    "derive the exact coordinates yourself and output a single valid TikZ/PGF "
    "figure that compiles and renders the described geometry. "
    "Output ONLY the TikZ code, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. No prose, no explanations, no markdown fences."
)
_USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/outputs": outputs_vol})
def infer_eval(
    adapter_dir: str | None,
    out_name: str,
    data_file: str = "/root/data/eval.jsonl",
    max_new_tokens: int = 384,
) -> str:
    import json

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
    rows = [json.loads(l) for l in open(data_file) if l.strip()]
    tag = adapter_dir or "base"
    batch_size = 32
    print(f"[{tag}] generating for {len(rows)} eval examples (batch={batch_size}) ...")

    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": _SYSTEM_PROMPT},
                 {"role": "user", "content": _USER_TEMPLATE.format(description=ex["description"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for ex in batch
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = g[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for ex, text in zip(batch, texts):
            out.append({"id": ex["id"], "description": ex["description"], "output": text,
                        "tikz": ex["tikz"], "points": ex["points"],
                        "chain": ex.get("chain"), "tags": ex.get("tags", [])})
        print(f"  [{tag}] {min(start + batch_size, len(rows))}/{len(rows)}")

    with open(f"/outputs/{out_name}", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    outputs_vol.commit()
    print(f"[{tag}] wrote {len(out)} preds -> /outputs/{out_name}")
    return out_name


@app.local_entrypoint()
def eval_infer() -> None:
    """Generate eval outputs for tuned (base+adapter) and base, on the GPU."""
    ev = "/root/data/eval_pgf.jsonl"
    infer_eval.remote(adapter_dir=RUN_NAME, out_name="eval_preds_pgf_tuned.jsonl", data_file=ev)
    infer_eval.remote(adapter_dir=None, out_name="eval_preds_pgf_base.jsonl", data_file=ev)
    print("\n" + "=" * 70)
    print("DONE. Download the predictions with:")
    print("  modal volume get -f geotikz-outputs eval_preds_tuned.jsonl ./outputs/")
    print("  modal volume get -f geotikz-outputs eval_preds_base.jsonl ./outputs/")
    print("=" * 70)
