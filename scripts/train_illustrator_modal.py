"""Train the ILLUSTRATOR specialist on Modal — Qwen3-1.7B + LoRA.

ADDITIVE: this is a new app with a NEW RUN_NAME (`qwen3-illustrator`) and a NEW
data file. It never touches the existing v1/v2 adapters (`qwen3-geotikz`,
`qwen3-1.7b-geotikz`, `qwen3-pgf-geotikz`) on the Volume.

Why 1.7B (vs the 0.6B specialist): illustrating arbitrary competition geometry
from free-form problem text needs more capacity than compiling the narrow
synthetic vocabulary did. Data is the lever, but the extra parameters help absorb
the far wider construction + language distribution of the distilled corpus.

The training records already embed the CONSTRUCTION system prompt
(build_construction_messages), so the model is tuned to emit coordinate-free
tkz-euclide / calc constructions from a raw problem statement.

Run (detached, survives laptop sleep / wifi drop):
  modal run --detach scripts/train_illustrator_modal.py

Download the adapter when done:
  modal volume get geotikz-outputs qwen3-illustrator ./outputs/qwen3-illustrator
"""

from __future__ import annotations

from pathlib import Path

import modal

# A100 (40GB) trains the full ~4k-example set in ~30-40 min; A10G also works
# (proven in smoke) but is ~2.5x slower. Falls back cleanly to A10G if changed.
GPU = "A100"
MODEL = "Qwen/Qwen3-1.7B"
RUN_NAME = "qwen3-illustrator"

app = modal.App("geotikz-train-illustrator")
outputs_vol = modal.Volume.from_name("geotikz-outputs", create_if_missing=True)

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
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={"/outputs": outputs_vol},
)
def train(
    run_name: str = RUN_NAME,
    epochs: float = 3.0,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 2e-4,
    max_len: int = 2560,
    data_file: str = "/root/data/illustrator_train_chat.jsonl",
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
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    out_dir = f"/outputs/{run_name}"
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
    outputs_vol.commit()
    print(f"saved adapter -> {out_dir} (on Volume 'geotikz-outputs')")
    return out_dir


@app.local_entrypoint()
def main(epochs: float = 3.0, run_name: str = RUN_NAME,
         data_file: str = "/root/data/illustrator_train_chat.jsonl") -> None:
    remote_path = train.remote(run_name=run_name, epochs=epochs, data_file=data_file)
    print("\n" + "=" * 70)
    print("DONE. Download the adapter with:")
    print(f"  modal volume get geotikz-outputs {RUN_NAME} ./outputs/{RUN_NAME}")
    print("Then run the illustrator eval (new out-dir, existing 14% run untouched):")
    print("  uv run python scripts/illustrate_aime.py --n 150 --backend modal \\")
    print("      --specialist-script scripts/infer_illustrator_modal.py \\")
    print("      --out-dir outputs/aime_gallery_illustrator \\")
    print("      --fallback-model openai-group/gpt-5.5")
    print("=" * 70)
    print(f"(adapter is at {remote_path} on the 'geotikz-outputs' Volume)")
