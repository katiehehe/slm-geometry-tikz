"""Train the ILLUSTRATOR specialist at 4B on Modal — Qwen3-4B + LoRA.

ADDITIVE capacity probe: identical data + learning hyper-parameters to
scripts/train_illustrator_modal.py (the 1.7B run), but on a larger base model to
test whether extra capacity raises FAITHFUL AIME illustration coverage toward the
frontier teacher's ceiling. This is a NEW app with a NEW RUN_NAME
(`qwen3-illustrator-4b`); it never touches the 1.7B adapter (`qwen3-illustrator`)
or any other existing adapter on the Volume.

Same-as-1.7B (so the only moving part is capacity):
  * data          data/illustrator_train_chat.jsonl (the exact 3,996-record set)
  * LoRA          r=32, alpha=64, dropout=0.05, all-linear
  * optimisation  lr=2e-4, cosine, warmup 0.05, epochs=2, effective batch 16,
                  max_len=2560, bf16
Changed (infra only — does not alter the learning problem):
  * base model    Qwen/Qwen3-1.7B  ->  Qwen/Qwen3-4B
  * GPU           A100 (40GB)      ->  A100-80GB (a 4B LoRA at batch 8 x 2560
                  needs ~50GB of activations; 80GB fits it with headroom).
                  gradient_checkpointing is left OFF to mirror the 1.7B run;
                  if a smaller GPU is used, drop batch_size to 4 (grad_accum 4)
                  to keep the effective batch at 16, or switch GPU to "H100".

Run (detached, survives laptop sleep / wifi drop):
  modal run --detach scripts/train_illustrator_4b_modal.py --epochs 2

Download the adapter when done:
  modal volume get geotikz-outputs qwen3-illustrator-4b ./outputs/qwen3-illustrator-4b
"""

from __future__ import annotations

from pathlib import Path

import modal

# A 4B LoRA (batch 8 x seq 2560, no grad-checkpointing) peaks ~50GB of
# activations + 8GB weights, so 40GB OOMs but 80GB fits. "H100" also works and is
# faster. If forced onto 40GB, set batch_size=4, grad_accum=4 (effective 16).
GPU = "A100-80GB"
MODEL = "Qwen/Qwen3-4B"
RUN_NAME = "qwen3-illustrator-4b"

app = modal.App("geotikz-train-illustrator-4b")
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
    timeout=4 * 60 * 60,
    volumes={"/outputs": outputs_vol},
)
def train(
    run_name: str = RUN_NAME,
    epochs: float = 2.0,
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

    # Identical to the 1.7B run (train_illustrator_modal.py).
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
def main(epochs: float = 2.0, run_name: str = RUN_NAME,
         data_file: str = "/root/data/illustrator_train_chat.jsonl") -> None:
    remote_path = train.remote(run_name=run_name, epochs=epochs, data_file=data_file)
    print("\n" + "=" * 70)
    print("DONE. Download the adapter with:")
    print(f"  modal volume get geotikz-outputs {RUN_NAME} ./outputs/{RUN_NAME}")
    print("Then run the 4B illustrator eval (new out-dir; 1.7B run untouched):")
    print("  uv run python scripts/illustrate_aime.py --n 150 --backend modal \\")
    print("      --specialist-script scripts/infer_illustrator_4b_modal.py \\")
    print("      --out-dir outputs/aime_gallery_illustrator_4b --max-new-tokens 1536 \\")
    print("      --fallback-model openai-group/gpt-5.5")
    print("=" * 70)
    print(f"(adapter is at {remote_path} on the 'geotikz-outputs' Volume)")
