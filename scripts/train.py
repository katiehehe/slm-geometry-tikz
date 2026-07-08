"""Local smoke SFT (LoRA) to prove the train step of the loop runs end to end.

This is NOT the real training run. On Apple Silicon there is no CUDA, so real
QLoRA on Qwen3 happens on a cloud GPU (see notebooks/train_colab_unsloth.py).
Here we fine-tune a tiny model on MPS/CPU just to close the loop on junk data.

Usage:
  uv run python scripts/train.py --data data/smoke_chat.jsonl --out outputs/smoke-adapter \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --max-steps 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from geotikz.infer import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/smoke_chat.jsonl")
    ap.add_argument("--out", type=str, default="outputs/smoke-adapter")
    ap.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    device = pick_device()
    print(f"device={device}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")

    def to_text(batch):
        texts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_length=1024,
        dataset_text_field="text",
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
    )
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
