"""REAL training run — paste into Google Colab (GPU runtime) or Modal/RunPod.

Apple Silicon has no CUDA, so QLoRA on Qwen3 runs on a cloud GPU. This mirrors
the local smoke (scripts/train.py) but uses Unsloth + Qwen3 on real data.

Colab setup:
  !pip install unsloth
  # upload data/train_chat.jsonl produced by scripts/generate.py

Then run this file.
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

MODEL = "unsloth/Qwen3-0.6B"  # or Qwen3-1.7B / 4B
MAX_SEQ = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

ds = load_dataset("json", data_files="data/train_chat.jsonl", split="train")
ds = ds.map(
    lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
    batched=True,
    remove_columns=ds.column_names,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="outputs/qwen3-geotikz",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=MAX_SEQ,
        dataset_text_field="text",
        bf16=True,
    ),
)
trainer.train()

model.save_pretrained("outputs/qwen3-geotikz")
tokenizer.save_pretrained("outputs/qwen3-geotikz")
# model.push_to_hub("<user>/qwen3-geotikz")  # for the submission package
