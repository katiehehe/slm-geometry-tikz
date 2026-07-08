"""Load a base or fine-tuned model and generate TikZ from scene descriptions."""

from __future__ import annotations

import torch

from .prompts import build_messages


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, adapter_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, device


@torch.no_grad()
def generate(model, tok, device: str, description: str, max_new_tokens: int = 384) -> str:
    messages = build_messages(description)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)
