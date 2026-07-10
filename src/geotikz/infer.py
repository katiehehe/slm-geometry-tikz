"""Generate TikZ from scene descriptions.

Two backends:
  - local:   load a HF base model (+ optional LoRA adapter) and run on device.
             Use this to evaluate YOUR fine-tuned adapter.
  - gateway: call a hosted, OpenAI-compatible model (e.g. via a TrueFoundry
             gateway). Use this for big-model baselines to compare against.
"""

from __future__ import annotations

import os

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
    # Qwen3 (and other hybrid-reasoning models) default to "thinking" mode, which
    # emits a <think>...</think> block before the answer. That makes the output
    # non-figure-only (is_figure_only checks the string STARTS with the figure)
    # and can burn the whole token budget before any TikZ appears. Disable it so
    # the empty think block lives in the prompt and the model emits pure TikZ.
    # (Harmless no-op for tokenizers whose template ignores this kwarg.)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


def generate_via_gateway(
    description: str,
    model: str,
    max_new_tokens: int = 1024,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate TikZ from a hosted OpenAI-compatible model (gateway baseline).

    Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment (.env) unless
    passed explicitly. `model` is the provider-qualified id, e.g.
    "openai-group/gpt-4o".
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_new_tokens,
        messages=build_messages(description),
    )
    return resp.choices[0].message.content or ""
