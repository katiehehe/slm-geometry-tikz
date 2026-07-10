"""Robust OpenAI-compatible chat calls for multi-model baselines.

A single gateway fronts many providers (OpenAI, Anthropic, Google, xAI, ...),
each with its own quirks. To measure them *fairly* we must not score a model as
"failing the task" when it actually just rejected one of our request params.
This module absorbs that:

  - token param:  some models want ``max_tokens``, others ``max_completion_tokens``
  - temperature:  some models only allow the default -> retry without it
  - truncation:   empty output with ``finish_reason == "length"`` -> raise the budget
  - transient:    429 / 5xx / timeouts -> exponential backoff with jitter

Plus a small bounded thread pool for fanning many prompts out concurrently.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_TRANSIENT = (
    "rate limit", "rate_limit", "429", "overloaded", "capacity",
    "timeout", "timed out", "temporarily", "502", "503", "504",
    "connection", "reset by peer", "econnreset", "server_error",
    "internal server error", "service unavailable",
)


def make_client(base_url: str | None = None, api_key: str | None = None):
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=180.0,
        max_retries=0,  # we do our own, smarter retries
    )


@dataclass
class ChatResult:
    ok: bool
    text: str
    model: str
    finish_reason: str | None = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0
    params: dict = field(default_factory=dict)


def chat(
    messages: list[dict],
    model: str,
    *,
    client=None,
    max_tokens: int = 2048,
    temperature: float | None = 0.0,
    retries: int = 6,
) -> ChatResult:
    """One chat completion, adapting to per-model param quirks. Never raises."""
    client = client or make_client()
    token_param = "max_tokens"
    use_temperature = temperature is not None
    budget = max_tokens
    last_err = "unknown"
    t0 = time.time()

    for attempt in range(1, retries + 1):
        kwargs: dict = {"model": model, "messages": messages, token_param: budget}
        if use_temperature:
            kwargs["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish = getattr(choice, "finish_reason", None)
            # Truncated (reasoning models burn the budget before finishing the
            # figure) -> give it more room. A complete answer finishes with
            # "stop", so "length" means the figure is almost certainly cut off.
            if finish == "length" and budget < 24576:
                budget = min(budget * 2, 24576)
                continue
            return ChatResult(
                ok=bool(text.strip()), text=text, model=model,
                finish_reason=finish, attempts=attempt,
                latency_s=round(time.time() - t0, 2),
                params={"token_param": token_param, "temperature": use_temperature and temperature},
            )
        except Exception as e:  # noqa: BLE001 - classify then adapt/backoff
            msg = str(e).lower()
            last_err = f"{type(e).__name__}: {e}"

            # --- param adaptations (don't consume the backoff budget) ---
            if token_param == "max_tokens" and "max_completion_tokens" in msg:
                token_param = "max_completion_tokens"
                continue
            if token_param == "max_completion_tokens" and (
                "unexpected" in msg or "unknown" in msg
            ) and "max_tokens" in msg:
                token_param = "max_tokens"
                continue
            if use_temperature and "temperature" in msg:
                # temperature is never essential for us; drop it and use the
                # provider default (covers "unsupported", "deprecated", "must be
                # the default", range errors, etc.)
                use_temperature = False
                continue

            # --- transient: back off and retry ---
            if any(s in msg for s in _TRANSIENT):
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue

            # --- unknown/permanent: one gentle retry, then give up ---
            if attempt < 2:
                time.sleep(1 + random.random())
                continue
            break

    return ChatResult(
        ok=False, text="", model=model, error=last_err, attempts=retries,
        latency_s=round(time.time() - t0, 2),
    )


def map_concurrent(
    fn: Callable[[T], R], items: Iterable[T], *, workers: int = 6
) -> list[R]:
    """Run ``fn`` over ``items`` with a bounded pool, preserving input order."""
    items = list(items)
    out: list[R | None] = [None] * len(items)
    if not items:
        return []  # type: ignore[return-value]
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut_to_i = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in cf.as_completed(fut_to_i):
            out[fut_to_i[fut]] = fut.result()
    return out  # type: ignore[return-value]
