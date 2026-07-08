"""Prompt templates. The system prompt encodes the Behavior Spec."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only "
    "through relationships and constraints (no explicit coordinates), you must "
    "derive the exact coordinates yourself and output a single valid TikZ/PGF "
    "figure that compiles and renders the described geometry. "
    "Output ONLY the TikZ code, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. No prose, no explanations, no markdown fences."
)

USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."


def build_messages(description: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]


def to_chat_example(ex: dict) -> dict:
    """Format one generated example as a chat SFT record (prompt + completion)."""
    return {
        "messages": build_messages(ex["description"])
        + [{"role": "assistant", "content": ex["tikz"]}]
    }
