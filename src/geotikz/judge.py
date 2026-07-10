"""LLM-as-judge against the Behavior Spec rubric.

Optional: only runs if an API key is set. Otherwise returns {"skipped": True}
so the loop still runs end to end without a key.

Works with plain OpenAI or any OpenAI-compatible gateway (e.g. a TrueFoundry
AI Gateway). Configure via environment:
  OPENAI_API_KEY   - your key/token (an OpenAI key, or a TrueFoundry PAT/VAT)
  OPENAI_BASE_URL  - optional; the gateway endpoint. Unset -> OpenAI directly.
  JUDGE_MODEL      - optional; model id. For a gateway use the provider-qualified
                     id (e.g. "openai-main/gpt-4o-mini"). Defaults to gpt-4o-mini.
"""

from __future__ import annotations

import json
import os

RUBRIC = """You are grading a geometry-to-TikZ model against a Behavior Spec.
Spec: given a coordinate-free geometry scene, the model must output ONLY a single
valid TikZ figure whose geometry matches the described construction.

Score each dimension 0, 1, or 2:
- spec_adherence: 0=states coords/prose/wrong task, 1=compiles but geometry off, 2=valid TikZ, geometry correct, figure only
- robustness: 0=breaks on multi-step/non-round, 1=wobbles, 2=holds
- task_quality: 0=doesn't compile/garbage, 1=roughly right, 2=clean correct figure
- consistency: 0=erratic, 1=mostly stable, 2=reliable

Return strict JSON: {"spec_adherence":int,"robustness":int,"task_quality":int,"consistency":int,"note":str}
"""


def judge(description: str, output: str, model: str | None = None) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"skipped": True, "reason": "no OPENAI_API_KEY"}
    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> OpenAI's default endpoint
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": RUBRIC},
                {
                    "role": "user",
                    "content": f"SCENE:\n{description}\n\nMODEL OUTPUT:\n{output}",
                },
            ],
        )
        content = resp.choices[0].message.content or "{}"
        content = content[content.find("{") : content.rfind("}") + 1]
        return json.loads(content)
    except Exception as e:  # noqa: BLE001 - judge is best-effort
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}
