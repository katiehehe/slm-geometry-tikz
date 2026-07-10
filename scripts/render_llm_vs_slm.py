"""A scene where a hosted FRONTIER LLM fails but the tiny tuned SLM passes.

Both were scored on the identical 800-item grid with the identical pass gate, so
this is apples-to-apples. We read cached outputs (no inference), find a scene the
SLM passed and a frontier model failed, and stitch:

    [ GROUND TRUTH | <FRONTIER LLM> FAIL | TUNED SLM (0.6B) PASS ]

Usage:
  uv run python scripts/render_llm_vs_slm.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling helpers

from PIL import Image, ImageDraw  # noqa: E402

from render_before_after import (  # noqa: E402
    F_BODY,
    F_TITLE,
    PANEL,
    base_panel,
    extract_tikz,
    fit,
    render_tikz,
)

ROOT = Path(__file__).resolve().parents[1]

# Try these frontier models in order; use the first that gives a clear, visible
# failure on a scene the SLM passes. Recognizable + capable models first.
MODEL_PREF = [
    "openai-group/gpt-4o",
    "openai-group/gpt-4.1",
    "openai-group/gpt-5.4",
    "claude-group/claude-opus-4-8",
    "gemini-group/gemini-3.5-flash",
    "deepseek-v3.2",
    "claude-group/claude-haiku-4-5",
]
PRETTY = {
    "openai-group/gpt-4o": "GPT-4o",
    "openai-group/gpt-4.1": "GPT-4.1",
    "openai-group/gpt-5.4": "GPT-5.4",
    "claude-group/claude-opus-4-8": "Claude-Opus-4-8",
    "gemini-group/gemini-3.5-flash": "Gemini-3.5-Flash",
    "deepseek-v3.2": "DeepSeek-v3.2",
    "claude-group/claude-haiku-4-5": "Claude-Haiku-4-5",
}
VISIBLE_ERR = 0.4  # a coord error this big is clearly visible in the render


def load_jsonl(p: Path) -> dict[int, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def passed_map(eval_json: Path) -> dict[int, bool]:
    d = json.loads(eval_json.read_text())
    return {r["id"]: r["passed"] for r in d["results"]}


def _err_from_note(note: str) -> float:
    if "off by" in note:
        try:
            return float(note.rsplit("off by", 1)[1])
        except ValueError:
            return 0.0
    return 0.0


def main() -> None:
    slm_pass = passed_map(ROOT / "outputs/eval_tuned.json")           # tuned Qwen3-0.6B
    slm_pred = load_jsonl(ROOT / "outputs/eval_preds_tuned.jsonl")    # its outputs + GT

    chosen = None
    for model in MODEL_PREF:
        stem = model.replace("/", "__")
        detail_p = ROOT / f"outputs/sweep/detail/{stem}.jsonl"
        raw_p = ROOT / f"outputs/sweep/raw/{stem}.jsonl"
        if not (detail_p.exists() and raw_p.exists()):
            continue
        fr_pass = {r["id"]: r["passed"] for r in
                   (json.loads(l) for l in detail_p.read_text().splitlines() if l.strip())}
        fr_raw = load_jsonl(raw_p)

        def rank(rid: int) -> tuple:
            tags = slm_pred[rid].get("tags", [])
            chain = slm_pred[rid].get("chain", 0) or 0
            op = 2 if "foot_altitude" in tags else (1 if "intersection" in tags else 0)
            return (op, {4: 2, 5: 1, 3: 1}.get(chain, 0), -len(tags))

        cands = [
            rid for rid in slm_pred
            if slm_pass.get(rid) is True
            and fr_pass.get(rid) is False
            and rid in fr_raw
            and (slm_pred[rid].get("chain") in (3, 4, 5))
        ]
        cands.sort(key=rank, reverse=True)

        best = None  # (err, pack)
        for rid in cands[:18]:  # bound tectonic compiles
            gt = slm_pred[rid]
            gt_img = render_tikz(gt["tikz"])
            slm_img = render_tikz(extract_tikz(gt["output"]) or "")
            if gt_img is None or slm_img is None:
                continue
            fr_panel, fr_note, kind = base_panel(fr_raw[rid]["raw"], gt["points"])
            if kind != "drawn_wrong":
                continue
            err = _err_from_note(fr_note)
            pack = (model, rid, gt, fit(gt_img), fr_panel, fr_note, fit(slm_img))
            if err >= VISIBLE_ERR:
                best = (err, pack)
                break
            if best is None or err > best[0]:
                best = (err, pack)
        if best is not None:
            chosen = best[1]
            break

    if chosen is None:
        print("no suitable LLM-fail / SLM-pass example found")
        return

    model, rid, gt, gt_panel, fr_panel, fr_note, slm_panel = chosen
    name = PRETTY.get(model, model)
    print(f"chose model={model} id={rid} chain={gt.get('chain')} "
          f"tags={gt.get('tags')} frontier_note={fr_note}")

    W = PANEL * 3 + 16
    lines = textwrap.wrap(gt["description"], width=150)[:3]
    head_h = 40 + 24 * len(lines)
    header = Image.new("RGB", (W, head_h), "white")
    hd = ImageDraw.Draw(header)
    hd.text((12, 10),
            f"Scene (no coordinates given)  ·  chain {gt.get('chain')}  ·  "
            f"ops: {', '.join(gt.get('tags', []))}",
            fill="#111111", font=F_TITLE)
    for i, ln in enumerate(lines):
        hd.text((12, 40 + 24 * i), ln, fill="#333333", font=F_BODY)

    def strip(text: str, color: str, x: int, dst: Image.Image) -> None:
        s = Image.new("RGB", (PANEL, 34), color)
        ImageDraw.Draw(s).text((10, 9), text, fill="white", font=F_BODY)
        dst.paste(s, (x, 0))

    short = fr_note.replace("point ", "")  # e.g. "D off by 1.37"
    strips = Image.new("RGB", (W, 34), "white")
    strip("GROUND TRUTH (correct)", "#3b5bdb", 0, strips)
    strip(f"{name} (frontier LLM) — FAIL: {short}", "#c92a2a", PANEL + 8, strips)
    strip("TUNED SLM 0.6B (local) — PASS", "#2b8a3e", 2 * PANEL + 16, strips)

    body = Image.new("RGB", (W, PANEL), "white")
    body.paste(gt_panel, (0, 0))
    body.paste(fr_panel, (PANEL + 8, 0))
    body.paste(slm_panel, (2 * PANEL + 16, 0))

    total = Image.new("RGB", (W, head_h + 34 + PANEL), "white")
    total.paste(header, (0, 0))
    total.paste(strips, (0, head_h))
    total.paste(body, (0, head_h + 34))

    out = ROOT / "outputs/renders/llm_vs_slm.png"
    total.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
