"""One decisive base-vs-tuned image on a single hard scene, from CACHED preds.

No live inference: we read the base and tuned model outputs already generated for
the held-out eval (outputs/eval_preds_*.jsonl), pick a foot-of-altitude scene the
BASE failed and the TUNED passed, compile all three, and stitch:

    [ GROUND TRUTH | BASE (no fine-tune) FAIL | TUNED (LoRA) PASS ]

Usage:
  uv run python scripts/render_before_after.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # pymupdf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from geotikz.metrics import coord_match, extract_tikz  # noqa: E402
from geotikz.tex import compile_tikz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = 440


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


F_TITLE = _font(20)
F_BODY = _font(17)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def passed_map(eval_json: Path) -> dict[int, bool]:
    d = json.loads(eval_json.read_text())
    return {r["id"]: r["passed"] for r in d["results"]}


def render_tikz(tikz: str) -> Image.Image | None:
    res = compile_tikz(tikz)
    try:
        if not res.ok or res.pdf_path is None:
            return None
        doc = fitz.open(res.pdf_path)
        pix = doc.load_page(0).get_pixmap(dpi=160)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return img
    finally:
        res.cleanup()


def is_blank(img: Image.Image) -> bool:
    """True if the figure is essentially empty (compiles but draws ~nothing)."""
    import numpy as np

    arr = np.asarray(img.convert("L"))
    return float((arr < 250).mean()) < 0.003


def fit(img: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL), "white")
    im = img.copy()
    im.thumbnail((PANEL - 28, PANEL - 28), Image.LANCZOS)
    canvas.paste(im, ((PANEL - im.width) // 2, (PANEL - im.height) // 2))
    return canvas


def placeholder(body: str) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL), "#fff5f5")
    d = ImageDraw.Draw(canvas)
    wrapped = "\n".join(textwrap.wrap(body, width=46)[:22]) or "(empty output)"
    d.text((16, 16), wrapped, fill="#7a2230", font=F_BODY)
    return canvas


def title_strip(text: str, color: str) -> Image.Image:
    strip = Image.new("RGB", (PANEL, 34), color)
    ImageDraw.Draw(strip).text((12, 8), text, fill="white", font=F_TITLE)
    return strip


def _worst_point(tikz: str, gt_points: dict) -> str:
    cm = coord_match(tikz, gt_points)
    worst_name, worst_err = None, -1.0
    for name, info in cm["per_point"].items():
        e = info["err"]
        if e is None:
            return f"point {name} missing"
        if e > worst_err:
            worst_name, worst_err = name, e
    return f"point {worst_name} off by {worst_err:.2f}"


def base_panel(raw: str, gt_points: dict) -> tuple[Image.Image, str, str]:
    """Return (panel, note, kind). kind in {drawn_wrong, blank, nocompile, nofig}."""
    tikz = extract_tikz(raw)
    if not tikz:
        return placeholder(raw or "(no TikZ emitted)"), "emitted no figure", "nofig"
    img = render_tikz(tikz)
    if img is None:
        return placeholder(tikz), "did not compile", "nocompile"
    if is_blank(img):
        return placeholder(tikz), "compiles but draws nothing", "blank"
    return fit(img), _worst_point(tikz, gt_points), "drawn_wrong"


def main() -> None:
    base_pass = passed_map(ROOT / "outputs/eval_base_new.json")
    tuned_pass = passed_map(ROOT / "outputs/eval_tuned.json")
    base_pred = {r["id"]: r for r in load_jsonl(ROOT / "outputs/eval_preds_base.jsonl")}
    tuned_pred = {r["id"]: r for r in load_jsonl(ROOT / "outputs/eval_preds_tuned.jsonl")}

    # Prefer an in-distribution, readable foot-of-altitude scene (chain 4, then 5)
    # whose BASE output is a *visibly wrong* drawn figure (kind=drawn_wrong).
    def score(rid: int) -> tuple:
        tags = tuned_pred[rid].get("tags", [])
        chain = tuned_pred[rid].get("chain", 0) or 0
        op_rank = 2 if "foot_altitude" in tags else (1 if "intersection" in tags else 0)
        # chain 4 first (short, in-distribution, easy to read), then 5.
        chain_rank = {4: 2, 5: 1}.get(chain, 0)
        n_ops = len(tags)
        return (op_rank, chain_rank, -n_ops)

    cands = [
        rid
        for rid in tuned_pred
        if rid in base_pred
        and tuned_pass.get(rid) is True
        and base_pass.get(rid) is False
        and (tuned_pred[rid].get("chain") in (4, 5))
    ]
    cands.sort(key=score, reverse=True)

    chosen = None
    fallback = None
    for rid in cands:
        gt = tuned_pred[rid]
        gt_img = render_tikz(gt["tikz"])
        tuned_img = render_tikz(extract_tikz(tuned_pred[rid]["output"]) or "")
        if gt_img is None or tuned_img is None:
            continue
        b_panel, b_note, kind = base_panel(base_pred[rid]["output"], gt["points"])
        pack = (rid, gt, fit(gt_img), b_panel, b_note, fit(tuned_img))
        if kind == "drawn_wrong":
            chosen = pack
            break
        if fallback is None:
            fallback = pack

    chosen = chosen or fallback
    if chosen is None:
        print("no suitable example found")
        return

    rid, gt, gt_panel, b_panel, b_note, tuned_panel = chosen
    tags = ", ".join(gt.get("tags", []))
    print(f"chose id={rid} chain={gt.get('chain')} tags=[{tags}] base_note={b_note}")

    # Header: scene description wrapped across the full width.
    W = PANEL * 3 + 16
    desc = gt["description"]
    lines = textwrap.wrap(desc, width=150)[:3]
    head_h = 40 + 24 * len(lines)
    header = Image.new("RGB", (W, head_h), "white")
    hd = ImageDraw.Draw(header)
    hd.text((12, 10), f"Scene (no coordinates given)  ·  chain {gt.get('chain')}  ·  ops: {tags}",
            fill="#111111", font=F_TITLE)
    for i, ln in enumerate(lines):
        hd.text((12, 40 + 24 * i), ln, fill="#333333", font=F_BODY)

    strips = Image.new("RGB", (W, 34), "white")
    strips.paste(title_strip("GROUND TRUTH", "#3b5bdb"), (0, 0))
    strips.paste(title_strip(f"BASE Qwen3-0.6B — FAIL ({b_note})", "#c92a2a"), (PANEL + 8, 0))
    strips.paste(title_strip("TUNED 0.6B + LoRA — PASS", "#2b8a3e"), (2 * PANEL + 16, 0))

    body = Image.new("RGB", (W, PANEL), "white")
    body.paste(gt_panel, (0, 0))
    body.paste(b_panel, (PANEL + 8, 0))
    body.paste(tuned_panel, (2 * PANEL + 16, 0))

    total = Image.new("RGB", (W, head_h + 34 + PANEL), "white")
    total.paste(header, (0, 0))
    total.paste(strips, (0, head_h))
    total.paste(body, (0, head_h + 34))

    out = ROOT / "outputs/renders/before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    total.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
