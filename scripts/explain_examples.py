"""Visual explainer: INPUT (coord-free spec) -> GROUND TRUTH (tikz+render) -> MODEL OUTPUT.

Stitches a 3-column grid so the whole task is legible at a glance:

  column 1  the model INPUT: a coordinate-free constraint description
  column 2  the GROUND TRUTH: TikZ computed from the known exact coords, rendered
  column 3  the MODEL OUTPUT: TikZ a model emits from the input alone, rendered
            (+ a PASS/FAIL badge from the same grader the eval harness uses)

Backends for column 3:
  --backend gateway   hosted OpenAI-compatible model (needs .env)   [default]
  --backend local     a local HF base model + optional LoRA adapter
  --backend none      skip the model; show only INPUT + GROUND TRUTH

Usage:
  uv run python scripts/explain_examples.py --n 6
  uv run python scripts/explain_examples.py --n 6 --backend local --adapter outputs/smoke-adapter
  uv run python scripts/explain_examples.py --n 6 --backend none
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # pymupdf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from geotikz import infer  # noqa: E402
from geotikz.metrics import coord_match, extract_tikz, is_figure_only  # noqa: E402
from geotikz.tex import compile_tikz  # noqa: E402

INPUT_W = 500
IMG_W = 360
HEADER_H = 30
ROW_H = 400
PAD = 12

INK = "#111111"
MUTED = "#555555"
GOOD = "#0a7d2c"
BAD = "#b00020"

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
_MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    for path in _MONO_CANDIDATES if mono else _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_px(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    """Word-wrap `text` so each line fits within max_w pixels."""
    lines: list[str] = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if draw.textlength(trial, font=fnt) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def draw_block(draw, xy, lines, fnt, fill, line_h) -> int:
    x, y = xy
    for ln in lines:
        draw.text((x, y), ln, font=fnt, fill=fill)
        y += line_h
    return y


def render_tikz(tikz: str) -> Image.Image | None:
    res = compile_tikz(tikz)
    if not res.ok or res.pdf_path is None:
        return None
    doc = fitz.open(res.pdf_path)
    pix = doc.load_page(0).get_pixmap(dpi=150)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def image_panel(img: Image.Image | None, box_w: int, box_h: int,
                fail_title: str = "", fail_body: str = "") -> Image.Image:
    canvas = Image.new("RGB", (box_w, box_h), "white")
    if img is not None:
        thumb = img.copy()
        thumb.thumbnail((box_w - 2 * PAD, box_h - 2 * PAD), Image.LANCZOS)
        canvas.paste(thumb, ((box_w - thumb.width) // 2, (box_h - thumb.height) // 2))
        return canvas
    d = ImageDraw.Draw(canvas)
    d.rectangle([4, 4, box_w - 5, box_h - 5], outline="#f0c0c0", width=1)
    d.text((PAD, PAD), fail_title, font=font(14), fill=BAD)
    body = "\n".join(textwrap.wrap(fail_body, width=44)[:14])
    d.text((PAD, PAD + 24), body, font=font(11, mono=True), fill=MUTED)
    return canvas


def header(text: str, w: int, bg: str) -> Image.Image:
    strip = Image.new("RGB", (w, HEADER_H), bg)
    ImageDraw.Draw(strip).text((PAD, 7), text, font=font(15), fill="white")
    return strip


def input_panel(ex: dict) -> Image.Image:
    panel = Image.new("RGB", (INPUT_W, ROW_H), "white")
    panel.paste(header("INPUT  ·  coordinate-free spec", INPUT_W, "#334155"), (0, 0))
    d = ImageDraw.Draw(panel)
    y = HEADER_H + PAD
    body = font(15)
    for i, c in enumerate(ex["constraints"], 1):
        lines = wrap_px(d, f"{i}. {c}", body, INPUT_W - 2 * PAD)
        y = draw_block(d, (PAD, y), lines, body, INK, 20) + 6
    meta = (f"chain={ex['chain']}   "
            f"{'irregular numbers' if ex['irregular'] else 'round numbers'}\n"
            f"steps: {', '.join(ex['tags'])}")
    y = max(y, ROW_H - 62)
    draw_block(d, (PAD, y), wrap_px(d, meta, font(12), INPUT_W - 2 * PAD),
               font(12), MUTED, 16)
    return panel


def gt_panel(ex: dict) -> Image.Image:
    panel = Image.new("RGB", (IMG_W, ROW_H), "white")
    panel.paste(header("GROUND TRUTH  ·  exact coords", IMG_W, "#0a7d2c"), (0, 0))
    img = render_tikz(ex["tikz"])
    body_h = ROW_H - HEADER_H - 44
    panel.paste(image_panel(img, IMG_W, body_h, "GT compile failed", ex["tikz"]),
                (0, HEADER_H))
    d = ImageDraw.Draw(panel)
    pts = "  ".join(f"{k}=({v[0]:g},{v[1]:g})" for k, v in ex["points"].items())
    d.text((PAD, ROW_H - 38), "points the model must recover:", font=font(11), fill=MUTED)
    d.text((PAD, ROW_H - 22), pts[:70], font=font(11, mono=True), fill=INK)
    return panel


def out_panel(ex: dict, raw: str | None) -> Image.Image:
    panel = Image.new("RGB", (IMG_W, ROW_H), "white")
    body_h = ROW_H - HEADER_H - 44
    if raw is None:
        panel.paste(header("MODEL OUTPUT  ·  (skipped)", IMG_W, "#64748b"), (0, 0))
        panel.paste(image_panel(None, IMG_W, body_h, "no backend",
                    "run with --backend gateway|local"), (0, HEADER_H))
        return panel

    pred = extract_tikz(raw)
    img = render_tikz(pred) if pred else None
    cm = coord_match(pred or "", ex["points"])
    figonly = is_figure_only(raw)
    passed = bool(figonly and img is not None and cm["all_correct"])

    badge = "PASS" if passed else "FAIL"
    bg = GOOD if passed else BAD
    panel.paste(header(f"MODEL OUTPUT  ·  {badge}", IMG_W, bg), (0, 0))
    fail_title = "no TikZ emitted" if not pred else "did not compile"
    panel.paste(image_panel(img, IMG_W, body_h, fail_title, pred or (raw or "(empty)")),
                (0, HEADER_H))
    d = ImageDraw.Draw(panel)
    acc = f"coords {cm['matched']}/{cm['total']} correct   figure-only={figonly}"
    d.text((PAD, ROW_H - 38), acc, font=font(11), fill=MUTED)
    if pred and cm["total"]:
        errs = "  ".join(
            f"{k}:{'ok' if v['ok'] else (('%.2f' % v['err']) if v['err'] is not None else 'miss')}"
            for k, v in cm["per_point"].items()
        )
        d.text((PAD, ROW_H - 22), errs[:70], font=font(11, mono=True), fill=INK)
    return panel


def pick_examples(rows: list[dict], n: int) -> list[dict]:
    """A spread across chain lengths, easiest first."""
    by_chain: dict[int, list[dict]] = {}
    for r in rows:
        by_chain.setdefault(r["chain"], []).append(r)
    chosen: list[dict] = []
    chains = sorted(by_chain)
    i = 0
    while len(chosen) < min(n, len(rows)):
        c = chains[i % len(chains)]
        if by_chain[c]:
            chosen.append(by_chain[c].pop(0))
        i += 1
        if all(not v for v in by_chain.values()):
            break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/eval.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--backend", choices=["gateway", "local", "none"], default="gateway")
    ap.add_argument("--model", type=str, default="openai-group/gpt-4o")
    ap.add_argument("--adapter", type=str, default="outputs/smoke-adapter")
    ap.add_argument("--out", type=str, default="outputs/renders/explainer.png")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(l) for l in (root / args.data).read_text().splitlines() if l.strip()]
    examples = pick_examples(rows, args.n)

    model = tok = device = None
    if args.backend == "local":
        print(f"loading local {args.model} + adapter {args.adapter} ...")
        model, tok, device = infer.load_model(args.model, args.adapter)

    def model_output(desc: str) -> str | None:
        if args.backend == "none":
            return None
        try:
            if args.backend == "gateway":
                return infer.generate_via_gateway(desc, args.model)
            return infer.generate(model, tok, device, desc)
        except Exception as e:  # noqa: BLE001 - explainer is best-effort
            return f"(backend error: {e})"

    banner_h = 54
    rows_imgs = []
    dump = []
    for k, ex in enumerate(examples):
        raw = model_output(ex["description"])
        row = Image.new("RGB", (INPUT_W + 2 * IMG_W, ROW_H), "white")
        row.paste(input_panel(ex), (0, 0))
        row.paste(gt_panel(ex), (INPUT_W, 0))
        row.paste(out_panel(ex, raw), (INPUT_W + IMG_W, 0))
        rows_imgs.append(row)
        pred = extract_tikz(raw) if raw else None
        cm = coord_match(pred or "", ex["points"])
        dump.append({
            "id": ex["id"], "chain": ex["chain"], "irregular": ex["irregular"],
            "input_description": ex["description"],
            "ground_truth_points": ex["points"],
            "ground_truth_tikz": ex["tikz"],
            "model_raw_output": raw,
            "model_tikz": pred,
            "coord_match": {"matched": cm["matched"], "total": cm["total"],
                            "per_point": cm["per_point"]},
        })
        print(f"[{k}] id={ex['id']} chain={ex['chain']} "
              f"{'irr' if ex['irregular'] else 'round'} rendered")

    total_w = INPUT_W + 2 * IMG_W
    total_h = banner_h + sum(r.height + 8 for r in rows_imgs)
    grid = Image.new("RGB", (total_w, total_h), "white")
    d = ImageDraw.Draw(grid)
    d.text((PAD, 10), "Spec-first geometry -> TikZ:  what the model sees, "
           "what's correct, and what it produces",
           font=font(17), fill=INK)
    d.text((PAD, 32), f"backend={args.backend}  model={args.model if args.backend!='none' else '-'}"
           "   ·   grader: figure-only AND compiles AND all coords within 0.05",
           font=font(12), fill=MUTED)
    y = banner_h
    for r in rows_imgs:
        grid.paste(r, (0, y))
        y += r.height + 8

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"\nsaved explainer -> {out_path}")

    dump_path = out_path.with_suffix(".json")
    dump_path.write_text(json.dumps(dump, indent=2))
    print(f"saved raw text  -> {dump_path}")


if __name__ == "__main__":
    main()
