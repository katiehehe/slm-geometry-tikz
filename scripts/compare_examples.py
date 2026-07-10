"""Side-by-side: ground-truth figure vs the tuned smoke model's actual output.

For each spec we render the ground truth, run the tuned model, and render its
output when it compiles (otherwise we draw a placeholder panel showing why it
failed and a snippet of what it produced). Panels are stitched into one PNG.

Usage:
  uv run python scripts/compare_examples.py --n 4 --adapter outputs/smoke-adapter
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from geotikz.generator import make_example  # noqa: E402
from geotikz.infer import generate, load_model  # noqa: E402
from geotikz.metrics import extract_tikz  # noqa: E402
from geotikz.tex import compile_tikz  # noqa: E402

PANEL = 360


def render_tikz_to_img(tikz: str) -> Image.Image | None:
    res = compile_tikz(tikz)
    if not res.ok or res.pdf_path is None:
        return None
    doc = fitz.open(res.pdf_path)
    pix = doc.load_page(0).get_pixmap(dpi=150)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def fit(img: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL), "white")
    img.thumbnail((PANEL - 20, PANEL - 20), Image.LANCZOS)
    canvas.paste(img, ((PANEL - img.width) // 2, (PANEL - img.height) // 2))
    return canvas


def placeholder(title: str, body: str) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL), "#fff5f5")
    d = ImageDraw.Draw(canvas)
    d.text((12, 12), title, fill="#b00020")
    wrapped = "\n".join(textwrap.wrap(body, width=44)[:16])
    d.text((12, 44), wrapped, fill="#333333")
    return canvas


def label(text: str, w: int) -> Image.Image:
    strip = Image.new("RGB", (w, 26), "#f0f0f0")
    ImageDraw.Draw(strip).text((8, 6), text, fill="#000000")
    return strip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", type=str, default="outputs/smoke-adapter")
    ap.add_argument("--out", type=str, default="outputs/renders/compare.png")
    args = ap.parse_args()

    print(f"loading {args.model} + adapter {args.adapter} ...")
    model, tok, device = load_model(args.model, args.adapter)
    print(f"device={device}")

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        chain = rng.randint(1, 2)
        irregular = rng.random() < 0.5
        ex = make_example(rng, chain, irregular)

        gt_img = render_tikz_to_img(ex["tikz"])
        gt_panel = fit(gt_img) if gt_img else placeholder("GT failed", ex["tikz"])

        raw = generate(model, tok, device, ex["description"])
        pred_tikz = extract_tikz(raw)
        if pred_tikz:
            pred_img = render_tikz_to_img(pred_tikz)
            pred_panel = fit(pred_img) if pred_img else placeholder(
                "model: no compile", pred_tikz
            )
        else:
            pred_panel = placeholder("model: no TikZ emitted", raw or "(empty output)")

        print(f"[{i}] chain={ex['chain']} irr={irregular} "
              f"emitted_tikz={pred_tikz is not None}")

        row = Image.new("RGB", (PANEL * 2, PANEL + 26), "white")
        row.paste(label(f"GROUND TRUTH  (chain={ex['chain']}, "
                        f"{'irregular' if irregular else 'round'})", PANEL), (0, 0))
        row.paste(label("TUNED SMOKE MODEL OUTPUT", PANEL), (PANEL, 0))
        row.paste(gt_panel, (0, 26))
        row.paste(pred_panel, (PANEL, 26))
        rows.append(row)

    total_h = sum(r.height for r in rows) + 8 * (len(rows) - 1)
    grid = Image.new("RGB", (PANEL * 2, total_h), "white")
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 8

    out_path = Path(__file__).resolve().parents[1] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"\nsaved comparison -> {out_path}")


if __name__ == "__main__":
    main()
