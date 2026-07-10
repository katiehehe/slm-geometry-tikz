"""Render a few ground-truth scenes to PNG so the figures can be viewed.

Unlike the eval harness (which renders to an in-memory array only for scoring),
this saves actual PNGs to outputs/renders/ for inspection.

Usage:
  uv run python scripts/render_examples.py --n 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # pymupdf  # noqa: E402

from geotikz.generator import make_example  # noqa: E402
from geotikz.tex import compile_tikz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=str, default="outputs/renders")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    saved = 0
    for i in range(args.n):
        chain = rng.randint(1, 3)
        irregular = rng.random() < 0.5
        ex = make_example(rng, chain, irregular)

        res = compile_tikz(ex["tikz"])
        if not res.ok or res.pdf_path is None:
            print(f"[{i}] compile failed: {res.reason}")
            continue

        png_path = out_dir / f"scene_{i:02d}_chain{ex['chain']}_{'irr' if irregular else 'round'}.png"
        doc = fitz.open(res.pdf_path)
        page = doc.load_page(0)
        page.get_pixmap(dpi=args.dpi).save(str(png_path))
        doc.close()

        saved += 1
        print(f"[{i}] chain={ex['chain']} irregular={irregular} -> {png_path.name}")
        print(f"     spec: {ex['description']}")
        print(f"     points: {ex['points']}")

    print(f"\nsaved {saved} PNG(s) -> {out_dir}")


if __name__ == "__main__":
    main()
