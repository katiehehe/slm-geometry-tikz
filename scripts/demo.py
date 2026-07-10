"""Interactive demo: geometry description -> specialist -> rendered figure + TikZ.

The specialist is the fine-tuned Qwen3-0.6B + `qwen3-pgf-geotikz` LoRA, run
LOCALLY (base + adapter) with the exact prompt it was trained with. It emits
coordinate-free PGF constructions, which are compiled with tectonic and
rasterised to a PNG under outputs/demo/.

Usage:
  # description as an argument
  uv run python scripts/demo.py "A circle centered at the origin with radius 3. \
Point A on the circle at 40 degrees. Point B diametrically opposite A."

  # or piped on stdin
  echo "Triangle with ... " | uv run python scripts/demo.py

  # also try a frontier model (construction prompt) for comparison / fallback
  uv run python scripts/demo.py "..." --frontier openai-group/gpt-5.5 --fallback
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    return "-".join(w for w in keep.split("-") if w)[:48] or "scene"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("description", nargs="*", help="geometry scene (else read stdin)")
    ap.add_argument("--adapter", default=serve.DEFAULT_ADAPTER)
    ap.add_argument("--base", default=serve.DEFAULT_BASE)
    ap.add_argument("--out-dir", default="outputs/demo")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--frontier", default=None,
                    help="also generate with this gateway model (construction prompt)")
    ap.add_argument("--fallback", action="store_true",
                    help="if the specialist figure is blank/uncompilable, use the frontier one")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    description = " ".join(args.description).strip() or sys.stdin.read().strip()
    if not description:
        ap.error("no description given (pass as args or on stdin)")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(description)

    print("=" * 72)
    print("SCENE:", description)
    print("=" * 72)

    # --- specialist (local) ---
    print("\n[specialist] loading base + adapter (first call downloads the base) ...")
    spec = serve.Specialist(base=args.base, adapter=args.adapter).load()
    t0 = time.time()
    tikz_out = spec.generate(description, max_new_tokens=args.max_new_tokens)
    gen_s = time.time() - t0
    png = out_dir / f"{stem}.specialist.png"
    r = serve.compile_and_render(tikz_out, png, dpi=args.dpi)
    figure = serve.metrics.extract_tikz(tikz_out) or tikz_out

    print(f"[specialist] generated in {gen_s:.1f}s  "
          f"compiles={r.compiles} degenerate={r.degenerate} "
          f"({'PNG -> ' + str(png) if r.ok else 'no usable figure: ' + r.reason})")
    print("\n--- SPECIALIST TikZ (coordinate-free construction) ---\n")
    print(figure)

    winner_png = png if r.ok else None

    # --- optional frontier (construction prompt) ---
    if args.frontier:
        print(f"\n[frontier:{args.frontier}] generating (construction prompt) ...")
        fres = serve.frontier_generate(description, args.frontier, construction=True)
        fpng = out_dir / f"{stem}.frontier.png"
        fr = serve.compile_and_render(fres.text, fpng, dpi=args.dpi)
        ffig = serve.metrics.extract_tikz(fres.text) or fres.text
        print(f"[frontier] {fres.latency_s:.1f}s  ok={fres.ok}  "
              f"compiles={fr.compiles} degenerate={fr.degenerate} "
              f"({'PNG -> ' + str(fpng) if fr.ok else 'no usable figure: ' + fr.reason})")
        print("\n--- FRONTIER TikZ (coordinate-free construction) ---\n")
        print(ffig)
        if args.fallback and not r.ok and fr.ok:
            winner_png = fpng

    print("\n" + "=" * 72)
    if winner_png:
        print(f"RENDERED FIGURE: {winner_png}")
    else:
        print("No usable figure produced. (Real AIME-style scenes are often "
              "out-of-distribution for the narrow specialist -- try --frontier.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
