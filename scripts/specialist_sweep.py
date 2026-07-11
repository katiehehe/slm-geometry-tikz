"""Difficulty sweep for the SPECIALIST (Qwen3-4B + `qwen3-illustrator-4b` LoRA).

Measures how hard a construction the fine-tuned specialist can reliably draw
before it fails — its *complexity ceiling* — on the SAME difficulty grid the
frontier sweep uses (chain length x number irregularity, plus op-targeted cells).
Reusing the identical grid makes specialist and frontier directly comparable on
the exact same examples.

This is the specialist counterpart to `scripts/difficulty_sweep.py`. Two things
had to change (both flagged as blockers in the task — `difficulty_sweep.py`
assumes a frontier gateway):

  1. INFERENCE. `difficulty_sweep.py` calls the hosted gateway (`gateway.chat`).
     Here we run the fine-tuned specialist on Modal GPU in ONE warm, batched A100
     container via the existing `scripts/infer_illustrator_4b_modal.py`
     local-entrypoint (model + adapter loaded once, all cells batched through it).
     The specialist is prompted with the CONSTRUCTION system prompt it was
     trained with (embedded verbatim in that Modal script).

  2. GRADING. `difficulty_sweep.py` grades with the STATIC parser
     (`metrics.coord_match`), which cannot recover coordinates from the
     coordinate-free tkz-euclide / `calc` constructions the specialist emits, so
     it would badly *undercount* the specialist. We instead use the
     compile-extract grader `extract.grade()` — the SAME falsifiable gate
     (figure-only AND compiles AND every named coord within `atol`) but it lets
     TeX place the points and reads the truth back out. This mirrors the
     established specialist eval (`scripts/eval_syn_illustrator.py`).

Writes `results.json` in the EXACT schema `scripts/sweep_report.py` consumes, so
the pass-rate heatmap / degradation / op-effect plots come straight from that
script, unchanged.

READ-ONLY: reuses the frontier grid (`outputs/sweep/grid.jsonl`) and does not
modify the app, training, data, or any adapter. It only runs specialist
inference on Modal and scores locally. Raw outputs are cached so re-scoring never
re-spends GPU.

Usage:
  # reuse the frontier grid (recommended: identical examples => comparable)
  uv run python scripts/specialist_sweep.py --grid outputs/sweep/grid.jsonl

  # score already-cached raw outputs only (no GPU)
  uv run python scripts/specialist_sweep.py --score-only

  # then render the heatmap + ceiling report
  uv run python scripts/sweep_report.py --dir outputs/specialist_sweep --threshold 0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import extract  # noqa: E402  (compile-extract, tkz-euclide-aware gate)

ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = "scripts/infer_illustrator_4b_modal.py"
_LANDSCAPE_RE = re.compile(r"c\d+_(rnd|irr)")


def _dhash(desc: str) -> str:
    return hashlib.md5(desc.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def load_grid(grid_path: Path) -> list[dict]:
    grid = [json.loads(l) for l in grid_path.read_text().splitlines() if l.strip()]
    for ex in grid:  # be tolerant of a grid built without the op-dial fields
        ex.setdefault("force_op", None)
        ex.setdefault("easy_only", False)
    return grid


def ordered_cells(grid: list[dict]) -> tuple[list[str], list[str]]:
    """(all_cells, landscape_cells): landscape sorted by (chain, irregular),
    op-targeted cells appended in (chain, name) order."""
    meta_by_cell: dict[str, tuple[int, bool]] = {}
    for ex in grid:
        meta_by_cell.setdefault(ex["cell"], (ex["chain"], ex["irregular"]))
    landscape = sorted((c for c in meta_by_cell if _LANDSCAPE_RE.fullmatch(c)),
                       key=lambda c: meta_by_cell[c])
    op = sorted((c for c in meta_by_cell if not _LANDSCAPE_RE.fullmatch(c)),
                key=lambda c: (meta_by_cell[c][0], c))
    return landscape + op, landscape


# --------------------------------------------------------------------------- #
# inference (specialist on Modal GPU, warm + batched)
# --------------------------------------------------------------------------- #
def run_modal(descs: list[str], adapter: str, max_new_tokens: int,
              batch_size: int, script: str) -> list[str]:
    """One warm, batched Modal container over ALL descriptions (model+adapter
    loaded once). Returns outputs aligned to `descs`."""
    tmp = Path(tempfile.mkdtemp(prefix="spec_sweep_"))
    inp, outp = tmp / "in.jsonl", tmp / "out.jsonl"
    inp.write_text("\n".join(json.dumps({"id": i, "description": d})
                             for i, d in enumerate(descs)) + "\n")
    cmd = ["modal", "run", script, "--input", str(inp), "--output", str(outp),
           "--max-new-tokens", str(max_new_tokens), "--batch-size", str(batch_size),
           "--adapter", adapter]
    print("  running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    by_i = {r["id"]: r["output"] for r in rows}
    return [by_i[i] for i in range(len(descs))]


def ensure_raw(grid: list[dict], cache_path: Path, adapter: str, max_new_tokens: int,
               batch_size: int, script: str, score_only: bool) -> dict[int, str]:
    """{id: raw_output}, calling Modal only for uncached examples.

    Cache is keyed by example id and validated against a hash of the description,
    so a stale entry from a different grid is never reused.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[int, str] = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            gex = next((e for e in grid if e["id"] == r["id"]), None)
            if gex is None or r.get("h") != _dhash(gex["description"]):
                continue
            cache[r["id"]] = r["output"]  # later lines win
    todo = [ex for ex in grid if ex["id"] not in cache]
    if todo and score_only:
        raise SystemExit(f"--score-only but {len(todo)} examples are uncached; "
                         f"drop --score-only to run the specialist on Modal first.")
    if todo:
        print(f"  specialist inference on Modal for {len(todo)} new prompts "
              f"(adapter={adapter}, batch={batch_size}, max_new_tokens={max_new_tokens}) ...",
              flush=True)
        outs = run_modal([ex["description"] for ex in todo], adapter,
                         max_new_tokens, batch_size, script)
        with cache_path.open("a") as f:
            for ex, o in zip(todo, outs):
                rec = {"id": ex["id"], "h": _dhash(ex["description"]), "output": o}
                f.write(json.dumps(rec) + "\n")
                cache[ex["id"]] = o
        print(f"  cached {len(todo)} raw outputs -> {cache_path}", flush=True)
    else:
        print(f"  all {len(grid)} raw outputs already cached ({cache_path})", flush=True)
    return cache


# --------------------------------------------------------------------------- #
# scoring (compile-extract gate) + aggregation
# --------------------------------------------------------------------------- #
def score_grid(grid: list[dict], raw: dict[int, str], atol: float,
               workers: int) -> list[dict]:
    def scorer(ex: dict) -> dict:
        out = raw.get(ex["id"], "") or ""
        gt = ex["points"]
        if ex.get("grade_only"):  # many-point figures grade a representative subset
            gt = {k: v for k, v in gt.items() if k in ex["grade_only"]}
        g = extract.grade(out, gt, atol=atol, unordered=ex.get("unordered"))
        return {"id": ex["id"], "cell": ex["cell"], "chain": ex["chain"],
                "irregular": ex["irregular"], "force_op": ex.get("force_op"),
                "easy_only": ex.get("easy_only", False), "latency_s": None,
                "api_fail": False,  # local inference: every scene gets an output
                "passed": bool(g["passed"]), "compiles": bool(g["compiles"]),
                "figure_only": bool(g["figure_only"]),
                "coord_accuracy": float(g["coord_accuracy"]),
                "coords_all_correct": bool(g["coords_all_correct"]), "ssim": None}

    detail: list[dict] = [None] * len(grid)  # type: ignore[list-item]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scorer, ex): i for i, ex in enumerate(grid)}
        for fut in as_completed(futs):
            detail[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(grid):
                print(f"    scored {done}/{len(grid)}", flush=True)
    return detail


def aggregate_cells(detail: list[dict]) -> dict:
    """Per-cell aggregate in the exact shape sweep_report.py expects."""
    by_cell: dict[str, list[dict]] = {}
    for d in detail:
        by_cell.setdefault(d["cell"], []).append(d)
    out: dict[str, dict] = {}
    for cell, rows in by_cell.items():
        live = [r for r in rows if not r["api_fail"]]
        n = len(live) or 1
        out[cell] = {
            "n": len(live),
            "n_api_fail": sum(r["api_fail"] for r in rows),
            "chain": rows[0]["chain"],
            "irregular": rows[0]["irregular"],
            "force_op": rows[0].get("force_op"),
            "easy_only": rows[0].get("easy_only", False),
            "pass_rate": sum(r["passed"] for r in live) / n,
            "compile_rate": sum(r["compiles"] for r in live) / n,
            "figure_only_rate": sum(r["figure_only"] for r in live) / n,
            "coord_accuracy_mean": sum(r["coord_accuracy"] for r in live) / n,
            "coords_all_correct_rate": sum(r["coords_all_correct"] for r in live) / n,
        }
    return out


def infer_grid_meta(grid: list[dict], landscape: list[str]) -> dict:
    """Best-effort {seed?, chains, k, op_dial} for the markdown report."""
    from collections import Counter
    per_cell = Counter(ex["cell"] for ex in grid)
    chains = sorted({ex["chain"] for ex in grid if _LANDSCAPE_RE.fullmatch(ex["cell"])})
    k = min((per_cell[c] for c in landscape), default=min(per_cell.values(), default=0))
    op_dial = any(not _LANDSCAPE_RE.fullmatch(c) for c in per_cell)
    return {"chains": chains, "k": k, "op_dial": op_dial}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="outputs/sweep/grid.jsonl",
                    help="reuse the frontier grid so specialist == frontier examples")
    ap.add_argument("--out", default="outputs/specialist_sweep")
    ap.add_argument("--adapter", default="qwen3-illustrator-4b",
                    help="LoRA RUN_NAME on the geotikz-outputs Volume")
    ap.add_argument("--model-label", default="qwen3-illustrator-4b",
                    help="row label in the heatmap / results.json")
    ap.add_argument("--infer-script", default=INFER_SCRIPT)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--atol", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8, help="concurrent compiles")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--score-only", action="store_true",
                    help="grade cached raw outputs; never call Modal")
    args = ap.parse_args()

    grid_path = ROOT / args.grid
    if not grid_path.exists():
        raise SystemExit(f"grid not found: {grid_path} (run difficulty_sweep to build one, "
                         f"or pass --grid)")
    grid = load_grid(grid_path)
    all_cells, landscape = ordered_cells(grid)
    print(f"grid: {len(grid)} examples, {len(all_cells)} cells "
          f"({len(landscape)} landscape + {len(all_cells) - len(landscape)} op-targeted)")

    out_dir = ROOT / args.out
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "detail").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw = ensure_raw(grid, out_dir / "raw" / f"{args.adapter}.jsonl", args.adapter,
                     args.max_new_tokens, args.batch_size, args.infer_script,
                     args.score_only)
    infer_s = time.time() - t0

    print("scoring with the compile-extract gate "
          "(figure-only AND compiles AND every coord within "
          f"{args.atol}) ...", flush=True)
    t1 = time.time()
    detail = score_grid(grid, raw, args.atol, args.workers)
    score_s = time.time() - t1

    (out_dir / "detail" / f"{args.model_label}.jsonl").write_text(
        "\n".join(json.dumps(d) for d in detail) + "\n")

    cellsagg = aggregate_cells(detail)
    n_live = sum(c["n"] for c in cellsagg.values())
    result = {
        "cells": cellsagg,
        "overall": {
            "n": n_live,
            "n_api_fail": 0,
            "pass_rate": sum(d["passed"] for d in detail) / (n_live or 1),
            "compile_rate": sum(d["compiles"] for d in detail) / (n_live or 1),
        },
        "avg_latency_s": round(infer_s / (len(grid) or 1), 3),
    }
    ov = result["overall"]
    print(f"\n  OVERALL pass={ov['pass_rate']:.3f} compile={ov['compile_rate']:.3f} "
          f"(n={n_live})  infer={infer_s:.0f}s score={score_s:.0f}s")

    grid_meta = infer_grid_meta(grid, landscape)
    meta = {
        "grid": grid_meta,
        "models": [args.model_label],
        "cells": all_cells,
        "landscape_cells": landscape,
        "op_dial": grid_meta["op_dial"],
        "threshold": args.threshold,
        "max_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "ssim": False,
        "grader": "extract.grade (compile-extract, tkz-euclide aware)",
        "inference": {"backend": "modal", "gpu": "A100", "adapter": args.adapter,
                      "base": "Qwen/Qwen3-4B", "batch_size": args.batch_size,
                      "grid_source": str(args.grid)},
    }
    results = {args.model_label: result}
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nwrote -> {out_dir / 'results.json'}")
    print(f"next: uv run python scripts/sweep_report.py --dir {args.out} "
          f"--threshold {args.threshold}")


if __name__ == "__main__":
    main()
