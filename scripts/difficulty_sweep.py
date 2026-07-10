"""Difficulty sweep: measure how SOTA LLMs degrade as the task gets harder.

Builds a difficulty grid (chain length x number irregularity), runs a set of
hosted models over it via the gateway, and scores each output with the same
falsifiable gate the project trains toward (figure-only AND compiles AND every
named coordinate correct within tolerance).

The point: find the *least complex* level at which frontier models stop being
reliable. That level is the right place to train the small specialist — below
it, prompting already wins; at/above it, a dataset earns its keep.

Raw model outputs are cached per (model, example) so re-scoring or adding models
never re-spends API budget.

Usage:
  # pilot: a few strong models, extended chains, small K
  uv run python scripts/difficulty_sweep.py --preset frontier_small \
      --chains 1 2 3 4 5 6 --k 12

  # explicit models
  uv run python scripts/difficulty_sweep.py \
      --models openai-group/gpt-5.5 claude-group/claude-opus-4-8 --k 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import gateway, harness  # noqa: E402
from geotikz.generator import make_example  # noqa: E402
from geotikz.prompts import build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Curated model sets. Ids are gateway (provider-qualified) ids; override with
# --models. Keep these to genuine chat/reasoning models (no embeddings/image).
PRESETS: dict[str, list[str]] = {
    "frontier": [
        "openai-group/gpt-5.5",
        "openai-group/gpt-5.4",
        "claude-group/claude-opus-4-8",
        "claude-group/claude-sonnet-5",
        "gemini-group/gemini-3.1-pro",
        "xai-group/grok-4.5",
        "deepseek-v3.2",
    ],
    "frontier_small": [
        "openai-group/gpt-5.5",
        "claude-group/claude-opus-4-8",
        "gemini-group/gemini-3.1-pro",
    ],
    "tiers": [  # frontier + mid + small, to show the whole gradient
        "openai-group/gpt-5.5",
        "claude-group/claude-opus-4-8",
        "gemini-group/gemini-3.1-pro",
        "xai-group/grok-4.5",
        "openai-group/gpt-5-mini",
        "gemini-group/gemini-3.5-flash",
        "claude-group/claude-haiku-4-5",
        "openai-group/gpt-4o",
    ],
}


def cell_name(chain: int, irregular: bool) -> str:
    return f"c{chain}_{'irr' if irregular else 'rnd'}"


def load_blocklist(paths: list[str]) -> set[str]:
    block: set[str] = set()
    for p in paths:
        fp = ROOT / p
        if fp.exists():
            block |= {
                json.loads(l)["description"]
                for l in fp.read_text().splitlines() if l.strip()
            }
    return block


def landscape_specs(chains: list[int]) -> list[dict]:
    """The chain x irregularity grid (random ops), the difficulty 'landscape'."""
    return [{"name": cell_name(c, irr), "chain": c, "irregular": irr,
             "force_op": None, "easy_only": False}
            for c in chains for irr in (False, True)]


def op_specs() -> list[dict]:
    """Op-targeted cells: isolate the effect of a hard operation at LOW chain.

    Compares, at matched (short) chain length + irregular numbers:
      easy_* : only easy ops (reflection/midpoint) -> control
      int_*  : a guaranteed line-intersection
      foot_* : a guaranteed foot-of-altitude (perpendicular projection)
    This is how we find the *shortest* scene that already breaks SOTA.
    """
    specs: list[dict] = []
    for c in (3, 4, 5):
        specs.append({"name": f"easy_c{c}_irr", "chain": c, "irregular": True,
                      "force_op": None, "easy_only": True})
    for c in (3, 4):
        specs.append({"name": f"int_c{c}_irr", "chain": c, "irregular": True,
                      "force_op": "intersection", "easy_only": False})
    for c in (4, 5):
        specs.append({"name": f"foot_c{c}_irr", "chain": c, "irregular": True,
                      "force_op": "foot_altitude", "easy_only": False})
    specs.append({"name": "foot_c4_rnd", "chain": 4, "irregular": False,
                  "force_op": "foot_altitude", "easy_only": False})
    return specs


def build_grid(specs: list[dict], k: int, seed: int, blocklist: set[str]) -> list[dict]:
    """Equal-K, deduplicated grid from cell specs, disjoint from blocklist."""
    rng = random.Random(seed)
    grid: list[dict] = []
    eid = 0
    for spec in specs:
        seen: set[str] = set()
        picked = 0
        cap = k * 400 + 4000
        tries = 0
        while picked < k and tries < cap:
            tries += 1
            ex = make_example(rng, spec["chain"], spec["irregular"],
                              force_op=spec["force_op"], easy_only=spec["easy_only"])
            if ex["chain"] != spec["chain"]:  # realized chain must match the target
                continue
            if spec["force_op"] and spec["force_op"] not in ex["tags"]:
                continue  # the forced hard op didn't materialize -> skip
            d = ex["description"]
            if d in blocklist or d in seen:
                continue
            seen.add(d)
            ex["id"] = eid
            ex["cell"] = spec["name"]
            ex["force_op"] = spec["force_op"]
            ex["easy_only"] = spec["easy_only"]
            grid.append(ex)
            eid += 1
            picked += 1
        if picked < k:
            print(f"  WARN {spec['name']}: only {picked}/{k} unique found")
    return grid


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _dhash(desc: str) -> str:
    return hashlib.md5(desc.encode()).hexdigest()[:12]


def ensure_raw(model: str, grid: list[dict], cache_dir: Path,
               workers: int, max_tokens: int, temperature: float) -> dict[int, dict]:
    """Return {id: raw-record}, calling the API only for uncached examples.

    Cache entries are keyed by example id and validated against a hash of the
    example description, so a stale cache from a different grid is never reused.
    """
    cache_path = cache_dir / f"{safe_name(model)}.jsonl"
    grid_by_id = {ex["id"]: ex for ex in grid}
    cache: dict[int, dict] = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from an interrupted run
            gex = grid_by_id.get(r["id"])
            if gex is None:
                continue
            if "h" in r and r["h"] != _dhash(gex["description"]):
                continue  # stale entry from a previous grid -> ignore (will re-call)
            cache[r["id"]] = r  # later lines win over earlier
    todo = [ex for ex in grid if ex["id"] not in cache]
    if todo:
        print(f"    calling {model} on {len(todo)} new prompts (workers={workers}) ...", flush=True)

        def worker(ex: dict) -> dict:
            res = gateway.chat(build_messages(ex["description"]), model,
                               max_tokens=max_tokens, temperature=temperature)
            return {"id": ex["id"], "h": _dhash(ex["description"]), "raw": res.text,
                    "ok": res.ok, "error": res.error, "finish": res.finish_reason,
                    "latency_s": res.latency_s, "attempts": res.attempts}

        # Stream results to disk as they complete so a crash never loses a run.
        import concurrent.futures as cf
        import threading

        lock = threading.Lock()
        done = 0
        with cache_path.open("a") as f, cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(worker, ex) for ex in todo]
            for fut in cf.as_completed(futs):
                rec = fut.result()
                with lock:
                    cache[rec["id"]] = rec
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done += 1
                    if done % 25 == 0 or done == len(todo):
                        print(f"      {done}/{len(todo)} done", flush=True)
        fails = sum(1 for ex in todo if not cache[ex["id"]]["ok"])
        if fails:
            print(f"    {fails}/{len(todo)} API failures (excluded from denominators)")
    return cache


def score_model(model: str, grid: list[dict], cache: dict[int, dict],
                workers: int, render: bool) -> list[dict]:
    def scorer(ex: dict) -> dict:
        rec = cache[ex["id"]]
        base = {"id": ex["id"], "cell": ex["cell"], "chain": ex["chain"],
                "irregular": ex["irregular"], "force_op": ex.get("force_op"),
                "easy_only": ex.get("easy_only", False), "latency_s": rec.get("latency_s")}
        if not rec.get("ok"):
            return {**base, "api_fail": True, "passed": False, "compiles": False,
                    "figure_only": False, "coord_accuracy": 0.0,
                    "coords_all_correct": False, "ssim": None}
        ev = harness.evaluate_one(
            ex_id=ex["id"], description=ex["description"], model_output=rec["raw"],
            gt_tikz=ex["tikz"], gt_points=ex["points"], render=render,
        )
        return {**base, "api_fail": False, "passed": ev.passed, "compiles": ev.compiles,
                "figure_only": ev.figure_only, "coord_accuracy": ev.coord_accuracy,
                "coords_all_correct": ev.coords_all_correct, "ssim": ev.ssim}

    return gateway.map_concurrent(scorer, grid, workers=workers)


def aggregate_cells(detail: list[dict]) -> dict:
    by_cell: dict[str, list[dict]] = {}
    for d in detail:
        by_cell.setdefault(d["cell"], []).append(d)
    out: dict[str, dict] = {}
    for cell, rows in by_cell.items():
        live = [r for r in rows if not r["api_fail"]]  # capability != availability
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


def print_pass_grid(models: list[str], results: dict, cells: list[str],
                    threshold: float) -> None:
    print(f"\n{'=' * 100}\nPASS RATE  (figure-only AND compiles AND all coords within 0.05)\n{'=' * 100}")
    head = f"{'model':<34}" + "".join(f"{c:>9}" for c in cells)
    print(head)
    print("-" * len(head))
    for m in models:
        cellmap = results[m]["cells"]
        row = f"{m:<34}"
        for c in cells:
            v = cellmap.get(c, {}).get("pass_rate")
            row += f"{('%.2f' % v) if v is not None else '  -  ':>9}"
        print(row)
    print("-" * len(head))
    # cross-model per-cell mean and best
    for label, fn in (("MEAN", lambda xs: sum(xs) / len(xs)), ("BEST", max)):
        row = f"{label + ' across models':<34}"
        for c in cells:
            xs = [results[m]["cells"][c]["pass_rate"] for m in models
                  if c in results[m]["cells"]]
            row += f"{('%.2f' % fn(xs)) if xs else '  -  ':>9}"
        print(row)
    print(f"\nreliability threshold = {threshold:.0%}  "
          f"(a cell is 'reliable' if a model's pass rate >= threshold)")


def _cell_meta(results: dict, models: list[str], cell: str) -> tuple[int, bool]:
    m = next(mm for mm in models if cell in results[mm]["cells"])
    c = results[m]["cells"][cell]
    return c["chain"], c["irregular"]


def _cell_stat(results: dict, models: list[str], cell: str, how: str) -> float:
    xs = [results[m]["cells"][cell]["pass_rate"] for m in models
          if cell in results[m]["cells"]]
    return (max(xs) if how == "best" else sum(xs) / len(xs)) if xs else 1.0


def recommend(models: list[str], results: dict, cells: list[str],
              threshold: float, how: str = "best") -> dict:
    """Least-complex cell from which the task stays unreliable.

    Robust to a noisy easy-cell dip: we take the *contiguous block of hardest
    cells* that are all below threshold, and return its easiest cell. Complexity
    order = chain first, then irregular. ``how`` = 'best' (even the strongest
    model is unreliable) or 'mean' (models on average are unreliable).
    """
    ordered = sorted(cells, key=lambda c: _cell_meta(results, models, c))
    rec = None
    for c in reversed(ordered):  # walk hardest -> easiest
        if _cell_stat(results, models, c, how) < threshold:
            rec = c
        else:
            break
    if rec is None:
        return {}
    return {"cell": rec, "how": how,
            "best_pass": _cell_stat(results, models, rec, "best"),
            "mean_pass": _cell_stat(results, models, rec, "mean")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="explicit gateway model ids")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--chains", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--k", type=int, default=12, help="examples per (chain x irregular) cell")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="outputs/sweep")
    ap.add_argument("--block", nargs="+", default=["data/train.jsonl"],
                    help="jsonl files whose descriptions to exclude from the grid")
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls")
    ap.add_argument("--score-workers", type=int, default=4, help="concurrent compiles")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ssim", action="store_true", help="also render+diff (slower)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--op-dial", action="store_true",
                    help="add op-targeted cells (guaranteed hard op at low chain)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score models already present in results.json")
    ap.add_argument("--rebuild-grid", action="store_true")
    ap.add_argument("--gather-only", action="store_true",
                    help="only fetch+cache raw model outputs; skip scoring (parallel prefetch)")
    args = ap.parse_args()

    models = args.models or (PRESETS[args.preset] if args.preset else PRESETS["frontier_small"])
    out_dir = ROOT / args.out
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "detail").mkdir(parents=True, exist_ok=True)

    specs = landscape_specs(args.chains) + (op_specs() if args.op_dial else [])
    cells = [s["name"] for s in specs]

    grid_path = out_dir / "grid.jsonl"
    meta_path = out_dir / "meta.json"
    want_meta = {"seed": args.seed, "chains": args.chains, "k": args.k, "op_dial": args.op_dial}
    if grid_path.exists() and not args.rebuild_grid:
        grid = [json.loads(l) for l in grid_path.read_text().splitlines() if l.strip()]
        old = None
        if meta_path.exists():
            try:
                old = json.loads(meta_path.read_text()).get("grid")
            except json.JSONDecodeError:
                old = None  # torn read from a concurrent writer -> skip the check
        if old is not None and old != want_meta:
            raise SystemExit(
                f"existing grid at {grid_path} was built with {old} != "
                f"requested {want_meta}. Use --rebuild-grid (clears caches) or a new --out."
            )
        if not meta_path.exists():  # write once; avoids a concurrent-write race across parallel groups
            meta_path.write_text(json.dumps({"grid": want_meta}, indent=2))
        print(f"reusing grid: {len(grid)} examples from {grid_path}")
    else:
        print(f"building grid: chains={args.chains} k={args.k} op_dial={args.op_dial} ...")
        block = load_blocklist(args.block)
        grid = build_grid(specs, args.k, args.seed, block)
        grid_path.write_text("\n".join(json.dumps(ex) for ex in grid) + "\n")
        meta_path.write_text(json.dumps({"grid": want_meta}, indent=2))  # persist now
        for p in (out_dir / "raw").glob("*.jsonl"):  # grid changed -> caches invalid
            p.unlink()
        print(f"built {len(grid)} examples -> {grid_path} (cleared raw caches)")

    landscape_cells = [cell_name(c, irr) for c in args.chains for irr in (False, True)]

    # Accumulate across runs so the sweep can be done in short, resumable batches
    # (the harness caps long background jobs). Prior models are preserved.
    results: dict = {}
    prior_models: list[str] = []
    results_path = out_dir / "results.json"
    if results_path.exists():
        prior = json.loads(results_path.read_text())
        results = prior.get("results", {})
        prior_models = prior.get("meta", {}).get("models", [])
    all_models = list(dict.fromkeys(prior_models + models))

    for model in models:
        detail_path = out_dir / "detail" / f"{safe_name(model)}.jsonl"
        if model in results and detail_path.exists() and not args.rescore:
            print(f"\n>>> {model} (already scored — skipping; use --rescore to redo)")
            continue
        print(f"\n>>> {model}")
        t0 = time.time()
        cache = ensure_raw(model, grid, out_dir / "raw", args.workers,
                           args.max_tokens, args.temperature)
        if args.gather_only:
            print(f"    gather-only: {len(cache)}/{len(grid)} cached ({time.time() - t0:.0f}s)")
            continue
        detail = score_model(model, grid, cache, args.score_workers, args.ssim)
        (out_dir / "detail" / f"{safe_name(model)}.jsonl").write_text(
            "\n".join(json.dumps(d) for d in detail) + "\n"
        )
        lats = [d["latency_s"] for d in detail if d.get("latency_s")]
        cellsagg = aggregate_cells(detail)
        n_live = sum(c["n"] for c in cellsagg.values())
        results[model] = {
            "cells": cellsagg,
            "overall": {
                "n": n_live,
                "n_api_fail": sum(c["n_api_fail"] for c in cellsagg.values()),
                "pass_rate": sum(d["passed"] for d in detail if not d["api_fail"]) / (n_live or 1),
                "compile_rate": sum(d["compiles"] for d in detail if not d["api_fail"]) / (n_live or 1),
            },
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else None,
        }
        ov = results[model]["overall"]
        print(f"    overall pass={ov['pass_rate']:.2f} compile={ov['compile_rate']:.2f} "
              f"api_fail={ov['n_api_fail']} avg_lat={results[model]['avg_latency_s']}s "
              f"({time.time() - t0:.0f}s)")

        # write incrementally so partial/batched runs are always analyzable
        meta = {"grid": want_meta, "models": all_models, "cells": cells,
                "landscape_cells": landscape_cells, "op_dial": args.op_dial,
                "threshold": args.threshold, "max_tokens": args.max_tokens,
                "temperature": args.temperature, "ssim": args.ssim}
        results_path.write_text(
            json.dumps({"meta": meta, "results": results}, indent=2))

    if args.gather_only:
        print(f"\ngather-only complete: raw caches populated for {len(models)} model(s). "
              f"Re-run without --gather-only to score.")
        return
    shown = [m for m in all_models if m in results]
    print_pass_grid(shown, results, landscape_cells, args.threshold)
    op_cells = [c for c in cells if c not in landscape_cells]
    if op_cells:
        print_pass_grid(shown, results, op_cells, args.threshold)
    rec = recommend(shown, results, landscape_cells, args.threshold)
    if rec:
        print(f"\nRECOMMENDATION: least-complex landscape cell where even the BEST model "
              f"is unreliable:\n  {rec['cell']}  (best pass {rec['best_pass']:.2f}, "
              f"mean pass {rec['mean_pass']:.2f})")
    else:
        print(f"\nNo landscape cell drops below the threshold for the best model — "
              f"extend --chains higher or lean on the op-dial cells.")
    print(f"\nwrote -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
