"""AIME auto-illustrator: try the specialist on every geometry problem, fall back
to a frontier model, and measure how much of the bank we can illustrate.

Pipeline
--------
1. Load the AIME bank (``gneubig/aime-1983-2024``) and keep geometry problems by
   keyword. Optionally sample (default 150) to keep gateway spend modest.
2. Run the SPECIALIST on each problem statement (local or Modal batch), compile
   the output, and count COVERAGE = fraction that yield a compiling,
   non-degenerate figure. This is EXPECTED TO BE LOW: the specialist's training
   distribution is narrow (origin-anchored circles + a fixed op set) and real
   AIME geometry is largely out-of-distribution. We report it honestly.
3. Route every remaining problem to a FRONTIER fallback (gateway), prompted with
   CONSTRUCTION_SYSTEM_PROMPT so it, too, returns coordinate-free constructions;
   compile those.
4. Emit a rendered gallery (PNGs + a contact-sheet HTML) and coverage stats
   (specialist vs fallback vs still-unillustratable, with a per-decade
   breakdown). Raw model outputs are cached so re-runs never re-spend.

Usage:
  # Modal batch for the specialist (fast), gpt-5.5 fallback:
  uv run python scripts/illustrate_aime.py --n 150 --backend modal \
      --fallback-model openai-group/gpt-5.5

  # fully local specialist (slow but no cloud):
  uv run python scripts/illustrate_aime.py --n 60 --backend local
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import gateway, serve, vision_judge  # noqa: E402
from geotikz.prompts import build_construction_messages  # noqa: E402
from geotikz.metrics import extract_tikz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Keywords that mark a problem as (plausibly) a geometry problem. Deliberately
# broad; coverage is measured honestly regardless of a few false positives.
GEO_KW = [
    "triangle", "circle", "angle", "tangent", "perpendicular", "polygon", "square",
    "rectangle", "circumscrib", "inscrib", "radius", "diameter", "chord", "vertex",
    "vertices", "parallel", "hexagon", "pentagon", "octagon", "rhombus", "trapezoid",
    "quadrilateral", "sphere", "cylinder", "cone", "isosceles", "equilateral",
    "midpoint", "bisector", "centroid", "incircle", "circumcircle", "altitude",
    "hypotenuse", "collinear", "concurrent", "cevian", "orthocenter", "circumcenter",
    "incenter", "pentagon", "coordinate", "segment", "arc ", "semicircle",
]


def is_geometry(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in GEO_KW)


def load_problems(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("gneubig/aime-1983-2024")["train"]
    geo = [r for r in ds if is_geometry(r.get("Question", ""))]
    print(f"AIME bank: {len(ds)} problems, {len(geo)} match geometry keywords")
    if n and n < len(geo):
        geo = random.Random(seed).sample(geo, n)
    probs = []
    for r in geo:
        pid = str(r.get("ID") or f"{r.get('Year')}-{r.get('Problem Number')}")
        probs.append({"id": pid, "year": int(r.get("Year", 0)),
                      "description": r.get("Question", "")})
    probs.sort(key=lambda p: p["id"])
    return probs


# --------------------------------------------------------------------------- #
# specialist outputs (cache-aware; local or modal backend)
# --------------------------------------------------------------------------- #
def specialist_outputs(probs: list[dict], backend: str, cache: serve.Cache,
                       max_new_tokens: int, batch_size: int,
                       modal_script: str = "scripts/infer_modal.py",
                       cache_prefix: str = "specialist") -> dict[str, str]:
    # cache_prefix separates different specialists (e.g. the narrow v2 model vs
    # the illustrator); the token budget is in the key too, so re-running with a
    # larger budget regenerates instead of reusing truncated outputs.
    keyed = {p["id"]: f"{cache_prefix}:t{max_new_tokens}:" + serve.dhash(p["description"])
             for p in probs}
    out: dict[str, str] = {}
    todo = []
    for p in probs:
        hit = cache.get(keyed[p["id"]])
        if hit:
            out[p["id"]] = hit["output"]
        else:
            todo.append(p)
    print(f"[specialist] {len(out)} cached, {len(todo)} to generate "
          f"(backend={backend}, script={modal_script})")
    if not todo:
        return out

    descs = [p["description"] for p in todo]
    if backend == "modal":
        gen = _specialist_modal(descs, max_new_tokens, batch_size, modal_script)
    else:
        spec = serve.Specialist().load()
        gen = spec.generate_batch(descs, max_new_tokens=max_new_tokens, batch_size=batch_size)
    for p, o in zip(todo, gen):
        cache.put(keyed[p["id"]], {"model": cache_prefix, "output": o})
        out[p["id"]] = o
    return out


def _specialist_modal(descs: list[str], max_new_tokens: int, batch_size: int,
                      modal_script: str = "scripts/infer_modal.py") -> list[str]:
    """Run the batched specialist on Modal via the given modal script.

    ``modal_script`` shares the ``--input/--output/--max-new-tokens/--batch-size``
    interface across scripts/infer_modal.py (narrow v2 specialist) and
    scripts/infer_illustrator_modal.py (the 1.7B illustrator), so the same driver
    works for both.
    """
    tmp = Path(tempfile.mkdtemp(prefix="aime_modal_"))
    inp, outp = tmp / "in.jsonl", tmp / "out.jsonl"
    inp.write_text("\n".join(json.dumps({"id": i, "description": d})
                             for i, d in enumerate(descs)) + "\n")
    cmd = ["modal", "run", modal_script, "--input", str(inp),
           "--output", str(outp), "--max-new-tokens", str(max_new_tokens),
           "--batch-size", str(batch_size)]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    rows = [json.loads(l) for l in outp.read_text().splitlines() if l.strip()]
    by_i = {r["id"]: r["output"] for r in rows}
    return [by_i[i] for i in range(len(descs))]


# --------------------------------------------------------------------------- #
# frontier fallback outputs (cache-aware, concurrent)
# --------------------------------------------------------------------------- #
def frontier_outputs(probs: list[dict], model: str, cache: serve.Cache,
                     workers: int, max_tokens: int) -> dict[str, str]:
    keyed = {p["id"]: f"{model}:" + serve.dhash(p["description"]) for p in probs}
    out: dict[str, str] = {}
    todo = []
    for p in probs:
        hit = cache.get(keyed[p["id"]])
        if hit:
            out[p["id"]] = hit["output"]
        else:
            todo.append(p)
    print(f"[frontier {model}] {len(out)} cached, {len(todo)} to generate")
    if not todo:
        return out

    def gen(p: dict) -> tuple[str, str]:
        res = gateway.chat(build_construction_messages(p["description"]), model,
                           max_tokens=max_tokens)
        cache.put(keyed[p["id"]], {"model": model, "output": res.text,
                                   "ok": res.ok, "error": res.error,
                                   "latency_s": res.latency_s})
        return p["id"], res.text

    results = gateway.map_concurrent(gen, todo, workers=workers)
    for pid, text in results:
        out[pid] = text
    return out


# --------------------------------------------------------------------------- #
# compile a batch of (id -> output) into (id -> RenderResult), in parallel
# --------------------------------------------------------------------------- #
def render_batch(items: list[tuple[str, str, Path]], workers: int) -> dict[str, serve.RenderResult]:
    out: dict[str, serve.RenderResult] = {}

    def one(item):
        pid, text, png = item
        return pid, serve.compile_and_render(text, png)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, it) for it in items]
        done = 0
        for fut in cf.as_completed(futs):
            pid, r = fut.result()
            out[pid] = r
            done += 1
            if done % 20 == 0 or done == len(items):
                print(f"    compiled {done}/{len(items)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# vision judge: does the (compiling, non-degenerate) figure faithfully depict
# the problem? Real AIME problems have NO ground-truth coordinates, so this is
# the only available correctness signal beyond "it drew something". Reported
# SEPARATELY from (and softer than) coordinate verification.
# --------------------------------------------------------------------------- #
def judge_coverage(items: list[tuple[str, str, str, str]], model: str,
                   cache: serve.Cache, workers: int) -> dict[str, dict]:
    """items: (pid, description, tikz, png_path) for figures that already render.

    Returns {pid: {approved, reason, mode}}; cached + concurrent.
    """
    todo = []
    out: dict[str, dict] = {}
    for pid, desc, tikz, png in items:
        ck = f"judge:{model}:" + serve.dhash((desc or "") + "|" + (tikz or ""))
        hit = cache.get(ck)
        if hit is not None:
            out[pid] = {"approved": hit["approved"], "reason": hit["reason"],
                        "mode": hit["mode"]}
        else:
            todo.append((pid, desc, tikz, png, ck))
    print(f"[judge {model}] {len(out)} cached, {len(todo)} to judge")

    def one(item):
        pid, desc, tikz, png, ck = item
        v = vision_judge.judge(desc, png, tikz or "", model, prefer_vision=True)
        rec = {"approved": v.approved, "reason": v.reason, "mode": v.mode}
        cache.put(ck, rec)
        return pid, rec

    for pid, rec in gateway.map_concurrent(one, todo, workers=workers):
        out[pid] = rec
    return out


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def write_contact_sheet(path: Path, entries: list[dict]) -> None:
    """entries: {id, route, png (relative), text, year}."""
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}
    h1{margin:0 0 4px} .sub{color:#666;margin-bottom:20px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
    .card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
    .card img{width:100%;height:200px;object-fit:contain;background:#fff;border-radius:6px}
    .id{font-weight:600;margin-top:8px} .txt{font-size:12px;color:#555;max-height:74px;overflow:auto;margin-top:6px}
    .badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;color:#fff}
    .specialist{background:#2e7d32} .frontier{background:#1565c0}
    """
    cards = []
    for e in entries:
        cards.append(
            f'<div class="card"><img src="{html.escape(e["png"])}" alt="{html.escape(e["id"])}">'
            f'<div class="id">{html.escape(e["id"])} '
            f'<span class="badge {e["route"]}">{e["route"]}</span></div>'
            f'<div class="txt">{html.escape(e["text"][:280])}</div></div>'
        )
    n_spec = sum(1 for e in entries if e["route"] == "specialist")
    n_front = sum(1 for e in entries if e["route"] == "frontier")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>AIME auto-illustrations</title>"
           f"<style>{css}</style></head><body>"
           f"<h1>AIME geometry - auto-illustrations</h1>"
           f"<div class='sub'>{len(entries)} figures &middot; "
           f"<span class='badge specialist'>specialist {n_spec}</span> "
           f"<span class='badge frontier'>frontier {n_front}</span></div>"
           f"<div class='grid'>{''.join(cards)}</div></body></html>")
    path.write_text(doc)


# --------------------------------------------------------------------------- #
def decade(year: int) -> str:
    return f"{(year // 10) * 10}s" if year else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="sample size (0 = all geometry)")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--backend", choices=["local", "modal"], default="modal",
                    help="specialist inference backend")
    ap.add_argument("--specialist-script", default="scripts/infer_modal.py",
                    help="Modal inference script for the specialist "
                         "(use scripts/infer_illustrator_modal.py for the 1.7B illustrator)")
    ap.add_argument("--fallback-model", default="openai-group/gpt-5.5")
    ap.add_argument("--no-fallback", action="store_true", help="skip the frontier fallback")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096, help="frontier budget")
    ap.add_argument("--workers", type=int, default=6, help="concurrent frontier calls")
    ap.add_argument("--compile-workers", type=int, default=4)
    ap.add_argument("--judge-coverage", dest="judge_coverage", action="store_true",
                    default=True, help="also report vision-judge-verified coverage")
    ap.add_argument("--no-judge-coverage", dest="judge_coverage", action="store_false")
    ap.add_argument("--judge-model", default="gemini-group/gemini-3.1-pro")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/aime_gallery")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    (out_dir / "specialist").mkdir(parents=True, exist_ok=True)
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    cache = serve.Cache(out_dir / "cache" / "raw.jsonl")
    t_start = time.time()

    probs = load_problems(args.n, args.seed)
    print(f"illustrating {len(probs)} geometry problems\n")

    # --- Stage 1+2: specialist ---
    cache_prefix = "illustrator" if "illustrator" in args.specialist_script else "specialist"
    spec_out = specialist_outputs(probs, args.backend, cache, args.max_new_tokens,
                                  args.batch_size, modal_script=args.specialist_script,
                                  cache_prefix=cache_prefix)
    print("[specialist] compiling ...")
    spec_items = [(p["id"], spec_out.get(p["id"], ""), out_dir / "specialist" / f"{p['id']}.png")
                  for p in probs]
    spec_render = render_batch(spec_items, args.compile_workers)
    covered_spec = {pid for pid, r in spec_render.items() if r.ok}
    print(f"[specialist] coverage: {len(covered_spec)}/{len(probs)} "
          f"({len(covered_spec) / len(probs) * 100:.1f}%)")

    by_id_desc = {p["id"]: p["description"] for p in probs}
    judge_cache = serve.Cache(out_dir / "cache" / "judge.jsonl")

    # --- Stage 3: vision-judge the SPECIALIST figures (gate routing on it) ---
    # Real AIME problems have no ground-truth coordinates, so coordinate
    # verification is not computable here; the vision judge is the available
    # correctness signal. We report BOTH raw (compile+non-degenerate) and
    # judge-verified coverage so the softer number is never conflated with the
    # stricter one. (Coordinate-verified coverage is reported on the synthetic
    # held-out set instead -- see scripts/eval_syn_illustrator.py.)
    spec_judge: dict[str, dict] = {}
    if args.judge_coverage:
        spec_items = [(pid, by_id_desc[pid], extract_tikz(spec_out.get(pid, "") or ""),
                       str(out_dir / "specialist" / f"{pid}.png")) for pid in covered_spec]
        spec_judge = judge_coverage(spec_items, args.judge_model, judge_cache, args.judge_workers)
    spec_faithful = ({pid for pid in covered_spec if spec_judge.get(pid, {}).get("approved")}
                     if args.judge_coverage else set(covered_spec))

    # --- Stage 4: frontier fallback with JUDGE-GATED routing ---
    # A compiling-but-unfaithful local figure must NOT pre-empt the frontier, so
    # the frontier is asked to handle every problem the specialist did not
    # *faithfully* cover (not merely every problem it failed to compile).
    route_gate = spec_faithful if args.judge_coverage else set(covered_spec)
    remaining = [p for p in probs if p["id"] not in route_gate]
    covered_front: set[str] = set()
    front_render: dict[str, serve.RenderResult] = {}
    front_out: dict[str, str] = {}
    front_judge: dict[str, dict] = {}
    front_faithful: set[str] = set()
    if remaining and not args.no_fallback:
        front_out = frontier_outputs(remaining, args.fallback_model, cache,
                                     args.workers, args.max_tokens)
        print(f"[frontier] compiling {len(remaining)} ...")
        front_items = [(p["id"], front_out.get(p["id"], ""),
                        out_dir / "frontier" / f"{p['id']}.png") for p in remaining]
        front_render = render_batch(front_items, args.compile_workers)
        covered_front = {pid for pid, r in front_render.items() if r.ok}
        print(f"[frontier] coverage: {len(covered_front)}/{len(remaining)} of the remainder")
        if args.judge_coverage:
            fitems = [(pid, by_id_desc[pid], extract_tikz(front_out.get(pid, "") or ""),
                       str(out_dir / "frontier" / f"{pid}.png")) for pid in covered_front]
            front_judge = judge_coverage(fitems, args.judge_model, judge_cache, args.judge_workers)
            front_faithful = {pid for pid in covered_front if front_judge.get(pid, {}).get("approved")}
        else:
            front_faithful = set(covered_front)

    # merged judged map for stats/back-compat
    judged: dict[str, dict] = {**spec_judge, **front_judge}

    # --- Stage 5: stats + gallery ---
    # Judge-gated routes (what actually gets shown / counted as covered):
    #   specialist if the specialist figure is judge-verified faithful,
    #   else frontier if the frontier figure is judge-verified faithful,
    #   else none. (When --no-judge-coverage, "faithful" == "compiles".)
    routes: dict[str, str] = {}
    for p in probs:
        if p["id"] in spec_faithful:
            routes[p["id"]] = "specialist"
        elif p["id"] in front_faithful:
            routes[p["id"]] = "frontier"
        else:
            routes[p["id"]] = "none"

    entries = []
    for p in probs:
        pid, route = p["id"], routes[p["id"]]
        if route == "none":
            continue
        entries.append({"id": pid, "route": route,
                        "png": f"{route}/{pid}.png", "text": p["description"],
                        "year": p["year"]})
    entries.sort(key=lambda e: (e["route"] != "specialist", e["id"]))
    write_contact_sheet(out_dir / "index.html", entries)

    n = len(probs)
    # compile + non-degenerate (any route drew a real figure)
    compile_covered = covered_spec | covered_front
    n_spec = len(covered_spec)                       # specialist compiled
    n_front_only = len(covered_front - covered_spec)  # frontier compiled, spec didn't
    n_compile_total = len(compile_covered)
    n_compile_none = n - n_compile_total
    # judge-verified (vision judge confirms faithful) -- disjoint by construction
    n_spec_j = len(spec_faithful)                    # HEADLINE: local, standalone
    n_front_j = len(front_faithful)                  # frontier fallback, faithful
    union_faithful = spec_faithful | front_faithful
    n_union_j = len(union_faithful)
    n_judge_none = n - n_union_j
    judge_mode = None
    if judged:
        modes = [v.get("mode") for v in judged.values()]
        judge_mode = "vision" if modes.count("vision") >= modes.count("text") else "text"
    # per-decade breakdown (judge-verified routing)
    dec: dict[str, dict] = {}
    for p in probs:
        d = decade(p["year"])
        dd = dec.setdefault(d, {"total": 0, "specialist": 0, "frontier": 0, "none": 0})
        dd["total"] += 1
        dd[routes[p["id"]]] += 1

    stats = {
        "n": n, "sample_seed": args.seed, "backend": args.backend,
        "specialist_script": args.specialist_script,
        "max_new_tokens": args.max_new_tokens,
        "fallback_model": None if args.no_fallback else args.fallback_model,
        "judge_model": args.judge_model if args.judge_coverage else None,
        "judge_mode": judge_mode,
        "routing": "judge-gated (local figure used only if judge-verified)"
        if args.judge_coverage else "compile-first",
        # (1) weaker signal: the model drew a real (non-blank/blob/oversized) figure
        "coverage_compile": {
            "specialist": n_spec, "specialist_pct": round(n_spec / n, 4),
            "frontier_additional": n_front_only,
            "total": n_compile_total, "total_pct": round(n_compile_total / n, 4),
            "none": n_compile_none,
        },
        # (2) softer-than-coordinate signal: compile + non-degenerate AND the
        # vision judge confirms the figure faithfully depicts the problem.
        # specialist_pct is the HEADLINE local coverage; total is the union.
        "coverage_judge_verified": {
            "specialist": n_spec_j, "specialist_pct": round(n_spec_j / n, 4),
            "frontier_fallback": n_front_j, "frontier_fallback_pct": round(n_front_j / n, 4),
            "union_total": n_union_j, "union_total_pct": round(n_union_j / n, 4),
            "none": n_judge_none,
        } if args.judge_coverage else None,
        "coverage_coordinate_verified": "n/a for real AIME (no ground-truth "
        "coordinates); see scripts/eval_syn_illustrator.py for the coordinate-"
        "verified pass rate on the held-out synthetic set. On AIME the union of "
        "coordinate-verified and judge-verified therefore equals judge-verified.",
        "by_decade": dec,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out_dir / "coverage_stats.json").write_text(json.dumps(stats, indent=2))

    # short markdown
    md = [
        "# AIME auto-illustrator - coverage\n",
        f"- Sample: {n} geometry problems from `gneubig/aime-1983-2024` "
        f"(seed={args.seed}, specialist backend={args.backend}, "
        f"script=`{args.specialist_script}`).",
        f"- Fallback model: `{stats['fallback_model']}` (construction prompt).",
        f"- Vision judge: `{stats['judge_model']}` (mode={judge_mode}).\n",
        "## Coverage - two signals, reported separately\n",
        "We report coverage under two correctness signals, from weaker to stronger:\n",
        "1. **compile + non-degenerate** - the model drew a real (non-blank, "
        "non-blob, non-oversized) figure. Necessary but not sufficient.",
        "2. **judge-verified** - additionally, a capable VISION model, shown the "
        "problem text + the rendered figure, confirmed it *faithfully depicts the "
        "described configuration* (including 3D / combinatorial figures). This is "
        "softer than coordinate verification (a judge can be fooled), but it is "
        "the only correctness signal available for real problems, which have **no "
        "ground-truth coordinates**. Coordinate-verified coverage is therefore "
        "reported on the held-out *synthetic* set instead (see "
        "`scripts/eval_syn_illustrator.py`).\n",
        "### (1) compile + non-degenerate (LOCAL specialist standalone)\n",
        "| model | count | share |",
        "|---|---|---|",
        f"| **specialist (local)** | **{n_spec}** | **{n_spec / n * 100:.1f}%** |",
        f"| + frontier (judge-gated fallback, additional) | {n_front_only} | "
        f"{n_front_only / n * 100:.1f}% |",
        f"| any route compiled | {n_compile_total} | {n_compile_total / n * 100:.1f}% |\n",
    ]
    if args.judge_coverage:
        md += [
            "### (2) judge-verified (compile + non-degenerate + vision judge faithful)\n",
            "| route | count | share |",
            "|---|---|---|",
            f"| **specialist (local, standalone)** | **{n_spec_j}** | "
            f"**{n_spec_j / n * 100:.1f}%** |",
            f"| frontier fallback (judge-gated) | {n_front_j} | {n_front_j / n * 100:.1f}% |",
            f"| **union total** | **{n_union_j}** | **{n_union_j / n * 100:.1f}%** |",
            f"| none (no faithful figure) | {n_judge_none} | {n_judge_none / n * 100:.1f}% |\n",
            "Routing is **judge-gated**: the local figure is used only when the "
            "vision judge approves it; otherwise the frontier handles the problem. "
            "So `union total` = local-faithful + frontier-faithful (disjoint). On "
            "real AIME there is no ground truth, so the union of coordinate- and "
            "judge-verified coverage equals this judge-verified number.\n",
        ]
    md += [
        "## Interpretation\n",
        f"The local specialist draws a compiling, non-degenerate figure for "
        f"**{n_spec / n * 100:.1f}%** of real AIME geometry, of which the vision "
        f"judge confirms **{n_spec_j / n * 100:.1f}%** faithfully depict the "
        "problem. The gap between the two is exactly the honesty the judge buys: "
        "a figure can compile yet not match. The frontier fallback covers the "
        "hard tail. `none` are problems no route drew acceptably "
        "(often genuinely hard 3D / heavily combinatorial configurations).\n",
        "## By decade (judge-verified routing)\n",
        "| decade | total | specialist | frontier | none |",
        "|---|---|---|---|---|",
    ]
    for d in sorted(dec):
        r = dec[d]
        md.append(f"| {d} | {r['total']} | {r['specialist']} | {r['frontier']} | {r['none']} |")
    md.append(f"\nGallery: `{args.out_dir}/index.html`  |  stats: `{args.out_dir}/coverage_stats.json`")
    (out_dir / "coverage_report.md").write_text("\n".join(md))

    print("\n" + "=" * 64)
    print(f"LOCAL specialist  compile+non-degenerate = {n_spec}/{n} "
          f"({n_spec / n * 100:.1f}%)")
    if args.judge_coverage:
        print(f"LOCAL specialist  judge-verified        = {n_spec_j}/{n} "
              f"({n_spec_j / n * 100:.1f}%)   <-- headline (faithful)")
        print(f"UNION (local + judge-gated frontier) judge-verified = "
              f"{n_union_j}/{n} ({n_union_j / n * 100:.1f}%)")
    print(f"compile coverage any route = {n_compile_total}/{n} "
          f"({n_compile_total / n * 100:.1f}%)")
    print(f"gallery  -> {out_dir / 'index.html'}")
    print(f"stats    -> {out_dir / 'coverage_stats.json'}")
    print(f"report   -> {out_dir / 'coverage_report.md'}")
    print(f"elapsed  {stats['elapsed_s']}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
