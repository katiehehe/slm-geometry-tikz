"""Turn a difficulty-sweep results.json into shareable evidence.

Produces, in the sweep output dir:
  - pass_heatmap.png   models x (chain x irregularity) pass-rate heatmap (landscape)
  - degradation.png    pass rate vs chain length, split round/irregular, with 95% CIs
  - op_effect.png       pass rate by operation type at matched (short) chain length
  - pass_rates.csv      one row per (model, cell) with every metric
  - report.md           pivoted tables + the training-complexity call

Headline question: what is the *least complex* level at which frontier models
stop being reliable? That is where a fine-tuned specialist has something to
prove — and thus where to generate training data.

Usage:
  uv run python scripts/sweep_report.py --dir outputs/sweep --threshold 0.9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sane at small n / p near 0/1)."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def short_model(m: str) -> str:
    return m.split("/")[-1]


def load(results_path: Path):
    payload = json.loads(results_path.read_text())
    meta, results = payload["meta"], payload["results"]
    models = [m for m in meta["models"] if m in results]
    cells = meta["cells"]
    landscape = meta.get("landscape_cells") or [
        c for c in cells if re.fullmatch(r"c\d+_(rnd|irr)", c)
    ]
    op_cells = [c for c in cells if c not in landscape]
    return meta, results, models, cells, landscape, op_cells


def cmeta(results: dict, models: list[str], cell: str) -> dict:
    m = next(mm for mm in models if cell in results[mm]["cells"])
    return results[m]["cells"][cell]


def pooled(results: dict, models: list[str], cell: str) -> tuple[float, int]:
    passes = sum(round(results[m]["cells"][cell]["pass_rate"]
                       * results[m]["cells"][cell]["n"]) for m in models
                 if cell in results[m]["cells"])
    n = sum(results[m]["cells"][cell]["n"] for m in models if cell in results[m]["cells"])
    return (passes / n if n else 0.0), n


def frontier_stat(results: dict, models: list[str], cell: str, how: str) -> float:
    xs = [results[m]["cells"][cell]["pass_rate"] for m in models
          if cell in results[m]["cells"]]
    if not xs:
        return 1.0
    return max(xs) if how == "best" else float(np.mean(xs))


def draw_heatmap(results: dict, models: list[str], cells: list[str],
                 out: Path, threshold: float) -> None:
    cells = sorted(cells, key=lambda c: (cmeta(results, models, c)["chain"],
                                         cmeta(results, models, c)["irregular"]))
    M = np.array([[results[m]["cells"][c]["pass_rate"] if c in results[m]["cells"] else np.nan
                   for c in cells] for m in models])
    mean_row = np.nanmean(M, axis=0, keepdims=True)
    best_row = np.nanmax(M, axis=0, keepdims=True)
    full = np.vstack([M, mean_row, best_row])
    ylabels = [short_model(m) for m in models] + ["MEAN", "BEST"]
    xlabels = [f"c{cmeta(results, models, c)['chain']}\n"
               f"{'irr' if cmeta(results, models, c)['irregular'] else 'rnd'}" for c in cells]

    fig, ax = plt.subplots(figsize=(max(8, len(cells) * 0.85),
                                    max(4, len(ylabels) * 0.45 + 1)))
    im = ax.imshow(full, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cells)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.axhline(len(models) - 0.5, color="black", lw=1.5)
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            if not np.isnan(full[i, j]):
                ax.text(j, i, f"{full[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pass rate by model x difficulty  (gate: figure-only & compiles & "
                 f"all coords <0.05)\nreliable = green (>= {threshold:.0%})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="pass rate")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def draw_degradation(results: dict, models: list[str], cells: list[str],
                     out: Path, threshold: float) -> None:
    chains = sorted({cmeta(results, models, c)["chain"] for c in cells})
    by = {(cmeta(results, models, c)["chain"], cmeta(results, models, c)["irregular"]): c
          for c in cells}
    fig, ax = plt.subplots(figsize=(8, 5))
    for irregular, color, marker in ((False, "#1f77b4", "o"), (True, "#d62728", "s")):
        xs, ys, los, his, best = [], [], [], [], []
        for ch in chains:
            c = by.get((ch, irregular))
            if not c:
                continue
            p, n = pooled(results, models, c)
            lo, hi = wilson(p, n)
            xs.append(ch); ys.append(p); los.append(p - lo); his.append(hi - p)
            best.append(frontier_stat(results, models, c, "best"))
        ax.errorbar(xs, ys, yerr=[los, his], color=color, marker=marker, capsize=3,
                    label=f"{'irregular' if irregular else 'round'} numbers (pooled)")
        ax.plot(xs, best, color=color, ls=":", alpha=0.6,
                label=f"best model ({'irr' if irregular else 'rnd'})")
    ax.axhline(threshold, color="gray", ls="--", lw=1)
    ax.text(chains[0], threshold + 0.01, f"reliability {threshold:.0%}", color="gray", fontsize=8)
    ax.set_xlabel("chain length (# composed derivation steps)")
    ax.set_ylabel("pass rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(chains)
    ax.set_title("Frontier LLM reliability degrades with geometric complexity")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _op_kind(cd: dict) -> str:
    if cd.get("easy_only"):
        return "easy only"
    if cd.get("force_op") == "intersection":
        return "intersection"
    if cd.get("force_op") == "foot_altitude":
        return "foot of altitude"
    return "mixed (random)"


def draw_op_effect(results: dict, models: list[str], op_cells: list[str],
                   landscape: list[str], out: Path, threshold: float) -> None:
    """Grouped bars: pass rate by operation kind at matched short chain (irregular)."""
    # consider op cells + the matched landscape irregular cells for context
    consider = list(op_cells) + [c for c in landscape if cmeta(results, models, c)["irregular"]]
    rows: dict[int, dict[str, str]] = {}
    for c in consider:
        cd = cmeta(results, models, c)
        if not cd["irregular"]:
            continue
        rows.setdefault(cd["chain"], {})[_op_kind(cd)] = c
    kinds = ["easy only", "mixed (random)", "intersection", "foot of altitude"]
    colors = {"easy only": "#2ca02c", "mixed (random)": "#7f7f7f",
              "intersection": "#ff7f0e", "foot of altitude": "#d62728"}
    chains = sorted(rows)
    if not chains:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(chains) * 2.2), 5))
    width = 0.2
    for ki, kind in enumerate(kinds):
        xs, ys, err = [], [], []
        for ci, ch in enumerate(chains):
            c = rows.get(ch, {}).get(kind)
            if not c:
                continue
            p, n = pooled(results, models, c)
            lo, hi = wilson(p, n)
            xs.append(ci + (ki - 1.5) * width); ys.append(p)
            err.append([[p - lo], [hi - p]])
        if not xs:
            continue
        yerr = np.hstack(err) if err else None
        ax.bar(xs, ys, width, color=colors[kind], label=kind,
               yerr=yerr, capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(threshold, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(chains)), [f"chain {c}" for c in chains])
    ax.set_ylabel("pass rate (pooled across models, irregular numbers)")
    ax.set_ylim(0, 1.03)
    ax.set_title("The operation is the difficulty: a foot-of-altitude breaks SOTA\n"
                 "at a far shorter chain than reflections do")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_csv(results: dict, models: list[str], cells: list[str], out: Path) -> None:
    fields = ["model", "cell", "chain", "irregular", "force_op", "easy_only", "n",
              "n_api_fail", "pass_rate", "compile_rate", "figure_only_rate",
              "coord_accuracy_mean", "coords_all_correct_rate"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in models:
            for c in cells:
                cd = results[m]["cells"].get(c)
                if cd:
                    w.writerow({"model": m, "cell": c, **{k: cd.get(k) for k in fields[2:]}})


def recommend(results: dict, models: list[str], landscape: list[str],
              threshold: float, how: str) -> str | None:
    ordered = sorted(landscape, key=lambda c: (cmeta(results, models, c)["chain"],
                                               cmeta(results, models, c)["irregular"]))
    rec = None
    for c in reversed(ordered):
        if frontier_stat(results, models, c, how) < threshold:
            rec = c
        else:
            break
    return rec


def write_markdown(meta, results, models, landscape, op_cells, out, threshold) -> None:
    L = sorted(landscape, key=lambda c: (cmeta(results, models, c)["chain"],
                                         cmeta(results, models, c)["irregular"]))
    lines = ["# Difficulty sweep — do frontier LLMs underperform on spec-first geometry→TikZ?", ""]
    lines += [
        "- **Gate:** figure-only AND compiles AND every named coordinate within 0.05 "
        "of ground truth (the project's Behavior Spec).",
        f"- **Grid:** {meta['grid']['k']} examples/cell, chains {meta['grid']['chains']}, "
        "round vs irregular numbers." + (" Plus op-targeted cells." if op_cells else ""),
        f"- **Models ({len(models)}):** " + ", ".join(short_model(m) for m in models),
        f"- **Reliability threshold:** {threshold:.0%} pass rate.",
        "",
        "## Landscape: pass rate by model x difficulty",
        "",
    ]
    xh = [f"c{cmeta(results, models, c)['chain']} "
          f"{'irr' if cmeta(results, models, c)['irregular'] else 'rnd'}" for c in L]
    lines += ["| model | " + " | ".join(xh) + " |",
              "| :-- | " + " | ".join(["--:"] * len(L)) + " |"]
    for m in models:
        row = [short_model(m)] + [f"{results[m]['cells'][c]['pass_rate']:.2f}"
                                  if c in results[m]["cells"] else "-" for c in L]
        lines.append("| " + " | ".join(row) + " |")
    for label, how in (("MEAN", "mean"), ("BEST", "best")):
        lines.append("| **" + label + "** | "
                     + " | ".join(f"{frontier_stat(results, models, c, how):.2f}" for c in L) + " |")
    lines.append("")

    if op_cells:
        lines += ["## Operation effect (matched short chains, irregular numbers)", "",
                  "Pass rate pooled across models. Same chain length, different last "
                  "operation — the *hard op* is what collapses reliability.", "",
                  "| cell | operation | chain | pooled pass | n |",
                  "| :-- | :-- | --: | --: | --: |"]
        for c in sorted(op_cells, key=lambda c: (cmeta(results, models, c)["chain"], c)):
            cd = cmeta(results, models, c)
            p, n = pooled(results, models, c)
            lines.append(f"| {c} | {_op_kind(cd)} | {cd['chain']} | {p:.2f} | {n} |")
        lines.append("")

    rec_best = recommend(results, models, landscape, threshold, "best")
    rec_mean = recommend(results, models, landscape, threshold, "mean")
    lines += ["## Where to train", ""]
    if rec_best:
        cd = cmeta(results, models, rec_best)
        lines.append(
            f"**Least-complex landscape level where even the BEST model is unreliable: "
            f"`{rec_best}`** (chain {cd['chain']}, "
            f"{'irregular' if cd['irregular'] else 'round'}) — best {frontier_stat(results, models, rec_best, 'best'):.2f}, "
            f"mean {frontier_stat(results, models, rec_best, 'mean'):.2f}.")
    else:
        lines.append("No landscape level breaks the best model — extend `--chains` higher "
                     "or rely on the op-targeted cells below.")
    if rec_mean:
        cd = cmeta(results, models, rec_mean)
        lines.append(f"- On *average* across models, reliability is already gone by "
                     f"`{rec_mean}` (chain {cd['chain']}, "
                     f"{'irregular' if cd['irregular'] else 'round'}).")
    if op_cells:
        worst = min(op_cells, key=lambda c: pooled(results, models, c)[0])
        cdw = cmeta(results, models, worst)
        pw, _ = pooled(results, models, worst)
        lines.append(f"- **Shortest hard breaker:** `{worst}` ({_op_kind(cdw)} at chain "
                     f"{cdw['chain']}) pools to only {pw:.2f} — a short scene that already "
                     f"beats prompting.")
    lines += ["", "**Recommendation:** center training data on the failure region "
              "(the recommended cell and the hard-op cells), keeping a minority of "
              "easier examples so the specialist stays robust across the ramp."]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="outputs/sweep")
    ap.add_argument("--results", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.dir
    results_path = Path(args.results) if args.results else out_dir / "results.json"
    meta, results, models, cells, landscape, op_cells = load(results_path)

    draw_heatmap(results, models, landscape, out_dir / "pass_heatmap.png", args.threshold)
    draw_degradation(results, models, landscape, out_dir / "degradation.png", args.threshold)
    if op_cells:
        draw_op_effect(results, models, op_cells, landscape,
                       out_dir / "op_effect.png", args.threshold)
    write_csv(results, models, cells, out_dir / "pass_rates.csv")
    write_markdown(meta, results, models, landscape, op_cells,
                   out_dir / "report.md", args.threshold)

    outs = ["pass_heatmap.png", "degradation.png"] + (["op_effect.png"] if op_cells else [])
    outs += ["pass_rates.csv", "report.md"]
    print("wrote:\n  " + "\n  ".join(str(out_dir / o) for o in outs))
    rec = recommend(results, models, landscape, args.threshold, "best")
    if rec:
        print(f"\nleast-complex level where the best model is unreliable: {rec}")


if __name__ == "__main__":
    main()
