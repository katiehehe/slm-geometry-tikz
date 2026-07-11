"""Turn specialist difficulty sweeps into a complexity-ceiling report.

Combines two sweeps of qwen3-illustrator-4b (both scored by the compile-extract
gate: figure-only AND compiles AND every named coord within 0.05 of GT):

  * CHAIN dir  (--chain-dir): chains composed of ROBUST ops the specialist handles
    singly (midpoint / reflect-over-line for the "isometry" regime; foot /
    line-intersection for the "+metric" regime). Degradation here isolates
    COMPOSITIONAL DEPTH -> the clean chain-length ceiling.
  * AUX dir    (--aux-dir): the 20 native construction FAMILIES (single
    constructions) + a generic-transform chain run that exposes PARAPHRASE
    BRITTLENESS (rotation / translation / point-symmetry phrased generically).

Outputs, in --chain-dir:
  chain_heatmap.png        pass rate on chain-length x op-complexity grid
  chain_degradation.png    pass vs chain length, isometry vs +metric, 95% Wilson CIs
  family_passrate.png      pass rate per native construction family
  op_robustness.png        chain-1 pass rate per single op (robust vs brittle)
  pass_by_cell.csv         every cell + metric + Wilson interval
  ceiling_report.md        the plain-English "reliable up to / breaks at" call

Usage:
  uv run python scripts/specialist_ceiling_report.py \
      --chain-dir outputs/specialist_ceiling_robust \
      --aux-dir outputs/specialist_ceiling --threshold 0.9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OP_MARKERS = [("midpoint", "midpoint of segment"), ("point_symmetry", "through point"),
              ("reflect_line", "over line"), ("rotation", "rotated"),
              ("translation", "translated by the vector"),
              ("foot", "foot of the perpendicular"), ("intersection", "intersection of lines")]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_joined(out_dir: Path, model: str) -> list[dict]:
    grid = {json.loads(l)["id"]: json.loads(l)
            for l in (out_dir / "grid.jsonl").read_text().splitlines() if l.strip()}
    rows = []
    for l in (out_dir / "detail" / f"{model}.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        g = grid.get(d["id"], {})
        rows.append({**d, "kind": g.get("kind"), "regime": g.get("regime"),
                     "tag": g.get("tag"), "n_derived": g.get("n_derived"),
                     "chain": g.get("chain", d.get("chain")),
                     "description": g.get("description", "")})
    return rows


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    passes = sum(r["passed"] for r in rows)
    p = passes / n if n else 0.0
    lo, hi = wilson(p, n)
    return {"n": n, "pass_rate": p, "lo": lo, "hi": hi,
            "compile_rate": sum(r["compiles"] for r in rows) / n if n else 0.0,
            "coord_accuracy_mean": sum(r["coord_accuracy"] for r in rows) / n if n else 0.0,
            "coords_all_correct_rate": sum(r["coords_all_correct"] for r in rows) / n if n else 0.0}


def single_op(desc: str) -> str | None:
    ops = [name for name, m in OP_MARKERS if m in desc]
    return ops[0] if len(ops) == 1 else None


# --------------------------------------------------------------------------- #
def draw_chain_heatmap(cells, chains, regimes, labels, out, thr):
    M = np.array([[cells.get((rg, c), {}).get("pass_rate", np.nan) for c in chains]
                  for rg in regimes])
    fig, ax = plt.subplots(figsize=(max(6, len(chains) * 1.3), 2.6 + 0.3 * len(regimes)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(chains)), [f"chain {c}" for c in chains])
    ax.set_yticks(range(len(regimes)), [labels.get(r, r) for r in regimes])
    for i, rg in enumerate(regimes):
        for j, c in enumerate(chains):
            cd = cells.get((rg, c))
            if cd:
                ax.text(j, i, f"{cd['pass_rate']:.2f}\nn={cd['n']}", ha="center",
                        va="center", fontsize=8)
    ax.set_title("Specialist pass rate: chain length x op complexity (robust ops)\n"
                 f"gate: figure-only & compiles & every coord <0.05  |  green = reliable (>={thr:.0%})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="pass rate")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def draw_degradation(cells, chains, regimes, labels, out, thr):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"affine": "#1f77b4", "mixed": "#d62728"}
    markers = {"affine": "o", "mixed": "s"}
    for rg in regimes:
        xs, ys, lo, hi = [], [], [], []
        for c in chains:
            cd = cells.get((rg, c))
            if not cd:
                continue
            xs.append(c); ys.append(cd["pass_rate"])
            lo.append(cd["pass_rate"] - cd["lo"]); hi.append(cd["hi"] - cd["pass_rate"])
        ax.errorbar(xs, ys, yerr=[lo, hi], color=colors.get(rg, "#333"),
                    marker=markers.get(rg, "o"), capsize=3, label=labels.get(rg, rg))
    ax.axhline(thr, color="gray", ls="--", lw=1)
    ax.text(chains[0], thr + 0.01, f"reliable {thr:.0%}", color="gray", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(chains[0], 0.51, "coin-flip 50%", color="gray", fontsize=8)
    ax.set_xlabel("chain length (# composed derived-point steps)")
    ax.set_ylabel("pass rate"); ax.set_ylim(-0.03, 1.03); ax.set_xticks(chains)
    ax.set_title("Specialist reliability vs compositional depth (95% Wilson CIs)")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def draw_family(fam, out, thr):
    items = sorted(fam.items(), key=lambda kv: kv[1]["pass_rate"])
    names = [k for k, _ in items]
    ys = [v["pass_rate"] for _, v in items]
    lo = [v["pass_rate"] - v["lo"] for _, v in items]
    hi = [v["hi"] - v["pass_rate"] for _, v in items]
    colors = ["#d62728" if y < 0.5 else "#ff7f0e" if y < thr else "#2ca02c" for y in ys]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.32)))
    ax.barh(range(len(names)), ys, xerr=[lo, hi], color=colors, capsize=2,
            error_kw={"elinewidth": 1})
    ax.set_yticks(range(len(names)), names)
    ax.axvline(thr, color="gray", ls="--", lw=1); ax.set_xlim(0, 1.03)
    ax.set_xlabel("pass rate (GT-verified, 95% Wilson CI)")
    ax.set_title(f"Specialist pass rate by native construction family\n"
                 f"green >= {thr:.0%}, orange 0.5-{thr:.0%}, red < 0.5")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def draw_op_robustness(ops, out, thr):
    order = sorted(ops.items(), key=lambda kv: -kv[1]["pass_rate"])
    names = [k for k, _ in order]
    ys = [v["pass_rate"] for _, v in order]
    lo = [v["pass_rate"] - v["lo"] for _, v in order]
    hi = [v["hi"] - v["pass_rate"] for _, v in order]
    colors = ["#d62728" if y < 0.5 else "#ff7f0e" if y < thr else "#2ca02c" for y in ys]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
    ax.barh(range(len(names)), ys, xerr=[lo, hi], color=colors, capsize=3,
            error_kw={"elinewidth": 1})
    ax.set_yticks(range(len(names)), [f"{n} (n={ops[n]['n']})" for n in names])
    ax.axvline(thr, color="gray", ls="--", lw=1); ax.set_xlim(0, 1.03)
    ax.invert_yaxis()
    ax.set_xlabel("chain-1 (single-op) pass rate, 95% Wilson CI")
    ax.set_title("Paraphrase robustness: single-op pass rate\n"
                 "robust ops (green) vs generically-phrased transforms the model mis-emits (red)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def ceiling(cells, chains, regime, thr):
    reliable = None
    for c in sorted(chains):
        cd = cells.get((regime, c))
        if cd and cd["pass_rate"] >= thr:
            reliable = c
        else:
            break
    breaks = None
    for c in sorted(chains):
        cd = cells.get((regime, c))
        if cd and cd["pass_rate"] < 0.5:
            breaks = c
            break
    return reliable, breaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", default="outputs/specialist_ceiling_robust")
    ap.add_argument("--aux-dir", default="outputs/specialist_ceiling")
    ap.add_argument("--model", default="qwen3-illustrator-4b")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    chain_dir = root / args.chain_dir
    aux_dir = root / args.aux_dir
    labels = {"affine": "isometry ops\n(midpoint/reflect)", "mixed": "+ metric op\n(foot/intersect)"}

    # ---- chain cells (robust) ----
    crows = [r for r in load_joined(chain_dir, args.model) if r["kind"] == "chain"]
    chains = sorted({r["chain"] for r in crows})
    regimes = [r for r in ["affine", "mixed"] if any(x["regime"] == r for x in crows)]
    cells = {(rg, c): agg([r for r in crows if r["regime"] == rg and r["chain"] == c])
             for rg in regimes for c in chains
             if any(r["regime"] == rg and r["chain"] == c for r in crows)}

    # ---- aux: families + single-op robustness ----
    arows = load_joined(aux_dir, args.model)
    fam_rows = [r for r in arows if r["kind"] == "family"]
    fam = {tag: {**agg([r for r in fam_rows if r["tag"] == tag]),
                 "n_derived": next(r["n_derived"] for r in fam_rows if r["tag"] == tag)}
           for tag in sorted({r["tag"] for r in fam_rows})}
    # single-op robustness from aux chain-1 (generic transforms present there)
    op_rows: dict[str, list] = {}
    for r in arows:
        if r["kind"] == "chain" and r["chain"] == 1:
            op = single_op(r["description"])
            if op:
                op_rows.setdefault(op, []).append(r)
    ops = {op: agg(rs) for op, rs in op_rows.items()}

    draw_chain_heatmap(cells, chains, regimes, labels, chain_dir / "chain_heatmap.png", args.threshold)
    draw_degradation(cells, chains, regimes, labels, chain_dir / "chain_degradation.png", args.threshold)
    if fam:
        draw_family(fam, chain_dir / "family_passrate.png", args.threshold)
    if ops:
        draw_op_robustness(ops, chain_dir / "op_robustness.png", args.threshold)

    with (chain_dir / "pass_by_cell.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "cell", "regime", "chain_or_nderived", "n", "pass_rate",
                    "wilson_lo", "wilson_hi", "compile_rate", "coord_accuracy_mean"])
        for rg in regimes:
            for c in chains:
                cd = cells.get((rg, c))
                if cd:
                    w.writerow([f"chain_{rg}", f"c{c}_{rg[:3]}", rg, c, cd["n"],
                                f"{cd['pass_rate']:.3f}", f"{cd['lo']:.3f}", f"{cd['hi']:.3f}",
                                f"{cd['compile_rate']:.3f}", f"{cd['coord_accuracy_mean']:.3f}"])
        for tag, cd in sorted(fam.items(), key=lambda kv: -kv[1]["pass_rate"]):
            w.writerow(["family", f"fam_{tag}", "", cd["n_derived"], cd["n"],
                        f"{cd['pass_rate']:.3f}", f"{cd['lo']:.3f}", f"{cd['hi']:.3f}",
                        f"{cd['compile_rate']:.3f}", f"{cd['coord_accuracy_mean']:.3f}"])
        for op, cd in sorted(ops.items(), key=lambda kv: -kv[1]["pass_rate"]):
            w.writerow(["single_op", op, "", 1, cd["n"], f"{cd['pass_rate']:.3f}",
                        f"{cd['lo']:.3f}", f"{cd['hi']:.3f}", f"{cd['compile_rate']:.3f}",
                        f"{cd['coord_accuracy_mean']:.3f}"])

    # ---- ceiling markdown ----
    n_cell = cells.get((regimes[0], chains[0]), {}).get("n", "?")
    L = ["# Specialist complexity ceiling — qwen3-illustrator-4b (Qwen3-4B + LoRA)", "",
         "- **Gate:** figure-only AND compiles AND every named coordinate within 0.05 of GT "
         "(compile-extract grader; identical falsifiable gate the project trains toward).",
         f"- **Reliability threshold:** {args.threshold:.0%} pass rate.",
         f"- **Chain grid (robust ops):** chains {chains}, regimes {regimes}, n={n_cell}/cell, "
         "every ground truth round-trip validated (emit->compile->read-back == GT).", "",
         "## Chain-length x op-complexity ceiling (clean: robust ops only)", "",
         "| regime | " + " | ".join(f"chain {c}" for c in chains) + " |",
         "| :-- | " + " | ".join("--:" for _ in chains) + " |"]
    for rg in regimes:
        L.append(f"| {rg} | " + " | ".join(
            f"{cells[(rg, c)]['pass_rate']:.2f}" if (rg, c) in cells else "-" for c in chains) + " |")
    L.append("")
    for rg in regimes:
        rel, brk = ceiling(cells, chains, rg, args.threshold)
        rel_s = f"chain {rel}" if rel else "not even chain 1"
        brk_s = f"chain {brk}" if brk else f">chain {max(chains)}"
        L.append(f"- **{rg} ({labels[rg].splitlines()[0]}):** reliable (>= {args.threshold:.0%}) "
                 f"up to **{rel_s}**; drops below 50% at **{brk_s}**.")
    L.append("")
    if fam:
        strong = [t for t, v in fam.items() if v["pass_rate"] >= args.threshold]
        weak = sorted([(t, v) for t, v in fam.items() if v["pass_rate"] < args.threshold],
                      key=lambda kv: kv[1]["pass_rate"])
        L += ["## Single-construction vocabulary (20 native families)", "",
              f"- **Strong (>= {args.threshold:.0%}):** {len(strong)}/{len(fam)} families.",
              "- **Weak:** " + (", ".join(f"{t} {v['pass_rate']:.2f}" for t, v in weak) or "none")
              + " (many simultaneous derived points).", ""]
    if ops:
        brittle = sorted([(o, v) for o, v in ops.items() if v["pass_rate"] < args.threshold],
                         key=lambda kv: kv[1]["pass_rate"])
        robust = [o for o, v in ops.items() if v["pass_rate"] >= args.threshold]
        L += ["## Paraphrase robustness (single op, chain 1)", "",
              f"- **Robust (>= {args.threshold:.0%}):** {', '.join(sorted(robust)) or 'none'}.",
              "- **Brittle (generically-phrased transforms the model mis-emits):** "
              + (", ".join(f"{o} {v['pass_rate']:.2f}" for o, v in brittle) or "none") + ".", ""]
    (chain_dir / "ceiling_report.md").write_text("\n".join(L) + "\n")

    print("wrote:")
    for o in ["chain_heatmap.png", "chain_degradation.png", "family_passrate.png",
              "op_robustness.png", "pass_by_cell.csv", "ceiling_report.md"]:
        print("  ", chain_dir / o)
    print()
    print("\n".join(L))


if __name__ == "__main__":
    main()
