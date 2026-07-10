"""Summarize the training-set composition + randomness, and render a slide figure.

Prints exact category counts from data/train.jsonl and saves a 1x3 composition
figure (chain length, number regularity, construction-op frequency) to
outputs/renders/data_composition.png.

Usage:
  uv run python scripts/analyze_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OP_LABEL = {
    "point_on_circle": "point on circle",
    "reflect_x": "reflect / x-axis",
    "reflect_y": "reflect / y-axis",
    "midpoint": "midpoint to center",
    "intersection": "line intersection",
    "foot_altitude": "foot of altitude",
}
HARD = {"intersection", "foot_altitude"}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    rows = load(ROOT / "data/train.jsonl")
    n = len(rows)

    chain_ct = Counter(r["chain"] for r in rows)
    irr_ct = Counter("irregular" if r["irregular"] else "round" for r in rows)
    op_doc = Counter()   # examples containing >=1 of the op
    op_tot = Counter()   # total occurrences
    for r in rows:
        for t in set(r["tags"]):
            op_doc[t] += 1
        for t in r["tags"]:
            op_tot[t] += 1
    hard_ex = sum(1 for r in rows if any(t in HARD for t in r["tags"]))

    print(f"train.jsonl: {n} examples\n")
    print("chain length:")
    for c in sorted(chain_ct):
        print(f"  chain {c}: {chain_ct[c]:>5}  ({chain_ct[c]/n:5.1%})")
    print("\nnumber regularity:")
    for k in ("irregular", "round"):
        print(f"  {k:9s}: {irr_ct[k]:>5}  ({irr_ct[k]/n:5.1%})")
    print(f"\nexamples with >=1 HARD op (intersection/foot): {hard_ex} ({hard_ex/n:.1%})")
    print("\nconstruction op (examples containing it / total occurrences):")
    for t, _ in op_doc.most_common():
        tag = "HARD" if t in HARD else "easy"
        print(f"  {OP_LABEL.get(t,t):18s} [{tag}]: {op_doc[t]:>5} ex ({op_doc[t]/n:5.1%})"
              f"   {op_tot[t]:>5} occ")

    # ---- figure ----
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Training-set composition  ·  data/train.jsonl  ·  n = {n:,} examples",
                 fontsize=15, fontweight="bold")

    chains = sorted(chain_ct)
    b1 = a1.bar([str(c) for c in chains], [chain_ct[c] for c in chains], color="#4c6ef5")
    a1.set_title("Chain length (# derivation steps)")
    a1.set_xlabel("chain")
    a1.set_ylabel("examples")
    for r_, c in zip(b1, chains):
        a1.text(r_.get_x() + r_.get_width()/2, r_.get_height(),
                f"{chain_ct[c]/n:.0%}", ha="center", va="bottom", fontsize=10)

    ks = ["irregular", "round"]
    b2 = a2.bar(ks, [irr_ct[k] for k in ks], color=["#e8590c", "#adb5bd"])
    a2.set_title("Number regularity")
    a2.set_ylabel("examples")
    for r_, k in zip(b2, ks):
        a2.text(r_.get_x() + r_.get_width()/2, r_.get_height(),
                f"{irr_ct[k]/n:.0%}", ha="center", va="bottom", fontsize=10)

    ordered = [t for t, _ in op_doc.most_common()][::-1]
    vals = [op_doc[t]/n for t in ordered]
    colors = ["#c92a2a" if t in HARD else "#2b8a3e" for t in ordered]
    a3.barh([OP_LABEL.get(t, t) for t in ordered], vals, color=colors)
    a3.set_title("Construction op — % of examples containing it")
    a3.set_xlabel("share of examples")
    a3.set_xlim(0, 1.0)
    for i, v in enumerate(vals):
        a3.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=10)
    a3.plot([], [], color="#c92a2a", label="hard op (breaks SOTA)", linewidth=6)
    a3.plot([], [], color="#2b8a3e", label="easy op", linewidth=6)
    a3.legend(loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "outputs/renders/data_composition.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
