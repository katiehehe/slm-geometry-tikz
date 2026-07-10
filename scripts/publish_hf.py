"""Publish the dataset + LoRA adapters to the Hugging Face Hub.

ADDITIVE / read-only over the repo: this script uploads existing artifacts. It does
NOT retrain, edit datasets, or touch adapters — it just pushes copies to the Hub.

Safety:
  * DRY-RUN by default. It prints exactly what it *would* upload and exits. Nothing is
    created or uploaded until you pass ``--push``.
  * Needs a token only when ``--push`` is set. Provide it via ``HF_TOKEN`` (or
    ``HUGGINGFACE_HUB_TOKEN``) in your environment / ``.env``, or ``--token``.

Typical use (the maintainer runs this with their own token):

  # 1. See the plan (no token needed):
  uv run python scripts/publish_hf.py --user YOURNAME

  # 2. Actually publish (token required):
  export HF_TOKEN=hf_...            # a WRITE token from https://huggingface.co/settings/tokens
  uv run python scripts/publish_hf.py --user YOURNAME --push

By default it publishes:
  * dataset repo  ``YOURNAME/spec-first-geometry-tikz``  (data/*.jsonl + dataset card)
  * model   repo  ``YOURNAME/qwen3-geotikz``  (LoRA adapters as subfolders + model card)

Adapters that live only on the Modal Volume (not downloaded locally) are skipped with a
``modal volume get`` hint so you can fetch and re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- what to publish -------------------------------------------------------- #

# LoRA adapters -> uploaded as subfolders of the single model repo. Each must have an
# adapter_config.json to count as a real adapter directory.
DEFAULT_ADAPTERS = [
    "qwen3-pgf-geotikz",     # v2 0.6B construction specialist (the headline)
    "qwen3-1.7b-geotikz",    # v1 1.7B numeric
    "qwen3-geotikz",         # v1 0.6B numeric
    "qwen3-illustrator",     # 1.7B AIME illustrator
    "qwen3-illustrator-4b",  # 4B illustrator (in progress; skipped if absent)
]

# Dataset files -> uploaded flat into the dataset repo. (name in data/, kept as-is on Hub.)
DEFAULT_DATA_FILES = [
    "train.jsonl",
    "train_pgf.jsonl",
    "eval.jsonl",
    "eval_pgf.jsonl",
    "golden_set.jsonl",
    "olympiad_eval.jsonl",
    "illustrator_train_chat.jsonl",
    "illustrator_syn_eval.jsonl",
]

MODEL_CARD = ROOT / "cards" / "model_card.md"
DATASET_CARD = ROOT / "cards" / "dataset_card.md"


def _token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    # Load .env if present (the repo already depends on python-dotenv).
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _plan(user: str, model_repo: str, dataset_repo: str, adapters: list[str],
          data_files: list[str]) -> tuple[list[tuple[str, Path]], list[str], list[Path]]:
    """Resolve what exists locally. Returns (adapter_dirs, missing_adapters, data_paths)."""
    adapter_dirs: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name in adapters:
        d = ROOT / "outputs" / name
        if (d / "adapter_config.json").exists():
            adapter_dirs.append((name, d))
        else:
            missing.append(name)

    data_paths: list[Path] = []
    for f in data_files:
        p = ROOT / "data" / f
        if p.exists():
            data_paths.append(p)
        else:
            print(f"  [warn] dataset file not found, skipping: data/{f}")
    return adapter_dirs, missing, data_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your Hugging Face username/org")
    ap.add_argument("--model-repo", default="qwen3-geotikz",
                    help="model repo name (adapters go in as subfolders)")
    ap.add_argument("--dataset-repo", default="spec-first-geometry-tikz")
    ap.add_argument("--adapters", nargs="*", default=DEFAULT_ADAPTERS)
    ap.add_argument("--data-files", nargs="*", default=DEFAULT_DATA_FILES)
    ap.add_argument("--private", action="store_true", help="create the repos as private")
    ap.add_argument("--token", default=None, help="HF write token (else env HF_TOKEN)")
    ap.add_argument("--push", action="store_true",
                    help="actually create repos and upload (default: dry-run)")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    model_id = f"{args.user}/{args.model_repo}"
    dataset_id = f"{args.user}/{args.dataset_repo}"

    adapter_dirs, missing, data_paths = _plan(
        args.user, model_id, dataset_id, args.adapters, args.data_files)

    print("=" * 74)
    print(f"PLAN ({'PUSH' if args.push else 'DRY-RUN'})")
    print("=" * 74)
    if not args.skip_dataset:
        print(f"\nDataset repo:  {dataset_id}  (private={args.private})")
        print(f"  card:  {DATASET_CARD.relative_to(ROOT)}  -> README.md"
              f"{'  [MISSING]' if not DATASET_CARD.exists() else ''}")
        for p in data_paths:
            print(f"  file:  data/{p.name}  ({p.stat().st_size/1e6:.2f} MB)")
    if not args.skip_model:
        print(f"\nModel repo:    {model_id}  (private={args.private})")
        print(f"  card:  {MODEL_CARD.relative_to(ROOT)}  -> README.md"
              f"{'  [MISSING]' if not MODEL_CARD.exists() else ''}")
        for name, d in adapter_dirs:
            print(f"  adapter subfolder:  {name}/  (from outputs/{name})")
        for name in missing:
            print(f"  [skip] adapter '{name}' not found locally. Fetch it first:")
            print(f"         modal volume get geotikz-outputs {name} ./outputs/{name}")

    if not args.push:
        print("\nDry-run only. Re-run with --push (and a token) to publish.")
        return

    tok = _token(args.token)
    if not tok:
        sys.exit("ERROR: --push set but no token. Set HF_TOKEN or pass --token.")

    from huggingface_hub import HfApi

    api = HfApi(token=tok)

    if not args.skip_dataset:
        print(f"\n[dataset] create_repo {dataset_id}")
        api.create_repo(dataset_id, repo_type="dataset", private=args.private,
                        exist_ok=True)
        if DATASET_CARD.exists():
            api.upload_file(path_or_fileobj=str(DATASET_CARD), path_in_repo="README.md",
                            repo_id=dataset_id, repo_type="dataset")
        for p in data_paths:
            print(f"[dataset] upload data/{p.name}")
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                            repo_id=dataset_id, repo_type="dataset")
        print(f"[dataset] done -> https://huggingface.co/datasets/{dataset_id}")

    if not args.skip_model:
        print(f"\n[model] create_repo {model_id}")
        api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
        if MODEL_CARD.exists():
            api.upload_file(path_or_fileobj=str(MODEL_CARD), path_in_repo="README.md",
                            repo_id=model_id, repo_type="model")
        for name, d in adapter_dirs:
            print(f"[model] upload_folder {name}/")
            api.upload_folder(folder_path=str(d), path_in_repo=name, repo_id=model_id,
                              repo_type="model")
        print(f"[model] done -> https://huggingface.co/{model_id}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
