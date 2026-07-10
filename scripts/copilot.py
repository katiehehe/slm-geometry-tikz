"""Geometry-figure copilot — LOCAL entry (frontier-first).

  uv run python scripts/copilot.py                 # open the printed local URL
  uv run python scripts/copilot.py --modal-specialist   # borrow the Modal GPU specialist
  uv run python scripts/copilot.py --auth me:secret     # password-protect the local app
  uv run python scripts/copilot.py --share              # public gradio.live tunnel

All the routing / rendering / attribution logic lives in the importable core
``geotikz.copilot`` (shared with the hosted Modal app, ``scripts/copilot_modal.py``).
This script only wires up a SPECIALIST BACKEND and launches the UI:

  * default            -> the LOCAL base+LoRA specialist (``serve.Specialist``,
    Qwen3-0.6B + qwen3-pgf-geotikz), loaded lazily on first use. Frontier-first:
    the specialist toggle is OFF by default.
  * ``--modal-specialist`` -> route the specialist to the DEPLOYED Modal GPU
    function instead, so you get "(specialist · Modal GPU)" attribution from the
    laptop without the local model load. Frontier still covers vision + edits and
    is the fallback if the specialist fails / degenerates.

Flow: TEXT scene -> specialist (if toggled) else frontier construction prompt;
IMAGE/PDF -> frontier vision; PASTE TikZ -> render + edit loop; EDIT -> frontier.
Every reply states which model produced it; non-compiling figures get one
self-repair pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotikz import copilot, serve  # noqa: E402

MODAL_APP = "geotikz-copilot"
MODAL_CLS = "Specialist"


def local_specialist_backend():
    """The laptop specialist: lazy base+LoRA (torch loads on first specialist use)."""
    spec = serve.Specialist()

    def fn(description: str) -> str:
        return spec.generate(description)  # raw text; the core extracts the tikz

    label = f"`{spec.base} + LoRA` (local specialist)"
    return fn, label


def modal_specialist_backend():
    """Route the specialist to the deployed Modal GPU function (no local load).

    Requires ``modal deploy scripts/copilot_modal.py`` to have run (so the class
    exists). Reports the ACTUAL adapter the GPU container loaded.
    """
    import modal

    spec = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
    holder = {"label": "`qwen3-illustrator-4b` (specialist · Modal GPU)"}

    def fn(description: str) -> str:
        res = spec.generate.remote(description)
        if isinstance(res, dict):
            holder["label"] = f"`{res.get('adapter', 'specialist')}` (specialist · Modal GPU)"
            return res.get("tikz", "")
        return res or ""

    return fn, (lambda: holder["label"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Geometry Figure Copilot (local).")
    ap.add_argument("--modal-specialist", action="store_true",
                    help="Use the deployed Modal GPU specialist instead of the local model.")
    ap.add_argument("--auth", default=None, metavar="USER:PASS",
                    help="Protect the local app with basic auth (default: none).")
    ap.add_argument("--share", action="store_true",
                    help="Expose a temporary public gradio.live URL.")
    args = ap.parse_args()

    if args.modal_specialist:
        specialist_fn, specialist_label = modal_specialist_backend()
        toggle = "Try the Modal-GPU specialist first"
    else:
        specialist_fn, specialist_label = local_specialist_backend()
        toggle = "Try the local specialist first (loads the model on first use)"

    auth = tuple(args.auth.split(":", 1)) if args.auth else None  # type: ignore[assignment]

    app = copilot.build_ui(
        specialist_fn=specialist_fn,
        specialist_label=specialist_label,
        auth=auth,
        specialist_toggle_label=toggle,
    )
    app.launch(auth=app._geo_auth, share=args.share, allowed_paths=[app._geo_out_dir])


if __name__ == "__main__":
    main()
