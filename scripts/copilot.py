"""Geometry-figure copilot — LOCAL entry (custom website by default).

  uv run python scripts/copilot.py                 # FastAPI + SPA on http://127.0.0.1:7860
  uv run python scripts/copilot.py --gradio        # legacy Gradio UI (deprecated)
  uv run python scripts/copilot.py --modal-specialist   # borrow the Modal GPU specialist
  uv run python scripts/copilot.py --auth me:secret     # password-protect the local app

All the routing / rendering / attribution logic lives in ``geotikz.copilot``
(shared with the hosted Modal app). This script only wires up a SPECIALIST
BACKEND and launches the UI:

  * default            -> the LOCAL base+LoRA specialist (``serve.Specialist``,
    Qwen3-0.6B + qwen3-pgf-geotikz), loaded lazily on first use. Frontier-first
    locally (specialist toggle OFF by default).
  * ``--modal-specialist`` -> route the specialist to the DEPLOYED Modal GPU
    function instead.

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

from geotikz import serve  # noqa: E402

MODAL_APP = "geotikz-copilot"
MODAL_CLS = "Specialist"
ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web"


def local_specialist_backend():
    """The laptop specialist: lazy base+LoRA (torch loads on first specialist use)."""
    spec = serve.Specialist()

    def fn(description: str) -> str:
        return spec.generate(description)

    label = f"`{spec.base} + LoRA` (local specialist)"
    return fn, label


def modal_specialist_backend():
    """Route the specialist to the deployed Modal GPU function (no local load)."""
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
    ap.add_argument("--gradio", action="store_true",
                    help="Launch the legacy Gradio UI instead of the custom website.")
    ap.add_argument("--share", action="store_true",
                    help="(Gradio only) Expose a temporary public gradio.live URL.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if args.modal_specialist:
        specialist_fn, specialist_label = modal_specialist_backend()
        toggle = "Try the Modal-GPU specialist first"
        specialist_default = True
    else:
        specialist_fn, specialist_label = local_specialist_backend()
        toggle = "Try the local specialist first (loads the model on first use)"
        specialist_default = False

    if args.gradio:
        from geotikz import copilot

        auth = tuple(args.auth.split(":", 1)) if args.auth else None  # type: ignore[assignment]
        app = copilot.build_ui(
            specialist_fn=specialist_fn,
            specialist_label=specialist_label,
            auth=auth,
            specialist_toggle_label=toggle,
            specialist_default=specialist_default,
        )
        app.launch(auth=app._geo_auth, share=args.share, allowed_paths=[app._geo_out_dir],
                   css=getattr(app, "_geo_css", None))
        return

    import uvicorn
    from geotikz.webapp import create_app

    user = pwd = None
    if args.auth:
        user, _, pwd = args.auth.partition(":")

    app = create_app(
        specialist_fn=specialist_fn,
        specialist_label=specialist_label,
        static_dir=STATIC,
        specialist_default=specialist_default,
        specialist_toggle_label=toggle,
        auth_user=user,
        auth_password=pwd,
    )
    print(f"Geometry Figure Copilot → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
