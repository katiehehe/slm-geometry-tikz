"""Minimal web UI for the specialist: scene text in -> figure + copyable TikZ out.

  uv run python scripts/demo_web.py            # then open the printed URL

The specialist (Qwen3-0.6B + qwen3-pgf-geotikz LoRA) is loaded locally once and
reused across requests. It emits coordinate-free PGF constructions, compiled with
tectonic and shown as a rendered figure. Optionally route to a frontier model
(construction prompt) as a fallback for out-of-distribution scenes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from geotikz import serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "demo_web"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Loaded lazily on the first request so the server starts instantly.
_SPEC = serve.Specialist()

FRONTIER_MODELS = [
    "openai-group/gpt-5.5",
    "claude-group/claude-opus-4-8",
    "gemini-group/gemini-3.1-pro",
    "openai-group/gpt-4o",
]

EXAMPLES = [
    ["There is a circle centered at the origin with radius 3. Point A lies on the circle at 40 degrees. Point B lies on the circle at 200 degrees. Point M is the midpoint of segment AB."],
    ["There is a circle centered at the origin with radius 2.5. Point A lies on the circle at 113 degrees. Point B lies on the circle at 17 degrees. Point C is the reflection of A across the x-axis. Point D is the foot of the perpendicular from B onto line CA."],
    ["There is a circle centered at the origin with radius 2. Point A on the circle at 90 degrees. Point B on the circle at 210 degrees. Point C on the circle at 330 degrees."],
]


def illustrate(description: str, use_fallback: bool, frontier_model: str):
    description = (description or "").strip()
    if not description:
        return None, "", "Enter a geometry scene description above."

    spec_res = serve.specialist_latency_generate(_SPEC, description)
    stem = serve.dhash(description)
    png = OUT_DIR / f"{stem}.specialist.png"
    r = serve.compile_and_render(spec_res.text, png, dpi=200)
    tikz = serve.metrics.extract_tikz(spec_res.text) or spec_res.text

    if r.ok:
        status = (f"**Specialist** (local, free) - {spec_res.latency_s:.1f}s - "
                  f"compiled, coordinate-free construction.")
        return str(png), tikz, status

    # specialist failed -> optional frontier fallback (construction prompt)
    status = (f"**Specialist** did not produce a usable figure "
              f"(`{r.reason}`) in {spec_res.latency_s:.1f}s - this scene is likely "
              f"out-of-distribution for the narrow specialist.")
    if use_fallback and frontier_model:
        fres = serve.frontier_generate(description, frontier_model, construction=True)
        fpng = OUT_DIR / f"{stem}.frontier.png"
        fr = serve.compile_and_render(fres.text, fpng, dpi=200)
        ftikz = serve.metrics.extract_tikz(fres.text) or fres.text
        if fr.ok:
            status += (f"\n\n**Frontier fallback** (`{frontier_model}`) - "
                       f"{fres.latency_s:.1f}s - compiled.")
            return str(fpng), ftikz, status
        status += (f"\n\n**Frontier fallback** (`{frontier_model}`) also failed "
                   f"(`{fr.reason}`).")
        return None, ftikz, status
    return None, tikz, status + "\n\n_Enable the frontier fallback to try a hosted model._"


def build() -> gr.Blocks:
    with gr.Blocks(title="Geometry -> TikZ specialist") as app:
        gr.Markdown(
            "# Geometry to TikZ - specialist demo\n"
            "A fine-tuned **Qwen3-0.6B** (+ `qwen3-pgf-geotikz` LoRA) that turns a "
            "coordinate-free geometry description into a **coordinate-free PGF/TikZ "
            "construction**, rendered locally. Free, offline, guaranteed-compile on "
            "its in-domain task. Out-of-distribution scenes can fall back to a "
            "frontier model (also prompted for coordinate-free constructions)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                desc = gr.Textbox(label="Geometry scene", lines=6,
                                  placeholder="Describe a scene using relationships, not coordinates ...")
                with gr.Row():
                    use_fb = gr.Checkbox(label="Frontier fallback if specialist fails", value=False)
                    fmodel = gr.Dropdown(FRONTIER_MODELS, value=FRONTIER_MODELS[0],
                                         label="Frontier model")
                btn = gr.Button("Illustrate", variant="primary")
                gr.Examples(EXAMPLES, inputs=[desc])
            with gr.Column(scale=1):
                img = gr.Image(label="Rendered figure", type="filepath")
                code = gr.Code(label="TikZ (copyable)")
                status = gr.Markdown()
        btn.click(illustrate, [desc, use_fb, fmodel], [img, code, status])
    return app


if __name__ == "__main__":
    build().launch()
