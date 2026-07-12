"""Geometry Figure Copilot — HOSTED on Modal (custom website + GPU specialist).

  modal serve  scripts/copilot_modal.py                 # ephemeral dev URL (live reload)
  modal deploy scripts/copilot_modal.py                 # PERSISTENT public URL
  modal run    scripts/copilot_modal.py::test_specialist  # smoke-test the GPU specialist
  modal app stop geotikz-copilot                        # tear the deployment down

This ships the copilot as a **custom** website (FastAPI + static SPA) with a
GPU-served LOCAL SPECIALIST — no Gradio chrome:

  1. ``Specialist`` — a GPU class (A10G) that loads the fine-tuned specialist ONCE
     (base ``Qwen/Qwen3-4B`` + LoRA ``qwen3-illustrator-4b`` from the Volume
     ``geotikz-outputs``; falls back to the 1.7B then 0.6B adapter if the 4B one
     is missing) and exposes ``generate(description) -> {tikz, adapter, ...}``.
     Uses the specialist's EXACT training prompt with ``enable_thinking=False``.
     ``scaledown_window`` keeps the container warm so we don't pay a cold start
     on every request.

  2. ``web`` — FastAPI JSON/multipart APIs + the static chat SPA under ``web/``.
     ``specialist_fn`` is wired to the GPU class, so a winning specialist figure is
     attributed "qwen3-illustrator-4b (specialist · Modal GPU)". Gateway creds
     live in a Modal Secret; HTTP Basic auth is ON by default.

Everything is ADDITIVE: it imports the repo's ``src`` (mounted into the web
image) and reuses the same routing/render/attribution core as the local app.

--- Auth ---------------------------------------------------------------------
Stateless HTTP **Basic auth** (ASGI middleware). Creds come from the
``geotikz-copilot`` Secret (COPILOT_USER / COPILOT_PASSWORD; default demo/geotikz).
  * change:  modal secret create geotikz-copilot COPILOT_USER=you COPILOT_PASSWORD=... \\
                 OPENAI_BASE_URL=... OPENAI_API_KEY=... JUDGE_MODEL=... --force
  * disable: add COPILOT_AUTH=off to that Secret (or the env) and redeploy.

--- Cost / access ------------------------------------------------------------
The public URL spends BOTH gateway budget (frontier text/vision/edit calls) and
Modal GPU time (the specialist). Auth gates access; ``modal app stop`` ends all
spend. The GPU scales to zero after ``scaledown_window`` idle (a cold start adds
model-load latency to the next specialist call).
"""

from __future__ import annotations

import base64

import modal

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
APP_NAME = "geotikz-copilot"
GPU = "A10G"  # 24GB: fits Qwen3-4B bf16 + KV for single-request inference.
SECRET_NAME = "geotikz-copilot"
OUT_DIR = "/tmp/geotikz-copilot"
CACHE_DIR = "/root/.tectonic-cache"

ADAPTERS = [
    ("qwen3-illustrator-4b-v2", "Qwen/Qwen3-4B", "construction"),
    ("qwen3-illustrator-4b", "Qwen/Qwen3-4B", "construction"),
    ("qwen3-illustrator", "Qwen/Qwen3-1.7B", "construction"),
    ("qwen3-pgf-geotikz", "Qwen/Qwen3-0.6B", "narrow"),
]

CONSTRUCTION_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler for olympiad constructions. You are given "
    "a geometry scene: some base points are given by coordinates, and one or more "
    "further points are described only by their geometric construction (e.g. the "
    "circumcenter, incenter, orthocenter, centroid, the foot of an altitude, where "
    "an angle bisector meets a side, a midpoint, or a point of tangency).\n\n"
    "Output ONE TikZ figure that realizes the scene and defines every requested "
    "named point at its correct location. You may work either way:\n"
    "  - compute the coordinates yourself and place them, e.g. "
    "\\coordinate (O) at (2.5,1.375); , or\n"
    "  - use coordinate-free constructions. The full tkz-euclide package and the "
    "tikz libraries calc, intersections, through, angles, positioning are ALREADY "
    "loaded, so macros like \\tkzDefPoint(0,0){A}, "
    "\\tkzDefTriangleCenter[circum](A,B,C)\\tkzGetPoint{O}, "
    "\\tkzDefTriangleCenter[in]/[ortho]/[centroid], "
    "\\tkzDefPointBy[projection=onto B--C](A)\\tkzGetPoint{F}, "
    "\\tkzDefMidPoint(B,C)\\tkzGetPoint{M}, \\tkzDefLine[bisector](B,A,C), "
    "\\tkzDefTangent[from = P](O,W)\\tkzGetPoints{T1}{T2}, and PGF calc "
    "($(a)!(c)!(b)$) are all available.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "  1. Every requested point MUST be a referenceable named coordinate/node using "
    "the EXACT name requested (case-sensitive), created by any of: "
    "\\coordinate (NAME) at (...);  \\tkzDefPoint(...){NAME}  \\tkzGetPoint{NAME}  "
    "or \\node (NAME) at (...) {}. Do not rename points.\n"
    "  2. Output ONLY the figure, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. Do NOT include \\documentclass, \\usepackage, or "
    "\\begin{document} — only the tikzpicture. No prose, no explanations, no "
    "markdown fences."
)
NARROW_SYSTEM_PROMPT = (
    "You are a geometry-to-TikZ compiler. Given a geometry scene described only "
    "through relationships and constraints (no explicit coordinates), you must "
    "derive the exact coordinates yourself and output a single valid TikZ/PGF "
    "figure that compiles and renders the described geometry. "
    "Output ONLY the TikZ code, starting with \\begin{tikzpicture} and ending "
    "with \\end{tikzpicture}. No prose, no explanations, no markdown fences."
)
USER_TEMPLATE = "Scene:\n{description}\n\nReturn the TikZ figure."

PREWARM_TEX = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{tkz-euclide}
\usetikzlibrary{calc,angles,quotes,intersections,through,positioning,arrows,arrows.meta,%
decorations.markings,decorations.pathreplacing,shapes.geometric,shapes.misc,patterns,%
patterns.meta,backgrounds,fit,math,3d,perspective}
\begin{document}
\begin{tikzpicture}
\tkzDefPoint(0,0){A}\tkzDefPoint(4,0){B}\tkzDefPoint(1,3){C}
\tkzDefTriangleCenter[circum](A,B,C)\tkzGetPoint{O}
\tkzDrawPolygon(A,B,C)\tkzDrawCircle(O,A)
\tkzDrawPoints(A,B,C,O)\tkzLabelPoints(A,B,C,O)
\draw ($(A)!0.5!(B)$) circle (2pt);
\end{tikzpicture}
\end{document}
"""

app = modal.App(APP_NAME)
outputs_vol = modal.Volume.from_name("geotikz-outputs", create_if_missing=True)
data_vol = modal.Volume.from_name("geotikz-copilot-data", create_if_missing=True)

gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.12.1",
    "transformers>=5.13.0",
    "peft>=0.19.1",
    "accelerate>=1.14.0",
)

_prewarm_b64 = base64.b64encode(PREWARM_TEX.encode()).decode()
web_image = (
    modal.Image.micromamba(python_version="3.12")
    .micromamba_install("tectonic", channels=["conda-forge"])
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "uvicorn[standard]",
        "openai>=2.44.0",
        "pymupdf>=1.28.0",
        "pillow>=12.3.0",
        "numpy>=2.5.1",
        "python-dotenv>=1.2.2",
        "pydantic>=2.0.0",
    )
    .env({"TECTONIC_CACHE_DIR": CACHE_DIR})
    .run_commands(
        "tectonic --version",
        f"echo {_prewarm_b64} | base64 -d > /root/warm.tex",
        "cd /root && tectonic -X compile --outfmt pdf warm.tex && echo PREWARM_DONE",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("web", remote_path="/root/web")
)


# --------------------------------------------------------------------------- #
# GPU specialist (loaded once, kept warm)
# --------------------------------------------------------------------------- #
@app.cls(
    image=gpu_image,
    gpu=GPU,
    volumes={"/outputs": outputs_vol},
    scaledown_window=300,
    timeout=600,
)
class Specialist:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        chosen = next(((a, b, m) for a, b, m in ADAPTERS if os.path.isdir(f"/outputs/{a}")), None)
        if chosen is None:
            raise RuntimeError("no specialist adapter found on Volume 'geotikz-outputs'")
        self.adapter, self.base, self.mode = chosen

        tok = AutoTokenizer.from_pretrained(self.base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.base, dtype=torch.bfloat16)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"/outputs/{self.adapter}")
        self.model = model.to("cuda").eval()
        self.tok = tok
        print(f"[specialist] loaded {self.adapter} on {self.base} (mode={self.mode})")

    @modal.method()
    def info(self) -> dict:
        return {"adapter": self.adapter, "base": self.base, "mode": self.mode}

    @modal.method()
    def generate(self, description: str, max_new_tokens: int = 640) -> dict:
        import time

        import torch

        sys_prompt = CONSTRUCTION_SYSTEM_PROMPT if self.mode == "construction" else NARROW_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(description=description)},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            g = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = g[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return {"tikz": text, "adapter": self.adapter, "base": self.base,
                "latency_s": round(time.time() - t0, 3)}


# --------------------------------------------------------------------------- #
# web endpoint (custom FastAPI + static SPA)
# --------------------------------------------------------------------------- #
@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={"/examples": data_vol},
    max_containers=1,
    scaledown_window=600,
    timeout=600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path.insert(0, "/root/src")

    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.pop("SSL_CERT_DIR", None)

    from geotikz.webapp import create_app

    os.makedirs(OUT_DIR, exist_ok=True)

    specialist = Specialist()
    holder = {"label": "`qwen3-illustrator-4b-v2` (specialist · Modal GPU)"}

    def specialist_fn(description: str) -> str:
        res = specialist.generate.remote(description)
        if isinstance(res, dict):
            holder["label"] = f"`{res.get('adapter', 'specialist')}` (specialist · Modal GPU)"
            return res.get("tikz", "")
        return res or ""

    user = os.environ.get("COPILOT_USER", "demo")
    pwd = os.environ.get("COPILOT_PASSWORD", "geotikz")
    auth_off = os.environ.get("COPILOT_AUTH", "on").strip().lower() in ("off", "0", "false", "no")

    return create_app(
        specialist_fn=specialist_fn,
        specialist_label=(lambda: holder["label"]),
        out_dir=OUT_DIR,
        static_dir="/root/web",
        specialist_default=True,
        specialist_toggle_label="Use the GPU specialist first (Modal)",
        examples_store_path="/examples/user_examples.json",
        commit_examples=data_vol.commit,
        auth_user=None if auth_off else user,
        auth_password=None if auth_off else pwd,
        title="Geometry Figure Copilot",
    )


@app.function(image=web_image, secrets=[modal.Secret.from_name(SECRET_NAME)])
def _gw_selftest() -> None:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.pop("SSL_CERT_DIR", None)
    from geotikz import gateway

    msgs = [{"role": "user", "content": "Reply with exactly: hi"}]
    for m in ["openai-group/gpt-5.5", "openai-group/gpt-4o"]:
        r = gateway.chat(msgs, m, max_tokens=50, retries=2)
        print(f"[{m}] ok={r.ok} attempts={r.attempts} lat={r.latency_s}s err={str(r.error)[:160]}")


@app.local_entrypoint()
def gw_selftest() -> None:
    _gw_selftest.remote()


@app.local_entrypoint()
def test_specialist(
    description: str = "Triangle ABC with vertices A=(0,0), B=(6,0), C=(2,5). "
                       "Let O be the circumcenter of triangle ABC and draw its circumcircle. "
                       "Define the named points A, B, C, O.",
) -> None:
    import time

    spec = Specialist()
    t0 = time.time()
    res = spec.generate.remote(description)
    wall = round(time.time() - t0, 2)
    print("=" * 70)
    print(f"adapter loaded : {res['adapter']}  (base {res['base']})")
    print(f"gpu latency    : {res['latency_s']}s")
    print(f"wall (inc. cold-start if any): {wall}s")
    print("-" * 70)
    print(res["tikz"][:600])
    print("=" * 70)
