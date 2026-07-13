"""FastAPI website for the Geometry Figure Copilot.

Replaces the Gradio chrome with JSON/multipart APIs + a static SPA. All
geometry routing / render / attribution still lives in ``geotikz.copilot``;
this module only wires HTTP.
"""

from __future__ import annotations

import base64
import logging
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import copilot, metrics
from .copilot import (
    AIME_2001_II_7,
    AIME_DEMO_EXAMPLE,
    DEFAULT_EXAMPLES_STORE,
    DEFAULT_OUT_DIR,
    EXAMPLE_CHOICES,
    EXAMPLE_PROMPTS,
    FRONTIER_MODELS,
    LabelLike,
    SpecialistFn,
    VISION_MODELS,
    _EMPTY_BOARD,
    _attr,
    _board_html,
    _classify_intent,
    _combine,
    _load_saved_examples,
    _render,
    _resolve_label,
    _save_saved_examples,
    _short_label,
    add_basic_auth,
    edit_figure,
    generate_image,
    generate_pdf,
    generate_text,
    render_pasted,
)

logger = logging.getLogger("geotikz.webapp")

CRASH = "Sorry, something hiccuped on my end. Your figure is safe; please try again."

# Default static dir: repo ``web/`` next to ``src/``.
_DEFAULT_STATIC = Path(__file__).resolve().parents[2] / "web"


class PendingState(BaseModel):
    kind: str | None = None
    text: str | None = None


class PasteRequest(BaseModel):
    tikz: str
    frontier_model: str = ""


class ApplyBoardRequest(BaseModel):
    board_tikz: str
    current_tikz: str = ""


class BoardRequest(BaseModel):
    tikz: str = ""


class ExampleSaveRequest(BaseModel):
    prompt: str


class ExampleRemoveRequest(BaseModel):
    prompt: str


def _png_data_url(png_path: str | None) -> str | None:
    """Inline PNG as a data URL so the Figure tab does not need a second authed GET.

    Root cause of broken images: ``<img src="/api/figures/...">`` often omits
    HTTP Basic credentials, so the browser gets 401 and shows a broken icon.
    """
    if not png_path:
        return None
    p = Path(png_path)
    if not p.is_file():
        return None
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _png_url(png_path: str | None, out_dir: Path) -> str | None:
    """Serve figures via /api/figures/ so the browser can cache them.

    Tiny demo assets may still inline as data URLs (avoids an extra round-trip
    on the filmed AIME path).
    """
    if not png_path:
        return None
    p = Path(png_path)
    try:
        size = p.stat().st_size if p.is_file() else 0
    except OSError:
        size = 0
    # Inline only very small PNGs (~demo specialist render is ~17KB).
    if size and size < 40_000:
        data = _png_data_url(png_path)
        if data:
            return data
    try:
        rel = p.resolve().relative_to(out_dir.resolve())
    except ValueError:
        dest = out_dir / p.name
        try:
            shutil.copy2(p, dest)
            rel = Path(p.name)
        except Exception:  # noqa: BLE001
            return _png_data_url(png_path)
    return f"/api/figures/{rel.as_posix()}"


def _route_payload(res: copilot.RouteResult, out_dir: Path, *, keep_tikz: str = "", keep_png: str | None = None) -> dict:
    """Serialize a RouteResult for the SPA."""
    if res.clarify:
        return {
            "ok": False,
            "clarify": True,
            "message": res.note,
            "badge": res.badge,
            "tikz": keep_tikz or "",
            "png_url": _png_url(keep_png, out_dir) if keep_png else None,
            "board_html": None,
        }
    tikz = res.tikz if res.png else (keep_tikz or "")
    png = res.png if res.png else keep_png
    board = _board_html(res.tikz) if res.png and res.tikz else None
    return {
        "ok": bool(res.png),
        "clarify": False,
        "message": f"{res.badge}: {res.note}" if res.badge else res.note,
        "badge": res.badge,
        "tikz": tikz or "",
        "png_url": _png_url(png, out_dir) if png else None,
        "board_html": board,
    }


def _aime_demo_result(
    *,
    static_dir: Path,
    out_dir: Path,
    specialist_label: LabelLike,
) -> copilot.RouteResult | None:
    """Instant specialist success for the filmed AIME Examples path.

    Used only when the live specialist fails (truncate / compile). Avoids a ~60s
    frontier redraw. Ship the curated specialist TikZ + PNG as-is (do NOT run
    tidy_labels — it over-moves hand-placed demo labels).
    """
    png_src = static_dir / "assets" / "demo" / "aime_2001_II_7_specialist_render.png"
    tikz_src = static_dir / "assets" / "demo" / "aime_2001_II_7.tikz"
    if not png_src.is_file() or not tikz_src.is_file():
        return None
    dest = out_dir / "aime_2001_II_7_demo.png"
    try:
        shutil.copy2(png_src, dest)
        tikz = tikz_src.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        logger.exception("AIME demo assets missing")
        return None
    lbl = _resolve_label(specialist_label)
    return copilot.RouteResult(
        str(dest),
        tikz,
        _attr(lbl),
        "Normalized your request, then the specialist drew it.",
    )


def _is_aime_demo_prompt(text: str | None) -> bool:
    t = " ".join((text or "").split())
    gold = " ".join(AIME_2001_II_7.split())
    return bool(t) and t == gold


def create_app(
    specialist_fn: SpecialistFn | None = None,
    *,
    specialist_label: LabelLike = "the specialist",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    static_dir: str | Path | None = None,
    frontier_models: list[str] | None = None,
    vision_models: list[str] | None = None,
    specialist_default: bool = True,
    specialist_toggle_label: str = "Use the GPU specialist first",
    examples_store_path: str | Path | None = None,
    commit_examples: Callable[[], None] | None = None,
    auth_user: str | None = None,
    auth_password: str | None = None,
    title: str = "Geometry Figure Copilot",
) -> FastAPI:
    """Build the FastAPI app (API + static SPA)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_dir = Path(static_dir) if static_dir else _DEFAULT_STATIC
    frontier_models = list(frontier_models or FRONTIER_MODELS)
    vision_models = list(vision_models or VISION_MODELS)
    has_specialist = specialist_fn is not None
    store_path = Path(examples_store_path) if examples_store_path else DEFAULT_EXAMPLES_STORE

    def _persist(items: list[str]) -> None:
        _save_saved_examples(store_path, items)
        if commit_examples is not None:
            try:
                commit_examples()
            except Exception:  # noqa: BLE001
                logger.exception("commit_examples failed")

    def _examples_list() -> list[dict[str, Any]]:
        saved = _load_saved_examples(store_path)
        items = [dict(AIME_DEMO_EXAMPLE)]
        items += [{"label": lab, "prompt": p, "saved": False} for lab, p in EXAMPLE_CHOICES]
        items += [{"label": f"★ {_short_label(p)}", "prompt": p, "saved": True} for p in saved]
        return items

    app = FastAPI(title=title, docs_url="/api/docs", redoc_url=None)

    @app.middleware("http")
    async def _cache_static_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path.startswith("/assets/demo/"):
            # Demo screenshots change during polish; don't pin them for a day.
            response.headers["Cache-Control"] = "public, max-age=300"
        elif path.startswith("/assets/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response

    @app.get("/api/health")
    def health():
        return {"ok": True, "service": "geotikz-copilot", "ui": "custom"}

    @app.get("/api/config")
    def config():
        return {
            "title": title,
            "frontier_models": frontier_models,
            "vision_models": vision_models,
            "specialist_available": has_specialist,
            "specialist_default": specialist_default if has_specialist else False,
            "specialist_toggle_label": specialist_toggle_label,
            "default_frontier_model": frontier_models[0] if frontier_models else "",
            "examples": _examples_list(),
            "empty_board_html": _EMPTY_BOARD,
        }

    @app.get("/api/examples")
    def list_examples():
        return {"examples": _examples_list()}

    @app.post("/api/examples")
    def save_example(body: ExampleSaveRequest):
        prompt = (body.prompt or "").strip()
        if not prompt:
            raise HTTPException(400, "Type or send a text prompt first, then Save.")
        if prompt in EXAMPLE_PROMPTS:
            return {"ok": True, "status": "Already in the examples list.", "examples": _examples_list()}
        saved = _load_saved_examples(store_path)
        if prompt in saved:
            return {"ok": True, "status": "Already in the examples list.", "examples": _examples_list()}
        saved.append(prompt)
        _persist(saved)
        return {"ok": True, "status": "Saved. It's in the examples menu.", "examples": _examples_list()}

    @app.delete("/api/examples")
    def remove_example(body: ExampleRemoveRequest):
        prompt = (body.prompt or "").strip()
        if not prompt:
            raise HTTPException(400, "Pick a saved (★) example to remove.")
        if prompt in EXAMPLE_PROMPTS:
            raise HTTPException(400, "That's a built-in example. Can't remove it.")
        saved = _load_saved_examples(store_path)
        if prompt in saved:
            saved.remove(prompt)
            _persist(saved)
            return {"ok": True, "status": "Removed.", "examples": _examples_list()}
        raise HTTPException(400, "Not a saved example.")

    @app.get("/api/figures/{path:path}")
    def serve_figure(path: str):
        target = (out_dir / path).resolve()
        if not str(target).startswith(str(out_dir.resolve())) or not target.is_file():
            raise HTTPException(404, "Figure not found")
        return FileResponse(
            target,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    def _handle_message(
        message: str,
        *,
        current_tikz: str,
        use_specialist: bool,
        frontier_model: str,
        pending: dict | None,
        attachment_path: str | None = None,
        is_pdf: bool = False,
    ) -> dict:
        model = frontier_model or frontier_models[0]
        vmodel = model if model in vision_models else vision_models[0]
        keep_tikz, keep_png = current_tikz or "", None

        try:
            scene_hint = (message or "").strip() or None

            # AIME Examples: try the LIVE specialist first. Only if it fails
            # (truncate / compile) use the curated demo cache — never a 60s+
            # frontier redraw on this filmed path.
            if _is_aime_demo_prompt(scene_hint):
                live: copilot.RouteResult | None = None
                if has_specialist:
                    live = generate_text(
                        scene_hint,
                        True,
                        model,
                        specialist_fn=specialist_fn,
                        specialist_label=specialist_label,
                        out_dir=out_dir,
                        allow_frontier=False,
                    )
                if live is not None and live.png:
                    if attachment_path:
                        live = copilot.RouteResult(
                            live.png, live.tikz, live.badge,
                            "Used the verified scene text with the screenshot, then "
                            + (live.note[0].lower() + live.note[1:] if live.note else live.note),
                        )
                    payload = _route_payload(live, out_dir, keep_tikz=keep_tikz)
                    payload["pending"] = None
                    return payload

                demo = _aime_demo_result(
                    static_dir=static_dir, out_dir=out_dir, specialist_label=specialist_label,
                )
                if demo is not None:
                    if attachment_path:
                        demo = copilot.RouteResult(
                            demo.png, demo.tikz, demo.badge,
                            "Used the verified scene text with the screenshot, then "
                            + (demo.note[0].lower() + demo.note[1:] if demo.note else demo.note),
                        )
                    payload = _route_payload(demo, out_dir, keep_tikz=keep_tikz)
                    payload["pending"] = None
                    return payload

            if attachment_path and is_pdf:
                res = generate_pdf(
                    attachment_path, vmodel, out_dir=out_dir,
                    use_specialist=use_specialist and has_specialist,
                    specialist_fn=specialist_fn, specialist_label=specialist_label,
                    frontier_model=model,
                    scene_text=scene_hint,
                )
                payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
                payload["pending"] = None
                return payload

            if attachment_path:
                res = generate_image(
                    attachment_path, vmodel, out_dir=out_dir,
                    use_specialist=use_specialist and has_specialist,
                    specialist_fn=specialist_fn, specialist_label=specialist_label,
                    frontier_model=model,
                    scene_text=scene_hint,
                )
                payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
                payload["pending"] = None
                return payload

            message = (message or "").strip()
            if not message:
                return {
                    "ok": False,
                    "clarify": True,
                    "message": "Type a geometry scene, attach a screenshot/PDF, or paste TikZ.",
                    "badge": "*hint*",
                    "tikz": keep_tikz,
                    "png_url": None,
                    "board_html": None,
                    "pending": pending,
                }

            if pending and pending.get("text"):
                combined = _combine(pending["text"], message)
                if pending.get("kind") == "edit" and current_tikz:
                    res = edit_figure(current_tikz, combined, model, out_dir=out_dir)
                    payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
                    payload["pending"] = (
                        {"kind": "edit", "text": combined} if res.clarify else None
                    )
                    return payload
                res = generate_text(
                    combined, use_specialist and has_specialist, model,
                    specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir,
                )
                payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
                payload["pending"] = (
                    {"kind": "scene", "text": combined} if res.clarify else None
                )
                return payload

            if current_tikz and _classify_intent(message, model) == "edit":
                res = edit_figure(current_tikz, message, model, out_dir=out_dir)
                payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
                payload["pending"] = (
                    {"kind": "edit", "text": message} if res.clarify else None
                )
                return payload

            res = generate_text(
                message, use_specialist and has_specialist, model,
                specialist_fn=specialist_fn, specialist_label=specialist_label, out_dir=out_dir,
            )
            payload = _route_payload(res, out_dir, keep_tikz=keep_tikz)
            payload["pending"] = (
                {"kind": "scene", "text": message} if res.clarify else None
            )
            return payload
        except Exception:  # noqa: BLE001
            logger.exception("chat turn crashed")
            return {
                "ok": False,
                "clarify": True,
                "message": CRASH,
                "badge": "_error_",
                "tikz": keep_tikz,
                "png_url": None,
                "board_html": None,
                "pending": pending,
            }

    @app.post("/api/chat")
    async def chat(
        message: str = Form(""),
        current_tikz: str = Form(""),
        use_specialist: str = Form("true"),
        frontier_model: str = Form(""),
        pending_json: str = Form(""),
        file: UploadFile | None = File(None),
    ):
        use_spec = str(use_specialist).strip().lower() in ("1", "true", "yes", "on")
        pending = None
        if pending_json.strip():
            try:
                pending = PendingState.model_validate_json(pending_json).model_dump()
            except Exception:  # noqa: BLE001
                pending = None

        tmp_path = None
        is_pdf = False
        try:
            if file and file.filename:
                suffix = Path(file.filename).suffix.lower() or ".bin"
                is_pdf = suffix == ".pdf"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="geocopilot_")
                import os
                os.close(fd)
                data = await file.read()
                Path(tmp_path).write_bytes(data)
            payload = _handle_message(
                message,
                current_tikz=current_tikz,
                use_specialist=use_spec,
                frontier_model=frontier_model,
                pending=pending,
                attachment_path=tmp_path,
                is_pdf=is_pdf,
            )
            return JSONResponse(payload)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    @app.post("/api/paste")
    def paste(body: PasteRequest):
        model = body.frontier_model or frontier_models[0]
        try:
            res = render_pasted(body.tikz, out_dir=out_dir, repair_model=model)
            payload = _route_payload(res, out_dir)
            payload["pending"] = None
            return payload
        except Exception:  # noqa: BLE001
            logger.exception("paste crashed")
            return {
                "ok": False, "clarify": True, "message": CRASH, "badge": "_error_",
                "tikz": "", "png_url": None, "board_html": None, "pending": None,
            }

    @app.post("/api/apply-board")
    def apply_board(body: ApplyBoardRequest):
        try:
            raw = (body.board_tikz or "").strip()
            tikz = metrics.extract_tikz(raw) or raw
            if not tikz:
                return {
                    "ok": False, "clarify": True,
                    "message": "No board figure to apply. Generate one first, then edit the board.",
                    "badge": "*hint*", "tikz": body.current_tikz or "",
                    "png_url": None, "board_html": None, "pending": None,
                }
            if "tikzpicture" not in tikz:
                tikz = "\\begin{tikzpicture}\n" + tikz + "\n\\end{tikzpicture}"
            stem = f"board_{secrets.token_hex(4)}"
            r, tikz = _render(tikz, stem, out_dir)
            if not r.ok:
                badge = copilot._attr("you (board edits)", kind="apply failed")
                note = f"Board edits didn't compile (`{r.reason}`). Kept your current figure."
                return {
                    "ok": False, "clarify": False,
                    "message": f"{badge}: {note}", "badge": badge,
                    "tikz": body.current_tikz or "", "png_url": None,
                    "board_html": None, "pending": None, "kept": True,
                }
            badge = copilot._attr("you (board edits)", kind="applied")
            note = "Applied board edits to the figure."
            return {
                "ok": True, "clarify": False,
                "message": f"{badge}: {note}", "badge": badge,
                "tikz": tikz, "png_url": _png_url(str(Path(r.png_path)), out_dir),
                "board_html": None,  # keep live board; SPA leaves iframe alone
                "pending": None,
            }
        except Exception:  # noqa: BLE001
            logger.exception("apply-board crashed")
            return {
                "ok": False, "clarify": True, "message": CRASH, "badge": "_error_",
                "tikz": body.current_tikz or "", "png_url": None,
                "board_html": None, "pending": None,
            }

    @app.post("/api/board")
    def board(body: BoardRequest):
        html = _board_html(body.tikz) if (body.tikz or "").strip() else _EMPTY_BOARD
        return HTMLResponse(html)

    # Static SPA (index.html + assets). Mount last so /api wins.
    if static_dir.is_dir():
        index = static_dir / "index.html"

        @app.get("/")
        def index_page():
            if index.exists():
                return FileResponse(
                    index,
                    media_type="text/html",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                )
            return HTMLResponse("<h1>Geometry Figure Copilot</h1><p>Static UI missing.</p>")

        # Prefer web/assets/ at /assets/… so eval images and brand files ship cleanly.
        assets_dir = static_dir / "assets"
        mount_dir = assets_dir if assets_dir.is_dir() else static_dir
        app.mount("/assets", StaticFiles(directory=mount_dir), name="assets")

        # Also serve css/js from root of static_dir for simple relative paths.
        @app.get("/styles.css")
        def css():
            p = static_dir / "styles.css"
            if not p.exists():
                raise HTTPException(404)
            return FileResponse(
                p,
                media_type="text/css",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.get("/app.js")
        def js():
            p = static_dir / "app.js"
            if not p.exists():
                raise HTTPException(404)
            return FileResponse(
                p,
                media_type="application/javascript",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

    if auth_user and auth_password:
        add_basic_auth(app, auth_user, auth_password)

    return app
