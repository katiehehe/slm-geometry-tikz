"""Compile TikZ to PDF (tectonic) and render to a grayscale numpy image (pymupdf).

Everything degrades gracefully: if tectonic isn't installed, compile returns a
CompileResult with ok=False and a reason, so the loop still runs end to end.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STANDALONE_TEMPLATE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,angles,quotes,intersections,through,positioning}
\begin{document}
%s
\end{document}
"""


def has_tectonic() -> bool:
    return shutil.which("tectonic") is not None


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Path | None
    log: str
    reason: str = ""


def wrap_standalone(tikz: str) -> str:
    return STANDALONE_TEMPLATE % tikz


def compile_tikz(tikz: str, workdir: Path | None = None, timeout: int = 60) -> CompileResult:
    if not has_tectonic():
        return CompileResult(False, None, "", reason="tectonic-not-installed")

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="geotikz_"))
    tmp.mkdir(parents=True, exist_ok=True)
    tex_path = tmp / "fig.tex"
    tex_path.write_text(wrap_standalone(tikz))

    try:
        proc = subprocess.run(
            ["tectonic", "-X", "compile", "--outfmt", "pdf", "-o", str(tmp), str(tex_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(False, None, "", reason="timeout")
    except FileNotFoundError:
        return CompileResult(False, None, "", reason="tectonic-not-installed")

    pdf_path = tmp / "fig.pdf"
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode == 0 and pdf_path.exists():
        return CompileResult(True, pdf_path, log)
    return CompileResult(False, None, log, reason=f"exit={proc.returncode}")


def render_pdf(pdf_path: Path, dpi: int = 96, size: int = 256) -> np.ndarray:
    """Render first PDF page to a fixed-size grayscale array in [0,1]."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()

    from PIL import Image

    img = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
