"""Spec-first geometry -> TikZ: generator, prompts, and eval harness."""

try:  # best-effort: load .env so gateway config works without manual exports
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from . import extract, generator, judge, metrics, olympiad, prompts, scene, tex

__all__ = ["extract", "generator", "judge", "metrics", "olympiad", "prompts",
           "scene", "tex"]
