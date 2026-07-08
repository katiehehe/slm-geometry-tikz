"""Evaluation harness: score one (or many) model outputs against ground truth.

Per output we compute:
  - figure_only   : output is just the TikZ figure (no prose)
  - compiles      : emitted TikZ compiles
  - ssim / mse    : render-and-diff vs the ground-truth render
  - coord_*       : coordinate assertion vs known ground-truth points
  - judge         : optional LLM-as-judge rubric scores
  - passed        : the falsifiable gate (figure_only & compiles & coords all correct)
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from . import judge as judge_mod
from . import metrics, tex


@dataclass
class OutputEval:
    id: int
    figure_only: bool
    has_tikz: bool
    compiles: bool
    compile_reason: str
    ssim: float | None
    mse: float | None
    coord_accuracy: float
    coords_all_correct: bool
    passed: bool
    judge: dict = field(default_factory=dict)


def evaluate_one(
    ex_id: int,
    description: str,
    model_output: str,
    gt_tikz: str,
    gt_points: dict[str, list[float]],
    atol: float = 0.05,
    use_judge: bool = False,
    render: bool = True,
) -> OutputEval:
    pred_tikz = metrics.extract_tikz(model_output)
    figure_only = metrics.is_figure_only(model_output)
    has_tikz = pred_tikz is not None

    compiles, reason = False, "no-tikz"
    ssim_v = mse_v = None

    if has_tikz:
        cr = tex.compile_tikz(pred_tikz)
        compiles, reason = cr.ok, cr.reason
        if compiles and render and cr.pdf_path:
            gt = tex.compile_tikz(gt_tikz)
            if gt.ok and gt.pdf_path:
                try:
                    pred_img = tex.render_pdf(cr.pdf_path)
                    gt_img = tex.render_pdf(gt.pdf_path)
                    ssim_v = metrics.ssim(pred_img, gt_img)
                    mse_v = metrics.mse(pred_img, gt_img)
                except Exception:  # noqa: BLE001 - render is best-effort
                    pass

    cm = metrics.coord_match(pred_tikz or "", gt_points, atol=atol)
    passed = bool(figure_only and compiles and cm["all_correct"])

    j = judge_mod.judge(description, model_output) if use_judge else {}

    return OutputEval(
        id=ex_id,
        figure_only=figure_only,
        has_tikz=has_tikz,
        compiles=compiles,
        compile_reason=reason,
        ssim=ssim_v,
        mse=mse_v,
        coord_accuracy=cm["accuracy"],
        coords_all_correct=cm["all_correct"],
        passed=passed,
        judge=j,
    )


def aggregate(results: list[OutputEval]) -> dict:
    n = len(results) or 1
    ssims = [r.ssim for r in results if r.ssim is not None]
    judged = [r.judge for r in results if r.judge and not r.judge.get("skipped")]

    summary = {
        "n": len(results),
        "figure_only_rate": sum(r.figure_only for r in results) / n,
        "compile_rate": sum(r.compiles for r in results) / n,
        "coord_accuracy_mean": sum(r.coord_accuracy for r in results) / n,
        "coords_all_correct_rate": sum(r.coords_all_correct for r in results) / n,
        "pass_rate": sum(r.passed for r in results) / n,
        "ssim_mean": statistics.mean(ssims) if ssims else None,
    }
    if judged:
        for dim in ["spec_adherence", "robustness", "task_quality", "consistency"]:
            vals = [j.get(dim) for j in judged if isinstance(j.get(dim), (int, float))]
            summary[f"judge_{dim}_mean"] = statistics.mean(vals) if vals else None
    return summary


def to_dicts(results: list[OutputEval]) -> list[dict]:
    return [asdict(r) for r in results]
