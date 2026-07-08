# Behavior Spec — Spec-First Geometry → TikZ

## The falsifiable gate (one sentence a stranger can grade)

> **Given a constraint-level geometry scene description with no explicit coordinates, the model returns a single valid TikZ/PGF figure that compiles and whose rendered geometry matches the ground-truth construction (points, incidences, and relations correct within tolerance), with no prose before or after the code.**

This one sentence is simultaneously the **data-generation rubric**, the **eval criterion**, and the **thesis**.

## What counts as PASS

An output passes iff **all** of the following hold:

1. **Figure only.** The output is exactly one TikZ figure — begins with `\begin{tikzpicture}` and ends with `\end{tikzpicture}` (optionally wrapped in a `document`), with **no prose** before or after.
2. **Compiles.** The emitted TikZ compiles to a PDF with no errors.
3. **Geometry correct.** Every named point's coordinates match the ground-truth construction within tolerance (default `atol = 0.05` in scene units). Incidences/relations (on-circle, reflection, midpoint, intersection, etc.) therefore hold.
4. **Derived, not transcribed.** The input never stated the coordinates; the model recovered them from the relationships.

## What counts as FAIL

- States coordinates that were not derivable / are wrong.
- Adds prose, explanations, or markdown fences around the code.
- Emits non-compiling TikZ.
- Renders a blank / degenerate figure (accidentally collinear, wrong incidence).
- Solves the wrong task.

## Difficulty dial (why a prompt can't guarantee this)

- **Chain length:** number of derivation steps that must compose (1 = easy, 3+ = base model breaks).
- **Number irregularity:** round numbers are fakeable; non-round (e.g. `A = (4.33, 2.50)` from a 30° point on r=5) forces real computation.

The clean base-vs-tuned delta lives at **multi-step + non-round**.

## Eval rubric (0/1/2 per dimension, reported base vs tuned)

| Dimension | 0 | 1 | 2 |
| :-- | :-- | :-- | :-- |
| Spec adherence | States coords / prose / wrong task | Compiles but geometry off | Valid TikZ, geometry correct, figure only |
| Robustness | Breaks on multi-step / non-round | Wobbles on harder cases | Holds under derivation pressure |
| Task quality | Doesn't compile / garbage | Compiles, roughly right | Clean, correct figure |
| Consistency | Different on similar scenes | Mostly stable | Reliable every time |

**A tuned model that beats the base on Spec adherence and Robustness is a win.**
