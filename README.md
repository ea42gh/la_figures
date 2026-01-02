# la_figures

`la_figures` is an **algorithmic** Python package that computes **trace descriptions** for linear-algebra procedures (e.g., echelon/back-substitution traces) intended to be **rendered by** [`matrixlayout`](https://github.com/ea42gh/matrixlayout).

This package owns the **math** (pivot logic, parameterization, substitution order). It does **not** render figures.

## Relationship to matrixlayout

- **`la_figures`**: computes *what happened* (algorithmic steps, symbolic expressions, traces)
- **`matrixlayout`**: decides *how to draw it* (layout/formatting) and renders via `jupyter_tikz`

This separation keeps the rendering layer stable and declarative while letting algorithms evolve independently.

## Scope

### What `la_figures` does
- Compute solver/trace artifacts for visualization (e.g., back-substitution traces)
- Decide pivots / free variables / parameterization conventions
- Perform substitution and produce “raw” vs “substituted” equations
- Produce data structures that are easy to pass from Julia (PythonCall) or Python

### What `la_figures` does not do
- No LaTeX toolchain management
- No TikZ/SVG rendering
- No layout decisions (spacing, braces, arrows, etc.)
- No dependency on `jupyter_tikz` (rendering belongs to `matrixlayout`)

## Installation

Development install (recommended):

```bash
pip install -e ".[dev]"
