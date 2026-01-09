# Overview

la_figures implements algorithmic logic and produces layout specs consumed by
*matrixlayout*. It does not render TeX directly unless using the convenience
wrappers that delegate to *matrixlayout* and *jupyter_tikz*.

## Layering

- *la_figures*: algorithms and spec construction.
- *matrixlayout*: formatting, layout, TeX emission, SVG rendering.
- *jupyter_tikz*: toolchain execution and SVG conversion.

## Data flow

Input matrices → algorithm trace → spec dictionary → matrixlayout renderer.

## Pitfalls

- QR with rank‑deficient `W` requires `rank_deficient` or `allow_rank_deficient`.
- SVD tables may include zero singular values; set digits to control formatting.
