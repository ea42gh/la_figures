# Overview

la_figures implements algorithmic logic and produces layout specs consumed by
*matrixlayout*. It does not render TeX directly unless using the convenience
wrappers that delegate to *matrixlayout* and *jupyter_tikz*.

## Layering

- *la_figures*: algorithms and spec construction.
- *matrixlayout*: formatting, layout, TeX emission, SVG rendering.
- *jupyter_tikz*: toolchain execution and SVG conversion.

## Package boundary

- Use `la_figures` when you want algorithmic outputs (traces, specs).
- Use `matrixlayout` when you want layout/rendering of a known spec or matrix grid.

## Data flow

Input matrices → algorithm trace → spec dictionary → matrixlayout renderer.

```
matrices -> la_figures (trace/spec) -> matrixlayout (tex/svg)
```

Example:

```
A -> ge_tbl_spec(A) -> grid_svg(spec["matrices"], specs=spec["specs"])
```

## Module map

- GE: `ge_trace`, `ge_tbl_spec`, `ge_tbl_svg`.
- QR: `compute_qr_matrices`, `qr_tbl_spec`, `qr_tbl_svg`.
- Eigen/SVD: `eig_tbl_spec`, `svd_tbl_spec`.
- Backsubstitution: `linear_system_tex`, `backsubstitution_tex`, `standard_solution_tex`.

## Pitfalls

- QR with rank‑deficient `W` requires `rank_deficient` or `allow_rank_deficient`.
- SVD tables may include zero singular values; set digits to control formatting.

## Common outputs

- `spec` dicts for GE/QR/EIG/SVD.
- `systeme`/`cascade`/solution TeX blocks for backsubstitution.
