# API

Key entry points (see module docstrings for details).

## GE

- `la_figures.ge_trace(A, **opts)`: generate a GE trace with pivots and steps. Inputs: matrix-like `A`. Returns: trace dict. Use when you need step-by-step elimination.
- `la_figures.trace_to_layer_matrices(trace)`: convert a trace to matrix layers. Inputs: trace dict. Returns: list of grid matrices.
- `la_figures.ge_tbl_spec(A, **opts)`: build a GE spec for matrixlayout. Inputs: matrix-like `A`. Returns: spec dict.
- `la_figures.ge_tbl_layout_spec(A, **opts)`: build a typed GE layout spec. Inputs: matrix-like `A`. Returns: typed spec.
- `la_figures.ge_tbl_bundle(A, **opts)`: build a spec plus TeX/submatrix metadata. Returns: `(spec, span_map, tex)`.

Example options:
`ge_tbl_spec(A, show_pivots=True, pivoting="partial", gj=False)`

## QR

- `la_figures.compute_qr_matrices(A, W)`: compute the QR matrix grid. Returns: list of grid matrices. Use when you already have `W`.
- `la_figures.gram_schmidt_qr_matrices(A, **opts)`: compute QR grid with rank-deficient handling. Returns: list of grid matrices.
- `la_figures.qr_tbl_spec(A, W, **opts)`: build a QR spec for matrixlayout. Returns: spec dict.
- `la_figures.qr_tbl_layout_spec(A, W, **opts)`: build a typed QR layout spec. Returns: typed spec.

Example options:
`qr_tbl_spec(A, W, array_names=True, rank_deficient=True)`

## Eigen/SVD

- `la_figures.eig_tbl_spec(A, **opts)`: build eigenproblem table specs. Returns: spec dict.
- `la_figures.svd_tbl_spec(A, **opts)`: build SVD table specs. Returns: spec dict.
- `la_figures.svd_tbl_spec_from_right_singular_vectors(V, **opts)`: build SVD specs from given vectors. Returns: spec dict.

## Backsubstitution

- `la_figures.linear_system_tex(A, b, **opts)`: build a `systeme` block for a linear system. Returns: TeX string.
- `la_figures.backsubstitution_tex(A, b, **opts)`: build cascade lines for back-substitution. Returns: list of `\\ShortCascade` lines.
- `la_figures.standard_solution_tex(A, b, **opts)`: build a standard-form solution block. Returns: TeX string.

Example return snippet:
`linear_system_tex(A, b)` returns a `\\systeme{...}` string.
