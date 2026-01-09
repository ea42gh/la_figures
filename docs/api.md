# API

Key entry points (see module docstrings for details).

## GE

- `la_figures.ge_trace`: generate a GE trace with pivots and steps.
- `la_figures.trace_to_layer_matrices`: convert a trace to matrix layers.
- `la_figures.ge_tbl_spec`: build a GE spec for matrixlayout.
- `la_figures.ge_tbl_layout_spec`: build a typed GE layout spec.
- `la_figures.ge_tbl_bundle`: build a spec plus TeX/submatrix metadata.

## QR

- `la_figures.compute_qr_matrices`: compute the legacy QR matrix grid.
- `la_figures.gram_schmidt_qr_matrices`: compute QR grid with rank‑deficient handling.
- `la_figures.qr_tbl_spec`: build a QR spec for matrixlayout.
- `la_figures.qr_tbl_layout_spec`: build a typed QR layout spec.

## Eigen/SVD

- `la_figures.eig_tbl_spec`: build eigenproblem table specs.
- `la_figures.svd_tbl_spec`: build SVD table specs.
- `la_figures.svd_tbl_spec_from_right_singular_vectors`: build SVD specs from given vectors.
