# Getting Started

Minimal examples for common entry points. Expected output: spec dictionaries
ready for matrixlayout rendering.

Common defaults:

- `ge_tbl_spec`: `pivoting="none"`, `gj=False`, `show_pivots=False`.
- `qr_tbl_spec`: `array_names=True`.
- `eig_tbl_spec`: `normal=False`.

## GE spec

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
spec = la_figures.ge_tbl_spec(A)
```

## QR spec

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
W = sym.Matrix([[1,-3],[3,1]])
spec = la_figures.qr_tbl_spec(A, W)
```

## Eigen/SVD specs

```python
import sympy as sym
import la_figures

A = sym.Matrix([[4, 2], [0, 9]])
eig_spec = la_figures.eig_tbl_spec(A)
svd_spec = la_figures.svd_tbl_spec(A)
```

## End-to-end render

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
svg = la_figures.ge_tbl_svg(A, output_dir="./_out", output_stem="ge_min")
```

## Troubleshooting

- If SVG rendering fails, call the corresponding `*_tex` wrapper and inspect TeX.
- Use `output_dir` to retain TeX/SVG artifacts for debugging.
