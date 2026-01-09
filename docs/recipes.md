# Spec Recipes

Compact examples for common tasks. Specs are passed directly to matrixlayout
renderers (e.g., `ge_grid_svg`, `qr_grid_svg`).

## GE with pivots

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
spec = la_figures.ge_tbl_spec(A, show_pivots=True)
```

## QR with labels

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
W = sym.eye(2)
spec = la_figures.qr_tbl_spec(A, W, array_names=True)
```

## Eigen/SVD

```python
import sympy as sym
import la_figures

A = sym.Matrix([[2, 0], [0, 3]])
eig_spec = la_figures.eig_tbl_spec(A)
svd_spec = la_figures.svd_tbl_spec(A)
```
