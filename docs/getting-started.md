# Getting Started

Minimal examples for common entry points. Expected output: spec dictionaries
ready for matrixlayout rendering.

Common defaults:

- `ge_tbl_spec`: `pivoting="none"`, `gj=False`, `show_pivots=True`.
- `qr_tbl_spec`: `array_names=True`.
- `eig_tbl_spec`: `normal=False`.

## What is a spec?

A spec is a plain dictionary that describes which matrices to render and which
labels/lines/callouts to attach. You can pass it to `matrixlayout` to emit TeX
or SVG.

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
spec = la_figures.ge_tbl_spec(A)
# spec["matrices"] holds the matrix grid
# spec["specs"] holds label/callout specs
```

## GE spec

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
spec = la_figures.ge_tbl_spec(A)
```

Minimal render from a spec:

```python
from matrixlayout.ge import grid_svg

svg = grid_svg(spec["matrices"], specs=spec["specs"])
```

## QR spec

```python
import sympy as sym
import la_figures

A = sym.Matrix([[1, 2], [3, 4]])
W = sym.Matrix([[1,-3],[3,1]])
spec = la_figures.qr_tbl_spec(A, W)
```

## QR end-to-end render

```python
import sympy as sym
from la_figures import qr_tbl_spec
from matrixlayout.qr import qr_grid_svg

A = sym.Matrix([[1, 2], [3, 4]])
W = sym.eye(2)
spec = qr_tbl_spec(A, W)
svg = qr_grid_svg(spec["matrices"], specs=spec["specs"])
```

## Eigen/SVD specs

```python
import sympy as sym
import la_figures

A = sym.Matrix([[4, 2], [0, 9]])
eig_spec = la_figures.eig_tbl_spec(A)
svd_spec = la_figures.svd_tbl_spec(A)
```

## Backsubstitution blocks

```python
from la_figures import linear_system_tex, backsubstitution_tex, standard_solution_tex
from matrixlayout.backsubst import backsubst_svg

A = [[1, 0, 1], [0, 1, 1]]
b = [1, 2]

svg = backsubst_svg(
    system_txt=linear_system_tex(A, b),
    cascade_txt=backsubstitution_tex(A, b),
    solution_txt=standard_solution_tex(A, b),
)
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
- See `matrixlayout/docs/specs.md` for spec keys and examples.
