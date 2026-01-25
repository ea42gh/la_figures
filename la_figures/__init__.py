"""la_figures: linear algebra algorithms + description objects for matrixlayout.

This package owns *algorithmic decisions* (row reduction, pivot/free-variable
choices, orthonormalization, eigen/SVD computations, etc.) and returns explicit
description objects intended to be consumed by :mod:`matrixlayout`.

The core rule is strict layering:

- la_figures does **not** render and does **not** call TeX toolchains.
- matrixlayout does **not** do linear algebra; it only formats and lays out.
"""

from .eig import EigenDecomposition, eig_spec_from_eigenvects, eig_tbl_spec, eigendecomposition
from .svd import svd_tbl_spec, svd_tbl_spec_from_right_singular_vectors
from .convenience import (
    eig_tbl_bundle,
    eig_tbl_svg,
    eig_tbl_tex,
    svd_tbl_bundle,
    svd_tbl_svg,
    svd_tbl_tex,
)
from .qr import compute_qr_matrices, gram_schmidt_qr_matrices, qr_tbl_layout_spec, qr_tbl_spec
from .convenience_qr import qr as qr_svg, qr_tbl_bundle as qr_tbl_bundle, qr_tbl_tex, qr_tbl_svg, gram_schmidt_qr
from .backsub import (
    backsubstitution_tex,
    linear_system_tex,
    standard_solution_tex,
)
from . import ge as ge
from .ge import ge_trace, trace_to_layer_matrices
from .ge_convenience import show_ge
from .convenience_ge import ge as svg, ge_tbl_bundle, ge_tbl_layout_spec, ge_tbl_spec, ge_tbl_tex, ge_tbl_svg
from .formatting import (
    decorate_tex_entries,
    latexify,
    make_decorator,
    decorator_box,
    decorator_color,
    decorator_bg,
    decorator_bf,
    sel_entry,
    sel_box,
    sel_row,
    sel_col,
    sel_rows,
    sel_cols,
    sel_all,
    sel_vec,
    sel_vec_range,
)

__all__ = [
    "EigenDecomposition",
    "eigendecomposition",
    "eig_spec_from_eigenvects",
    "eig_tbl_spec",
    "svd_tbl_spec",
    "svd_tbl_spec_from_right_singular_vectors",
    "eig_tbl_tex",
    "eig_tbl_svg",
    "eig_tbl_bundle",
    "svd_tbl_tex",
    "svd_tbl_svg",
    "svd_tbl_bundle",
    "compute_qr_matrices",
    "gram_schmidt_qr_matrices",
    "qr_tbl_spec",
    "qr_tbl_layout_spec",
    "qr_tbl_tex",
    "qr_tbl_svg",
    "qr_tbl_bundle",
    "gram_schmidt_qr",
    "backsubstitution_tex",
    "linear_system_tex",
    "standard_solution_tex",
    "ge_trace",
    "trace_to_layer_matrices",
    "ge_tbl_spec",
    "ge_tbl_layout_spec",
    "ge_tbl_tex",
    "ge_tbl_svg",
    "show_ge",
    "ge_tbl_bundle",
    "ge",
    "svg",
    "qr_svg",
    "latexify",
    "make_decorator",
    "decorate_tex_entries",
    "decorator_box",
    "decorator_color",
    "decorator_bg",
    "decorator_bf",
    "sel_entry",
    "sel_box",
    "sel_row",
    "sel_col",
    "sel_rows",
    "sel_cols",
    "sel_all",
    "sel_vec",
    "sel_vec_range",
]
