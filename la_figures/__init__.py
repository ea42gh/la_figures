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
from .backsub import (
    backsub_trace_from_ref,
    backsub_solution_tex,
    backsub_system_tex,
)
from .ge import ge_trace, trace_to_layer_matrices
from .convenience_ge import ge_tbl_bundle, ge_tbl_layout_spec, ge_tbl_spec, ge_tbl_tex, ge_tbl_svg

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
    "backsub_trace_from_ref",
    "backsub_solution_tex",
    "backsub_system_tex",
    "ge_trace",
    "trace_to_layer_matrices",
    "ge_tbl_spec",
    "ge_tbl_layout_spec",
    "ge_tbl_tex",
    "ge_tbl_svg",
    "ge_tbl_bundle",
]
