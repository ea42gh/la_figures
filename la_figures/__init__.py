"""la_figures: linear algebra algorithms + description objects for matrixlayout.

This package owns *algorithmic decisions* (row reduction, pivot/free-variable
choices, orthonormalization, eigen/SVD computations, etc.) and returns explicit
description objects intended to be consumed by :mod:`matrixlayout`.

The core rule is strict layering:

- la_figures does **not** render and does **not** call TeX toolchains.
- matrixlayout does **not** do linear algebra; it only formats and lays out.
"""

from .eig import EigenDecomposition, eig_tbl_spec, eig_spec_from_eigenvects, eigendecomposition
from .svd import svd_tbl_spec, svd_tbl_spec_from_right_singular_vectors
# Backwards-compatible alias (not exported):
from .svd import svd_tbl_spec_from_AtA_eigenvects  # noqa: F401
from .backsub import (
    backsub_trace_from_ref,
    backsub_solution_tex,
    backsub_system_tex,
)
from .ge import ge_trace, trace_to_layer_matrices

__all__ = [
    "EigenDecomposition",
    "eig_tbl_spec",
    "eig_spec_from_eigenvects",
    "eigendecomposition",
    "svd_tbl_spec",
    "svd_tbl_spec_from_right_singular_vectors",
    "backsub_trace_from_ref",
    "backsub_solution_tex",
    "backsub_system_tex",
    "ge_trace",
    "trace_to_layer_matrices",
]
