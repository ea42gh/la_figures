"""Convenience wrappers for Gaussian elimination figures.

This module provides *interop-friendly* top-level helpers that:

1) compute a Gaussian-elimination trace (algorithmic work in :mod:`la_figures.ge`),
2) convert that trace to a legacy-style matrix stack, and
3) delegate layout-only rendering to :mod:`matrixlayout`.

These helpers are designed to be callable from both Python and Julia (via
PyCall or PythonCall) using primitive containers (strings, ints, nested lists
and tuples).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import sympy as sym

from .ge import GETrace, ge_trace, trace_to_layer_matrices


def ge_tbl_spec(
    A: Any,
    rhs: Any = None,
    *,
    pivoting: str = "partial",
    augmented: bool = True,
) -> Mapping[str, Any]:
    """Compute the GE matrix-stack spec consumed by :func:`matrixlayout.ge_grid_*`.

    Parameters
    ----------
    A:
        Coefficient matrix.
    rhs:
        Optional right-hand side (vector or matrix).
    pivoting:
        Pivoting mode: ``"partial"`` or ``"none"``.
    augmented:
        If True, include RHS columns in the A-block when present.
    """

    tr: GETrace = ge_trace(A, rhs, pivoting=pivoting)  # type: ignore[arg-type]
    return trace_to_layer_matrices(tr, augmented=augmented)


def ge_tbl_tex(
    A: Any,
    rhs: Any = None,
    *,
    pivoting: str = "partial",
    augmented: bool = True,
    formater: Any = sym.latex,
    Nrhs: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """Compute a GE TeX document.

    This is a convenience wrapper around :func:`ge_tbl_spec` and
    :func:`matrixlayout.ge_grid_tex`.
    """

    spec = ge_tbl_spec(A, rhs, pivoting=pivoting, augmented=augmented)
    if Nrhs is None:
        Nrhs = spec.get("Nrhs", 0)

    from matrixlayout import ge_grid_tex

    return ge_grid_tex(spec["matrices"], Nrhs=Nrhs, formater=formater, **kwargs)


def ge_tbl_svg(
    A: Any,
    rhs: Any = None,
    *,
    pivoting: str = "partial",
    augmented: bool = True,
    formater: Any = sym.latex,
    Nrhs: Optional[Any] = None,
    toolchain_name: Optional[Any] = None,
    crop: Optional[Any] = None,
    padding: Any = None,
    **kwargs: Any,
) -> str:
    """Render a GE figure to SVG.

    This is a convenience wrapper around :func:`ge_tbl_spec` and
    :func:`matrixlayout.ge_grid_svg`.
    """

    spec = ge_tbl_spec(A, rhs, pivoting=pivoting, augmented=augmented)
    if Nrhs is None:
        Nrhs = spec.get("Nrhs", 0)

    from matrixlayout import ge_grid_svg

    return ge_grid_svg(
        spec["matrices"],
        Nrhs=Nrhs,
        formater=formater,
        toolchain_name=toolchain_name,
        crop=crop,
        padding=padding,
        **kwargs,
    )
