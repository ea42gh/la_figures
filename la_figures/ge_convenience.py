"""Convenience GE wrappers producing TeX (no rendering).

This module is intentionally thin: it ties together

- :func:`la_figures.ge.ge_trace` (algorithmic trace)
- :func:`la_figures.ge.decorate_ge` (data-only decorations)
- :func:`matrixlayout.ge.ge_grid_tex` (layout-only TeX template)

It does **not** call any toolchains.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import sympy as sym

from .ge import ge_trace, trace_to_layer_matrices


def ge_tbl_bundle(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    pivoting: str = "partial",
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}\n",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
) -> Dict[str, Any]:
    """Return a bundle containing trace, decorations, and TeX.

    Parameters
    ----------
    ref_A, ref_rhs:
        Passed through to :func:`la_figures.ge.ge_trace`.
    pivoting:
        Pivoting mode for the trace builder.
    index_base:
        Coordinate base for decorations. Use 1 for nicematrix coordinates.
    pivot_style:
        Extra TikZ node style for pivot boxes.
    preamble, nice_options:
        Passed to the matrixlayout GE template.
    outer_delims:
        Whether to draw an outer delimiter SubMatrix.
    outer_hspace_mm:
        Horizontal space between the E and A columns in the outer grid.
    """

    tr = ge_trace(ref_A, ref_rhs, pivoting=pivoting)

    try:
        from .ge import decorate_ge  # added in Step 2
    except Exception as e:
        raise ImportError("decorate_ge is required for ge_tbl_bundle") from e

    decor = decorate_ge(tr, index_base=index_base, pivot_style=pivot_style)
    layers = trace_to_layer_matrices(tr, augmented=True)

    from matrixlayout.ge import ge_grid_tex

    tex = ge_grid_tex(
        matrices=layers["matrices"],
        Nrhs=tr.Nrhs,
        preamble=preamble,
        nice_options=nice_options,
        pivot_locs=decor.get("pivot_locs"),
        txt_with_locs=decor.get("txt_with_locs"),
        rowechelon_paths=decor.get("rowechelon_paths"),
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
    )

    return {
        "trace": tr,
        "decor": decor,
        "layers": layers,
        "tex": tex,
    }


def ge_tbl_tex(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    pivoting: str = "partial",
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}\n",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
) -> str:
    """Compute a GE-table TeX document (no rendering).

    This is a convenience wrapper around :func:`ge_tbl_bundle`.
    """

    return ge_tbl_bundle(
        ref_A,
        ref_rhs,
        pivoting=pivoting,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
    )["tex"]
