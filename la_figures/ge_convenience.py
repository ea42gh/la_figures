"""Convenience GE wrappers producing TeX/SVG.

This module is intentionally thin: it ties together

- :func:`la_figures.ge.ge_trace` (algorithmic trace)
- :func:`la_figures.ge.decorate_ge` (data-only decorations)
- :func:`la_figures.ge.trace_to_layer_matrices` (matrix stack)
- :func:`matrixlayout.ge.ge_grid_tex` / :func:`matrixlayout.ge.ge_grid_svg` (layout/render)

The TeX helpers do **not** call any toolchains.
SVG helpers call the strict rendering boundary in :mod:`matrixlayout`.

All outputs are designed to remain Julia-friendly (primitives + nested lists/tuples)
so Julia can later compute traces/decorations and call the same renderer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .ge import GETrace, decorate_ge, ge_trace, trace_to_layer_matrices


def _build_ge_bundle(
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
    cell_align: str = "r",
) -> Dict[str, Any]:
    tr: GETrace = ge_trace(ref_A, ref_rhs, pivoting=pivoting)  # type: ignore[arg-type]

    decor = decorate_ge(tr, index_base=index_base, pivot_style=pivot_style)
    layers = trace_to_layer_matrices(tr, augmented=True)

    spec: Dict[str, Any] = dict(
        matrices=layers["matrices"],
        Nrhs=int(tr.Nrhs or 0),
        preamble=preamble,
        nice_options=nice_options,
        pivot_locs=decor.get("pivot_locs"),
        txt_with_locs=decor.get("txt_with_locs"),
        rowechelon_paths=decor.get("rowechelon_paths"),
        outer_delims=bool(outer_delims),
        outer_hspace_mm=int(outer_hspace_mm),
        cell_align=str(cell_align),
    )

    return {
        "trace": tr,
        "decor": decor,
        "layers": layers,
        "spec": spec,
    }


def ge_tbl_spec(
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
    cell_align: str = "r",
) -> Dict[str, Any]:
    """Return a layout spec for :func:`matrixlayout.ge.ge_grid_tex`.

    The returned dict contains only primitives + nested lists.
    """

    return _build_ge_bundle(
        ref_A,
        ref_rhs,
        pivoting=pivoting,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
    )["spec"]


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
    cell_align: str = "r",
) -> Dict[str, Any]:
    """Return a bundle containing the trace, decorations, spec, and TeX."""

    bundle = _build_ge_bundle(
        ref_A,
        ref_rhs,
        pivoting=pivoting,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
    )

    from matrixlayout.ge import ge_grid_tex

    tex = ge_grid_tex(**bundle["spec"])
    bundle["tex"] = tex
    return bundle


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
    cell_align: str = "r",
) -> str:
    """Compute a GE-table TeX document (no rendering)."""

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
        cell_align=cell_align,
    )["tex"]


def ge_tbl_svg(
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
    cell_align: str = "r",
    toolchain_name: Optional[str] = None,
    crop: Optional[str] = None,
    padding: Tuple[int, int, int, int] = (2, 2, 2, 2),
) -> str:
    """Render a GE-table to SVG via matrixlayout (strict rendering boundary)."""

    spec = ge_tbl_spec(
        ref_A,
        ref_rhs,
        pivoting=pivoting,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
    )

    from matrixlayout.ge import ge_grid_svg

    return ge_grid_svg(toolchain_name=toolchain_name, crop=crop, padding=padding, **spec)
