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
    rhs: Any = None,
    # Default pivoting strategy must match the original implementation:
    # no pivoting unless explicitly requested.
    pivoting: str = "none",
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
) -> Dict[str, Any]:
    # Julia interop typically passes the RHS as a keyword named `rhs`.
    # Keep the internal naming (`ref_rhs`) but accept `rhs` as an alias.
    if (ref_rhs is not None) and (rhs is not None):
        raise TypeError("Provide only one of ref_rhs or rhs")
    if ref_rhs is None:
        ref_rhs = rhs

    tr: GETrace = ge_trace(ref_A, ref_rhs, pivoting=pivoting)  # type: ignore[arg-type]

    # Decorations are produced in *coefficient-matrix* coordinates.
    # We rebase pivot boxes to the final (last-layer) A-block in the GE grid.
    # Backwards-compatible pivot toggle:
    # - tests and legacy notebooks pass show_pivots=True/False.
    # - if show_pivots is True and no explicit pivot_style is given, use a visible default.
    pivots_enabled = bool(show_pivots)
    eff_pivot_style = pivot_style
    if pivots_enabled and (not str(eff_pivot_style).strip()):
        eff_pivot_style = "draw, thick, rounded corners"

    decor = decorate_ge(tr, index_base=index_base, pivot_style=eff_pivot_style)
    layers = trace_to_layer_matrices(tr, augmented=True)

    # Rebase pivot locations to the last layer and to the A-block column offset.
    pivot_positions = (decor.get("pivot_positions") or []) if pivots_enabled else []
    n_layers = len(layers.get("matrices") or [])
    m = int(getattr(tr.initial, "rows", 0) or 0)
    if m <= 0:
        # Fallback: infer from the first A-block.
        try:
            A0 = layers["matrices"][0][1]
            m = int(getattr(A0, "rows", 0) or len(A0))
        except Exception:
            m = 0
    row_off = (max(n_layers - 1, 0) * m)

    # E-block width (in columns). Prefer a concrete E if present; otherwise assume square m×m.
    Ew = 0
    for row in (layers.get("matrices") or []):
        if row and len(row) >= 1 and row[0] is not None:
            Ew = int(getattr(row[0], "cols", 0) or 0)
            break
    if Ew <= 0:
        Ew = m

    def _cell_global(r: int, c: int) -> str:
        rr = int(r) + row_off + int(index_base)
        cc = int(c) + Ew + int(index_base)
        return f"({rr}-{cc})"

    pivot_locs = [
        (f"{_cell_global(r, c)}{_cell_global(r, c)}", str(eff_pivot_style).strip())
        for (r, c) in pivot_positions
    ]

    spec: Dict[str, Any] = dict(
        matrices=layers["matrices"],
        Nrhs=int(tr.Nrhs or 0),
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        pivot_locs=pivot_locs,
        txt_with_locs=decor.get("txt_with_locs"),
        rowechelon_paths=decor.get("rowechelon_paths"),
        callouts=callouts,
        fig_scale=fig_scale,
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
    rhs: Any = None,
    pivoting: str = "none",
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a layout spec for :func:`matrixlayout.ge.ge_grid_tex`.

    The returned dict contains only primitives + nested lists.
    """

    return _build_ge_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        fig_scale=fig_scale,
    )["spec"]


def ge_tbl_bundle(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a bundle containing the trace, decorations, spec, and TeX."""

    bundle = _build_ge_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        fig_scale=fig_scale,
    )

    # Prefer the notebook-oriented bundle API when available. This avoids
    # regex-parsing generated TeX to discover \SubMatrix names.
    from dataclasses import asdict

    try:
        from matrixlayout.ge import ge_grid_bundle

        gb = ge_grid_bundle(**bundle["spec"])
        bundle["tex"] = gb.tex
        bundle["submatrix_spans"] = [
            {
                **asdict(s),
                "left_delim_node": s.left_delim_node,
                "right_delim_node": s.right_delim_node,
                "start_token": s.start_token,
                "end_token": s.end_token,
            }
            for s in gb.submatrix_spans
        ]
    except ImportError:
        # Fall back to plain TeX generation for older matrixlayout versions.
        from matrixlayout.ge import ge_grid_tex

        tex = ge_grid_tex(**bundle["spec"])
        bundle["tex"] = tex
    return bundle


def ge_tbl_tex(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
) -> str:
    """Compute a GE-table TeX document (no rendering)."""

    return ge_tbl_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        fig_scale=fig_scale,
    )["tex"]


def ge_tbl_svg(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    toolchain_name: Optional[str] = None,
    crop: Optional[str] = None,
    padding: Tuple[int, int, int, int] = (2, 2, 2, 2),
    callouts: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
) -> str:
    """Render a GE-table to SVG via matrixlayout (strict rendering boundary)."""

    spec = ge_tbl_spec(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        fig_scale=fig_scale,
    )

    from matrixlayout.ge import ge_grid_svg

    return ge_grid_svg(toolchain_name=toolchain_name, crop=crop, padding=padding, **spec)
