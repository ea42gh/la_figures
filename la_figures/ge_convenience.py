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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .ge import GETrace, decorate_ge, ge_trace, trace_to_layer_matrices


def _matrix_shape(mat: Any) -> Tuple[int, int]:
    if mat is None:
        return (0, 0)
    if hasattr(mat, "shape"):
        try:
            shp = mat.shape
            if len(shp) == 2:
                return int(shp[0]), int(shp[1])
        except Exception:
            pass
    if isinstance(mat, (list, tuple)):
        rows = list(mat)
        if not rows:
            return (0, 0)
        if isinstance(rows[0], (list, tuple)):
            return (len(rows), max((len(r) for r in rows), default=0))
        return (len(rows), 1)
    return (0, 0)


def _grid_offsets(
    matrices: Sequence[Sequence[Any]],
    *,
    index_base: int = 1,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    grid: List[List[Any]] = [list(r) for r in (matrices or [])]
    if not grid:
        return ([], [], [], [])
    n_block_rows = len(grid)
    n_block_cols = max((len(r) for r in grid), default=0)
    for r in range(n_block_rows):
        if len(grid[r]) < n_block_cols:
            grid[r].extend([None] * (n_block_cols - len(grid[r])))

    block_heights = [0] * n_block_rows
    block_widths = [0] * n_block_cols
    for br in range(n_block_rows):
        for bc in range(n_block_cols):
            h, w = _matrix_shape(grid[br][bc])
            block_heights[br] = max(block_heights[br], h)
            block_widths[bc] = max(block_widths[bc], w)

    max_h = max(block_heights) if block_heights else 0
    for bc in range(n_block_cols):
        if block_widths[bc] == 0 and max_h > 0:
            block_widths[bc] = max_h

    row_starts: List[int] = []
    acc = int(index_base)
    for h in block_heights:
        row_starts.append(acc)
        acc += h
    col_starts: List[int] = []
    acc = int(index_base)
    for w in block_widths:
        col_starts.append(acc)
        acc += w

    return block_heights, block_widths, row_starts, col_starts


def _grid_cell_coord(
    matrices: Sequence[Sequence[Any]],
    *,
    gM: int,
    gN: int,
    i: int,
    j: int,
    index_base: int = 1,
) -> str:
    block_heights, block_widths, row_starts, col_starts = _grid_offsets(matrices, index_base=index_base)
    if gM >= len(row_starts) or gN >= len(col_starts):
        raise ValueError("grid position out of range")
    rr = row_starts[gM] + int(i)
    cc = col_starts[gN] + int(j)
    if block_heights and (int(i) >= block_heights[gM]):
        rr = row_starts[gM] + block_heights[gM] - 1
    if block_widths and (int(j) >= block_widths[gN]):
        cc = col_starts[gN] + block_widths[gN] - 1
    return f"({rr}-{cc})"


def _legacy_array_name_specs(
    n_rows: int,
    lhs: str,
    rhs: Sequence[str],
    *,
    start_index: Optional[int],
) -> List[Tuple[Tuple[int, int], str, str]]:
    names: List[List[str]] = [["", ""] for _ in range(n_rows)]

    def _pe(i: int) -> str:
        if start_index is None:
            return " ".join([f" {lhs}" for _ in range(i, 0, -1)])
        return " ".join([f" {lhs}_{k + start_index - 1}" for k in range(i, 0, -1)])

    def _pa(e_prod: str, i: int) -> str:
        rr = list(rhs)
        if i > 0 and rr and rr[-1] == "I":
            rr[-1] = ""
        return r" \mid ".join([e_prod + " " + k for k in rr])

    for i in range(n_rows):
        if start_index is None:
            names[i][0] = f"{lhs}"
        else:
            names[i][0] = f"{lhs}_{start_index + i - 1}"

        e_prod = _pe(i)
        names[i][1] = _pa(e_prod, i)

    if len(rhs) > 1:
        for i in range(n_rows):
            names[i][1] = r"\left( " + names[i][1] + r" \right)"

    for i in range(n_rows):
        for j in range(2):
            names[i][j] = r"\mathbf{ " + names[i][j] + " }"

    terms: List[Tuple[Tuple[int, int], str, str]] = [((0, 1), "ar", "$" + names[0][1] + "$")]
    for i in range(1, n_rows):
        terms.append(((i, 0), "al", "$" + names[i][0] + "$"))
        terms.append(((i, 1), "ar", "$" + names[i][1] + "$"))
    return terms


def _legacy_name_specs_to_paths(
    matrices: Sequence[Sequence[Any]],
    name_specs: Sequence[Tuple[Tuple[int, int], str, str]],
    *,
    color: str = "blue",
) -> List[str]:
    from matrixlayout.ge import ge_grid_submatrix_spans

    spans = ge_grid_submatrix_spans(matrices)
    name_map = {(s.block_row, s.block_col): s.name for s in spans}

    ar = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north east) + (0.02,0.02) $) -- +(0.6cm,0.3cm)    node[COLOR, above right=-3pt]{TXT};"
    al = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north west) + (-0.02,0.02) $) -- +(-0.6cm,0.3cm) node[COLOR, above left=-3pt] {TXT};"
    a = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.north) + (0,0) $) -- +(0cm,0.6cm) node[COLOR, above=1pt] {TXT};"

    bl = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south west) + (-0.02,-0.02) $) -- +(-0.6cm,-0.3cm)  node[COLOR, below left=-3pt]{TXT};"
    br = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south east) + (0.02,-0.02) $) -- +(0.6cm,-0.3cm)   node[COLOR, below right=-3pt]{TXT};"
    b = r"\tikz \draw[<-,>=stealth,COLOR,thick] ($ (NAME.south) + (0,0) $) -- +(0cm,-0.6cm) node[COLOR, below=1pt] {TXT};"

    out: List[str] = []
    for (gM, gN), pos, txt in name_specs:
        name = name_map.get((gM, gN))
        if not name:
            continue
        t = None
        if pos == "a":
            t = a
        elif pos == "al":
            t = al
        elif pos == "ar":
            t = ar
        elif pos == "b":
            t = b
        elif pos == "bl":
            t = bl
        elif pos == "br":
            t = br
        if t is None:
            continue
        t = t.replace("COLOR", color).replace("NAME", name).replace("TXT", txt)
        out.append(t)
    return out


def _legacy_ref_path_list_to_rowechelon_paths(
    matrices: Sequence[Sequence[Any]],
    ref_path_list: Sequence[Any],
) -> List[str]:
    out: List[str] = []
    for spec in ref_path_list:
        if not isinstance(spec, (list, tuple)) or len(spec) < 3:
            continue
        gM, gN = int(spec[0]), int(spec[1])
        pivots = spec[2]
        case = spec[3] if len(spec) > 3 else "hh"
        color = spec[4] if len(spec) > 4 else "violet,line width=0.4mm"
        _, _, row_starts, col_starts = _grid_offsets(matrices, index_base=1)
        block_heights, block_widths, _, _ = _grid_offsets(matrices, index_base=1)
        if gM >= len(block_heights) or gN >= len(block_widths):
            continue
        shape = (block_heights[gM], block_widths[gN])
        if not pivots:
            continue

        cur = pivots[0]
        ll = [cur] if (case in ("vv", "vh")) else []
        for nxt in pivots[1:]:
            if cur[0] != nxt[0]:
                cur = (cur[0] + 1, cur[1])
                ll.append(cur)
            if nxt[1] != cur[1]:
                cur = (cur[0], nxt[1])
                ll.append(cur)
            if cur != nxt:
                ll.append(nxt)
            cur = nxt

        if len(ll) == 0 and case == "hv":
            ll = [(pivots[0][0] + 1, pivots[0][0]), (shape[0], pivots[0][1])]

        if case in ("hh", "vh"):
            if cur[0] != shape[0]:
                cur = (cur[0] + 1, cur[1])
                ll.append(cur)
            ll.append((cur[0], shape[1]))
        else:
            ll.append((shape[0], cur[1]))

        coords = [
            _grid_cell_coord(matrices, gM=gM, gN=gN, i=int(i), j=int(j), index_base=1)
            for (i, j) in ll
        ]
        out.append(rf"\draw[{color}] " + " -- ".join(coords) + ";")
    return out


def _legacy_pivot_list_to_pivot_locs(
    matrices: Sequence[Sequence[Any]],
    pivot_list: Sequence[Any],
    *,
    index_base: int = 1,
    pivot_style: str = "",
) -> List[Tuple[str, str]]:
    _, _, row_starts, col_starts = _grid_offsets(matrices, index_base=index_base)

    out: List[Tuple[str, str]] = []
    for spec in pivot_list:
        if not spec or len(spec) < 2:
            continue
        grid_pos, pivots = spec[0], spec[1]
        if not isinstance(grid_pos, (list, tuple)) or len(grid_pos) != 2:
            continue
        gM, gN = int(grid_pos[0]), int(grid_pos[1])
        if gM < 0 or gN < 0:
            continue
        if gM >= len(row_starts) or gN >= len(col_starts):
            continue
        for (i, j) in pivots:
            coord = _grid_cell_coord(matrices, gM=gM, gN=gN, i=int(i), j=int(j), index_base=index_base)
            out.append((f"{coord}{coord}", pivot_style))
    return out


def _build_typed_layout_spec(
    *,
    pivot_locs: Optional[Any],
    txt_with_locs: Optional[Any],
    rowechelon_paths: Optional[Any],
    callouts: Optional[Any],
    nice_options: str,
    preamble: str,
    extension: str,
    fig_scale: Optional[Any],
    outer_delims: bool,
) -> "GELayoutSpec":
    from matrixlayout.specs import GELayoutSpec, PivotBox, RowEchelonPath, TextAt

    typed_pivots = [PivotBox(*item) for item in (pivot_locs or [])]
    typed_txt = [TextAt(*item) for item in (txt_with_locs or [])]
    typed_paths = [RowEchelonPath(str(p)) for p in (rowechelon_paths or [])]

    return GELayoutSpec(
        nice_options=nice_options,
        preamble=preamble,
        extension=extension,
        pivot_locs=typed_pivots,
        txt_with_locs=typed_txt,
        rowechelon_paths=typed_paths,
        callouts=callouts,
        outer_delims=outer_delims,
    )


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

    typed_layout = _build_typed_layout_spec(
        pivot_locs=pivot_locs,
        txt_with_locs=decor.get("txt_with_locs"),
        rowechelon_paths=decor.get("rowechelon_paths"),
        callouts=callouts,
        nice_options=nice_options,
        preamble=preamble,
        extension=extension,
        fig_scale=fig_scale,
        outer_delims=bool(outer_delims),
    )

    return {
        "trace": tr,
        "decor": decor,
        "layers": layers,
        "spec": spec,
        "typed_layout": typed_layout,
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


def ge_tbl_layout_spec(
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
    """Return a layout spec using :class:`matrixlayout.specs.GELayoutSpec`."""

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

    return {
        "matrices": bundle["layers"]["matrices"],
        "Nrhs": int(bundle["trace"].Nrhs or 0),
        "layout": bundle["typed_layout"],
        "outer_hspace_mm": int(outer_hspace_mm),
        "cell_align": str(cell_align),
    }


def ge(
    matrices: Sequence[Sequence[Any]],
    *,
    Nrhs: Any = 0,
    formater: Any = str,
    pivot_list: Optional[Sequence[Any]] = None,
    bg_for_entries: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "blue"),
    pivot_text_color: str = "red",
    ref_path_list: Optional[Any] = None,
    comment_list: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    array_names: Optional[Any] = None,
    start_index: Optional[int] = 1,
    func: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
    tmp_dir: Optional[str] = None,
    keep_file: Optional[str] = None,
    **render_opts: Any,
) -> str:
    """Compatibility wrapper for legacy ``itikz.nicematrix.ge``."""

    unsupported = {
        "func": func,
        "tmp_dir": tmp_dir,
        "keep_file": keep_file,
    }
    missing = [k for k, v in unsupported.items() if v not in (None, [], {}, ())]
    if missing:
        raise NotImplementedError(f"Legacy options not yet supported in new GE: {', '.join(missing)}")

    pivot_style = f"draw={pivot_text_color}" if pivot_text_color else ""
    pivot_locs = (
        _legacy_pivot_list_to_pivot_locs(matrices, pivot_list, index_base=1, pivot_style=pivot_style)
        if pivot_list
        else None
    )

    codebefore: List[str] = []
    if bg_for_entries:
        specs = bg_for_entries
        if isinstance(specs, list) and specs and all(isinstance(elem, list) for elem in specs):
            if len(specs) == 5 and not any(isinstance(e, list) for e in specs[0:2]):
                specs = [specs]
        if not isinstance(specs, list):
            specs = [specs]
        for spec in specs:
            if not isinstance(spec, (list, tuple)) or len(spec) < 3:
                continue
            gM, gN, entries = spec[0], spec[1], spec[2]
            color = spec[3] if len(spec) > 3 else "red!15"
            pt = spec[4] if len(spec) > 4 else 0
            if not isinstance(entries, (list, tuple)):
                entries = [entries]
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) == 2 and all(isinstance(x, (list, tuple)) for x in entry):
                    (i0, j0), (i1, j1) = entry
                    c0 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i0), j=int(j0), index_base=1)
                    c1 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i1), j=int(j1), index_base=1)
                    fit = f"{c0} {c1}"
                else:
                    i0, j0 = entry
                    c0 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i0), j=int(j0), index_base=1)
                    fit = f"{c0}"
                codebefore.append(
                    rf"\tikz \node [fill={color}, inner sep = {pt}pt, fit = {fit}] {{}} ;"
                )

    txt_with_locs: List[Tuple[str, str, str]] = []
    if comment_list:
        _, block_widths, row_starts, col_starts = _grid_offsets(matrices, index_base=1)
        if block_widths and col_starts:
            last_col_start = col_starts[-1]
            last_col_width = block_widths[-1]
            comment_col = last_col_start + max(last_col_width - 1, 0)
            for g, txt in enumerate(comment_list):
                if g >= len(row_starts):
                    break
                row = row_starts[g]
                coord = f"({row}-{comment_col}.east)"
                txt_with_locs.append((coord, str(txt), "text=violet"))

    if variable_summary:
        _, block_widths, row_starts, col_starts = _grid_offsets(matrices, index_base=1)
        if block_widths and col_starts and row_starts:
            last_col_start = col_starts[-1]
            last_col_width = block_widths[-1]
            last_row = row_starts[-1] + max(_grid_offsets(matrices, index_base=1)[0][-1] - 1, 0)
            base_yshift = -1.2
            for j, basic in enumerate(variable_summary):
                if j >= last_col_width:
                    break
                col = last_col_start + j
                arrow = r"\Uparrow" if basic is True else r"\uparrow"
                color = variable_colors[0] if basic is True else variable_colors[1]
                txt_with_locs.append((f"({last_row}-{col})", rf"\textcolor{{{color}}}{{{arrow}}}", f"yshift={base_yshift}em"))
                txt_with_locs.append((f"({last_row}-{col})", rf"\textcolor{{{color}}}{{x_{j+1}}}", f"yshift={2*base_yshift}em"))

    rowechelon_paths: List[str] = []
    if ref_path_list:
        specs = ref_path_list if isinstance(ref_path_list, list) else [ref_path_list]
        rowechelon_paths.extend(_legacy_ref_path_list_to_rowechelon_paths(matrices, specs))

    if array_names is not None:
        try:
            lhs, rhs = array_names
            rhs_list = list(rhs)
        except Exception:
            lhs, rhs_list = "E", ["A"]
        n_rows = len(matrices or [])
        name_specs = _legacy_array_name_specs(n_rows, str(lhs), [str(x) for x in rhs_list], start_index=start_index)
        rowechelon_paths.extend(_legacy_name_specs_to_paths(matrices, name_specs, color="blue"))

    from matrixlayout.ge import ge_grid_svg

    return ge_grid_svg(
        matrices=matrices,
        Nrhs=Nrhs,
        formater=formater,
        pivot_locs=pivot_locs,
        codebefore=codebefore,
        txt_with_locs=txt_with_locs,
        rowechelon_paths=rowechelon_paths,
        fig_scale=fig_scale,
        **render_opts,
    )


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
