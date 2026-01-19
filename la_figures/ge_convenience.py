"""Convenience GE wrappers producing TeX/SVG.

This module is intentionally thin: it ties together

- :func:`la_figures.ge.ge_trace` (algorithmic trace)
- :func:`la_figures.ge.decorate_ge` (data-only decorations)
- :func:`la_figures.ge.trace_to_layer_matrices` (matrix stack)
- :func:`matrixlayout.ge.grid_tex` / :func:`matrixlayout.ge.grid_svg` (layout/render)

The TeX helpers do **not** call any toolchains.
SVG helpers call the strict rendering boundary in :mod:`matrixlayout`.

All outputs are designed to remain Julia-friendly (primitives + nested lists/tuples)
so Julia can later compute traces/decorations and call the same renderer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .ge import GETrace, decorate_ge, ge_trace, trace_to_layer_matrices
from .formatting import latexify, make_decorator


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


def _nrhs_count(Nrhs: Any) -> int:
    if Nrhs is None:
        return 0
    if isinstance(Nrhs, (list, tuple)):
        return int(sum(int(x) for x in Nrhs))
    try:
        import numpy as np

        if isinstance(Nrhs, np.ndarray):
            return int(np.sum(Nrhs))
    except Exception:
        pass
    try:
        return int(Nrhs)
    except Exception:
        return 0


def _coerce_rhs_labels(rhs_list: Sequence[str], Nrhs: Any) -> List[str]:
    labels = list(rhs_list)
    count = _nrhs_count(Nrhs)
    if count <= 0:
        return labels
    default_rhs = "b" if count == 1 else "B"
    if len(labels) <= 1:
        return [labels[0] if labels else "A", default_rhs]
    last = labels[-1]
    if count == 1 and last == "B":
        labels[-1] = "b"
    elif count > 1 and last == "b":
        labels[-1] = "B"
    return labels


def _legacy_name_specs_to_callouts(
    matrices: Sequence[Sequence[Any]],
    name_specs: Sequence[Tuple[Tuple[int, int], str, str]],
    *,
    color: str = "blue",
    legacy_submatrix_names: bool = True,
) -> List[Dict[str, Any]]:
    from matrixlayout.ge import grid_submatrix_spans

    spans = grid_submatrix_spans(matrices, legacy_submatrix_names=legacy_submatrix_names)
    name_map = {(s.block_row, s.block_col): s.name for s in spans}

    def _strip_math(s: str) -> Tuple[str, bool]:
        s = str(s).strip()
        if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
            return s[1:-1].strip(), True
        return s, False

    anchor_map = {
        "a": ("auto", "north"),
        "al": ("left", "north"),
        "ar": ("right", "north"),
        "b": ("auto", "south"),
        "bl": ("left", "south"),
        "br": ("right", "south"),
    }

    out: List[Dict[str, Any]] = []
    for (gM, gN), pos, txt in name_specs:
        name = name_map.get((gM, gN))
        if not name:
            continue
        side, anchor = anchor_map.get(pos, ("auto", "north"))
        label, had_math = _strip_math(txt)
        out.append(
            {
                "name": name,
                "label": label,
                "side": side,
                "anchor": anchor,
                "angle_deg": -35.0,
                "length_mm": 6.0,
                "label_shift_y_mm": 1.0,
                "color": color,
                "math_mode": had_math,
            }
        )
    return out


def _legacy_ref_path_list_to_rowechelon_paths(
    matrices: Sequence[Sequence[Any]],
    ref_path_list: Sequence[Any],
    *,
    legacy_submatrix_names: bool = True,
) -> List[str]:
    out: List[str] = []
    from matrixlayout.ge import grid_submatrix_spans

    spans = grid_submatrix_spans(matrices, legacy_submatrix_names=legacy_submatrix_names)
    span_map = {(s.block_row, s.block_col): s for s in spans}
    for spec in ref_path_list:
        if not isinstance(spec, (list, tuple)) or len(spec) < 3:
            continue
        gM, gN = int(spec[0]), int(spec[1])
        pivots = spec[2]
        case = spec[3] if len(spec) > 3 else "hh"
        color = spec[4] if len(spec) > 4 else "blue,line width=0.4mm"
        adj = spec[5] if len(spec) > 5 else 0.1
        left_pad = spec[6] if len(spec) > 6 else 0.0
        span = span_map.get((gM, gN))
        if span is None:
            continue
        shape = (span.row_end - span.row_start + 1, span.col_end - span.col_start + 1)
        if not pivots:
            continue

        tlr = span.row_start - 1
        tlc = span.col_start - 1
        name = span.name

        def coords(i: int, j: int) -> str:
            if i >= shape[0]:
                if gN == 0 and j == 0:
                    x = r"\x1"
                else:
                    x = r"\x2" if j >= shape[1] else r"\x4"
                y = r"\y2"
                p = f"({x},{y})"
            elif j >= shape[1]:
                x = r"\x2"
                y = r"\y4"
                p = f"({x},{y})"
            elif j == 0:
                x = r"\x1"
                y = r"\y1" if i == 0 else r"\y3"
                p = f"({x},{y})"
            else:
                x = f"{i + 1 + tlr}"
                y = f"{j + 1 + tlc}"
                p = f"({x}-|{y})"

            if j == 0 and left_pad:
                p = f"($ {p} + (-{left_pad:2},0) $)"
            elif j != 0 and j < shape[1] and adj != 0:
                p = f"($ {p} + ({adj:2},0) $)"
            return p

        cur = pivots[0]
        ll = [cur] if (case == "vv") or (case == "vh") else []
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

        if (case == "hh") or (case == "vh"):
            if cur[0] != shape[0]:
                cur = (cur[0] + 1, cur[1])
                ll.append(cur)
            ll.append((cur[0], shape[1]))
        else:
            ll.append((shape[0], cur[1]))

        corners = (
            f"let \\p1 = ({name}.north west), "
            f"\\p2 = ({name}.south east), "
        )

        if (case == "vv") or (case == "vh"):
            p3 = f"\\p3 = ({ll[1][0] + tlr + 1}-|{ll[1][1] + tlc + 1}), "
        else:
            p3 = f"\\p3 = ({ll[0][0] + tlr + 1}-|{ll[0][1] + tlc + 1}), "

        if (case == "vh") or (case == "hh"):
            i, j = ll[-2]
            p4 = f"\\p4 = ({i + tlr + 1}-|{j + tlc + 1}) in "
        else:
            i, j = ll[-1]
            p4 = f"\\p4 = ({i + tlr + 1}-|{j + tlc + 1}) in "

        cmd = "\\tikz \\draw[" + color + "] " + corners + p3 + p4 + " -- ".join([coords(*p) for p in ll]) + ";"
        out.append(cmd)
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


def _pivot_list_to_decorators(
    pivot_list: Sequence[Any],
    *,
    pivot_text_color: str = "red",
) -> List[Dict[str, Any]]:
    decorators: List[Dict[str, Any]] = []
    dec = make_decorator(boxed=True, bf=True, text_color=pivot_text_color)
    for spec in pivot_list:
        if not spec or len(spec) < 2:
            continue
        grid_pos, pivots = spec[0], spec[1]
        if not isinstance(grid_pos, (list, tuple)) or len(grid_pos) != 2:
            continue
        gM, gN = int(grid_pos[0]), int(grid_pos[1])
        if not isinstance(pivots, (list, tuple)):
            continue
        decorators.append({"grid": (gM, gN), "entries": list(pivots), "decorator": dec})
    return decorators


def _legacy_bg_list_to_codebefore(
    matrices: Sequence[Sequence[Any]],
    bg_list: Sequence[Any],
) -> List[str]:
    codebefore: List[str] = []

    def _emit_spec(spec: Sequence[Any]) -> None:
        if not spec or len(spec) < 4:
            return
        gM, gN = int(spec[0]), int(spec[1])
        entries = spec[2]
        color = spec[3]
        pt = spec[4] if len(spec) > 4 else 0
        if not isinstance(entries, (list, tuple)):
            entries = [entries]
        for entry in entries:
            cmd_1 = rf"\tikz \node [fill={color}, inner sep = {pt}pt, fit = "
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and all(isinstance(x, (list, tuple)) for x in entry):
                (i0, j0), (i1, j1) = entry
                c0 = _grid_cell_coord(matrices, gM=gM, gN=gN, i=int(i0), j=int(j0), index_base=1)
                c1 = _grid_cell_coord(matrices, gM=gM, gN=gN, i=int(i1), j=int(j1), index_base=1)
                cmd_2 = f"({c0[1:-1]}-medium) ({c1[1:-1]}-medium)"
            else:
                i0, j0 = entry
                c0 = _grid_cell_coord(matrices, gM=gM, gN=gN, i=int(i0), j=int(j0), index_base=1)
                cmd_2 = f"({c0[1:-1]}-medium)"
            codebefore.append(cmd_1 + cmd_2 + " ] {} ;")

    for spec in bg_list:
        if isinstance(spec, list) and spec and all(isinstance(elem, list) for elem in spec):
            for s in spec:
                _emit_spec(s)
        else:
            _emit_spec(spec)
    return codebefore


def _variable_summary_label_rows(
    matrices: Sequence[Sequence[Any]],
    variable_summary: Sequence[Any],
    variable_colors: Sequence[str],
) -> List[Dict[str, Any]]:
    _, block_widths, row_starts, _ = _grid_offsets(matrices, index_base=1)
    if not (block_widths and row_starts):
        return []
    n_block_rows = len(matrices or [])
    n_block_cols = max((len(r) for r in (matrices or [])), default=0)
    if n_block_rows == 0 or n_block_cols == 0:
        return []
    last_col_width = block_widths[-1]
    arrows: List[str] = []
    labels: List[str] = []
    for j, basic in enumerate(variable_summary):
        if j >= last_col_width:
            break
        arrow = r"\Uparrow" if basic is True else r"\uparrow"
        color = variable_colors[0] if basic is True else variable_colors[1]
        arrows.append(rf"$\textcolor{{{color}}}{{{arrow}}}$")
        labels.append(rf"$\textcolor{{{color}}}{{x_{{{j+1}}}}}$")
    if not arrows:
        return []
    return [
        {
            "grid": (n_block_rows - 1, n_block_cols - 1),
            "rows": [arrows, labels],
        }
    ]


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


def _merge_extension(extension: str, row_stretch: Optional[float]) -> str:
    if row_stretch is None:
        return extension
    cmd = rf"\renewcommand{{\arraystretch}}{{{row_stretch}}}"
    ext = extension or ""
    if cmd in ext:
        return ext
    if ext and not ext.endswith("\n"):
        ext += "\n"
    return ext + cmd + "\n"


def _build_ge_bundle(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    # Default pivoting strategy must match the original implementation:
    # no pivoting unless explicitly requested.
    pivoting: str = "none",
    gj: bool = False,
    show_pivots: Optional[bool] = True,
    index_base: int = 1,
    pivot_style: str = "",
    pivot_text_color: str = "red",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt, left-margin=6pt, right-margin=6pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    # Julia interop typically passes the RHS as a keyword named `rhs`.
    # Keep the internal naming (`ref_rhs`) but accept `rhs` as an alias.
    if (ref_rhs is not None) and (rhs is not None):
        raise TypeError("Provide only one of ref_rhs or rhs")
    if ref_rhs is None:
        ref_rhs = rhs

    extension = _merge_extension(extension, row_stretch)

    tr: GETrace = ge_trace(ref_A, ref_rhs, pivoting=pivoting, gj=gj)  # type: ignore[arg-type]

    # Decorations are produced in *coefficient-matrix* coordinates.
    # We rebase pivot boxes to the final (last-layer) A-block in the GE grid.
    # Backwards-compatible pivot toggle:
    # - tests and legacy notebooks pass show_pivots=True/False.
    # - pivot_style controls explicit TikZ pivot boxes; otherwise we use entry decorators.
    pivots_enabled = bool(show_pivots)
    eff_pivot_style = str(pivot_style or "").strip()

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

    pivot_locs: List[Tuple[str, str]] = []
    if pivots_enabled and str(eff_pivot_style).strip():
        pivot_locs = [
            (f"{_cell_global(r, c)}{_cell_global(r, c)}", str(eff_pivot_style).strip())
            for (r, c) in pivot_positions
        ]

    extra_callouts: List[Dict[str, Any]] = []
    if array_names is not None:
        explicit_names = array_names is not True
        try:
            lhs, rhs = array_names
            rhs_list = list(rhs)
        except Exception:
            lhs, rhs_list = "E", ["A"]
        rhs_list = [str(x) for x in rhs_list]
        if not explicit_names:
            rhs_list = _coerce_rhs_labels(rhs_list, tr.Nrhs)
        name_specs = _legacy_array_name_specs(len(layers["matrices"]), str(lhs), rhs_list, start_index=index_base)
        extra_callouts = _legacy_name_specs_to_callouts(
            layers["matrices"],
            name_specs,
            color="blue",
            legacy_submatrix_names=False,
        )

    if callouts is None or callouts is True:
        if extra_callouts:
            for item in extra_callouts:
                if "name" not in item and "grid_pos" in item:
                    item.pop("grid_pos", None)
            callouts = extra_callouts
        else:
            callouts = callouts
    elif isinstance(callouts, list) and extra_callouts:
        callouts = list(callouts) + extra_callouts

    bg_list = decor.get("bg_list") or []
    codebefore: List[str] = _legacy_bg_list_to_codebefore(layers["matrices"], bg_list) if bg_list else []

    ref_path_list = decor.get("ref_path_list") or []
    rowechelon_paths = (
        _legacy_ref_path_list_to_rowechelon_paths(
            layers["matrices"],
            ref_path_list,
            legacy_submatrix_names=False,
        )
        if ref_path_list
        else decor.get("rowechelon_paths") or []
    )

    pivot_decorators: List[Dict[str, Any]] = []
    if pivots_enabled and decor.get("pivot_list"):
        pivot_decorators = _pivot_list_to_decorators(
            decor.get("pivot_list") or [],
            pivot_text_color=pivot_text_color,
        )

    if pivot_decorators:
        if decorators is None:
            decorators = pivot_decorators
        else:
            decorators = list(decorators) + pivot_decorators

    txt_with_locs = list(decor.get("txt_with_locs") or [])
    variable_labels: Optional[List[Dict[str, Any]]] = None
    eff_variable_summary = variable_summary
    if eff_variable_summary is None:
        eff_variable_summary = decor.get("variable_types") or decor.get("variable_summary")
    if eff_variable_summary:
        variable_labels = _variable_summary_label_rows(
            layers["matrices"],
            eff_variable_summary,
            variable_colors,
        )

    spec: Dict[str, Any] = dict(
        matrices=layers["matrices"],
        Nrhs=int(tr.Nrhs or 0),
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        pivot_locs=pivot_locs,
        txt_with_locs=txt_with_locs,
        variable_labels=variable_labels,
        rowechelon_paths=rowechelon_paths,
        callouts=callouts,
        decorators=decorators,
        fig_scale=fig_scale,
        outer_delims=bool(outer_delims),
        outer_hspace_mm=int(outer_hspace_mm),
        cell_align=str(cell_align),
        strict=bool(strict) if strict is not None else False,
        codebefore=codebefore,
        create_cell_nodes=True if rowechelon_paths else None,
        create_medium_nodes=True if codebefore else None,
    )

    typed_layout = _build_typed_layout_spec(
        pivot_locs=pivot_locs,
        txt_with_locs=txt_with_locs,
        rowechelon_paths=rowechelon_paths,
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
    gj: bool = False,
    show_pivots: Optional[bool] = True,
    index_base: int = 1,
    pivot_style: str = "",
    pivot_text_color: str = "red",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt, left-margin=6pt, right-margin=6pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a layout spec for :func:`matrixlayout.ge.grid_tex`.

    The returned dict contains only primitives + nested lists.
    """

    return _build_ge_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        gj=gj,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        row_stretch=row_stretch,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        array_names=array_names,
        decorators=decorators,
        fig_scale=fig_scale,
        variable_summary=variable_summary,
        variable_colors=variable_colors,
        strict=bool(strict) if strict is not None else False,
    )["spec"]


def ge_tbl_layout_spec(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    gj: bool = False,
    show_pivots: Optional[bool] = True,
    index_base: int = 1,
    pivot_style: str = "",
    pivot_text_color: str = "red",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt, left-margin=6pt, right-margin=6pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a layout spec using :class:`matrixlayout.specs.GELayoutSpec`."""

    bundle = _build_ge_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        gj=gj,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        row_stretch=row_stretch,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        array_names=array_names,
        decorators=decorators,
        fig_scale=fig_scale,
        variable_summary=variable_summary,
        variable_colors=variable_colors,
        strict=bool(strict) if strict is not None else False,
    )

    return {
        "matrices": bundle["layers"]["matrices"],
        "Nrhs": int(bundle["trace"].Nrhs or 0),
        "layout": bundle["typed_layout"],
        "outer_hspace_mm": int(outer_hspace_mm),
        "cell_align": str(cell_align),
        "strict": bool(strict) if strict is not None else False,
        "codebefore": bundle["spec"].get("codebefore"),
        "create_cell_nodes": bundle["spec"].get("create_cell_nodes"),
        "create_medium_nodes": bundle["spec"].get("create_medium_nodes"),
        "variable_labels": bundle["spec"].get("variable_labels"),
    }


def ge(
    matrices: Sequence[Sequence[Any]],
    *,
    Nrhs: Any = 0,
    formatter: Any = latexify,
    pivot_list: Optional[Sequence[Any]] = None,
    bg_for_entries: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    pivot_text_color: str = "red",
    ref_path_list: Optional[Any] = None,
    comment_list: Optional[Any] = None,
    comment_shift_x_mm: float = 50.0,
    comment_shift_y_mm: float = 0.0,
    variable_summary: Optional[Any] = None,
    array_names: Optional[Any] = None,
    start_index: Optional[int] = 1,
    func: Optional[Any] = None,
    fig_scale: Optional[Any] = None,
    outer_hspace_mm: int = 9,
    tmp_dir: Optional[str] = None,
    keep_file: Optional[str] = None,
    output_dir: Optional[Any] = None,
    frame: Any = None,
    decorators: Optional[Sequence[Any]] = None,
    strict: Optional[bool] = None,
    **render_opts: Any,
) -> str:
    """Compatibility wrapper for the original ``itikz.nicematrix.ge`` surface."""

    output_stem: Optional[str] = None
    if keep_file:
        import os
        from pathlib import Path

        p = Path(str(keep_file))
        suffix = p.suffix.lower()
        if suffix in (".tex", ".svg", ".pdf", ".dvi", ".xdv"):
            output_stem = p.stem
        else:
            output_stem = p.name
        if output_dir is None:
            if p.is_absolute():
                output_dir = str(p.parent)
            else:
                if str(p.parent) not in ("", "."):
                    if tmp_dir:
                        output_dir = str(Path(tmp_dir) / p.parent)
                    else:
                        output_dir = str(p.parent)
                elif tmp_dir:
                    output_dir = tmp_dir

    if output_dir is None and tmp_dir:
        output_dir = tmp_dir

    pivot_style = f"draw={pivot_text_color}, inner sep=2pt, outer sep=0pt" if pivot_text_color else ""
    pivot_locs = (
        _legacy_pivot_list_to_pivot_locs(matrices, pivot_list, index_base=1, pivot_style=pivot_style)
        if pivot_list
        else None
    )

    codebefore: List[str] = []
    needs_medium_nodes = False
    if bg_for_entries:
        needs_medium_nodes = True
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
                    fit = f"({c0[1:-1]}-medium) ({c1[1:-1]}-medium)"
                else:
                    i0, j0 = entry
                    c0 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i0), j=int(j0), index_base=1)
                    fit = f"({c0[1:-1]}-medium)"
                codebefore.append(
                    rf"\tikz \node [fill={color}, inner sep = {pt}pt, fit = {fit}] {{}} ;"
                )

    txt_with_locs: List[Tuple[str, str, str]] = []
    variable_labels: Optional[List[Dict[str, Any]]] = None
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
                style = f"right,align=left,text=violet, xshift={comment_shift_x_mm}mm"
                if comment_shift_y_mm:
                    style += f", yshift={comment_shift_y_mm}mm"
                txt_with_locs.append((coord, r"\qquad " + str(txt), style))

    if variable_summary:
        variable_labels = _variable_summary_label_rows(matrices, variable_summary, variable_colors)

    rowechelon_paths: List[str] = []
    if ref_path_list:
        specs = ref_path_list if isinstance(ref_path_list, list) else [ref_path_list]
        rowechelon_paths.extend(
            _legacy_ref_path_list_to_rowechelon_paths(matrices, specs, legacy_submatrix_names=True)
        )

    callouts: List[Dict[str, Any]] = []
    if array_names is not None:
        explicit_names = array_names is not True
        try:
            lhs, rhs = array_names
            rhs_list = list(rhs)
        except Exception:
            lhs, rhs_list = "E", ["A"]
        rhs_list = [str(x) for x in rhs_list]
        if not explicit_names:
            rhs_list = _coerce_rhs_labels(rhs_list, Nrhs)
        n_rows = len(matrices or [])
        name_specs = _legacy_array_name_specs(n_rows, str(lhs), rhs_list, start_index=start_index)
        callouts.extend(_legacy_name_specs_to_callouts(matrices, name_specs, color="blue"))

    if func is not None:
        class _LegacyFuncAdapter:
            def __init__(self):
                self.codebefore = codebefore
                self.txt_with_locs = txt_with_locs
                self.rowechelon_paths = rowechelon_paths

            def nm_background(self, gM, gN, loc_list, color="red!15", pt=0):
                nonlocal needs_medium_nodes
                needs_medium_nodes = True
                specs = [(gM, gN, loc_list, color, pt)]
                for spec in specs:
                    gM, gN, entries, color, pt = spec
                    if not isinstance(entries, (list, tuple)):
                        entries = [entries]
                    for entry in entries:
                        if isinstance(entry, (list, tuple)) and len(entry) == 2 and all(isinstance(x, (list, tuple)) for x in entry):
                            (i0, j0), (i1, j1) = entry
                            c0 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i0), j=int(j0), index_base=1)
                            c1 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i1), j=int(j1), index_base=1)
                            fit = f"({c0[1:-1]}-medium) ({c1[1:-1]}-medium)"
                        else:
                            i0, j0 = entry
                            c0 = _grid_cell_coord(matrices, gM=int(gM), gN=int(gN), i=int(i0), j=int(j0), index_base=1)
                            fit = f"({c0[1:-1]}-medium)"
                        self.codebefore.append(
                            rf"\tikz \node [fill={color}, inner sep = {pt}pt, fit = {fit}] {{}} ;"
                        )

            def nm_add_rowechelon_path(self, gM, gN, pivots, case="hh", color="blue,line width=0.4mm", adj=0.1):
                specs = [(gM, gN, pivots, case, color, adj)]
                self.rowechelon_paths.extend(
                    _legacy_ref_path_list_to_rowechelon_paths(matrices, specs, legacy_submatrix_names=True)
                )

            def nm_text(self, txt_list, color="violet"):
                if not txt_list:
                    return
                _, block_widths, row_starts, col_starts = _grid_offsets(matrices, index_base=1)
                if block_widths and col_starts:
                    last_col_start = col_starts[-1]
                    last_col_width = block_widths[-1]
                    comment_col = last_col_start + max(last_col_width - 1, 0)
                    for g, txt in enumerate(txt_list):
                        if g >= len(row_starts):
                            break
                        row = row_starts[g]
                        coord = f"({row}-{comment_col}.east)"
                        style = f"right,align=left,text={color}, xshift={comment_shift_x_mm}mm"
                        if comment_shift_y_mm:
                            style += f", yshift={comment_shift_y_mm}mm"
                        self.txt_with_locs.append((coord, r"\qquad " + str(txt), style))

        func(_LegacyFuncAdapter())

    from matrixlayout.ge import grid_svg

    return grid_svg(
        matrices=matrices,
        Nrhs=Nrhs,
        formatter=formatter,
        outer_hspace_mm=outer_hspace_mm,
        legacy_submatrix_names=True,
        legacy_format=True,
        pivot_locs=pivot_locs,
        codebefore=codebefore,
        txt_with_locs=txt_with_locs,
        variable_labels=variable_labels,
        rowechelon_paths=rowechelon_paths,
        callouts=callouts or None,
        create_extra_nodes=True if (ref_path_list or needs_medium_nodes) else None,
        create_medium_nodes=True if (ref_path_list or needs_medium_nodes) else None,
        fig_scale=fig_scale,
        decorators=decorators,
        strict=bool(strict) if strict is not None else False,
        output_dir=output_dir,
        output_stem=output_stem or "output",
        frame=frame,
        **render_opts,
    )


def ge_tbl_bundle(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    gj: bool = False,
    show_pivots: Optional[bool] = True,
    index_base: int = 1,
    pivot_style: str = "",
    pivot_text_color: str = "red",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt, left-margin=6pt, right-margin=6pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a bundle containing the trace, decorations, spec, and TeX."""

    bundle = _build_ge_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        gj=gj,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        row_stretch=row_stretch,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        array_names=array_names,
        decorators=decorators,
        fig_scale=fig_scale,
        variable_summary=variable_summary,
        variable_colors=variable_colors,
        strict=bool(strict) if strict is not None else False,
    )

    # Prefer the notebook-oriented bundle API when available. This avoids
    # regex-parsing generated TeX to discover \SubMatrix names.
    from dataclasses import asdict

    try:
        from matrixlayout.ge import grid_bundle

        gb = grid_bundle(**bundle["spec"])
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
        from matrixlayout.ge import grid_tex

        tex = grid_tex(**bundle["spec"])
        bundle["tex"] = tex
    return bundle


def ge_tbl_tex(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    gj: bool = False,
    show_pivots: Optional[bool] = True,
    index_base: int = 1,
    pivot_style: str = "",
    pivot_text_color: str = "red",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt, left-margin=6pt, right-margin=6pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> str:
    """Compute a GE-table TeX document (no rendering)."""

    return ge_tbl_bundle(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        gj=gj,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        row_stretch=row_stretch,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        array_names=array_names,
        decorators=decorators,
        fig_scale=fig_scale,
        variable_summary=variable_summary,
        variable_colors=variable_colors,
        strict=bool(strict) if strict is not None else False,
    )["tex"]


def ge_tbl_svg(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    rhs: Any = None,
    pivoting: str = "none",
    gj: bool = False,
    show_pivots: Optional[bool] = False,
    index_base: int = 1,
    pivot_style: str = "",
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    extension: str = "",
    row_stretch: Optional[float] = None,
    nice_options: str = "vlines-in-sub-matrix = I",
    outer_delims: bool = False,
    outer_hspace_mm: int = 6,
    cell_align: str = "r",
    toolchain_name: Optional[str] = None,
    crop: Optional[str] = None,
    padding: Tuple[int, int, int, int] = (2, 2, 2, 2),
    frame: Any = None,
    callouts: Optional[Any] = None,
    array_names: Optional[Any] = None,
    decorators: Optional[Sequence[Any]] = None,
    fig_scale: Optional[Any] = None,
    variable_summary: Optional[Any] = None,
    variable_colors: Sequence[str] = ("red", "black"),
    strict: Optional[bool] = None,
) -> str:
    """Render a GE-table to SVG via matrixlayout (strict rendering boundary)."""

    spec = ge_tbl_spec(
        ref_A,
        ref_rhs,
        rhs=rhs,
        pivoting=pivoting,
        gj=gj,
        show_pivots=show_pivots,
        index_base=index_base,
        pivot_style=pivot_style,
        preamble=preamble,
        extension=extension,
        row_stretch=row_stretch,
        nice_options=nice_options,
        outer_delims=outer_delims,
        outer_hspace_mm=outer_hspace_mm,
        cell_align=cell_align,
        callouts=callouts,
        array_names=array_names,
        decorators=decorators,
        fig_scale=fig_scale,
        variable_summary=variable_summary,
        variable_colors=variable_colors,
        strict=bool(strict) if strict is not None else False,
    )

    from matrixlayout.ge import grid_svg

    return grid_svg(toolchain_name=toolchain_name, crop=crop, padding=padding, frame=frame, **spec)
