"""Gaussian elimination trace builders.

This module owns the *algorithmic* portion of the GE migration.

It computes a step-by-step elimination trace (elementary matrices + intermediate
states) suitable for Gaussian-elimination figures.

The returned values are **description objects**:

- they contain SymPy matrices and pivot metadata,
- they do not produce TeX strings,
- they do not call any rendering/toolchain functions.

Once the grid layout engine is migrated into :mod:`matrixlayout`, callers can
use :func:`ge_trace` to compute the elimination steps and pass the resulting
trace into a layout builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import sympy as sym

from ._sympy_utils import to_sympy_col, to_sympy_matrix


Pivoting = Literal["none", "partial"]

@dataclass(frozen=True)
class GEEvent:
    """One logical operation/event emitted by :func:`ge_trace`.

    This is intentionally primitive-only (op string + dict payload) so it
    can cross language boundaries (e.g., Julia via PyCall/PythonCall).
    """

    op: str
    level: int
    data: Mapping[str, Any]



@dataclass(frozen=True)
class GEStep:
    """One elimination step.

    Attributes
    ----------
    E:
        The elementary matrix applied to the *previous* augmented matrix.
        The updated state satisfies ``Ab_next == E * Ab_prev``.
    Ab:
        The augmented matrix state *after* applying ``E``.
    pivot:
        Pivot position ``(row, col)`` in the coefficient columns of ``Ab``.
    """

    E: sym.Matrix
    Ab: sym.Matrix
    pivot: Tuple[int, int]


@dataclass(frozen=True)
class GETrace:
    """Gaussian elimination trace description.

    This is a purely algorithmic object intended for downstream presentation.

    Attributes
    ----------
    initial:
        The initial augmented matrix ``[A | b]`` (or just ``A`` if no RHS was
        provided).
    steps:
        A sequence of elimination steps.
    pivot_cols:
        Pivot columns in the coefficient block.
    free_cols:
        Non-pivot columns in the coefficient block.
    pivot_positions:
        Pivot positions as ``(row, col)`` pairs.
    Nrhs:
        Number of right-hand-side columns appended to ``A`` (0 if none).
    meta:
        Miscellaneous metadata (shapes, pivoting mode, etc.).
    """

    initial: sym.Matrix
    steps: Sequence[GEStep]
    events: Sequence[GEEvent]
    pivot_cols: Sequence[int]
    free_cols: Sequence[int]
    pivot_positions: Sequence[Tuple[int, int]]
    Nrhs: int
    meta: Mapping[str, Any]


    def events_as_dicts(self) -> List[Dict[str, Any]]:
        """Return the event stream as JSON-friendly dictionaries."""

        out: List[Dict[str, Any]] = []
        for ev in getattr(self, "events", ()):
            data: Dict[str, Any] = {}
            for k, v in dict(ev.data).items():
                # Convert common SymPy scalars to plain Python values.
                if isinstance(v, sym.Integer):
                    data[k] = int(v)
                elif isinstance(v, sym.Rational):
                    data[k] = (int(v.p), int(v.q))
                elif isinstance(v, sym.Basic):
                    data[k] = str(v)
                else:
                    data[k] = v
            out.append({"op": str(ev.op), "level": int(ev.level), "data": data})
        return out


def _is_nonzero(x: sym.Expr) -> bool:
    """Best-effort check for whether ``x`` is nonzero.

    SymPy may not be able to decide nonzeroness for symbolic expressions.
    For GE traces used in figures, callers typically supply numeric/rational
    matrices, in which case this works reliably.

    If the value is undecidable, this returns ``True`` to avoid accidentally
    skipping a candidate pivot.
    """

    z = getattr(x, "is_zero", None)
    if z is True:
        return False
    if z is False:
        return True
    try:
        # For most numeric/rational values this will be a Python bool.
        return bool(x != 0)
    except Exception:
        return True


def _find_pivot_row(Ab: sym.Matrix, row0: int, col: int, *, pivoting: Pivoting) -> Optional[int]:
    """Choose a pivot row at or below ``row0`` for a given column.

    Parameters
    ----------
    Ab:
        Current augmented matrix state.
    row0:
        First row eligible to be a pivot.
    col:
        Coefficient-column index.
    pivoting:
        ``"none"`` chooses the first nonzero entry.
        ``"partial"`` chooses the row with largest ``abs(entry)`` when entries
        are numeric; otherwise it falls back to the first nonzero entry.
    """

    m = Ab.rows
    candidates: List[int] = [r for r in range(row0, m) if _is_nonzero(Ab[r, col])]
    if not candidates:
        return None

    if pivoting != "partial":
        return candidates[0]

    # Partial pivoting: only safe when candidates are numeric.
    def _is_numeric(v: sym.Expr) -> bool:
        return bool(getattr(v, "is_number", False)) and v.is_real is not False

    if all(_is_numeric(Ab[r, col]) for r in candidates):
        return max(candidates, key=lambda r: abs(Ab[r, col]))

    return candidates[0]


def _swap_matrix(m: int, i: int, j: int) -> sym.Matrix:
    """Return the elementary matrix that swaps rows ``i`` and ``j``."""

    P = sym.eye(m)
    if i == j:
        return P
    P[i, i] = 0
    P[j, j] = 0
    P[i, j] = 1
    P[j, i] = 1
    return P


def _elimination_matrices(
    Ab: sym.Matrix,
    pivot_row: int,
    pivot_col: int,
    *,
    gj: bool = False,
    consolidate: bool = True,
) -> List[Tuple[sym.Matrix, int, sym.Expr]]:
    """Return elementary matrices to eliminate entries below a pivot.

    Each elementary matrix eliminates a single row:
        row_r <- row_r - (Ab[r,p]/Ab[p,p]) * row_p

    Returning per-row matrices matches the Julia reference behavior and
    preserves a step-by-step elimination trace.
    """

    m = Ab.rows
    piv = Ab[pivot_row, pivot_col]
    if not _is_nonzero(piv):
        return []

    if gj:
        E = sym.eye(m)
        for r in range(m):
            if r == pivot_row:
                continue
            a = Ab[r, pivot_col]
            if not _is_nonzero(a):
                continue
            factor = a / piv
            E[r, pivot_row] = -factor
        return [(E, pivot_row, sym.Integer(0))]

    if consolidate:
        E = sym.eye(m)
        for r in range(pivot_row + 1, m):
            a = Ab[r, pivot_col]
            if not _is_nonzero(a):
                continue
            factor = a / piv
            E[r, pivot_row] = -factor
        if E == sym.eye(m):
            return []
        return [(E, pivot_row, sym.Integer(0))]

    out: List[Tuple[sym.Matrix, int, sym.Expr]] = []
    for r in range(pivot_row + 1, m):
        a = Ab[r, pivot_col]
        if not _is_nonzero(a):
            continue
        E = sym.eye(m)
        factor = a / piv
        E[r, pivot_row] = -factor
        out.append((E, r, factor))
    return out


def _has_nonzero_entry(Ab: sym.Matrix, pivot_row: int, pivot_col: int, *, gj: bool = False) -> bool:
    m = Ab.rows
    for r in range(pivot_row + 1, m):
        if _is_nonzero(Ab[r, pivot_col]):
            return True
    if gj:
        for r in range(0, pivot_row):
            if _is_nonzero(Ab[r, pivot_col]):
                return True
    return False


def ge_trace(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    pivoting: Pivoting = "none",
    gj: bool = False,
    consolidate_elimination: bool = True,
) -> GETrace:
    """Compute a Gaussian elimination trace for ``A`` (optionally augmented).

    Parameters
    ----------
    ref_A:
        Coefficient matrix ``A``.
    ref_rhs:
        Optional RHS. May be a vector (m,) or (m,1) or a matrix (m,k).
    pivoting:
        Pivot selection strategy. ``"partial"`` uses max-abs pivoting for numeric
        matrices; otherwise it behaves like ``"none"``.
    gj:
        If True, perform Gauss-Jordan elimination (eliminate above the pivot and
        emit a scaling step when needed).

    Returns
    -------
    GETrace
        A trace containing the initial augmented matrix, a list of elementary
        matrices and intermediate states, and pivot/free variable metadata.
    """

    A = to_sympy_matrix(ref_A)
    if A is None:
        raise ValueError("ref_A must not be None")

    rhs = to_sympy_matrix(ref_rhs)
    if rhs is None:
        Ab = sym.Matrix(A)
        Nrhs = 0
    else:
        # Accept vectors and matrix RHS.
        if rhs.rows != A.rows:
            raise ValueError(f"rhs has {rhs.rows} rows but A has {A.rows}")
        Ab = sym.Matrix(A).row_join(sym.Matrix(rhs))
        Nrhs = rhs.cols

    m, n_aug = Ab.shape
    n_coef = A.cols

    steps: List[GEStep] = []
    events: List[GEEvent] = []
    pivot_cols: List[int] = []
    pivot_positions: List[Tuple[int, int]] = []

    row = 0
    cur = sym.Matrix(Ab)

    for col in range(n_coef):
        if row >= m:
            break

        pr = _find_pivot_row(cur, row, col, pivoting=pivoting)
        if pr is None:
            continue

        # Record pivot metadata once per pivot *column* (not once per displayed layer).
        # This intentionally matches the Julia reference implementation: the final
        # REF/RREF pivot set is independent of whether an explicit elementary
        # operation is required at that pivot (e.g., last pivot column has no rows
        # below to eliminate).
        pivot_cols.append(col)
        pivot_positions.append((row, col))

        # ------------------------------------------------------------------
        # Row exchange step (if needed): emitted as its own layer.
        # ------------------------------------------------------------------
        if pr != row:
            events.append(
                GEEvent(
                    op="RequireRowExchange",
                    level=len(steps),
                    data={"row_1": int(row), "row_2": int(pr), "col": int(col)},
                )
            )
            P = _swap_matrix(m, row, pr)
            cur = P * cur
            steps.append(GEStep(E=P, Ab=cur, pivot=(row, col)))
            events.append(
                GEEvent(
                    op="DoRowExchange",
                    level=len(steps),
                    data={"row_1": int(row), "row_2": int(pr), "col": int(col)},
                )
            )

        # Found pivot (reported after any exchange so the pivot is on `row`).
        events.append(
            GEEvent(
                op="FoundPivot",
                level=len(steps),
                data={
                    "row": int(row),
                    "pivot_row": int(pr),
                    "pivot_col": int(col),
                    "rank": int(len(pivot_cols)),
                },
            )
        )

        # ------------------------------------------------------------------
        # Elimination step (if needed): emitted as its own layer.
        # Skip the step entirely when the elimination matrix would be identity.
        # ------------------------------------------------------------------
        needs_elim = _has_nonzero_entry(cur, row, col, gj=gj)
        elim_ops = _elimination_matrices(cur, row, col, gj=gj, consolidate=consolidate_elimination) if needs_elim else []
        if elim_ops:
            events.append(
                GEEvent(
                    op="RequireElimination",
                    level=len(steps),
                    data={
                        "pivot_row": int(row),
                        "pivot_col": int(col),
                        "count": int(len(elim_ops)),
                        "gj": bool(gj),
                        "yes": True,
                    },
                )
            )
            for Eelim, target_row, factor in elim_ops:
                cur = Eelim * cur
                steps.append(GEStep(E=Eelim, Ab=cur, pivot=(row, col)))
                events.append(
                    GEEvent(
                        op="DoElimination",
                        level=len(steps),
                        data={
                            "pivot_row": int(row),
                            "pivot_col": int(col),
                            "target_row": int(target_row),
                            "targets": list(range(int(row) + 1, m)) if consolidate_elimination else [int(target_row)],
                            "factor": factor,
                            "gj": bool(gj),
                        },
                    )
                )
        else:
            # Retain an explicit “no elimination required” event for consumers.
            events.append(
                GEEvent(
                    op="RequireElimination",
                    level=len(steps),
                    data={"pivot_row": int(row), "pivot_col": int(col), "yes": False, "gj": bool(gj)},
                )
            )

        row += 1

    free_cols = [j for j in range(n_coef) if j not in pivot_cols]

    if gj and m > 0:
        require_scaling = False
        scale_factors: List[sym.Expr] = []
        for i, col_idx in enumerate(pivot_cols):
            if i >= m:
                break
            piv = cur[i, col_idx]
            if not _is_nonzero(piv) or not sym.simplify(piv - 1).is_zero:
                require_scaling = True
            scale_factors.append(piv)
        if require_scaling and scale_factors:
            E = sym.eye(m)
            for i, piv in enumerate(scale_factors):
                if not _is_nonzero(piv):
                    continue
                E[i, i] = sym.Integer(1) / piv
            events.append(
                GEEvent(
                    op="RequireScaling",
                    level=len(steps),
                    data={"pivot_cols": list(pivot_cols)},
                )
            )
            cur = E * cur
            steps.append(GEStep(E=E, Ab=cur, pivot=(pivot_positions[-1] if pivot_positions else (0, 0))))
            events.append(GEEvent(op="DoScaling", level=len(steps), data={}))

    events.append(GEEvent(op="Finished", level=len(steps), data={"rank": int(len(pivot_cols))}))

    return GETrace(
        initial=Ab,
        steps=tuple(steps),
        events=tuple(events),
        pivot_cols=tuple(pivot_cols),
        free_cols=tuple(free_cols),
        pivot_positions=tuple(pivot_positions),
        Nrhs=Nrhs,
        meta={
            "shape": tuple(A.shape),
            "aug_shape": tuple(Ab.shape),
            "rank": len(pivot_cols),
            "pivoting": pivoting,
            "gj": bool(gj),
        },
    )


def trace_to_layer_matrices(
    trace: GETrace,
    *,
    augmented: bool = True,
) -> Dict[str, Any]:
    """Convert a :class:`GETrace` to a legacy-style ``matrices`` layout input.

    This returns the nested list shape historically used by
    ``itikz.nicematrix.ge`` / ``MatrixGridLayout``:

        [ [None, A0], [E1, A1], [E2, A2], ... ]

    where each ``Ei`` is the elementary matrix applied at step ``i`` and each
    ``Ai`` is the matrix after that step.

    Parameters
    ----------
    trace:
        The elimination trace.
    augmented:
        If True, ``Ai`` is the augmented matrix ``[A|b]`` (when RHS exists).
        If False, ``Ai`` is just the coefficient matrix block.

    Returns
    -------
    dict
        Keys:
        - ``matrices``: nested list of SymPy matrices
        - ``Nrhs``: RHS column count (for future partition-line insertion)
        - ``pivot_positions`` / ``pivot_cols`` / ``free_cols``
        - ``meta``: metadata passthrough
    """

    n_coef = trace.meta.get("shape", (None, None))[1]
    if n_coef is None:
        n_coef = trace.initial.cols - trace.Nrhs

    def _coef_block(M: sym.Matrix) -> sym.Matrix:
        return M[:, :n_coef]

    layers: List[List[Optional[sym.Matrix]]] = []

    A0 = trace.initial if augmented else _coef_block(trace.initial)
    layers.append([None, A0])

    for st in trace.steps:
        Ak = st.Ab if augmented else _coef_block(st.Ab)
        layers.append([st.E, Ak])

    return {
        "matrices": layers,
        "Nrhs": trace.Nrhs,
        "pivot_positions": list(trace.pivot_positions),
        "pivot_cols": list(trace.pivot_cols),
        "free_cols": list(trace.free_cols),
        "meta": dict(trace.meta),
    }


def decorate_ge(
    trace: GETrace,
    *,
    index_base: int = 1,
    pivot_style: str = "",
    ref_path_case: str = "vh",
    ref_path_block_col: int = 1,
    pivot_block_col: int = 1,
    pivot_color: str = "yellow!40",
    missing_pivot_color: str = "gray!20",
    path_color: str = "blue,line width=0.5mm",
) -> Dict[str, Any]:
    """Compute presentation decorations from a :class:`GETrace`.

    This is the Python analogue of the Julia-side ``decorate_ge`` helper.
    It is intentionally *data-only*: it returns plain Python containers
    (strings, ints, lists, tuples) so the result can be passed across
    language boundaries (e.g. Julia via PyCall/PythonCall) and later
    consumed by layout/rendering code.

    Parameters
    ----------
    trace:
        Gaussian-elimination trace returned by :func:`ge_trace`.
    index_base:
        Coordinate base for cell references. Use 1 for nicematrix coordinates.
        Set to 0 if your upstream producer emits 0-based coordinates.
    pivot_style:
        Optional TikZ style fragment appended to pivot "fit" nodes when the
        decoration is rendered (e.g. ``"thick, draw=red"``).

    Returns
    -------
    dict
        Keys are chosen to be directly usable by downstream layout builders:

        - ``pivot_locs``: list of ``(fit_target, extra_style)`` where
          ``fit_target`` is a string like ``"(2-3)(2-3)"``.
        - ``variable_types``: list of booleans indicating pivot columns for each
          coefficient column (RHS columns excluded).

        Additional keys mirror the Julia ``decorate_ge`` helper:
        ``pivot_list``, ``bg_list``, ``path_list`` (aka ``ref_path_list``),
        ``variable_summary``. Placeholder keys ``txt_with_locs`` and
        ``rowechelon_paths`` are included for future increments.
    """

    # Number of coefficient columns. Prefer the recorded shape metadata.
    n_coef = None
    shp = trace.meta.get("shape") if isinstance(trace.meta, dict) else None
    if shp and len(shp) == 2:
        n_coef = int(shp[1])
    if n_coef is None:
        n_coef = int(trace.initial.cols - int(trace.Nrhs or 0))

    pivot_set = set(int(c) for c in trace.pivot_cols)
    variable_types: List[bool] = [(j in pivot_set) for j in range(n_coef)]

    def _cell(r: int, c: int) -> str:
        rr = int(r) + int(index_base)
        cc = int(c) + int(index_base)
        return f"({rr}-{cc})"

    pivot_locs: List[Tuple[str, str]] = [
        (f"{_cell(r, c)}{_cell(r, c)}", str(pivot_style).strip())
        for (r, c) in trace.pivot_positions
    ]

    ref_path_list: List[Tuple[int, int, List[Tuple[int, int]], str]] = []
    if trace.pivot_positions:
        ref_path_list.append(
            (
                len(trace.steps),
                int(ref_path_block_col),
                list(trace.pivot_positions),
                str(ref_path_case),
            )
        )

    # Julia-style decorations for legacy compatibility.
    M = int(shp[0]) if shp and len(shp) == 2 else int(trace.initial.rows)
    N = int(n_coef)
    pivot_cols_1b: List[int] = [int(c) + 1 for c in trace.pivot_cols]

    def _plist(cols_1b: List[int]) -> List[Tuple[int, int]]:
        return [(row, cols_1b[row] - 1) for row in range(len(cols_1b))]

    pivot_dict: Dict[Tuple[int, int], Any] = {}
    bg_dict: Dict[Tuple[int, int], Any] = {}
    path_dict: Dict[Tuple[int, int], Any] = {}

    current_cols: List[int] = []
    update = True

    def _ensure_col(col_0b: int) -> None:
        col_1b = int(col_0b) + 1
        if col_1b not in current_cols:
            current_cols.append(col_1b)

    for ev in trace.events:
        level = int(ev.level)
        if ev.op == "RequireRowExchange":
            try:
                _ensure_col(ev.data["col"])
                row_1 = int(ev.data["row_1"])
                row_2 = int(ev.data["row_2"])
                col = int(ev.data["col"])
            except Exception:
                continue
            length = len(current_cols)
            if length >= 2:
                bg_dict[(level, 1)] = [
                    [level, 1, [(row_1, col), (row_2, col)], missing_pivot_color],
                    [level, 1, _plist(current_cols[:-1]), pivot_color],
                ]
            elif length == 1:
                bg_dict[(level, 1)] = [level, 1, [(row_1, col), (row_2, col)], missing_pivot_color]
            if length != 0:
                pl = _plist(current_cols)
                pivot_dict[(level, 1)] = [(level, 1), pl]
                bg_dict[(level, 1)] = [level, 1, pl, pivot_color]
                path_dict[(level, 1)] = [level, 1, pl, "vv", path_color]
            update = True
        elif ev.op == "FoundPivot":
            try:
                _ensure_col(ev.data["pivot_col"])
            except Exception:
                pass
            pl = _plist(current_cols)
            pivot_dict[(level, 1)] = [(level, 1), pl]
            bg_dict[(level, 1)] = [level, 1, pl, pivot_color]
            update = True
        elif ev.op == "RequireElimination":
            try:
                row = int(ev.data["pivot_row"])
                col = int(ev.data["pivot_col"])
            except Exception:
                continue
            first = 0 if ev.data.get("gj") else row
            new_bg = [level, 1, [(row, col), [(first, col), (M - 1, col)]], pivot_color, 1]
            if (level, 1) in bg_dict:
                bg_dict[(level, 1)] = [bg_dict[(level, 1)], new_bg]
            else:
                bg_dict[(level, 1)] = new_bg
            yes = ev.data.get("yes", True)
            pl = _plist(current_cols)
            path_dict[(level, 1)] = [level, 1, pl, "vh" if yes is False else "vv", path_color]
        elif ev.op == "DoElimination":
            try:
                c = int(ev.data["pivot_row"])
            except Exception:
                continue
            pivot_dict[(level, 0)] = [(level, 0), [(c, c)]]
            if ev.data.get("gj"):
                path_dict[(level, 0)] = [level, 0, [(0, c)], "vv", path_color]
            else:
                path_dict[(level, 0)] = [level, 0, [(c, c)], "vv", path_color]
            if ev.data.get("gj"):
                bg_dict[(level, 0)] = [level, 0, [(c, c), [(0, c), (M - 1, c)]], pivot_color, 1]
            else:
                bg_dict[(level, 0)] = [level, 0, [(c, c), [(c, c), (M - 1, c)]], pivot_color, 1]
        elif ev.op == "DoRowExchange":
            try:
                row_1 = int(ev.data["row_1"])
                row_2 = int(ev.data["row_2"])
            except Exception:
                continue
            pl = [(row_1, row_2), (row_2, row_1)]
            pivot_dict[(level, 0)] = [(level, 0), pl]
            bg_dict[(level, 0)] = [level, 0, pl, missing_pivot_color]
        elif ev.op == "RequireScaling":
            if current_cols:
                pl = _plist(current_cols)
                if update:
                    pivot_dict[(level, 1)] = [(level, 1), pl]
                    bg_dict[(level, 1)] = [level, 1, pl, pivot_color]
                path_dict[(level, 1)] = [level, 1, pl, "vh", path_color]
            update = True
        elif ev.op == "DoScaling":
            pl = [(c, c) for c in range(M)]
            pivot_dict[(level, 0)] = [(level, 0), pl]
            bg_dict[(level, 0)] = [level, 0, pl, pivot_color]
        elif ev.op == "Finished":
            if current_cols:
                pl = _plist(current_cols)
                pivot_dict[(level, 1)] = [(level, 1), pl]
                bg_dict[(level, 1)] = [level, 1, pl, pivot_color]
                path_dict[(level, 1)] = [level, 1, pl, "vh", path_color]
            update = True

    pivot_list = [pivot_dict[k] for k in sorted(pivot_dict.keys())]
    bg_list = [bg_dict[k] for k in sorted(bg_dict.keys())]
    path_list = [path_dict[k] for k in sorted(path_dict.keys())]

    variable_summary: List[bool] = list(variable_types)

    return {
        "pivot_locs": pivot_locs,
        "variable_types": variable_types,
        "pivot_list": pivot_list,
        "bg_list": bg_list,
        "path_list": path_list,
        "ref_path_list": ref_path_list or path_list,
        "variable_summary": variable_summary,
        # Retain trace metadata for downstream consumers.
        "pivot_positions": list(trace.pivot_positions),
        "pivot_cols": list(trace.pivot_cols),
        "free_cols": list(trace.free_cols),
        # Placeholders for later GE decoration increments.
        "txt_with_locs": [],
        "rowechelon_paths": [],
        "callouts": [],
    }
