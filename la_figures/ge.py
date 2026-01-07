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


def ge_trace(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    pivoting: Pivoting = "none",
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
        elim_ops = _elimination_matrices(cur, row, col)
        if elim_ops:
            events.append(
                GEEvent(
                    op="RequireElimination",
                    level=len(steps),
                    data={
                        "pivot_row": int(row),
                        "pivot_col": int(col),
                        "count": int(len(elim_ops)),
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
                            "factor": factor,
                        },
                    )
                )
        else:
            # Retain an explicit “no elimination required” event for consumers.
            events.append(
                GEEvent(
                    op="RequireElimination",
                    level=len(steps),
                    data={"pivot_row": int(row), "pivot_col": int(col), "yes": False},
                )
            )

        row += 1

    free_cols = [j for j in range(n_coef) if j not in pivot_cols]

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
        - ``variable_types``: list of ``"pivot"`` / ``"free"`` for each
          coefficient column (RHS columns excluded).

        Additional keys are returned as empty lists for future increments:
        ``bg_list``, ``path_list``, ``txt_with_locs``, ``rowechelon_paths``.
    """

    # Number of coefficient columns. Prefer the recorded shape metadata.
    n_coef = None
    shp = trace.meta.get("shape") if isinstance(trace.meta, dict) else None
    if shp and len(shp) == 2:
        n_coef = int(shp[1])
    if n_coef is None:
        n_coef = int(trace.initial.cols - int(trace.Nrhs or 0))

    pivot_set = set(int(c) for c in trace.pivot_cols)
    variable_types: List[str] = [
        ("pivot" if j in pivot_set else "free") for j in range(n_coef)
    ]

    def _cell(r: int, c: int) -> str:
        rr = int(r) + int(index_base)
        cc = int(c) + int(index_base)
        return f"({rr}-{cc})"

    pivot_locs: List[Tuple[str, str]] = [
        (f"{_cell(r, c)}{_cell(r, c)}", str(pivot_style).strip())
        for (r, c) in trace.pivot_positions
    ]

    return {
        "pivot_locs": pivot_locs,
        "variable_types": variable_types,
        # Retain trace metadata for downstream consumers.
        "pivot_positions": list(trace.pivot_positions),
        "pivot_cols": list(trace.pivot_cols),
        "free_cols": list(trace.free_cols),
        # Placeholders for later GE decoration increments.
        "bg_list": [],
        "path_list": [],
        "txt_with_locs": [],
        "rowechelon_paths": [],
        "callouts": [],
    }
