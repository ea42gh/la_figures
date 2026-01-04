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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import sympy as sym

from ._sympy_utils import to_sympy_col, to_sympy_matrix


Pivoting = Literal["none", "partial"]


@dataclass(frozen=True)
class GEEvent:
    """A single action/event emitted during elimination.

    This event stream is designed to be easily serializable and usable for
    downstream figure-decoration logic.

    The ``level`` field is an integer suitable for indexing a matrix-stack
    representation where level 0 is the initial matrix and level k corresponds
    to the state after applying the k-th stored elementary matrix.
    """

    op: str
    level: int
    data: Mapping[str, Any] = field(default_factory=dict)


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
    pivot_cols: Sequence[int]
    free_cols: Sequence[int]
    pivot_positions: Sequence[Tuple[int, int]]
    Nrhs: int
    meta: Mapping[str, Any]
    events: Sequence[GEEvent] = field(default_factory=tuple)

    def events_as_dicts(self) -> List[Dict[str, Any]]:
        """Return the event stream as JSON-friendly dictionaries."""

        return [
            {"op": ev.op, "level": ev.level, **({} if not ev.data else {"data": dict(ev.data)})}
            for ev in self.events
        ]


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


def _elimination_matrix(
    Ab: sym.Matrix,
    pivot_row: int,
    pivot_col: int,
    *,
    gj: bool = False,
) -> sym.Matrix:
    """Return an elementary matrix eliminating entries below a pivot.

    The returned matrix ``E`` performs:
        row_r <- row_r - (Ab[r,p]/Ab[p,p]) * row_p
    for every row r below the pivot row where the entry is nonzero.

    The elimination is aggregated into a single lower-triangular elementary
    matrix so that one :class:`GEStep` corresponds to one pivot column.
    """

    m = Ab.rows
    E = sym.eye(m)

    piv = Ab[pivot_row, pivot_col]
    if not _is_nonzero(piv):
        # Caller asked to eliminate around a zero pivot; return identity.
        return E

    # Below pivot.
    for r in range(pivot_row + 1, m):
        a = Ab[r, pivot_col]
        if _is_nonzero(a):
            E[r, pivot_row] = -a / piv

    # Above pivot (Gauss-Jordan).
    if gj and pivot_row > 0:
        for r in range(0, pivot_row):
            a = Ab[r, pivot_col]
            if _is_nonzero(a):
                E[r, pivot_row] = -a / piv

    return E


def ge_trace(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    pivoting: Pivoting = "partial",
    gj: bool = False,
    n: Optional[int] = None,
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
        If True, perform Gauss-Jordan elimination (also eliminate above pivots).
        (Pivot normalization is intentionally left to a later migration step.)
    n:
        If provided, only the first ``n`` coefficient columns are reduced.

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

    m, _n_aug = Ab.shape
    n_coef = A.cols
    if n is not None:
        n_coef = min(n_coef, int(n))

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

        # The step index is the level of the *resulting* state after the
        # net elementary transformation.
        level = len(steps) + 1

        if pr != row:
            events.append(
                GEEvent(
                    op="RequireRowExchange",
                    level=level,
                    data={
                        "row_1": row,
                        "row_2": pr,
                        "col": col,
                        "cur_rank": len(pivot_cols) + 1,
                        "pivot_cols": tuple(pivot_cols) + (col,),
                    },
                )
            )

        # Row swap (if needed).
        P = _swap_matrix(m, row, pr)
        cur_swapped = P * cur

        if pr != row:
            events.append(
                GEEvent(
                    op="DoRowExchange",
                    level=level,
                    data={
                        "row_1": row,
                        "row_2": pr,
                        "col": col,
                        "cur_rank": len(pivot_cols) + 1,
                    },
                )
            )

        events.append(
            GEEvent(
                op="FoundPivot",
                level=level,
                data={
                    "row": row,
                    "pivot_row": row,
                    "pivot_col": col,
                    "cur_rank": len(pivot_cols) + 1,
                    "pivot_cols": tuple(pivot_cols) + (col,),
                },
            )
        )

        # Eliminate below (and optionally above) pivot.
        Eelim = _elimination_matrix(cur_swapped, row, col, gj=gj)
        Enet = Eelim * P
        cur = Eelim * cur_swapped

        # Determine whether an elimination was needed.
        def _needs_elim() -> bool:
            below = any(_is_nonzero(cur_swapped[r, col]) for r in range(row + 1, m))
            above = gj and any(_is_nonzero(cur_swapped[r, col]) for r in range(0, row))
            return bool(below or above)

        need_elim = _needs_elim()
        events.append(
            GEEvent(
                op="RequireElimination",
                level=level,
                data={
                    "gj": gj,
                    "yes": need_elim,
                    "row": row,
                    "col": col,
                    "cur_rank": len(pivot_cols) + 1,
                    "pivot_cols": tuple(pivot_cols) + (col,),
                },
            )
        )

        steps.append(GEStep(E=Enet, Ab=cur, pivot=(row, col)))
        if need_elim:
            events.append(
                GEEvent(
                    op="DoElimination",
                    level=level,
                    data={
                        "pivot_row": row,
                        "pivot_col": col,
                        "gj": gj,
                    },
                )
            )
        pivot_cols.append(col)
        pivot_positions.append((row, col))
        row += 1

    free_cols = [j for j in range(n_coef) if j not in pivot_cols]

    return GETrace(
        initial=Ab,
        steps=tuple(steps),
        pivot_cols=tuple(pivot_cols),
        free_cols=tuple(free_cols),
        pivot_positions=tuple(pivot_positions),
        Nrhs=Nrhs,
        meta={
            "shape": tuple(A.shape),
            "aug_shape": tuple(Ab.shape),
            "rank": len(pivot_cols),
            "pivoting": pivoting,
            "gj": gj,
            "n": n_coef,
        },
        events=tuple(events + [GEEvent(op="Finished", level=len(steps), data={"pivot_cols": tuple(pivot_cols)})]),
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
