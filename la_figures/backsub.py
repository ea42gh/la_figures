"""Back-substitution helpers for system, cascade, and solution blocks.

It provides:
  - A cascade block generator compatible with :func:`matrixlayout.mk_shortcascade_lines`.
  - Convenience helpers to build TeX for the system and solution blocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sym

from matrixlayout.shortcascade import mk_shortcascade_lines

from .formatting import latexify

from ._sympy_utils import to_sympy_col, to_sympy_matrix


def _bs_rhs(
    A: sym.Matrix,
    pivot_row: int,
    pivot_col: int,
    rhs: Optional[sym.Matrix],
    *,
    symbol_prefix: str,
) -> str:
    """Build the right-hand side expression for one back-substitution equation.

    This mirrors the structure of the historical ``_bs_equation`` helper:
    - it forms ``t = rhs[i] - sum_j A[i,j] * <symbol>``
    - it divides by the pivot coefficient
    - it returns a LaTeX string (not an Eq object) to keep downstream formatting
      predictable.
    """

    t = sym.Integer(0) if rhs is None else rhs[pivot_row]
    for j in range(pivot_col + 1, A.shape[1]):
        t = t - A[pivot_row, j] * sym.Symbol(f"{symbol_prefix}_{j+1}")

    if getattr(t, "is_zero", False):
        return " 0 "

    pivot = A[pivot_row, pivot_col]
    factor = sym.Integer(1) / pivot

    if factor == 1:
        return latexify(t)
    if factor == -1:
        return f"- \\left( {latexify(t)} \\right)"
    return f"{latexify(factor)} \\left( {latexify(t)} \\right)"


def _backsub_trace_from_ref(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    var_name: str = "x",
    param_name: str = r"\alpha",
) -> Dict[str, Any]:
    """Produce an Option-A back-substitution trace from a (possibly augmented) system.

    Parameters
    ----------
    ref_A:
        Coefficient matrix in *reference* (typically echelon) form.
        The function will compute an RREF internally for the substituted lines.
    ref_rhs:
        Optional RHS vector. If provided, the RREF is computed for the augmented
        matrix and pivots are derived accordingly.
    var_name:
        Variable symbol prefix (e.g. ``x`` gives variables ``x_1, x_2, ...``).
    param_name:
        Parameter symbol prefix for free variables (default ``\\alpha``).

    Returns
    -------
    dict
        ``{"base": <tex>, "steps": [(raw, substituted), ...], "meta": {...}}``
    """

    A = to_sympy_matrix(ref_A)
    if A is None:
        raise ValueError("ref_A must not be None")
    b = to_sympy_col(ref_rhs)

    if b is None:
        rref_A, pivot_cols = A.rref()
        rref_b = None
    else:
        Ab = A.row_join(b)
        rref_Ab, pivot_cols_ab = Ab.rref()
        rref_A = rref_Ab[:, 0:-1]
        rref_b = rref_Ab[:, -1]

        # If the RHS column is marked as a pivot, drop it for pivot var logic.
        if pivot_cols_ab and pivot_cols_ab[-1] == rref_A.shape[1]:
            pivot_cols = pivot_cols_ab[:-1]
        else:
            pivot_cols = pivot_cols_ab

    free_cols = [i for i in range(A.shape[1]) if i not in pivot_cols]
    rank = len(pivot_cols)

    # Base assignment(s)
    if free_cols:
        base = ",\\;".join([f"{var_name}_{i+1} = {param_name}_{i+1}" for i in free_cols])
        start = rank - 1
    else:
        # Full-column-rank: the last pivot variable becomes the base.
        base = f"{var_name}_{rank} = {_bs_rhs(A, rank-1, pivot_cols[-1], b, symbol_prefix=param_name)}"
        start = rank - 2

    steps: List[Tuple[str, str]] = []
    for i in range(start, -1, -1):
        pc = pivot_cols[i]
        raw = f"{var_name}_{pc+1} = {_bs_rhs(A, i, pc, b, symbol_prefix=var_name)}"
        sub = f"{var_name}_{pc+1} = {_bs_rhs(rref_A, i, pc, rref_b, symbol_prefix=param_name)}"
        steps.append((raw, sub))

    return {
        "base": base,
        "steps": steps,
        "meta": {
            "pivot_cols": list(pivot_cols),
            "free_cols": list(free_cols),
            "rank": rank,
            "shape": tuple(A.shape),
        },
    }


def backsubstitution_tex(
    ref_A: Any,
    ref_rhs: Any = None,
    *,
    var_name: str = "x",
    param_name: str = r"\alpha",
) -> list[str]:
    """Return a TeX ``cascade`` block (as ``\\ShortCascade`` lines)."""

    trace = _backsub_trace_from_ref(
        ref_A,
        ref_rhs,
        var_name=var_name,
        param_name=param_name,
    )
    return mk_shortcascade_lines(trace)


def linear_system_tex(A: Any, b: Any, *, var_name: str = "x") -> str:
    """Return a TeX ``systeme`` representation of ``A x = b``.

    This is a presentation helper, colocated with the back-substitution logic.
    """

    A = to_sympy_matrix(A)
    bb = to_sympy_col(b)
    if A is None or bb is None:
        raise ValueError("A and b must not be None")

    var = set()

    def _term(j: int, v: Any) -> List[str]:
        var.add(f"{var_name}_{j}")
        try:
            if v > 0:
                if v == 1:
                    return [" + ", f"{var_name}_{j}"]
                return [" + ", latexify(v), f"{var_name}_{j}"]
            if v < 0:
                if v == -1:
                    return [" - ", f"{var_name}_{j}"]
                return [" - ", latexify(-v), f"{var_name}_{j}"]
            return [""]
        except Exception:
            return [" + ", latexify(v), f"{var_name}_{j}"]

    eqs: List[str] = []
    for i in range(A.rows):
        terms: List[str] = []
        for j in range(A.cols):
            if not getattr(A[i, j], "is_zero", False):
                terms.extend(_term(j + 1, A[i, j]))

        if not terms:
            terms = [" 0 "]
        if terms and terms[0] == " + ":
            terms = terms[1:]
        eqs.append("".join(terms) + " = " + latexify(bb[i, 0]))

    vars_sorted = "".join(sorted(var))
    return r"\systeme[" + vars_sorted + "]{ " + ",".join(eqs) + "}"


def standard_solution_tex(
    ref_A: Any,
    ref_rhs: Any,
    *,
    var_name: str = "x",
    param_name: str = r"\alpha",
) -> str:
    """Return a TeX solution expression in the standard style.

    The output matches the ``_gen_solution_eqs`` structure:
      ``x = p + sum(alpha_j * h_j)``
    where ``p`` is a particular solution and the ``h_j`` span the nullspace.
    """

    A = to_sympy_matrix(ref_A)
    b = to_sympy_col(ref_rhs)
    if A is None or b is None:
        raise ValueError("ref_A and ref_rhs must not be None")

    Ab = A.row_join(b)
    rref_Ab, pivot_cols_ab = Ab.rref()
    rref_A = rref_Ab[:, 0:-1]
    rref_b = rref_Ab[:, -1]

    if pivot_cols_ab and pivot_cols_ab[-1] == rref_A.shape[1]:
        pivot_cols = pivot_cols_ab[:-1]
    else:
        pivot_cols = pivot_cols_ab
    free_cols = [i for i in range(A.shape[1]) if i not in pivot_cols]
    rank = len(pivot_cols)

    lft = r"\begin{pNiceArray}{r}"
    rgt = r"\end{pNiceArray}"

    xvec = lft + r" \\ ".join([f" {var_name}_{i+1}" for i in range(A.cols)]) + rgt

    # Particular solution (set free vars = 0)
    ps = sym.Matrix.zeros(A.cols, 1)
    if rank > 0:
        sol = (A[0:rank, pivot_cols] ** -1) * b[0:rank, 0]
        for i, c in enumerate(pivot_cols):
            ps[c, 0] = sol[i, 0]
    ptxt = lft + r" \\ ".join([latexify(ps[i, 0]) for i in range(A.cols)]) + rgt

    # Homogeneous solution basis
    if free_cols:
        hs = sym.Matrix.zeros(A.cols, len(free_cols))
        for j, col in enumerate(free_cols):
            hs[col, j] = 1
        for i, col in enumerate(pivot_cols):
            hs[col, :] = -rref_A[i, free_cols]

        h_terms: List[str] = []
        for j, free_col in enumerate(free_cols):
            hvec = lft + r" \\ ".join([latexify(hs[i, j]) for i in range(A.cols)]) + rgt
            h_terms.append(f"{param_name}_{free_col+1} " + hvec)
        htxt = " + ".join(h_terms)
        plus = " + "
    else:
        htxt = ""
        plus = ""

    return "$ " + xvec + " = " + ptxt + plus + htxt + " $"
