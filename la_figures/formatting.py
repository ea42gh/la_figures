"""Shared TeX formatting helpers for la_figures.

This module defers to ``matrixlayout.formatting`` when available to keep the
formatting policy consistent across packages.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple


def latexify(x: Any) -> str:
    """Return a TeX-ready representation for a scalar-like value."""
    try:
        from matrixlayout.formatting import latexify as _latexify  # type: ignore
    except Exception:
        _latexify = None

    if _latexify is not None:
        return _latexify(x)

    # Fallback to the local implementation (mirrors matrixlayout.formatting).
    from fractions import Fraction
    import numbers

    if isinstance(x, str):
        return x

    if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, int) for v in x):
        num, den = x
        return rf"\frac{{{num}}}{{{den}}}"

    if isinstance(x, Fraction):
        return rf"\frac{{{x.numerator}}}{{{x.denominator}}}"

    try:
        import sympy as sym  # type: ignore

        if isinstance(x, sym.Basic):
            return sym.latex(x)
    except Exception:
        pass

    if isinstance(x, numbers.Number) and not isinstance(x, bool):
        return format(x, "g")

    return str(x)

def make_decorator(**kwargs: Any) -> Callable[[str], str]:
    """Return a decorator function for TeX strings (matrixlayout-compatible)."""
    try:
        from matrixlayout.formatting import make_decorator as _make_decorator  # type: ignore
    except Exception:
        _make_decorator = None
    if _make_decorator is not None:
        return _make_decorator(**kwargs)
    raise RuntimeError("matrixlayout is required for make_decorator")


def decorator_box(*, color: Optional[str] = None) -> Callable[[str], str]:
    """Return a decorator that draws a box (optionally colored) around TeX."""
    try:
        from matrixlayout.formatting import decorator_box as _decorator_box  # type: ignore
    except Exception:
        _decorator_box = None
    if _decorator_box is not None:
        return _decorator_box(color=color)
    if color:
        return make_decorator(box_color=color)
    return make_decorator(boxed=True)


def decorator_color(color: str) -> Callable[[str], str]:
    """Return a decorator that colors TeX."""
    try:
        from matrixlayout.formatting import decorator_color as _decorator_color  # type: ignore
    except Exception:
        _decorator_color = None
    if _decorator_color is not None:
        return _decorator_color(color)
    return make_decorator(text_color=color)


def decorator_bg(color: str) -> Callable[[str], str]:
    """Return a decorator that adds a background color to TeX."""
    try:
        from matrixlayout.formatting import decorator_bg as _decorator_bg  # type: ignore
    except Exception:
        _decorator_bg = None
    if _decorator_bg is not None:
        return _decorator_bg(color)
    return make_decorator(bg_color=color)


def decorator_bf() -> Callable[[str], str]:
    """Return a decorator that boldfaces TeX."""
    try:
        from matrixlayout.formatting import decorator_bf as _decorator_bf  # type: ignore
    except Exception:
        _decorator_bf = None
    if _decorator_bf is not None:
        return _decorator_bf()
    return make_decorator(bf=True)


def sel_entry(i: int, j: int) -> Any:
    """Selector for a single matrix entry."""
    try:
        from matrixlayout.formatting import sel_entry as _sel_entry  # type: ignore
    except Exception:
        _sel_entry = None
    if _sel_entry is not None:
        return _sel_entry(i, j)
    return (int(i), int(j))


def sel_box(top_left: Any, bottom_right: Any) -> Any:
    """Selector for a rectangular entry region."""
    try:
        from matrixlayout.formatting import sel_box as _sel_box  # type: ignore
    except Exception:
        _sel_box = None
    if _sel_box is not None:
        return _sel_box(top_left, bottom_right)
    return (tuple(top_left), tuple(bottom_right))


def sel_row(i: int) -> Any:
    """Selector for an entire row."""
    try:
        from matrixlayout.formatting import sel_row as _sel_row  # type: ignore
    except Exception:
        _sel_row = None
    if _sel_row is not None:
        return _sel_row(i)
    return {"row": int(i)}


def sel_col(j: int) -> Any:
    """Selector for an entire column."""
    try:
        from matrixlayout.formatting import sel_col as _sel_col  # type: ignore
    except Exception:
        _sel_col = None
    if _sel_col is not None:
        return _sel_col(j)
    return {"col": int(j)}


def sel_rows(rows: Any) -> Any:
    """Selector for multiple rows."""
    try:
        from matrixlayout.formatting import sel_rows as _sel_rows  # type: ignore
    except Exception:
        _sel_rows = None
    if _sel_rows is not None:
        return _sel_rows(rows)
    return {"rows": [int(r) for r in rows]}


def sel_cols(cols: Any) -> Any:
    """Selector for multiple columns."""
    try:
        from matrixlayout.formatting import sel_cols as _sel_cols  # type: ignore
    except Exception:
        _sel_cols = None
    if _sel_cols is not None:
        return _sel_cols(cols)
    return {"cols": [int(c) for c in cols]}


def sel_all() -> Any:
    """Selector for all entries."""
    try:
        from matrixlayout.formatting import sel_all as _sel_all  # type: ignore
    except Exception:
        _sel_all = None
    if _sel_all is not None:
        return _sel_all()
    return {"all": True}


def sel_vec(group: int, vec: int, entry: int) -> Any:
    """Selector for a vector entry in an eigenproblem row."""
    try:
        from matrixlayout.formatting import sel_vec as _sel_vec  # type: ignore
    except Exception:
        _sel_vec = None
    if _sel_vec is not None:
        return _sel_vec(group, vec, entry)
    return (int(group), int(vec), int(entry))


def sel_vec_range(start: Any, end: Any) -> Any:
    """Selector for a contiguous vector entry range."""
    try:
        from matrixlayout.formatting import sel_vec_range as _sel_vec_range  # type: ignore
    except Exception:
        _sel_vec_range = None
    if _sel_vec_range is not None:
        return _sel_vec_range(start, end)
    return (tuple(start), tuple(end))


def decorate_tex_entries(
    matrices: Any,
    gM: int,
    gN: int,
    decorator: Callable[[str], str],
    *,
    entries: Optional[Iterable[Tuple[int, int]]] = None,
    formatter: Callable[[Any], str] = latexify,
) -> Any:
    """Apply a decorator to selected entries in a grid matrix (in-place)."""
    try:
        from matrixlayout.formatting import decorate_tex_entries as _decorate_tex_entries  # type: ignore
    except Exception:
        _decorate_tex_entries = None
    if _decorate_tex_entries is None:
        raise RuntimeError("matrixlayout is required for decorate_tex_entries")
    return _decorate_tex_entries(
        matrices,
        gM,
        gN,
        decorator,
        entries=entries,
        formatter=formatter,
    )


__all__ = [
    "latexify",
    "make_decorator",
    "decorate_tex_entries",
    "decorator_box",
    "decorator_color",
    "decorator_bg",
    "decorator_bf",
    "sel_entry",
    "sel_box",
    "sel_row",
    "sel_col",
    "sel_rows",
    "sel_cols",
    "sel_all",
    "sel_vec",
    "sel_vec_range",
]
