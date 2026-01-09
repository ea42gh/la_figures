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
    """Return a decorator function for TeX strings (legacy-compatible)."""
    try:
        from matrixlayout.formatting import make_decorator as _make_decorator  # type: ignore
    except Exception:
        _make_decorator = None
    if _make_decorator is not None:
        return _make_decorator(**kwargs)
    raise RuntimeError("matrixlayout is required for make_decorator")


def decorate_tex_entries(
    matrices: Any,
    gM: int,
    gN: int,
    decorator: Callable[[str], str],
    *,
    entries: Optional[Iterable[Tuple[int, int]]] = None,
    formater: Callable[[Any], str] = latexify,
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
        formater=formater,
    )


__all__ = ["latexify", "make_decorator", "decorate_tex_entries"]
