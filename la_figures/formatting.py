"""Shared TeX formatting helpers for la_figures.

This module defers to ``matrixlayout.formatting`` when available to keep the
formatting policy consistent across packages.
"""

from __future__ import annotations

from typing import Any


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


__all__ = ["latexify"]
