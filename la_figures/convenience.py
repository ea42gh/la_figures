"""Julia-friendly convenience wrappers.

This module provides higher-level functions that combine :mod:`la_figures`
algorithmic specs with :mod:`matrixlayout` rendering.

These wrappers are intended for notebook users (Python and Julia) who want a
single function call that yields TeX or SVG without manually piping a spec into
``matrixlayout.eigproblem_tex/svg``.

Notes on Julia interop
----------------------
When calling these functions from Julia:

* With **PyCall**, Julia ``Symbol`` values are typically converted to Python
  strings.
* With **PythonCall**, Julia ``Symbol`` values are wrapped as
  ``juliacall.AnyValue`` (i.e. they are *not* converted to ``str`` by default).

For this reason, string-like arguments in this module are normalized via
``_julia_str``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import sympy as sym

from .eig import eig_tbl_spec
from .svd import svd_tbl_spec


def _julia_str(x: Any) -> Optional[str]:
    """Convert Julia-origin string-ish values to a Python ``str``.

    Accepts:
      * ``None``
      * ``str``
      * Julia ``Symbol`` coming in via PyCall (already a ``str``)
      * Julia ``Symbol`` coming in via PythonCall (a ``juliacall.*Value`` wrapper)

    For wrapped symbols, ``str(x)`` commonly yields ``":name"``.
    """

    if x is None:
        return None
    if isinstance(x, str):
        return x

    s = str(x)
    # Common Julia Symbol textual forms.
    if s.startswith(":") and len(s) > 1:
        return s[1:]
    if s.startswith("Symbol(") and s.endswith(")"):
        # e.g. Symbol("tight") or Symbol(:tight)
        inner = s[len("Symbol(") : -1].strip()
        inner = inner.strip("\"'")
        if inner.startswith(":"):
            inner = inner[1:]
        return inner
    return s


def _padding_4(x: Any) -> Any:
    """Normalize padding-like inputs to a 4-tuple when possible."""

    if x is None:
        return None
    # Julia often passes vectors; PythonCall wraps them but supports iteration.
    try:
        seq = list(x)  # type: ignore[arg-type]
        if len(seq) == 4:
            return tuple(seq)
    except Exception:
        pass
    return x


def _ml_eigproblem_tex():
    """Import ``eigproblem_tex`` robustly.

    In some development setups (and some Julia↔Python bridge configurations),
    users may have the project checkout on ``sys.path`` without the project
    being installed. In those cases, ``import matrixlayout`` can resolve but
    the package's top-level re-exports may not be available.
    """

    try:
        from matrixlayout import eigproblem_tex as _eigproblem_tex  # type: ignore

        return _eigproblem_tex
    except Exception:
        from matrixlayout.eigproblem import eigproblem_tex as _eigproblem_tex  # type: ignore

        return _eigproblem_tex


def _ml_eigproblem_svg():
    """Import ``eigproblem_svg`` robustly. See :func:`_ml_eigproblem_tex`."""

    try:
        from matrixlayout import eigproblem_svg as _eigproblem_svg  # type: ignore

        return _eigproblem_svg
    except Exception:
        from matrixlayout.eigproblem import eigproblem_svg as _eigproblem_svg  # type: ignore

        return _eigproblem_svg


def eig_tbl_tex(
    A: Any,
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    case: Optional[Any] = None,
    formater: Any = sym.latex,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
) -> str:
    """Compute an eigen-table TeX document.

    This is a convenience wrapper around ``la_figures.eig_tbl_spec`` and
    ``matrixlayout.eigproblem_tex``.
    """

    eigproblem_tex = _ml_eigproblem_tex()

    spec = eig_tbl_spec(A, normal=normal, Ascale=Ascale, eig_digits=eig_digits, vec_digits=vec_digits)
    resolved_case = _julia_str(case) or ("Q" if normal else "S")
    return eigproblem_tex(
        spec,
        case=resolved_case,
        formater=formater,
        color=_julia_str(color) or "blue",
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
    )


def eig_tbl_svg(
    A: Any,
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    case: Optional[Any] = None,
    formater: Any = sym.latex,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    toolchain_name: Optional[Any] = None,
    crop: Optional[Any] = "tight",
    padding: Any = (2, 2, 2, 2),
) -> str:
    """Compute and render an eigen-table to SVG."""

    eigproblem_svg = _ml_eigproblem_svg()

    spec = eig_tbl_spec(A, normal=normal, Ascale=Ascale, eig_digits=eig_digits, vec_digits=vec_digits)
    resolved_case = _julia_str(case) or ("Q" if normal else "S")
    return eigproblem_svg(
        spec,
        case=resolved_case,
        formater=formater,
        color=_julia_str(color) or "blue",
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        toolchain_name=_julia_str(toolchain_name),
        crop=_julia_str(crop),
        padding=_padding_4(padding),
    )


def svd_tbl_tex(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    case: Optional[Any] = "SVD",
    formater: Any = sym.latex,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    sz: Optional[Tuple[int, int]] = None,
) -> str:
    """Compute an SVD-table TeX document."""

    eigproblem_tex = _ml_eigproblem_tex()

    spec = svd_tbl_spec(
        A,
        Ascale=Ascale,
        sigma2_digits=sigma2_digits,
        eig_digits=eig_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
    resolved_case = _julia_str(case) or "SVD"
    resolved_sz = tuple(spec.get("sz")) if sz is None and "sz" in spec else sz
    return eigproblem_tex(
        spec,
        case=resolved_case,
        formater=formater,
        color=_julia_str(color) or "blue",
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        sz=resolved_sz,
    )


def svd_tbl_svg(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    case: Optional[Any] = "SVD",
    formater: Any = sym.latex,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    sz: Optional[Tuple[int, int]] = None,
    toolchain_name: Optional[Any] = None,
    crop: Optional[Any] = "tight",
    padding: Any = (2, 2, 2, 2),
) -> str:
    """Compute and render an SVD-table to SVG."""

    eigproblem_svg = _ml_eigproblem_svg()

    spec = svd_tbl_spec(
        A,
        Ascale=Ascale,
        sigma2_digits=sigma2_digits,
        eig_digits=eig_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
    resolved_case = _julia_str(case) or "SVD"
    resolved_sz = tuple(spec.get("sz")) if sz is None and "sz" in spec else sz
    return eigproblem_svg(
        spec,
        case=resolved_case,
        formater=formater,
        color=_julia_str(color) or "blue",
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        sz=resolved_sz,
        toolchain_name=_julia_str(toolchain_name),
        crop=_julia_str(crop),
        padding=_padding_4(padding),
    )


def eig_tbl_bundle(
    A: Any,
    *,
    normal: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return ``{"spec": ..., "tex": ..., "svg": ...}`` for eigen tables."""

    spec = eig_tbl_spec(A, normal=normal, Ascale=kwargs.get("Ascale"), eig_digits=kwargs.get("eig_digits"), vec_digits=kwargs.get("vec_digits"))
    tex = eig_tbl_tex(A, normal=normal, **kwargs)
    svg = eig_tbl_svg(A, normal=normal, **kwargs)
    return {"spec": spec, "tex": tex, "svg": svg}


def svd_tbl_bundle(A: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return ``{"spec": ..., "tex": ..., "svg": ...}`` for SVD tables."""

    spec = svd_tbl_spec(
        A,
        Ascale=kwargs.get("Ascale"),
        sigma2_digits=kwargs.get("sigma2_digits"),
        eig_digits=kwargs.get("eig_digits"),
        sigma_digits=kwargs.get("sigma_digits"),
        vec_digits=kwargs.get("vec_digits"),
    )
    tex = svd_tbl_tex(A, **kwargs)
    svg = svd_tbl_svg(A, **kwargs)
    return {"spec": spec, "tex": tex, "svg": svg}
