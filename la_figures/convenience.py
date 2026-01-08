"""Convenience wrappers that produce TeX/SVG from :mod:`la_figures` specs.

The core la_figures API returns *spec dictionaries* intended to be consumed by
:mod:`matrixlayout`. These wrappers exist to make interactive use (including
Julia via PyCall/PythonCall) more ergonomic by providing single-call helpers.

Design constraints:
- Keep argument types "interop friendly": strings, numbers, and nested lists / tuples.
- Normalize Julia Symbols (often stringifying as ":name") to plain strings.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .formatting import latexify

from .eig import eig_tbl_spec
from .svd import svd_tbl_spec


def _norm_str(x: Any) -> Any:
    """Normalize Julia/PythonCall string-like values.

    - Preserve None.
    - Convert non-str to str.
    - Strip a leading ':' (common for Julia Symbols).
    """

    if x is None:
        return None
    if isinstance(x, str):
        s = x
    else:
        s = str(x)
    if s.startswith(":"):
        s = s[1:]
    return s




def _julia_str(x: Any) -> Any:
    """Alias for :func:`_norm_str` kept for Julia interop tests/backward-compat."""

    return _norm_str(x)


def _norm_padding(padding: Any) -> Any:
    if padding is None:
        return None
    if isinstance(padding, tuple) and len(padding) == 4:
        return padding
    try:
        seq = list(padding)
    except Exception:
        return padding
    return tuple(seq)


def _import_eigproblem_tex():
    try:
        from matrixlayout import eigproblem_tex  # type: ignore

        return eigproblem_tex
    except Exception:
        from matrixlayout.eigproblem import eigproblem_tex  # type: ignore

        return eigproblem_tex


def _import_eigproblem_svg():
    try:
        from matrixlayout import eigproblem_svg  # type: ignore

        return eigproblem_svg
    except Exception:
        from matrixlayout.eigproblem import eigproblem_svg  # type: ignore

        return eigproblem_svg


def eig_tbl_tex(
    A: Any,
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    case: Optional[Any] = None,
    formater: Any = latexify,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
) -> str:
    """Compute an eigen-table TeX document."""

    eigproblem_tex = _import_eigproblem_tex()

    if case is None:
        case = "Q" if normal else "E"

    spec = eig_tbl_spec(
        A,
        normal=normal,
        Ascale=Ascale,
        eig_digits=eig_digits,
        vec_digits=vec_digits,
    )

    return eigproblem_tex(
        spec,
        case=_norm_str(case),
        formater=formater,
        color=_norm_str(color),
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
    formater: Any = latexify,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    sz: Optional[Tuple[int, int]] = None,
    toolchain_name: Optional[Any] = None,
    crop: Any = "tight",
    padding: Any = (2, 2, 2, 2),
) -> str:
    """Compute an eigen-table SVG via :mod:`matrixlayout` and :mod:`jupyter_tikz`."""

    eigproblem_svg = _import_eigproblem_svg()

    if case is None:
        case = "Q" if normal else "E"

    spec = eig_tbl_spec(
        A,
        normal=normal,
        Ascale=Ascale,
        eig_digits=eig_digits,
        vec_digits=vec_digits,
    )

    return eigproblem_svg(
        spec,
        case=_norm_str(case),
        formater=formater,
        color=_norm_str(color),
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        sz=sz,
        toolchain_name=_norm_str(toolchain_name),
        crop=_norm_str(crop),
        padding=_norm_padding(padding),
    )


def eig_tbl_bundle(
    A: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return a dict with spec/tex/svg for the eigen-table."""

    # Generate spec once.
    spec = eig_tbl_spec(
        A,
        normal=kwargs.get("normal", False),
        Ascale=kwargs.get("Ascale"),
        eig_digits=kwargs.get("eig_digits"),
        vec_digits=kwargs.get("vec_digits"),
    )

    tex = eig_tbl_tex(A, **kwargs)
    svg = None
    try:
        svg = eig_tbl_svg(A, **kwargs)
    except Exception:
        # SVG rendering depends on external toolchains; keep bundle usable without them.
        svg = None

    return {"spec": spec, "tex": tex, "svg": svg}


def svd_tbl_tex(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    formater: Any = latexify,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    sz: Optional[Tuple[int, int]] = None,
) -> str:
    """Compute an SVD-table TeX document."""

    eigproblem_tex = _import_eigproblem_tex()

    spec = svd_tbl_spec(
        A,
        Ascale=Ascale,
        eig_digits=eig_digits,
        sigma2_digits=sigma2_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
    if sz is None:
        sz = spec.get("sz")

    return eigproblem_tex(
        spec,
        case="SVD",
        formater=formater,
        color=_norm_str(color),
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        sz=sz,
    )


def svd_tbl_svg(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    formater: Any = latexify,
    color: Any = "blue",
    mmLambda: int = 8,
    mmS: int = 4,
    fig_scale: Optional[Union[int, float]] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 1pt}" + "\n",
    sz: Optional[Tuple[int, int]] = None,
    toolchain_name: Optional[Any] = None,
    crop: Any = "tight",
    padding: Any = (2, 2, 2, 2),
) -> str:
    """Compute an SVD-table SVG via :mod:`matrixlayout` and :mod:`jupyter_tikz`."""

    eigproblem_svg = _import_eigproblem_svg()

    spec = svd_tbl_spec(
        A,
        Ascale=Ascale,
        eig_digits=eig_digits,
        sigma2_digits=sigma2_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
    if sz is None:
        sz = spec.get("sz")

    return eigproblem_svg(
        spec,
        case="SVD",
        formater=formater,
        color=_norm_str(color),
        mmLambda=mmLambda,
        mmS=mmS,
        fig_scale=fig_scale,
        preamble=preamble,
        sz=sz,
        toolchain_name=_norm_str(toolchain_name),
        crop=_norm_str(crop),
        padding=_norm_padding(padding),
    )


def svd_tbl_bundle(A: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return a dict with spec/tex/svg for the SVD table."""

    spec = svd_tbl_spec(
        A,
        Ascale=kwargs.get("Ascale"),
        eig_digits=kwargs.get("eig_digits"),
        sigma2_digits=kwargs.get("sigma2_digits"),
        sigma_digits=kwargs.get("sigma_digits"),
        vec_digits=kwargs.get("vec_digits"),
    )
    tex = svd_tbl_tex(A, **kwargs)
    svg = None
    try:
        svg = svd_tbl_svg(A, **kwargs)
    except Exception:
        svg = None
    return {"spec": spec, "tex": tex, "svg": svg}
