"""Convenience wrappers for QR/Gram–Schmidt figures."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .formatting import latexify
from .qr import gram_schmidt_qr_matrices, qr_tbl_spec, qr_tbl_spec_from_matrices


def _norm_str(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        s = x
    else:
        s = str(x)
    if s.startswith(":"):
        s = s[1:]
    return s


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


def qr_tbl_tex(
    A: Any,
    W: Any,
    *,
    array_names: Any = True,
    formater: Any = latexify,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Any] = None,
) -> str:
    """Compute a QR table TeX document."""

    from matrixlayout.qr import qr_grid_tex

    spec = qr_tbl_spec(
        A,
        W,
        array_names=array_names,
        fig_scale=fig_scale,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        label_color=label_color,
        label_text_color=label_text_color,
        known_zero_color=known_zero_color,
        decorators=decorators,
    )

    return qr_grid_tex(
        matrices=spec["matrices"],
        formater=formater,
        array_names=spec["array_names"],
        fig_scale=spec["fig_scale"],
        preamble=spec["preamble"],
        extension=spec["extension"],
        nice_options=spec["nice_options"],
        label_color=spec["label_color"],
        label_text_color=spec["label_text_color"],
        known_zero_color=spec["known_zero_color"],
        decorators=spec.get("decorators"),
    )


def qr_tbl_svg(
    A: Any,
    W: Any,
    *,
    array_names: Any = True,
    formater: Any = latexify,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Any] = None,
    toolchain_name: Optional[Any] = None,
    crop: Any = "tight",
    padding: Any = (2, 2, 2, 2),
    frame: Any = None,
    output_dir: Optional[Any] = None,
) -> str:
    """Compute a QR table SVG."""

    from matrixlayout.qr import qr_grid_svg

    spec = qr_tbl_spec(
        A,
        W,
        array_names=array_names,
        fig_scale=fig_scale,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        label_color=label_color,
        label_text_color=label_text_color,
        known_zero_color=known_zero_color,
        decorators=decorators,
    )

    return qr_grid_svg(
        matrices=spec["matrices"],
        formater=formater,
        array_names=spec["array_names"],
        fig_scale=spec["fig_scale"],
        preamble=spec["preamble"],
        extension=spec["extension"],
        nice_options=spec["nice_options"],
        label_color=spec["label_color"],
        label_text_color=spec["label_text_color"],
        known_zero_color=spec["known_zero_color"],
        decorators=spec.get("decorators"),
        toolchain_name=_norm_str(toolchain_name),
        crop=_norm_str(crop),
        padding=_norm_padding(padding),
        frame=frame,
        output_dir=output_dir,
    )


def qr_tbl_bundle(
    A: Any,
    W: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return a dict with spec/tex/svg for the QR table."""

    spec = qr_tbl_spec(A, W, **kwargs)
    tex = qr_tbl_tex(A, W, **kwargs)
    svg = None
    try:
        svg = qr_tbl_svg(A, W, **kwargs)
    except Exception:
        svg = None
    return {"spec": spec, "tex": tex, "svg": svg}


def qr(
    matrices: Any,
    *,
    formater: Any = latexify,
    array_names: Any = True,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Any] = None,
    toolchain_name: Optional[Any] = None,
    crop: Any = "tight",
    padding: Any = (2, 2, 2, 2),
    frame: Any = None,
    output_dir: Optional[Any] = None,
) -> str:
    """Render a QR grid from a precomputed matrix stack."""

    from matrixlayout.qr import qr_grid_svg

    spec = qr_tbl_spec_from_matrices(
        matrices,
        array_names=array_names,
        fig_scale=fig_scale,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        label_color=label_color,
        label_text_color=label_text_color,
        known_zero_color=known_zero_color,
        decorators=decorators,
    )

    return qr_grid_svg(
        matrices=spec["matrices"],
        formater=formater,
        array_names=spec["array_names"],
        fig_scale=spec["fig_scale"],
        preamble=spec["preamble"],
        extension=spec["extension"],
        nice_options=spec["nice_options"],
        label_color=spec["label_color"],
        label_text_color=spec["label_text_color"],
        known_zero_color=spec["known_zero_color"],
        decorators=spec.get("decorators"),
        toolchain_name=_norm_str(toolchain_name),
        crop=_norm_str(crop),
        padding=_norm_padding(padding),
        frame=frame,
        output_dir=output_dir,
    )


def gram_schmidt_qr(
    A: Any,
    W: Any,
    *,
    formater: Any = latexify,
    array_names: Any = True,
    allow_rank_deficient: bool = False,
    rank_deficient: Optional[str] = None,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Any] = None,
    toolchain_name: Optional[Any] = None,
    crop: Any = "tight",
    padding: Any = (2, 2, 2, 2),
    frame: Any = None,
    output_dir: Optional[Any] = None,
) -> str:
    """Render a QR grid using Gram–Schmidt scaling."""

    from matrixlayout.qr import qr_grid_svg

    matrices = gram_schmidt_qr_matrices(
        A,
        W,
        allow_rank_deficient=allow_rank_deficient,
        rank_deficient=rank_deficient,
    )
    spec = qr_tbl_spec_from_matrices(
        matrices,
        array_names=array_names,
        fig_scale=fig_scale,
        preamble=preamble,
        extension=extension,
        nice_options=nice_options,
        label_color=label_color,
        label_text_color=label_text_color,
        known_zero_color=known_zero_color,
        decorators=decorators,
    )

    return qr_grid_svg(
        matrices=spec["matrices"],
        formater=formater,
        array_names=spec["array_names"],
        fig_scale=spec["fig_scale"],
        preamble=spec["preamble"],
        extension=spec["extension"],
        nice_options=spec["nice_options"],
        label_color=spec["label_color"],
        label_text_color=spec["label_text_color"],
        known_zero_color=spec["known_zero_color"],
        decorators=spec.get("decorators"),
        toolchain_name=_norm_str(toolchain_name),
        crop=_norm_str(crop),
        padding=_norm_padding(padding),
        frame=frame,
        output_dir=output_dir,
    )
