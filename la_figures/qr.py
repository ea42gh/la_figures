"""QR/Gram–Schmidt helpers (algorithmic layer)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import sympy as sym

from ._sympy_utils import to_sympy_matrix

def compute_qr_matrices(A: Any, W: Any) -> Sequence[Sequence[sym.Matrix]]:
    """Return the matrix grid used by the legacy QR layout.

    Parameters
    ----------
    A:
        Original matrix.
    W:
        Matrix with orthogonal columns spanning the same column space as ``A``.
    """

    A_mat = to_sympy_matrix(A)
    W_mat = to_sympy_matrix(W)
    if A_mat is None or W_mat is None:
        raise ValueError("compute_qr_matrices requires non-empty A and W")

    WtW = W_mat.T @ W_mat
    WtA = W_mat.T @ A_mat
    S = sym.Matrix.diag([1 / sym.sqrt(x) for x in sym.Matrix.diagonal(WtW)])

    Qt = S * W_mat.T
    R = S * WtA

    return [
        [None, None, A_mat, W_mat],
        [None, W_mat.T, WtA, WtW],
        [S, Qt, R, None],
    ]


def gram_schmidt_qr_matrices(
    A: Any,
    W: Any,
    *,
    allow_rank_deficient: bool = False,
    rank_deficient: Optional[str] = None,
) -> Sequence[Sequence[sym.Matrix]]:
    """Return the QR grid using the legacy Gram–Schmidt-style scaling.

    If ``rank_deficient`` is set, it selects the fallback behavior when
    ``W^T W`` is singular:
      - "drop": drop dependent columns from W (and corresponding A columns).
      - "pinv": use the pseudo-inverse.
      - "diag": invert only nonzero diagonals.
    If ``allow_rank_deficient`` is True and no ``rank_deficient`` mode is set,
    it falls back to "pinv" (legacy compatibility).
    """

    A_mat = to_sympy_matrix(A)
    W_mat = to_sympy_matrix(W)
    if A_mat is None or W_mat is None:
        raise ValueError("gram_schmidt_qr_matrices requires non-empty A and W")

    mode = (rank_deficient or "").strip().lower() or None
    orig_cols = W_mat.cols
    if mode == "drop":
        pivots = list(W_mat.rref()[1])
        if pivots:
            W_mat = W_mat[:, pivots]
            if A_mat.cols == orig_cols:
                A_mat = A_mat[:, pivots]

    WtW = W_mat.T @ W_mat
    WtA = W_mat.T @ A_mat
    try:
        S = WtW**(-1)
    except Exception as exc:
        if mode is None and allow_rank_deficient:
            mode = "pinv"
        handled = False
        if mode == "pinv":
            pinv = getattr(WtW, "pinv", None)
            if callable(pinv):
                try:
                    S = pinv()
                    handled = True
                except Exception:
                    mode = "diag"
            else:
                mode = "diag"
        if mode == "diag":
            S = sym.zeros(*WtW.shape)
            n = min(WtW.shape)
            for i in range(n):
                if WtW[i, i] != 0:
                    S[i, i] = 1 / WtW[i, i]
            handled = True
        if not handled:
            raise ValueError("W^T W is singular; set rank_deficient or allow_rank_deficient") from exc

    for i in range(min(S.shape)):
        S[i, i] = sym.sqrt(S[i, i])

    Qt = S * W_mat.T
    R = S * WtA

    return [
        [None, None, A_mat, W_mat],
        [None, W_mat.T, WtA, WtW],
        [S, Qt, R, None],
    ]


def qr_tbl_spec(
    A: Any,
    W: Any,
    *,
    array_names: Any = True,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Return a layout spec for :func:`matrixlayout.qr.qr_grid_tex`."""

    matrices = compute_qr_matrices(A, W)
    return {
        "matrices": matrices,
        "array_names": array_names,
        "fig_scale": fig_scale,
        "preamble": preamble,
        "extension": extension,
        "nice_options": nice_options,
        "label_color": label_color,
        "label_text_color": label_text_color,
        "known_zero_color": known_zero_color,
        "decorators": decorators,
    }


def qr_tbl_layout_spec(
    A: Any,
    W: Any,
    *,
    array_names: Any = True,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Sequence[Any]] = None,
) -> Any:
    """Return a typed QR layout spec (``QRGridSpec``) for matrixlayout."""

    from matrixlayout.specs import QRGridSpec

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
    return QRGridSpec.from_dict(spec)


def qr_tbl_spec_from_matrices(
    matrices: Sequence[Sequence[Any]],
    *,
    array_names: Any = True,
    fig_scale: Optional[Any] = None,
    preamble: str = r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n",
    extension: str = "",
    nice_options: Optional[str] = "vlines-in-sub-matrix = I",
    label_color: str = "blue",
    label_text_color: str = "red",
    known_zero_color: str = "brown",
    decorators: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Return a layout spec from a precomputed matrix grid."""

    return {
        "matrices": matrices,
        "array_names": array_names,
        "fig_scale": fig_scale,
        "preamble": preamble,
        "extension": extension,
        "nice_options": nice_options,
        "label_color": label_color,
        "label_text_color": label_text_color,
        "known_zero_color": known_zero_color,
        "decorators": decorators,
    }
