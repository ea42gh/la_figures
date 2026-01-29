"""Orthonormalization helpers.

This module provides a minimal Gram–Schmidt procedure for producing an
orthonormal basis inside an eigenspace.
"""

from __future__ import annotations

from typing import List, Sequence

import sympy as sym


def q_gram_schmidt(v_list: Sequence[sym.Matrix]) -> List[sym.Matrix]:
    """Return an orthonormal list spanning the same subspace as ``v_list``.

    Notes
    -----
    - This is the same basic algorithm used by earlier QR helpers.
    - It assumes the input vectors are linearly independent within the
      eigenspace basis returned by SymPy.
    - If a vector has zero norm (or the orthogonalization produces a zero
      vector), a ``ValueError`` is raised.
    """

    w: List[sym.Matrix] = []
    for j in range(len(v_list)):
        w_j = sym.Matrix(v_list[j])
        for k in range(j):
            w_j = w_j - (w[k].dot(v_list[j])) * w[k]

        nrm = w_j.norm()
        if nrm == 0:
            raise ValueError("Gram–Schmidt encountered a zero-norm vector")
        w.append((1 / nrm) * w_j)
    return w
