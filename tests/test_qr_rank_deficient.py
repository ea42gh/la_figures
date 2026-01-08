import sympy as sym
import pytest

from la_figures.qr import gram_schmidt_qr_matrices


def test_gram_schmidt_qr_rank_deficient_raises_by_default():
    A = sym.Matrix([[1, 2], [2, 4]])
    W = sym.Matrix([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        gram_schmidt_qr_matrices(A, W)


def test_gram_schmidt_qr_rank_deficient_allow():
    A = sym.Matrix([[1, 2], [2, 4]])
    W = sym.Matrix([[1, 2], [2, 4]])
    mats = gram_schmidt_qr_matrices(A, W, allow_rank_deficient=True)
    assert mats[0][2].shape == A.shape
    assert mats[0][3].shape == W.shape


def test_gram_schmidt_qr_rank_deficient_drop():
    A = sym.Matrix([[1, 2], [2, 4]])
    W = sym.Matrix([[1, 2], [2, 4]])
    mats = gram_schmidt_qr_matrices(A, W, rank_deficient="drop")
    assert mats[0][2].shape == (2, 1)
    assert mats[0][3].shape == (2, 1)
