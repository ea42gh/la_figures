import numpy as np
import sympy as sym


def test_to_sympy_matrix_tuple_rationals_2d_list():
    from la_figures._sympy_utils import to_sympy_matrix

    A = [[(1, 2), (0, 1)], [(3, 4), (5, 6)]]
    M = to_sympy_matrix(A)
    assert M.shape == (2, 2)
    assert M[0, 0] == sym.Rational(1, 2)
    assert M[1, 0] == sym.Rational(3, 4)


def test_to_sympy_matrix_tuple_rationals_numpy_array():
    from la_figures._sympy_utils import to_sympy_matrix

    A = np.array([[(1, 2), (0, 1)], [(3, 4), (5, 6)]], dtype=object)
    M = to_sympy_matrix(A)
    assert M.shape == (2, 2)
    assert M[0, 0] == sym.Rational(1, 2)
    assert M[1, 1] == sym.Rational(5, 6)


def test_eig_tbl_spec_accepts_tuple_rationals():
    import la_figures

    A = [[(1, 1), (0, 1)], [(0, 1), (2, 1)]]
    spec = la_figures.eig_tbl_spec(A, normal=True)
    assert "lambda" in spec and spec["lambda"]
    assert "qvecs" in spec  # normal=True should populate qvecs


def test_svd_tbl_spec_accepts_tuple_rationals():
    import la_figures

    A = [[(1, 1), (0, 1)], [(0, 1), (0, 1)]]
    spec = la_figures.svd_tbl_spec(A)
    assert spec["sz"] == (2, 2)
    # For this A, the only nonzero singular value is 1.
    assert sym.simplify(spec["sigma"][0] - 1) == 0


def test_to_sympy_matrix_accepts_juliacall_arrayvalue_wrapper():
    """Simulate PythonCall/JuliaCall ArrayValue (1-based indexing)"""

    from la_figures._sympy_utils import to_sympy_matrix

    class FakeArrayValue:
        __module__ = "juliacall"

        def __init__(self, data):
            self._data = data
            self.shape = (len(data), len(data[0]))

        def __getitem__(self, idx):
            # Expect idx either (i,j) 1-based, or i 1-based for 1D
            if isinstance(idx, tuple):
                i, j = idx
                if i < 1 or j < 1:
                    raise IndexError("1-based")
                return self._data[i - 1][j - 1]
            if idx < 1:
                raise IndexError("1-based")
            return self._data[idx - 1]

    A = FakeArrayValue([[1, 2], [3, 4]])
    M = to_sympy_matrix(A)
    assert M.shape == (2, 2)
    assert int(M[0, 0]) == 1
    assert int(M[1, 1]) == 4


def test_julia_symbol_normalization_in_convenience_wrappers():
    from la_figures.convenience import _julia_str

    class FakeSymbol:
        __module__ = "juliacall"

        def __str__(self):
            return ":tight"

    class FakePyCallSymbol:
        __module__ = "pycall"

        def __str__(self):
            return "Symbol(:tight)"

    assert _julia_str(FakeSymbol()) == "tight"
    assert _julia_str(FakePyCallSymbol()) == "tight"


def test_convenience_tex_wrappers_smoke():
    import pytest

    pytest.importorskip("matrixlayout")
    import la_figures

    A = [[1, 0], [0, 2]]
    tex_eig = la_figures.eig_tbl_tex(A)
    assert "\\begin{tabular}" in tex_eig

    tex_svd = la_figures.svd_tbl_tex([[1, 0], [0, 0]])
    assert "\\begin{tabular}" in tex_svd
