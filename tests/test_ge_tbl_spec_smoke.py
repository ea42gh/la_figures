import sympy as sym


def test_ge_tbl_spec_and_tex_smoke():
    import pytest

    pytest.importorskip("matrixlayout")
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])

    spec = la_figures.ge_tbl_spec(A)
    assert isinstance(spec, dict)
    assert "matrices" in spec
    assert "Nrhs" in spec
    assert spec["Nrhs"] == 0

    tex = la_figures.ge_tbl_tex(A)
    assert "\\begin{NiceArray}" in tex
    assert "\\end{NiceArray}" in tex
