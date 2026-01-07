import sympy as sym


def test_ge_tbl_tex_includes_expected_submatrix_names():
    import pytest

    pytest.importorskip("matrixlayout")
    from la_figures.ge_convenience import ge_tbl_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    tex = ge_tbl_tex(A, show_pivots=True)

    assert "name=A0" in tex
    assert "name=E1" in tex
    assert "name=A1" in tex


def test_ge_tbl_spec_pivot_locs_rebased_to_final_layer():
    from la_figures.ge_convenience import ge_tbl_spec

    A = sym.Matrix([[1, 2], [3, 4]])
    spec = ge_tbl_spec(A, show_pivots=True)

    assert spec["pivot_locs"] == [
        ("(3-3)(3-3)", "draw, thick, rounded corners"),
        ("(4-4)(4-4)", "draw, thick, rounded corners"),
    ]
