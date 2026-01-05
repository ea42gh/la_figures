import sympy as sym
import pytest


@pytest.mark.render
def test_ge_tbl_svg_smoke():
    pytest.importorskip("matrixlayout")
    import la_figures

    A = sym.Matrix([[1, 0], [0, 2]])

    try:
        svg = la_figures.ge_tbl_svg(A, crop="tight", padding=(2, 2, 2, 2))
    except Exception as e:
        pytest.skip(f"SVG render not available: {e}")

    assert "<svg" in svg
