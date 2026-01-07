import shutil

import sympy as sym
import pytest


def _pick_toolchain_name_or_skip() -> str:
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk not found")

    if shutil.which("dvisvgm") is not None:
        return "pdftex_dvisvgm"
    if shutil.which("pdftocairo") is not None:
        return "pdftex_pdftocairo"
    if shutil.which("pdf2svg") is not None:
        return "pdftex_pdf2svg"

    pytest.skip("no SVG converter found (need dvisvgm, pdftocairo, or pdf2svg)")
    raise AssertionError("unreachable")


@pytest.mark.render
def test_ge_tbl_svg_smoke():
    pytest.importorskip("matrixlayout")
    import la_figures

    A = sym.Matrix([[1, 0], [0, 2]])

    svg = la_figures.ge_tbl_svg(
        A,
        crop="tight",
        padding=(2, 2, 2, 2),
        toolchain_name=_pick_toolchain_name_or_skip(),
    )

    assert "<svg" in svg
