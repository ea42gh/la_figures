import shutil

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


def test_svd_tbl_tex_smoke():
    import la_figures

    tex = la_figures.svd_tbl_tex([[1, 0], [0, 0]])
    assert "\\begin{tabular}" in tex
    assert "\\Sigma" in tex


@pytest.mark.render
def test_svd_tbl_svg_smoke():
    pytest.importorskip("jupyter_tikz")

    import la_figures

    svg = la_figures.svd_tbl_svg(
        [[1, 0], [0, 0]],
        toolchain_name=_pick_toolchain_name_or_skip(),
        crop="tight",
        padding=(2, 2, 2, 2),
    )
    assert isinstance(svg, str)
    assert "<svg" in svg
