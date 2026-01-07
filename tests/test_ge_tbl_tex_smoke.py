import pathlib

import pytest
import sympy as sym


def _has_ge_template() -> bool:
    try:
        import matrixlayout
    except Exception:
        return False
    tpl = pathlib.Path(matrixlayout.__file__).resolve().parent / "templates" / "ge.tex.j2"
    return tpl.exists()


def test_ge_tbl_tex_smoke():
    pytest.importorskip("matrixlayout")
    if not _has_ge_template():
        pytest.skip("matrixlayout GE template not found")

    from la_figures.ge_convenience import ge_tbl_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    tex = ge_tbl_tex(A, show_pivots=True)
    assert "\\begin{NiceArray}" in tex
    assert r"\SubMatrix({1-3}{2-4})[name=A0]" in tex
    # pivot boxes are emitted in CodeAfter
    assert "fit=" in tex


def test_ge_tbl_tex_auto_callouts_smoke():
    """callouts=True should emit delimiter-attached \\draw snippets."""

    pytest.importorskip("matrixlayout")
    if not _has_ge_template():
        pytest.skip("matrixlayout GE template not found")

    from la_figures.ge_convenience import ge_tbl_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    tex = ge_tbl_tex(A, callouts=True)

    # Callouts are rendered into rowechelon_paths and emitted as TikZ \draw commands.
    assert "\\draw[" in tex
    # GE stack naming convention: the first A-block is named A0.
    assert "A0-right" in tex
