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

    spec = la_figures.ge_tbl_spec(A, array_names=True)
    assert spec.get("callouts")

    spec = la_figures.ge_tbl_spec(A, show_pivots=True)
    assert spec.get("codebefore")
    assert spec.get("rowechelon_paths")


def test_ge_tbl_layout_spec_uses_typed_layout():
    import pytest

    pytest.importorskip("matrixlayout")
    from la_figures.ge_convenience import ge_tbl_layout_spec
    from matrixlayout.specs import GELayoutSpec
    from matrixlayout.ge import grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    spec = ge_tbl_layout_spec(A, show_pivots=True)

    assert isinstance(spec["layout"], GELayoutSpec)
    tex = grid_tex(**spec)
    assert "\\begin{NiceArray}" in tex


def test_ge_tbl_spec_variable_summary_labels():
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])
    spec = la_figures.ge_tbl_spec(
        A,
        variable_summary=[True, False],
        variable_colors=("blue", "orange"),
    )

    labels = spec.get("variable_labels") or []
    assert labels
    rows = labels[0]["rows"]
    assert rows[0] == [
        r"$\textcolor{blue}{\Uparrow}$",
        r"$\textcolor{orange}{\uparrow}$",
    ]
    assert rows[1] == [
        r"$\textcolor{blue}{x_{1}}$",
        r"$\textcolor{orange}{x_{2}}$",
    ]


def test_ge_tbl_spec_rhs_vline():
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])
    spec = la_figures.ge_tbl_spec(A, rhs=[5, 6])

    decorations = spec.get("decorations") or []
    assert {"grid": (0, 1), "vlines": 2} in decorations
    assert {"grid": (1, 1), "vlines": 2} in decorations


def test_ge_tbl_spec_layout_and_bundle_match_tex():
    import pytest

    pytest.importorskip("matrixlayout")
    import la_figures
    from la_figures.ge_convenience import ge_tbl_layout_spec
    from matrixlayout.ge import grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    rhs = sym.Matrix([[5], [6]])

    spec = la_figures.ge_tbl_spec(A, ref_rhs=rhs, show_pivots=True)
    layout_spec = ge_tbl_layout_spec(A, ref_rhs=rhs, show_pivots=True)
    bundle = la_figures.ge_tbl_bundle(A, ref_rhs=rhs, show_pivots=True)

    tex_spec = grid_tex(**spec)
    tex_layout = grid_tex(**layout_spec)
    tex_bundle = grid_tex(**bundle["spec"])

    assert tex_spec == tex_layout
    assert tex_spec == tex_bundle
