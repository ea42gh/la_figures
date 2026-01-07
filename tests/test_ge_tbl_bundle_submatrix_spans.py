import sympy as sym


def test_ge_tbl_bundle_includes_submatrix_spans_metadata():
    import pytest

    # The spans metadata depends on the matrixlayout bundle API.
    pytest.importorskip("matrixlayout")
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])

    bundle = la_figures.ge_tbl_bundle(A)
    assert isinstance(bundle, dict)
    assert "tex" in bundle
    assert "spec" in bundle

    spans = bundle.get("submatrix_spans")
    assert isinstance(spans, list)
    assert spans, "expected at least one SubMatrix span"

    names = {s.get("name") for s in spans}
    # In the legacy 2-column GE-grid layout, the initial layer includes A0.
    assert "A0" in names

    # Convenience fields for notebook authors.
    a0 = next(s for s in spans if s.get("name") == "A0")
    assert a0["left_delim_node"] == "A0-left"
    assert a0["right_delim_node"] == "A0-right"
