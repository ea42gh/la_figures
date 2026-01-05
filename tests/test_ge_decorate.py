import sympy as sym


def test_decorate_ge_pivots_and_variable_types():
    from la_figures import ge as ge_mod

    A = sym.Matrix([[1, 2], [3, 4]])
    tr = ge_mod.ge_trace(A, pivoting="partial")

    decor = ge_mod.decorate_ge(tr, index_base=1)

    assert decor["variable_types"] == ["pivot", "pivot"]
    assert decor["pivot_locs"] == [
        ("(1-1)(1-1)", ""),
        ("(2-2)(2-2)", ""),
    ]


def test_decorate_ge_excludes_rhs_from_variable_types():
    from la_figures import ge as ge_mod

    A = sym.Matrix([[1, 2], [3, 4]])
    rhs = sym.Matrix([[5], [6]])

    tr = ge_mod.ge_trace(A, rhs, pivoting="partial")
    assert tr.Nrhs == 1

    decor = ge_mod.decorate_ge(tr, index_base=1)

    # RHS columns must not be classified as variables.
    assert len(decor["variable_types"]) == A.cols
    assert decor["variable_types"] == ["pivot", "pivot"]
