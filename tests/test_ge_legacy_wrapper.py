import pytest
import sympy as sym


def test_legacy_pivot_list_to_pivot_locs():
    from la_figures.ge_convenience import _legacy_pivot_list_to_pivot_locs

    A0 = sym.Matrix([[1, 2], [3, 4]])
    E1 = sym.eye(2)
    A1 = sym.Matrix([[1, 2], [0, 1]])
    matrices = [[None, A0], [E1, A1]]

    pivot_list = [
        ((1, 1), [(0, 0), (1, 1)]),
    ]

    pivot_locs = _legacy_pivot_list_to_pivot_locs(matrices, pivot_list, index_base=1, pivot_style="draw=red")
    assert pivot_locs == [
        ("(3-3)(3-3)", "draw=red"),
        ("(4-4)(4-4)", "draw=red"),
    ]


def test_ge_legacy_wrapper_rejects_unsupported_options():
    from la_figures.convenience_ge import ge

    A = sym.Matrix([[1, 2], [3, 4]])
    with pytest.raises(NotImplementedError):
        ge([[None, A]], bg_for_entries=[1])
