import sympy as sym

from la_figures import ge_trace, trace_to_layer_matrices


def test_ge_trace_basic_augmented():
    A = sym.Matrix([[2, 1], [4, 3]])
    b = sym.Matrix([1, 2])

    tr = ge_trace(A, b, pivoting="none")

    # Initial augmented matrix: [A|b]
    assert tr.initial.shape == (2, 3)
    assert tr.Nrhs == 1

    # Two pivots (full rank)
    assert list(tr.pivot_cols) == [0, 1]
    assert list(tr.free_cols) == []
    assert list(tr.pivot_positions) == [(0, 0), (1, 1)]

    # First elimination matrix should eliminate row 2, col 1.
    E0 = tr.steps[0].E
    Ab1 = tr.steps[0].Ab
    assert E0.shape == (2, 2)
    assert E0[1, 0] == -sym.Rational(2, 1)
    assert Ab1[1, 0] == 0

    # Second step should be identity (no elimination needed).
    assert tr.steps[1].E == sym.eye(2)

    # Conversion to legacy-style layer matrices.
    spec = trace_to_layer_matrices(tr, augmented=True)
    mats = spec["matrices"]
    assert len(mats) == 1 + len(tr.steps)
    assert mats[0][0] is None
    assert mats[0][1] == tr.initial


def test_ge_trace_row_swap():
    A = sym.Matrix([[0, 1], [2, 3]])

    tr = ge_trace(A, None, pivoting="none")

    # Pivot in col 0 requires a row swap.
    assert list(tr.pivot_cols) == [0, 1]
    assert tr.steps[0].pivot == (0, 0)

    E0 = tr.steps[0].E
    Ab1 = tr.steps[0].Ab

    # E0 should swap rows 0 and 1.
    assert E0 == sym.Matrix([[0, 1], [1, 0]])
    assert Ab1[0, 0] == 2
    assert Ab1[1, 0] == 0
