import sympy as sym


def test_ge_trace_pivots_and_steps():
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])
    trace = la_figures.ge_trace(A, pivoting="none")

    assert trace.Nrhs == 0
    assert trace.pivot_cols == (0, 1)
    assert trace.free_cols == ()
    assert len(trace.steps) == 2

    layers = la_figures.trace_to_layer_matrices(trace)
    assert layers["Nrhs"] == 0
    assert layers["matrices"][0][0] is None
    assert layers["matrices"][0][1] == trace.initial
    assert layers["matrices"][1][0] is not None
    assert layers["matrices"][1][1].shape == trace.initial.shape


def test_ge_trace_emits_row_exchange_events():
    import la_figures

    # The first pivot in column 0 is in row 1, requiring a swap.
    A = sym.Matrix([[0, 1], [1, 0]])
    trace = la_figures.ge_trace(A, pivoting="none")

    ops = [ev.op for ev in trace.events]
    assert "RequireRowExchange" in ops
    assert "DoRowExchange" in ops

    # The event stream should be serializable to dictionaries.
    payload = trace.events_as_dicts()
    assert isinstance(payload, list)
    assert all("op" in item and "level" in item for item in payload)
