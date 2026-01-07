import sympy as sym


def test_ge_trace_pivots_and_steps():
    import la_figures

    A = sym.Matrix([[1, 2], [3, 4]])
    trace = la_figures.ge_trace(A, pivoting="none")

    assert trace.Nrhs == 0
    assert trace.pivot_cols == (0, 1)
    assert trace.free_cols == ()
    assert len(trace.steps) == 1

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


def test_ge_trace_emits_per_row_elimination_steps():
    import la_figures

    A = sym.Matrix([[1, 0, 0], [2, 1, 0], [3, 0, 1]])
    trace = la_figures.ge_trace(A, pivoting="none")

    elim_events = [ev for ev in trace.events if ev.op == "DoElimination"]
    # Eliminate rows 1 and 2 in the first pivot column.
    assert [ev.data["target_row"] for ev in elim_events] == [1, 2]
    assert len(trace.steps) == 2


def test_ge_trace_separates_row_swap_and_elimination_steps():
    import la_figures

    A = sym.Matrix([[0, 1, 0], [1, 1, 0], [2, 0, 1]])
    trace = la_figures.ge_trace(A, pivoting="none")

    ops = [ev.op for ev in trace.events]
    assert "RequireRowExchange" in ops
    assert "DoRowExchange" in ops
    assert "RequireElimination" in ops
    assert "DoElimination" in ops

    assert ops.index("RequireRowExchange") < ops.index("DoRowExchange")
    assert ops.index("DoRowExchange") < ops.index("RequireElimination")
