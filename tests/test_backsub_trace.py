def test_backsub_trace_has_base_and_steps():
    from la_figures import backsub_trace_from_ref

    # System with one free variable (x_3)
    A = [[1, 0, 1], [0, 1, 1]]
    b = [1, 2]
    trace = backsub_trace_from_ref(A, b)

    assert "base" in trace and "steps" in trace
    assert "x_3" in trace["base"]
    assert r"\alpha_3" in trace["base"]
    assert len(trace["steps"]) == 2
    raw0, sub0 = trace["steps"][0]
    assert raw0.startswith("x_2 =")
    assert sub0.startswith("x_2 =")