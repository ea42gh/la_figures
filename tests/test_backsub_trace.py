def test_backsubstitution_tex_has_base_and_steps():
    from la_figures import backsubstitution_tex

    # System with one free variable (x_3)
    A = [[1, 0, 1], [0, 1, 1]]
    b = [1, 2]
    lines = backsubstitution_tex(A, b)

    text = "\n".join(lines)
    assert "x_3" in text
    assert r"\alpha_3" in text
    assert "x_2 =" in text
    assert "x_1 =" in text
