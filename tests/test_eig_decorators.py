import sympy as sym


def test_eig_tbl_tex_decorates_eigenbasis_entry():
    from la_figures import eig_tbl_tex

    A = sym.Matrix([[2]])

    def dec(tex: str) -> str:
        return rf"\boxed{{{tex}}}"

    tex = eig_tbl_tex(
        A,
        formater=sym.latex,
        decorators=[{"target": "eigenbasis", "entries": [(0, 0, 0)], "decorator": dec}],
        preamble="",
    )

    assert r"\boxed{1}" in tex or r"\boxed{1.0}" in tex


def test_svd_tbl_tex_decorates_sigma_matrix_entry():
    from la_figures import svd_tbl_tex

    A = sym.Matrix([[3, 0], [0, 0]])

    def dec(tex: str) -> str:
        return rf"\boxed{{{tex}}}"

    tex = svd_tbl_tex(
        A,
        formater=sym.latex,
        decorators=[{"matrix": "sigma", "entries": [(0, 0)], "decorator": dec}],
        preamble="",
    )

    assert r"\boxed" in tex
