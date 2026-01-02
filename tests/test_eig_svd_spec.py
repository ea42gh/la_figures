import sympy as sym


def test_eig_tbl_spec_basic():
    from la_figures import eig_tbl_spec

    A = [[2, 0], [0, 3]]
    spec = eig_tbl_spec(A)
    assert set(spec.keys()) == {"lambda", "ma", "evecs"}
    assert len(spec["lambda"]) == 2
    assert sum(spec["ma"]) == 2
    # Each group contains at least one vector.
    assert all(len(g) >= 1 for g in spec["evecs"])


def test_eig_tbl_spec_normal_adds_qvecs():
    from la_figures import eig_tbl_spec

    A = [[1, 0], [0, 1]]
    spec = eig_tbl_spec(A, normal=True)
    assert "qvecs" in spec
    # Identity has a 2D eigenspace.
    assert len(spec["qvecs"][0]) == 2


def test_svd_tbl_spec_has_sz_and_sigma():
    from la_figures import svd_tbl_spec

    A = [[3, 0], [0, 0]]
    spec = svd_tbl_spec(A)
    assert spec["sz"] == (2, 2)
    assert "sigma" in spec and len(spec["sigma"]) >= 1
    assert "qvecs" in spec and "uvecs" in spec