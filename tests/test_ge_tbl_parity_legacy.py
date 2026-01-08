import re

import pytest
import sympy as sym


def _legacy_ge_tex(matrices, *, Nrhs=0):
    itikz = pytest.importorskip("itikz")
    import numpy as np

    nM = itikz.nicematrix
    m = nM.MatrixGridLayout(matrices, extra_rows=None, extra_cols=None)

    if isinstance(Nrhs, np.ndarray):
        Nrhs = list(Nrhs.flatten())

    if not isinstance(Nrhs, list):
        partitions = {} if Nrhs == 0 else {1: [m.mat_col_width[-1] - Nrhs]}
    else:
        nrhs = Nrhs.copy()
        cuts = [m.mat_col_width[-1] - sum(nrhs)]
        for cut in nrhs[0:-1]:
            cuts.append(cuts[-1] + cut)
        partitions = {1: cuts}

    m.array_format_string_list(partitions=partitions)
    m.array_of_tex_entries(formater=str)
    m.nm_submatrix_locs("A", color="blue", name_specs=None)
    m.tex_repr()
    return m.nm_latexdoc(template=nM.GE_TEMPLATE, fig_scale=None)


def _submatrix_names(tex: str):
    return re.findall(r"name=([A-Za-z0-9_]+)", tex)


def _submatrix_spans(tex: str):
    return re.findall(r"\\SubMatrix\(\{([^}]+)\}\{([^}]+)\}\)", tex)


def test_ge_submatrix_names_match_legacy_no_rhs():
    import la_figures
    from matrixlayout.ge import ge_grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    tr = la_figures.ge_trace(A, pivoting="none")
    layers = la_figures.trace_to_layer_matrices(tr, augmented=True)["matrices"]

    legacy = _legacy_ge_tex(layers, Nrhs=0)
    new = ge_grid_tex(matrices=layers, Nrhs=0)

    legacy_names = _submatrix_names(legacy)
    new_names = _submatrix_names(new)
    legacy_spans = _submatrix_spans(legacy)
    new_spans = _submatrix_spans(new)
    assert len(new_names) == len(legacy_names)
    assert len(set(new_names)) == len(new_names)
    assert new_spans == legacy_spans


def test_ge_submatrix_names_match_legacy_with_rhs():
    import la_figures
    from matrixlayout.ge import ge_grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    rhs = sym.Matrix([[5], [6]])
    tr = la_figures.ge_trace(A, rhs, pivoting="none")
    layers = la_figures.trace_to_layer_matrices(tr, augmented=True)["matrices"]

    legacy = _legacy_ge_tex(layers, Nrhs=1)
    new = ge_grid_tex(matrices=layers, Nrhs=1)

    legacy_names = _submatrix_names(legacy)
    new_names = _submatrix_names(new)
    legacy_spans = _submatrix_spans(legacy)
    new_spans = _submatrix_spans(new)
    assert len(new_names) == len(legacy_names)
    assert len(set(new_names)) == len(new_names)
    assert new_spans == legacy_spans
