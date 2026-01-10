"""Characterization tests: legacy itikz QR vs migrated matrixlayout QR."""

from __future__ import annotations

import sys


def _ensure_repo_on_path() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    itikz_root = root / "itikz"

    for p in (root, itikz_root):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)

    for p in (root, itikz_root):
        sys.path.insert(0, str(p))


def _extract_mat_format(tex: str) -> str:
    start = tex.find(r"\begin{NiceArray")
    if start < 0:
        return ""
    i = start + len(r"\begin{NiceArray")
    if i < len(tex) and tex[i] == "}":
        i += 1
    if i < len(tex) and tex[i] == "[":
        depth = 1
        i += 1
        while i < len(tex) and depth:
            if tex[i] == "[":
                depth += 1
            elif tex[i] == "]":
                depth -= 1
            i += 1
    while i < len(tex) and tex[i].isspace():
        i += 1
    if i >= len(tex) or tex[i] != "{":
        return ""
    i += 1
    brace = 1
    buf = []
    while i < len(tex) and brace:
        ch = tex[i]
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                i += 1
                break
        if brace:
            buf.append(ch)
        i += 1
    return "".join(buf).strip()


def _extract_mat_rep(tex: str) -> str:
    start = tex.find(r"\Body%")
    if start >= 0:
        start = start + len(r"\Body%")
    else:
        start = tex.find(r"\begin{NiceArray")
        if start < 0:
            return ""
        start = tex.find("\n", start)
        if start < 0:
            return ""
        start += 1
    end = tex.find(r"\CodeAfter", start)
    if end < 0:
        body = tex[start:]
    else:
        body = tex[start:end]
    body = body.replace("%", "").strip()
    return body


def _normalize_mat_rep(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def _submatrix_spans(tex: str):
    spans = []
    for line in tex.splitlines():
        if "\\SubMatrix" not in line:
            continue
        start = line.find("(")
        end = line.find(")", start + 1)
        if start < 0 or end < 0:
            continue
        inner = line[start + 1 : end]
        import re

        m = re.search(r"\{([^}]+)\}\{([^}]+)\}", inner)
        if not m:
            continue
        spans.append(f"{{{m.group(1)}}}{{{m.group(2)}}}")
    return spans


def _submatrix_names(tex: str):
    import re

    return re.findall(r"name=([A-Za-z0-9_]+)", tex)


def _legacy_qr_tex(matrices):
    import itikz.nicematrix as nM
    m = nM.MatrixGridLayout(matrices, extra_rows=[0, 0, 0, 0])
    m.preamble = nM.preamble + "\n" + r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n"

    n = matrices[0][2].shape[1]
    m.array_format_string_list()
    m.array_of_tex_entries(formater=str)

    brown = nM.make_decorator(text_color="brown", bf=True)
    l_WtA = [(1, 2), [(i, j) for i in range(matrices[1][2].shape[0]) for j in range(matrices[1][2].shape[0]) if i > j]]
    l_WtW = [(1, 3), [(i, j) for i in range(matrices[1][3].shape[0]) for j in range(matrices[1][3].shape[0]) if i != j]]
    for spec in (l_WtA, l_WtW):
        m.decorate_tex_entries(*spec[0], brown, entries=spec[1])

    dec = nM.make_decorator(bf=True, delim="$")
    name_specs = [
        [(0, 2), "al", dec("A")],
        [(0, 3), "ar", dec("W")],
        [(1, 1), "al", dec("W^t")],
        [(1, 2), "al", dec("W^t A")],
        [(1, 3), "ar", dec("W^t W")],
        [(2, 0), "al", dec(r"S = \left( W^t W \right)^{-\tfrac{1}{2}}")],
        [(2, 1), "br", dec(r"Q^t = S W^t")],
        [(2, 2), "br", dec(r"R = S W^t A")],
    ]
    m.nm_submatrix_locs("QR", color="blue", name_specs=name_specs)
    m.tex_repr(blockseps=r"\noalign{\vskip3mm} ")
    return m.nm_latexdoc(template=nM.GE_TEMPLATE, fig_scale=None)


def _compare_blocks(legacy: str, new: str) -> bool:
    return (
        _extract_mat_format(legacy) == _extract_mat_format(new)
        and _normalize_mat_rep(_extract_mat_rep(legacy)) == _normalize_mat_rep(_extract_mat_rep(new))
        and _submatrix_spans(legacy) == _submatrix_spans(new)
        and _submatrix_names(legacy) == _submatrix_names(new)
    )


def test_qr_tex_matches_legacy_for_2x2():
    _ensure_repo_on_path()
    try:
        import itikz  # noqa: F401
    except Exception:
        import pytest

        pytest.skip("legacy itikz.nicematrix not importable")

    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    W = sym.Matrix([[1, 0], [0, 1]])
    matrices = compute_qr_matrices(A, W)

    legacy = _legacy_qr_tex(matrices)
    new = qr_grid_tex(matrices=matrices, formatter=str)
    assert _compare_blocks(legacy, new)


def test_qr_tex_matches_legacy_for_3x2():
    _ensure_repo_on_path()
    try:
        import itikz  # noqa: F401
    except Exception:
        import pytest

        pytest.skip("legacy itikz.nicematrix not importable")

    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [3, 4], [5, 6]])
    W = sym.Matrix([[1, 0], [0, 1], [0, 0]])
    matrices = compute_qr_matrices(A, W)

    legacy = _legacy_qr_tex(matrices)
    new = qr_grid_tex(matrices=matrices, formatter=str)
    assert _compare_blocks(legacy, new)


def test_qr_tex_matches_legacy_for_rank_deficient():
    _ensure_repo_on_path()
    try:
        import itikz  # noqa: F401
    except Exception:
        import pytest

        pytest.skip("legacy itikz.nicematrix not importable")

    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [2, 4]])
    W = sym.Matrix([[1, 2], [2, 4]])
    matrices = compute_qr_matrices(A, W)

    legacy = _legacy_qr_tex(matrices)
    new = qr_grid_tex(matrices=matrices, formatter=str)
    assert _compare_blocks(legacy, new)
