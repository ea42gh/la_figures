"""Characterization tests for migrated matrixlayout QR output."""

from __future__ import annotations


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


def _assert_qr_tex_has_markers(tex: str) -> None:
    assert _extract_mat_format(tex)
    assert _extract_mat_rep(tex)
    assert "W^T W" in tex
    assert "v_1" in tex
    assert _submatrix_spans(tex)


def test_qr_tex_matches_legacy_for_2x2():
    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [3, 4]])
    W = sym.Matrix([[1, 0], [0, 1]])
    matrices = compute_qr_matrices(A, W)

    new = qr_grid_tex(matrices=matrices, formatter=str)
    _assert_qr_tex_has_markers(new)


def test_qr_tex_matches_legacy_for_3x2():
    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [3, 4], [5, 6]])
    W = sym.Matrix([[1, 0], [0, 1], [0, 0]])
    matrices = compute_qr_matrices(A, W)

    new = qr_grid_tex(matrices=matrices, formatter=str)
    _assert_qr_tex_has_markers(new)


def test_qr_tex_matches_legacy_for_rank_deficient():
    import sympy as sym

    from la_figures import compute_qr_matrices
    from matrixlayout.qr import qr_grid_tex

    A = sym.Matrix([[1, 2], [2, 4]])
    W = sym.Matrix([[1, 2], [2, 4]])
    matrices = compute_qr_matrices(A, W)

    new = qr_grid_tex(matrices=matrices, formatter=str)
    _assert_qr_tex_has_markers(new)
