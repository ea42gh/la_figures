import re

import pytest
import sympy as sym


def _legacy_qr_layout(matrices, *, array_names=True):
    itikz = pytest.importorskip("itikz")
    nM = itikz.nicematrix

    m = nM.MatrixGridLayout(matrices, extra_rows=[1, 0, 0, 0])
    m.preamble = nM.preamble + "\n" + r" \NiceMatrixOptions{cell-space-limits = 2pt}" + "\n"

    n = matrices[0][2].shape[1]
    m.array_format_string_list()
    m.array_of_tex_entries(formater=str)

    brown = nM.make_decorator(text_color="brown", bf=True)

    def _known_zeros(WtA, WtW):
        l_WtA = [(1, 2), [(i, j) for i in range(WtA.shape[0]) for j in range(WtA.shape[0]) if i > j]]
        l_WtW = [(1, 3), [(i, j) for i in range(WtW.shape[0]) for j in range(WtW.shape[0]) if i != j]]
        return [l_WtA, l_WtW]

    for spec in _known_zeros(matrices[1][2], matrices[1][3]):
        m.decorate_tex_entries(*spec[0], brown, entries=spec[1])

    red = nM.make_decorator(text_color="red", bf=True)
    red_rgt = nM.make_decorator(text_color="red", bf=True, move_right=True)
    m.add_row_above(
        0,
        2,
        [red(f"v_{i+1}") for i in range(n)] + [red(f"w_{i+1}") for i in range(n)],
        formater=lambda a: a,
    )
    m.add_col_left(1, 1, [red_rgt(f"w^t_{i+1}") for i in range(n)], formater=lambda a: a)

    if array_names:
        dec = nM.make_decorator(bf=True, delim="$")
        name_specs = [
            [(0, 2), "al", dec("A")],
            [(0, 3), "ar", dec("W")],
            [(1, 1), "al", dec("W^t")],
            [(1, 2), "al", dec("W^t A")],
            [(1, 3), "ar", dec("W^t W")],
            [(2, 0), "al", dec(r"S = \left( W^t W \right)^{-\tfrac{1}{2}}")],
            [(2, 1), "br", dec(r"Q^t = S W^t")],
            [(2, 2), "br", dec("R = S W^t A")],
        ]
        m.nm_submatrix_locs("QR", color="blue", name_specs=name_specs)
    else:
        m.nm_submatrix_locs()

    m.tex_repr(blockseps=r"\noalign{\vskip3mm} ")
    return m


def _extract_mat_format(tex: str) -> str:
    start = tex.find(r"\begin{NiceArray")
    if start < 0:
        return ""
    i = start + len(r"\begin{NiceArray")
    if i < len(tex) and tex[i] == "}":
        i += 1
    # Skip optional [..]
    if i < len(tex) and tex[i] == "[":
        depth = 1
        i += 1
        while i < len(tex) and depth:
            if tex[i] == "[":
                depth += 1
            elif tex[i] == "]":
                depth -= 1
            i += 1
    # Skip whitespace
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
    if start < 0:
        return ""
    start = start + len(r"\Body%")
    end = tex.find(r"\CodeAfter", start)
    if end < 0:
        body = tex[start:]
    else:
        body = tex[start:end]
    # Drop template comment markers emitted after the body (e.g., "%% %").
    body = re.sub(r"\s*%+\s*$", "", body, flags=re.MULTILINE)
    return body.strip()


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _submatrix_names(tex: str):
    return re.findall(r"name=([A-Za-z0-9_]+)", tex)


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
        m = re.search(r"\{([^}]+)\}\{([^}]+)\}", inner)
        if not m:
            continue
        spans.append(f"{{{m.group(1)}}}{{{m.group(2)}}}")
    return spans


@pytest.mark.parametrize(
    ("A", "W"),
    [
        (sym.Matrix([[1, 2], [3, 4]]), sym.Matrix([[1, 0], [0, 1]])),
        (sym.Matrix([[1, 2], [3, 4], [5, 6]]), sym.Matrix([[1, 0], [0, 1], [0, 0]])),
        (sym.Matrix([[1, 2], [2, 4]]), sym.Matrix([[1, 2], [2, 4]])),
    ],
)
def test_qr_layout_matches_legacy(A, W):
    import la_figures
    from matrixlayout.qr import qr_grid_tex

    matrices = la_figures.compute_qr_matrices(A, W)
    legacy = _legacy_qr_layout(matrices)
    new = qr_grid_tex(matrices=matrices, formatter=str)

    assert _normalize(_extract_mat_format(new)) == _normalize(legacy.format)
    assert _normalize(_extract_mat_rep(new)) == _normalize("\n".join(legacy.tex_list))
    assert _submatrix_spans(new) == [loc[1] for loc in legacy.locs]
    assert _submatrix_names(new) == [re.search(r"name=([A-Za-z0-9_]+)", loc[0]).group(1) for loc in legacy.locs]
