# Specs

la_figures returns dictionaries suitable for matrixlayout.

Selectors and decorators use 0-based entry indices within each matrix block.
Grid positions are addressed by (row, column) in the outer list.

## GE

`ge_tbl_spec` parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `ref_A` | matrix | required | Reference matrix. |
| `ref_rhs` | matrix | None | Reference RHS (augmented). |
| `rhs` | matrix | None | RHS to eliminate. |
| `pivoting` | str | "none" | "none", "partial", or "full". |
| `gj` | bool | False | Use Gauss-Jordan. |
| `show_pivots` | bool | True | Emit pivot boxes (entry decorators). |
| `index_base` | int | 1 | Index base for labels. |
| `pivot_style` | str | "" | Use CodeAfter fit boxes when set. |
| `pivot_text_color` | str | "red" | Pivot entry text color. |
| `preamble` | str | nicematrix opts | Body preamble. |
| `extension` | str | "" | LaTeX preamble injection. |
| `row_stretch` | float | None | Row spacing multiplier. |
| `nice_options` | str | "" | NiceArray options. |
| `callouts` | list/bool | None | Matrix labels/callouts. |
| `array_names` | list/bool | None | Matrix name labels. |
| `decorators` | list | None | Entry decorators. |
| `format_nrhs` | bool | True | Use format-level RHS separators (disabled when line decorations are emitted). |
| `fig_scale` | float | None | Figure scale. |
| `outer_hspace_mm` | int | 6 | Horizontal block spacing (mm). |
| `cell_align` | str | "r" | Cell alignment in TeX. |
| `variable_summary` | list | None | Override pivot/free indicator row. |
| `variable_colors` | tuple | ("red","black") | Colors for pivot/free indicators. |
| `strict` | bool | None | Error on invalid decorators. |

When `strict=True`, invalid decorator selectors raise errors instead of being ignored.

Pivot rendering:

- Default (`pivot_style=""`): boxed entry decorators in the matrix body.
- Explicit `pivot_style`: CodeAfter `fit` boxes using the provided TikZ style.

Defaults (selected):

- `ge_tbl_spec`: `pivoting="none"`, `gj=False`, `show_pivots=True`
- `qr_tbl_spec`: `array_names=True`
- `eig_tbl_spec`: `normal=False`

GE convenience wrappers render TeX to SVG via matrixlayout. Shared renderer
parameters: `toolchain_name`, `crop`, `padding`, `output_dir`, `output_stem`,
`frame`.

`svg` parameters (subset shown):

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `matrices` | grid | required | Matrix grid. |
| `Nrhs` | int/list | 0 | RHS partitioning. |
| `pivot_list` | list | None | Legacy pivot specs. |
| `bg_for_entries` | list | None | Background highlights. |
| `ref_path_list` | list | None | Row‑echelon paths. |
| `comment_list` | list | None | Right‑side comments. |
| `array_names` | list/bool | None | Matrix labels. |
| `decorators` | list | None | Entry decorators. |

`ge_tbl_tex`, `ge_tbl_svg`, `ge_tbl_bundle` accept the same algorithmic
parameters as `ge_tbl_spec` plus renderer options (for SVG).

Note on RHS separators:

`ge_tbl_spec` now emits RHS separator lines via `decorations` (per-block vlines)
and disables format-level separators to keep label rows clean. The resulting
vertical lines only appear in the matrix rows.

## QR

`qr_tbl_spec` parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `A` | matrix | required | Input matrix. |
| `W` | matrix | required | Orthogonal columns spanning the column space of `A`. |
| `array_names` | bool/list | True | Enable labels or custom list. |
| `fig_scale` | float | None | Figure scale. |
| `preamble` | str | nicematrix opts | Body preamble. |
| `extension` | str | "" | LaTeX preamble injection. |
| `nice_options` | str | "vlines-in-sub-matrix = I" | NiceArray options. |
| `label_color` | str | "blue" | Label color. |
| `label_text_color` | str | "red" | Label text color. |
| `known_zero_color` | str | "brown" | Known-zero highlight. |
| `decorators` | list | None | Entry decorators. |
| `strict` | bool | None | Error on invalid decorators. |

`qr_svg` parameters (subset shown):

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `A` | matrix | required | Input matrix. |
| `W` | matrix | required | Orthogonal basis. |
| `array_names` | bool/list | True | Labels. |
| `decorators` | list | None | Entry decorators. |

`qr_tbl_tex`, `qr_tbl_svg`, `qr_tbl_bundle` follow `qr_tbl_spec` plus renderer
options (for SVG).

QR callout labels are nudged vertically to align with their arrows. For 2x2
inputs, the Q^T label uses a longer arrow to avoid overlapping the R label.

## Eigen/SVD

`eig_tbl_spec` parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `A` | matrix | required | Input matrix. |
| `normal` | bool | False | Normal matrix handling. |
| `Ascale` | scalar | None | Scaling factor. |
| `eig_digits` | int | None | Rounding for eigenvalues. |
| `vec_digits` | int | None | Rounding for eigenvectors. |

`svd_tbl_spec` parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `A` | matrix | required | Input matrix. |
| `Ascale` | scalar | None | Scaling factor. |

`eig_tbl_tex`, `eig_tbl_svg`, `svd_tbl_tex`, `svd_tbl_svg` accept `A` plus the
corresponding spec parameters, along with renderer options (for SVG).
