"""Import forwarding for Gaussian elimination convenience wrappers.

Some callers import GE wrappers from :mod:`la_figures.convenience_ge`. The
current implementation lives in :mod:`la_figures.ge_convenience`.

This module re-exports the public wrappers so both import paths work.
"""

from .ge_convenience import (
    ge,
    ge_tbl_bundle,
    ge_tbl_layout_spec,
    ge_tbl_spec,
    ge_tbl_tex,
    ge_tbl_svg,
)

__all__ = [
    "ge_tbl_bundle",
    "ge_tbl_layout_spec",
    "ge_tbl_spec",
    "ge_tbl_tex",
    "ge_tbl_svg",
    "ge",
]
