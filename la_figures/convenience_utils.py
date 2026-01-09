"""Shared normalization helpers for convenience wrappers."""

from __future__ import annotations

import re
from typing import Any


def norm_str(x: Any) -> Any:
    """Normalize Julia/PythonCall/PyCall string-like values."""
    if x is None:
        return None
    s = x if isinstance(x, str) else str(x)
    s = s.strip()
    if s.startswith(":"):
        return s[1:]
    if "Symbol(" in s:
        match = re.search(r"Symbol\((.*)\)", s)
        if match:
            inner = match.group(1).strip()
            if inner.startswith(":"):
                inner = inner[1:]
            if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in ("'", '"'):
                inner = inner[1:-1]
            return inner
    return s


def norm_padding(padding: Any) -> Any:
    """Normalize padding inputs into a 4-tuple when possible."""
    if padding is None:
        return None
    if isinstance(padding, tuple) and len(padding) == 4:
        return padding
    try:
        seq = list(padding)
    except Exception:
        return padding
    return tuple(seq)


__all__ = ["norm_str", "norm_padding"]
