"""
neurong_core
============

Simulation-only Python facade for NEURONG runtime helpers.

The module uses lazy re-exports to avoid import cycles between
`neurong_sim` and `neurong_core.binding`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NeurongManager",
    "ObjWrapper",
    "MonitorWrapper",
    "RecorderWrapper",
    "VecPlayWrapper",
    "HandleBatch",
    "VecPlayBatch",
    "get_variable_handle",
]


def __getattr__(name: str) -> Any:
    if name in {"NeurongManager", "ObjWrapper", "MonitorWrapper", "RecorderWrapper", "VecPlayWrapper"}:
        mod = import_module("neurong_sim")
        return getattr(mod, name)
    if name in {"HandleBatch", "VecPlayBatch"}:
        mod = import_module(".batch", __name__)
        return getattr(mod, name)
    if name == "get_variable_handle":
        mod = import_module(".binding", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
