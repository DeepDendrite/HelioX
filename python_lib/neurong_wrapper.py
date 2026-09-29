"""
Backward-compatible facade for the NEURONG simulation/runtime API.

Historically this repo exposed a large monolithic module named
`neurong_wrapper.py`. The implementation has been split into the
`neurong_sim` package; this file remains as a stable import path for
existing code.

Existing code can keep using:
    from neurong_wrapper import NeurongManager

New code should prefer:
    import neurong_sim

If you want a single aggregated entrypoint, this facade also lazy-exports
selected `neurong_learn` symbols.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from neurong_sim import (  # noqa: F401
    GapJunctionInterface,
    MonitorWrapper,
    NeurongManager,
    ObjWrapper,
    RecorderWrapper,
    VecPlayWrapper,
    WindowRecorderWrapper,
)

__all__ = [
    "NeurongManager",
    "ObjWrapper",
    "MonitorWrapper",
    "RecorderWrapper",
    "WindowRecorderWrapper",
    "VecPlayWrapper",
    "GapJunctionInterface",
    "learn",
    "Runtime",
    "RuntimeConfig",
    "SegmentRunner",
    "SegmentSpec",
    "SegmentWindow",
    "MonitorHandleGroup",
    "LearningBackend",
    "LearningOps",
    "OnlineVJPLearner",
    "VJPSegment",
    "VJPForwardResult",
    "VJPLossResult",
    "VJPStepResult",
]


def __getattr__(name: str) -> Any:
    if name == "learn":
        return import_module("neurong_learn")
    if name in {
        "Runtime",
        "RuntimeConfig",
        "SegmentRunner",
        "SegmentSpec",
        "SegmentWindow",
        "MonitorHandleGroup",
        "LearningBackend",
        "LearningOps",
        "OnlineVJPLearner",
        "VJPSegment",
        "VJPForwardResult",
        "VJPLossResult",
        "VJPStepResult",
    }:
        mod = import_module("neurong_learn")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
