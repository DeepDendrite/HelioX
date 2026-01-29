"""
Backward-compatible facade for the HELIOX simulation runtime.

Historically, this repo exposed a large monolithic module named `heliox_wrapper.py`.
It has been split into smaller, responsibility-focused modules under `heliox_sim/`.

Existing code can keep using:
    from heliox_wrapper import HelioXManager

New code should prefer:
    import heliox_sim
"""

from __future__ import annotations

from heliox_sim import (  # noqa: F401
    GapJunctionInterface,
    MonitorWrapper,
    HelioXManager,
    ObjWrapper,
    RecorderWrapper,
    VecPlayWrapper,
)

__all__ = [
    "HelioXManager",
    "ObjWrapper",
    "MonitorWrapper",
    "RecorderWrapper",
    "VecPlayWrapper",
    "GapJunctionInterface",
]

