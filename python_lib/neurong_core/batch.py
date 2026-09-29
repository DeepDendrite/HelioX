"""
neurong_core.batch
=================

Lightweight helpers for batch IO against a NEURONG runtime.

Notes
-----
- NEURONG already exposes batch APIs on `NeurongManager`:
  - `get_variables_by_handles_f32(handles)`
  - `set_variables_by_handles(handles, values)`
- This module provides small convenience wrappers to make those "first class"
  in application code (and easier to later swap implementations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any

import numpy as np


@dataclass(frozen=True)
class HandleBatch:
    """A pre-bound batch of NEURONG variable handles."""

    manager: Any
    handles: tuple[int, ...]

    @classmethod
    def from_iterable(cls, manager: Any, handles: Iterable[int]) -> "HandleBatch":
        return cls(manager=manager, handles=tuple(int(h) for h in handles))

    def read_f32(self) -> np.ndarray:
        """Read all handles into a float32 numpy array."""
        return self.manager.get_variables_by_handles_f32(list(self.handles))

    def read(self) -> list[float]:
        """Read all handles into a Python list (fallback/compat)."""
        return self.manager.get_variables_by_handles(list(self.handles))

    def write(self, values: Sequence[float]) -> int:
        """Write values (same length as handles) via batch API."""
        if len(values) != len(self.handles):
            raise ValueError(f"expected {len(self.handles)} values, got {len(values)}")
        return int(self.manager.set_variables_by_handles(list(self.handles), list(values)))


@dataclass
class VecPlayBatch:
    """
    Convenience wrapper for controlling multiple VecPlayWrapper channels.

    This does not change asymptotic performance by itself; it just centralizes
    the shape checks and makes call sites cleaner.
    """

    vecplays: list[Any]

    def play_matrix(self, x: np.ndarray, *, dt_ms: float) -> None:
        """
        Play input matrix x over all vecplays.

        Parameters
        ----------
        x:
            shape (n_channels, T) float array (amps per channel per step).
        dt_ms:
            timestep in ms for tvec.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected x shape (C, T), got {x.shape}")
        if len(self.vecplays) != x.shape[0]:
            raise ValueError(f"expected {len(self.vecplays)} channels, got {x.shape[0]}")
        tvec = (np.arange(x.shape[1], dtype=np.float64) * float(dt_ms)).tolist()
        for i, vp in enumerate(self.vecplays):
            vp.play(tvec, x[i, :].tolist())

