from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ObjectiveSpec:
    """Lightweight, paper-friendly objective descriptor."""

    name: str


@runtime_checkable
class Objective(Protocol):
    """Objective/Loss interface.

    An objective consumes task probes (e.g. Vout(t)) and produces:
    - a scalar loss (or mean error)
    - gradients w.r.t. probe traces (e.g. dL/dVout(t))

    The concrete data type of traces can be numpy arrays, torch tensors, or
    backend-specific views; tasks should treat it as an opaque "trace" object.
    """

    @property
    def spec(self) -> ObjectiveSpec: ...

    def loss_and_grad(self, traces: Any) -> tuple[float, Any]: ...

