from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MethodSpec:
    """Lightweight, paper-friendly gradient-method descriptor."""

    name: str
    precise: bool = True


@runtime_checkable
class GradientMethod(Protocol):
    """Simulator-driven gradient method (e.g. replay/LRDS).

    A GradientMethod is responsible for:
    - declaring which internal signals need to be captured from the runtime
    - defining how to compute gradients (dw/dx/...) given dL/d(probes)

    The framework intentionally treats captured signals as opaque objects to
    avoid leaking method-specific intermediate tensors into task code.
    """

    @property
    def spec(self) -> MethodSpec: ...

    def capture(self, runtime: Any, *, state: Any) -> Any: ...

    def backward(self, runtime: Any, *, captured: Any, dL_dprobe: Any) -> Any: ...

