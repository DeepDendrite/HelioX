from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Trainable:
    """Declarative description of a trainable parameter.

    Notes:
    - `handle` is a logical identifier resolved by the runtime/bundle mapping.
    - `bounds` is applied by the task or optimizer (framework will standardize).
    """

    name: str
    handle: str
    bounds: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class Probe:
    """Declarative description of a probe/observable signal."""

    name: str
    handle: str


@dataclass(frozen=True)
class TaskSpec:
    """High-level task specification.

    This is intentionally lightweight: it is the piece we want to show in papers
    as the user-facing task definition.
    """

    name: str
    trainables: Sequence[Trainable] = ()
    probes: Sequence[Probe] = ()


class LearningTask(Protocol):
    """Task interface consumed by the generic Trainer.

    The goal is to keep application-specific details out of the framework core:
    the task owns how to run one epoch; Trainer owns lifecycle/checkpoint/logging.
    """

    @property
    def spec(self) -> TaskSpec: ...

    def setup(self, runtime: Any) -> Any: ...

    def run_epoch(self, runtime: Any, state: Any, epoch: int) -> tuple[Any, dict[str, float]]: ...
