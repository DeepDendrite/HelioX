from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backend import LearningBackend
from .ops import LearningOps
from .segment import SegmentRunner, SegmentWindow


@dataclass(frozen=True)
class VJPSegment:
    """One forward-time learning segment.

    The public contract is step-based because training targets and loss arrays
    are naturally indexed by sample step. Backend time conversion stays here.
    """

    index: int
    start_step: int
    stop_step: int
    dt_ms: float

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError(f"segment index must be non-negative, got {self.index}")
        if int(self.stop_step) <= int(self.start_step):
            raise ValueError(
                f"segment stop_step must be greater than start_step, "
                f"got {self.start_step}->{self.stop_step}"
            )
        if float(self.dt_ms) <= 0.0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")

    @property
    def step_count(self) -> int:
        return int(self.stop_step) - int(self.start_step)

    @property
    def start_ms(self) -> float:
        return float(self.start_step) * float(self.dt_ms)

    @property
    def stop_ms(self) -> float:
        return float(self.stop_step) * float(self.dt_ms)

    @property
    def duration_ms(self) -> float:
        return float(self.step_count) * float(self.dt_ms)

    @property
    def target_slice(self) -> slice:
        return slice(int(self.start_step), int(self.stop_step))


@dataclass(frozen=True)
class VJPForwardResult:
    """Forward result for one VJP segment."""

    segment: VJPSegment
    window: SegmentWindow

    @property
    def monitors(self) -> dict[str, np.ndarray]:
        return self.window.monitors

    @property
    def step_count(self) -> int:
        return self.segment.step_count


@dataclass(frozen=True)
class VJPLossResult:
    """Result returned by a high-level loss callback."""

    loss: float
    dldv: np.ndarray
    metrics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class VJPStepResult:
    """Forward + backward result for one segment."""

    forward: VJPForwardResult
    loss: float
    metrics: Mapping[str, float]


LossGradFn = Callable[
    [VJPForwardResult],
    VJPLossResult | tuple[float, Any] | tuple[float, Any, Mapping[str, float]],
]


class OnlineVJPLearner:
    """High-level coordinator for full-window and online segmented VJP training.

    This class intentionally keeps the learning task in user control: callers
    choose the segment schedule, compute the loss, and update parameters. It
    hides the backend-only details needed to run one segment correctly.
    """

    def __init__(
        self,
        manager: Any,
        *,
        voltage_grad_handles: Any,
        monitor_groups: Mapping[str, Any],
        vjp_mechanisms: Sequence[str],
        dt_ms: float,
        sim_client: Any | None = None,
        backend: LearningBackend | None = None,
        ops: LearningOps | None = None,
        _group_index: int = 0,
    ):
        if float(dt_ms) <= 0.0:
            raise ValueError(f"dt_ms must be positive, got {dt_ms}")
        self.manager = manager
        self.dt_ms = float(dt_ms)
        self.voltage_grad_handles = self._as_int_array(voltage_grad_handles, "voltage_grad_handles")
        if self.voltage_grad_handles.size == 0:
            raise ValueError("voltage_grad_handles must not be empty")

        self.monitor_groups = {
            str(name): SegmentRunner.normalize_monitor_group(handles, f"monitor_groups[{name!r}]")
            for name, handles in dict(monitor_groups).items()
        }
        if not self.monitor_groups:
            raise ValueError("monitor_groups must not be empty")

        self.vjp_mechanisms = tuple(str(name) for name in vjp_mechanisms if str(name))
        if not self.vjp_mechanisms:
            raise ValueError("vjp_mechanisms must not be empty")

        self._group_index = int(_group_index)
        self.runner = SegmentRunner(
            manager,
            sim_client=sim_client,
            backend=backend,
            dt_ms=self.dt_ms,
        )
        self.ops = ops or LearningOps.from_manager(manager)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        voltage_grad_handles: Any,
        monitor_groups: Mapping[str, Any],
        vjp_mechanisms: Sequence[str],
        dt_ms: float | None = None,
        **kwargs: Any,
    ) -> "OnlineVJPLearner":
        if not getattr(runtime, "loaded", False):
            runtime.load()
        dt_use = float(runtime.dt_ms if dt_ms is None else dt_ms)
        return cls(
            runtime.manager,
            voltage_grad_handles=voltage_grad_handles,
            monitor_groups=monitor_groups,
            vjp_mechanisms=vjp_mechanisms,
            dt_ms=dt_use,
            ops=getattr(runtime, "ops", None),
            **kwargs,
        )

    @staticmethod
    def _as_int_array(values: Any, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int32).reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be array-like")
        return arr

    def initialize(self, v_init: float) -> None:
        self.manager.finitialize(float(v_init))

    def play_inputs(
        self,
        stim_ctrls: Sequence[Sequence[Any]],
        inputs: Sequence[Sequence[Any]],
        *,
        base_time_ms: float = 0.0,
    ) -> None:
        self.runner.play_inputs(stim_ctrls, inputs, base_time_ms=float(base_time_ms))

    def current_time_ms(self) -> float:
        return self.runner.global_t()

    def current_step(self) -> int:
        return int(round(self.current_time_ms() / self.dt_ms))

    def advance_to_step(self, step: int) -> None:
        target_ms = float(int(step)) * self.dt_ms
        delta = target_ms - self.current_time_ms()
        if delta < -1e-9:
            raise RuntimeError(
                f"cannot advance backward in time: current={self.current_time_ms():.9g} ms, "
                f"target={target_ms:.9g} ms"
            )
        self.runner.advance(max(0.0, delta))

    def advance_to_time(self, time_ms: float) -> None:
        delta = float(time_ms) - self.current_time_ms()
        if delta < -1e-9:
            raise RuntimeError(
                f"cannot advance backward in time: current={self.current_time_ms():.9g} ms, "
                f"target={float(time_ms):.9g} ms"
            )
        self.runner.advance(max(0.0, delta))

    def segments(self, start_step: int, stop_step: int, *, count: int = 1) -> list[VJPSegment]:
        """Split [start_step, stop_step) into `count` contiguous VJP segments."""
        start = int(start_step)
        stop = int(stop_step)
        nseg = int(count)
        if nseg <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if stop <= start:
            raise ValueError(f"stop_step must be greater than start_step, got {start}->{stop}")
        edges = np.linspace(start, stop, nseg + 1, dtype=np.int64)
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"count={count} is too large for step range {start}->{stop}")
        return [
            VJPSegment(index=i, start_step=int(edges[i]), stop_step=int(edges[i + 1]), dt_ms=self.dt_ms)
            for i in range(nseg)
        ]

    def forward(self, segment: VJPSegment) -> VJPForwardResult:
        """Advance to `segment.start_step`, run forward, and return monitor data."""
        self.advance_to_step(segment.start_step)
        window = self.runner.run_window(
            group_index=self._group_index,
            local_grad_handles=self.voltage_grad_handles.tolist(),
            step_count=segment.step_count,
            current_vjp_mechs=self.vjp_mechanisms,
            duration_ms=segment.duration_ms,
            monitor_groups=self.monitor_groups,
        )
        return VJPForwardResult(segment=segment, window=window)

    def backward_voltage_grads(self, dldv_forward: Any) -> None:
        """Run VJP from forward-time dL/dV for the most recent forward segment."""
        self.runner.backward_voltage_grads(dldv_forward)

    def step(self, segment: VJPSegment, loss_grad_fn: LossGradFn) -> VJPStepResult:
        """Run one segment forward, call `loss_grad_fn`, then run VJP backward.

        `loss_grad_fn` receives a `VJPForwardResult` and returns either:
        - `VJPLossResult(loss, dldv, metrics)`
        - `(loss, dldv)`
        - `(loss, dldv, metrics)`
        """
        forward = self.forward(segment)
        loss_result = loss_grad_fn(forward)
        loss, dldv, metrics = self._normalize_loss_result(loss_result)
        self.backward_voltage_grads(dldv)
        return VJPStepResult(forward=forward, loss=loss, metrics=metrics)

    @staticmethod
    def _normalize_loss_result(
        result: VJPLossResult | tuple[float, Any] | tuple[float, Any, Mapping[str, float]],
    ) -> tuple[float, Any, Mapping[str, float]]:
        if isinstance(result, VJPLossResult):
            return float(result.loss), result.dldv, dict(result.metrics or {})
        if not isinstance(result, tuple):
            raise TypeError(
                "loss_grad_fn must return VJPLossResult, (loss, dldv), "
                "or (loss, dldv, metrics)"
            )
        if len(result) == 2:
            loss, dldv = result
            return float(loss), dldv, {}
        if len(result) == 3:
            loss, dldv, metrics = result
            return float(loss), dldv, dict(metrics)
        raise TypeError("loss_grad_fn tuple result must have length 2 or 3")
