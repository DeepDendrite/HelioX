from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import LearningBackend


@dataclass(frozen=True)
class SegmentSpec:
    """A local-time learning segment.

    NEURONG keeps one monotonic backend time. Segment-local times are shifted to
    backend time only at the API boundary.
    """

    duration_ms: float
    learn_start_ms: float = 0.0
    learn_stop_ms: float | None = None

    @property
    def learn_duration_ms(self) -> float:
        stop = self.duration_ms if self.learn_stop_ms is None else float(self.learn_stop_ms)
        return stop - float(self.learn_start_ms)


@dataclass(frozen=True)
class SegmentResult:
    global_start_ms: float
    global_stop_ms: float
    recorder_cursor: int


@dataclass(frozen=True)
class SegmentWindow:
    global_start_ms: float
    global_stop_ms: float
    recorder_cursor: int
    step_count: int
    monitors: dict[str, np.ndarray]


@dataclass(frozen=True)
class MonitorHandleGroup:
    kind: str
    handles: np.ndarray


class SegmentRunner:
    """Small coordinator for online VJP segments.

    This class owns no model state. It only keeps the local/global time mapping,
    monitor cursors, and the prepare/continue ordering for one VJP window.
    """

    def __init__(
        self,
        manager: Any,
        *,
        sim_client: Any | None = None,
        backend: LearningBackend | None = None,
        dt_ms: float | None = None,
    ):
        self.manager = manager
        self.sim_client = sim_client or getattr(manager, "client", None) or getattr(manager, "_client", None)
        if backend is None and self.sim_client is None:
            raise RuntimeError("SegmentRunner requires a sim_client or a LearningBackend")
        self.backend = backend or LearningBackend(sim_client=self.sim_client)
        self.dt_ms = None if dt_ms is None else float(dt_ms)

    def global_t(self) -> float:
        return float(self.manager.get_t()) if hasattr(self.manager, "get_t") else 0.0

    def to_global(self, local_t_ms: float, *, segment_start_ms: float) -> float:
        return float(segment_start_ms) + float(local_t_ms)

    @staticmethod
    def _valid_monitor_handles(handles: Any) -> np.ndarray:
        if isinstance(handles, MonitorHandleGroup):
            handles = handles.handles
        flat = np.asarray(handles, dtype=np.int32).reshape(-1)
        return flat[flat >= 0]

    @staticmethod
    def normalize_monitor_group(handles: Any, name: str = "monitor_group") -> MonitorHandleGroup:
        if isinstance(handles, MonitorHandleGroup):
            return handles
        arr = np.asarray(handles, dtype=object).reshape(-1)
        kinds: list[str] = []
        out: list[int] = []
        for item in arr:
            kind = str(getattr(item, "handle_kind", "history"))
            if hasattr(item, "get_handle"):
                handle = int(item.get_handle())
            elif kind == "window" and hasattr(item, "window_observer_id"):
                handle = int(item.window_observer_id)
            elif hasattr(item, "monitor_id"):
                handle = int(item.monitor_id)
            else:
                handle = int(item)
            kinds.append(kind)
            out.append(handle)
        unique_kinds = set(kinds) if kinds else {"history"}
        if len(unique_kinds) != 1:
            raise ValueError(f"{name} mixes monitor handle kinds: {sorted(unique_kinds)}")
        kind = unique_kinds.pop()
        if kind not in {"history", "window"}:
            raise ValueError(f"{name} has unsupported monitor handle kind {kind!r}")
        return MonitorHandleGroup(kind=kind, handles=np.asarray(out, dtype=np.int32))

    def play_inputs(self, stim_ctrls: Sequence[Sequence[Any]], inputs: Sequence[Sequence[Any]], *, base_time_ms: float) -> None:
        base = float(base_time_ms)
        for ibatch, ctrls in enumerate(stim_ctrls):
            for i, ctrl in enumerate(ctrls):
                spikes = np.asarray(inputs[ibatch][i], dtype=np.float64)
                ctrl.play(spikes + base if base != 0.0 else spikes)

    def monitor_cursor(self, handles: Any, *, include_current_sample: bool = True) -> int:
        flat = self._valid_monitor_handles(handles)
        if flat.size == 0:
            return 0
        if hasattr(self.manager, "get_monitor_cursor"):
            cursor = int(self.manager.get_monitor_cursor(flat.tolist()))
        else:
            cursor = min(len(self.manager.get_monitor_data(int(h))) for h in flat.tolist())
        if include_current_sample and cursor > 0:
            return cursor - 1
        return cursor

    def read_monitors_f32(self, handles: Any, *, n_steps: int, start_step: int) -> np.ndarray:
        flat = self._valid_monitor_handles(handles)
        if flat.size == 0:
            return np.zeros((0, int(n_steps)), dtype=np.float32)
        return self.manager.get_multiple_monitor_data_f32(
            flat.tolist(),
            n_steps=int(n_steps),
            start_step=int(start_step),
        )

    def read_window_recorders_f32(self, handles: Any, *, n_steps: int) -> np.ndarray:
        flat = self._valid_monitor_handles(handles)
        if flat.size == 0:
            return np.zeros((0, int(n_steps)), dtype=np.float32)
        return self.manager.get_window_recording_f32(flat.tolist(), n_steps=int(n_steps))

    def validate_history_window_sampling(self) -> None:
        stride = int(getattr(self.manager, "_recorder_stride_steps", 1))
        if stride != 1:
            raise ValueError(
                "history monitor groups require recorder stride 1 for per-step training windows; "
                "use create_window_recorder() for segment-local loss traces when persistent recorder "
                f"stride is {stride}"
            )

    def advance(self, duration_ms: float) -> None:
        duration = float(duration_ms)
        if duration > 0.0:
            self.manager.continue_run(duration)

    def run_window(
        self,
        *,
        group_index: int,
        local_grad_handles: Sequence[int],
        step_count: int,
        current_vjp_mechs: Sequence[str],
        duration_ms: float,
        monitor_groups: Mapping[str, Any],
    ) -> SegmentWindow:
        if not monitor_groups:
            raise ValueError("monitor_groups must not be empty")

        groups = {
            str(name): self.normalize_monitor_group(handles, f"monitor_groups[{name!r}]")
            for name, handles in monitor_groups.items()
        }
        history_parts = [self._valid_monitor_handles(group) for group in groups.values() if group.kind == "history"]
        window_parts = [self._valid_monitor_handles(group) for group in groups.values() if group.kind == "window"]
        history_handles = np.concatenate(history_parts) if history_parts else np.zeros((0,), dtype=np.int32)
        window_handles = np.concatenate(window_parts) if window_parts else np.zeros((0,), dtype=np.int32)
        if history_handles.size:
            self.validate_history_window_sampling()
        cursor = self.monitor_cursor(history_handles, include_current_sample=True) if history_handles.size else 0
        start_ms = self.global_t()
        if window_handles.size:
            self.manager.begin_window_recording(window_handles.tolist(), int(step_count))
        try:
            self.run_vjp_window(
                group_index=group_index,
                local_grad_handles=local_grad_handles,
                step_count=step_count,
                current_vjp_mechs=current_vjp_mechs,
                duration_ms=duration_ms,
            )
            if history_handles.size > 0:
                recorded_steps = self.monitor_cursor(history_handles, include_current_sample=False)
                expected_steps = cursor + int(step_count)
                if recorded_steps < expected_steps:
                    raise RuntimeError(
                        "monitor recorder is shorter than requested segment window: "
                        f"cursor={cursor}, step_count={int(step_count)}, recorded_steps={recorded_steps}"
                    )
            monitors = {}
            for name, group in groups.items():
                if group.kind == "window":
                    monitors[str(name)] = self.read_window_recorders_f32(group, n_steps=int(step_count))
                else:
                    monitors[str(name)] = self.read_monitors_f32(group, n_steps=int(step_count), start_step=cursor)
        finally:
            if window_handles.size:
                self.manager.end_window_recording()
        return SegmentWindow(
            global_start_ms=start_ms,
            global_stop_ms=self.global_t(),
            recorder_cursor=cursor,
            step_count=int(step_count),
            monitors=monitors,
        )

    def backward_voltage_grads(self, dldv_forward: Any) -> None:
        self.backend.run_backward_from_forward_grads(dldv_forward)

    def run_vjp_window(
        self,
        *,
        group_index: int,
        local_grad_handles: Sequence[int],
        step_count: int,
        current_vjp_mechs: Sequence[str],
        duration_ms: float,
    ) -> None:
        self.backend.prepare_vjp_window(
            group_index=int(group_index),
            local_grad_handles=local_grad_handles,
            step_count=int(step_count),
            current_vjp_mechs=current_vjp_mechs,
        )
        self.advance(duration_ms)
