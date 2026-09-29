from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LearningBackend:
    """Thin learning-only bridge on top of the low-level neurong.Sim client.

    `neurong_sim` remains simulation-focused; learning orchestration calls are
    collected here to keep a clean boundary.
    """

    sim_client: Any

    @classmethod
    def from_manager(cls, manager: Any) -> "LearningBackend":
        sim_client = getattr(manager, "client", None) or getattr(manager, "_client", None)
        if sim_client is None:
            raise RuntimeError("LearningBackend requires manager.client")
        return cls(sim_client=sim_client)

    def require_methods(self, names: Sequence[str]) -> None:
        missing = [n for n in names if not hasattr(self.sim_client, n)]
        if missing:
            raise RuntimeError("learning backend missing client APIs: " + ", ".join(missing))

    def prepare_vjp_window(
        self,
        *,
        group_index: int,
        local_grad_handles: Sequence[int],
        step_count: int,
        current_vjp_mechs: Sequence[str],
    ) -> None:
        self.require_methods(("prepare_vjp_from_handles",))
        self.sim_client.prepare_vjp_from_handles(
            int(group_index),
            [int(h) for h in local_grad_handles],
            int(step_count),
            [str(m) for m in current_vjp_mechs],
        )

    def set_spike_vjp_surrogate(self, kind: str, width_mv: float) -> None:
        self.require_methods(("set_spike_vjp_surrogate",))
        self.sim_client.set_spike_vjp_surrogate(str(kind), float(width_mv))

    def run_backward_from_forward_grads(self, dldv_forward: Any) -> None:
        grads = np.ascontiguousarray(dldv_forward, dtype=np.float64)
        if hasattr(self.sim_client, "run_vjp_from_forward_grads"):
            self.sim_client.run_vjp_from_forward_grads(grads)
            return

        self.require_methods(("run_vjp",))
        backward = grads[..., ::-1]
        flat = backward.reshape(-1, backward.shape[-1])
        self.sim_client.run_vjp(np.ascontiguousarray(flat.T, dtype=np.float64).tolist())
