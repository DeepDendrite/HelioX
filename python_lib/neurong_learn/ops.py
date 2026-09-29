from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LearningOps:
    """Learning-side ops facade over a loaded runtime manager/client.

    This keeps training-oriented entrypoints in `neurong_learn` while reusing
    backend kernels exposed by the low-level `neurong.Sim` client.
    """

    manager: Any
    sim_client: Any

    @classmethod
    def from_manager(cls, manager: Any) -> "LearningOps":
        sim_client = getattr(manager, "client", None) or getattr(manager, "_client", None)
        if sim_client is None:
            raise RuntimeError("LearningOps requires manager.client")
        return cls(manager=manager, sim_client=sim_client)

    def _require(self, names: Sequence[str]) -> None:
        missing = [n for n in names if not hasattr(self.sim_client, n)]
        if missing:
            raise RuntimeError("learning ops missing client APIs: " + ", ".join(missing))

    def get_learnable_grad_handles(self, variable_handles: Sequence[int]) -> np.ndarray:
        if not variable_handles:
            return np.zeros((0,), dtype=np.int32)
        self._require(("get_learnable_grad_handles_from_variable_handles",))
        out = self.sim_client.get_learnable_grad_handles_from_variable_handles(
            [int(h) for h in variable_handles]
        )
        return np.asarray(out, dtype=np.int32)

    def zero_learnable_grads(self) -> None:
        self._require(("zero_learnable_grads_cpu",))
        self.sim_client.zero_learnable_grads_cpu()

    def get_values_by_handles_f32(self, handles: Sequence[int]) -> np.ndarray:
        if not handles:
            return np.zeros((0,), dtype=np.float32)
        handle_list = [int(h) for h in handles]
        if hasattr(self.sim_client, "get_variables_by_handles_f32_into"):
            out = np.empty((len(handle_list),), dtype=np.float32)
            self.sim_client.get_variables_by_handles_f32_into(handle_list, out)
            return out
        vals = self.sim_client.get_variables_by_handles(handle_list)
        return np.asarray(vals, dtype=np.float32)

    def compute_spk_readout_loss_and_grads(
        self,
        spk_bct,
        target_bot,
        w_oc,
        b_o,
        *,
        k_out: float,
        dt: float,
    ):
        """Compute readout MSE loss and gradients for leaky readout."""
        spk = np.asarray(spk_bct, dtype=np.float32, order="C")
        target = np.asarray(target_bot, dtype=np.float32, order="C")
        w = np.asarray(w_oc, dtype=np.float32, order="C")
        b = np.asarray(b_o, dtype=np.float32, order="C")
        if spk.ndim != 3 or target.ndim != 3:
            raise ValueError(f"spk/target must be 3D, got {spk.shape} / {target.shape}")
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError(f"w/b must be 2D/1D, got {w.shape} / {b.shape}")
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        n_batch, n_cell, t_len = spk.shape
        if target.shape[0] != n_batch or target.shape[2] != t_len:
            raise ValueError(f"target shape mismatch: {target.shape} vs ({n_batch}, n_out, {t_len})")
        if w.shape[1] != n_cell or w.shape[0] != target.shape[1]:
            raise ValueError(f"w shape mismatch: {w.shape}, expected ({target.shape[1]}, {n_cell})")
        if b.shape[0] != target.shape[1]:
            raise ValueError(f"b shape mismatch: {b.shape}, expected ({target.shape[1]},)")

        pred = np.empty_like(target, dtype=np.float32, order="C")
        dldspk = np.zeros_like(spk, dtype=np.float32, order="C")
        dw = np.zeros_like(w, dtype=np.float32, order="C")
        db = np.zeros_like(b, dtype=np.float32, order="C")

        if hasattr(self.sim_client, "compute_spk_readout_loss_and_grads_into"):
            loss = float(
                self.sim_client.compute_spk_readout_loss_and_grads_into(
                    spk, target, w, b, float(k_out), float(dt), pred, dldspk, dw, db
                )
            )
            if loss < 0.0:
                raise RuntimeError("compute_spk_readout_loss_and_grads_into failed")
            return loss, pred, dldspk, dw, db

        # Numpy fallback (kept here so learning users can stay in neurong_learn API).
        spk64 = spk.astype(np.float64, copy=False)
        target64 = target.astype(np.float64, copy=False)
        w64 = w.astype(np.float64, copy=False)
        b64 = b.astype(np.float64, copy=False)
        n_out = w64.shape[0]
        alpha = float((1.0 - float(k_out)) / float(dt))

        pred64 = np.zeros((n_batch, n_out, t_len), dtype=np.float64)
        z0 = spk64[:, :, 0] @ w64.T + b64[None, :]
        pred64[:, :, 0] = alpha * z0
        for t in range(1, t_len):
            zt = spk64[:, :, t] @ w64.T + b64[None, :]
            pred64[:, :, t] = float(k_out) * pred64[:, :, t - 1] + alpha * zt

        err = pred64 - target64
        loss = float(0.5 * np.sum(err * err))

        grad_pred = np.zeros_like(pred64)
        grad_next = np.zeros((n_batch, n_out), dtype=np.float64)
        for t in range(t_len - 1, -1, -1):
            grad_cur = err[:, :, t] + float(k_out) * grad_next
            grad_pred[:, :, t] = grad_cur
            grad_next = grad_cur

        grad_z = alpha * grad_pred
        dw64 = np.einsum("bot,bct->oc", grad_z, spk64, optimize=True)
        db64 = np.sum(grad_z, axis=(0, 2))
        dldspk64 = np.einsum("bot,oc->bct", grad_z, w64, optimize=True)

        return (
            loss,
            pred64.astype(np.float32, copy=False),
            dldspk64.astype(np.float32, copy=False),
            dw64.astype(np.float32, copy=False),
            db64.astype(np.float32, copy=False),
        )

    def create_optimizer(self, optimizer_type: str = "sgd") -> int:
        self._require(("create_optimizer",))
        return int(self.sim_client.create_optimizer(str(optimizer_type)))

    def optimizer_add_param_batch(
        self,
        optimizer_id: int,
        weight_handles: Sequence[int],
        grad_handles: Sequence[int],
        *,
        impedance: float,
    ) -> int:
        if len(weight_handles) != len(grad_handles):
            raise ValueError("weight_handles and grad_handles must have the same length")
        if not weight_handles:
            raise ValueError("optimizer_add_param_batch requires at least one handle")
        self._require(("optimizer_add_param_batch",))
        return int(
            self.sim_client.optimizer_add_param_batch(
                int(optimizer_id),
                [int(h) for h in weight_handles],
                [int(h) for h in grad_handles],
                float(impedance),
            )
        )

    def configure_optimizer(
        self,
        optimizer_id: int,
        *,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> int:
        self._require(("configure_optimizer",))
        return int(
            self.sim_client.configure_optimizer(
                int(optimizer_id),
                float(momentum),
                float(beta1),
                float(beta2),
                float(epsilon),
            )
        )

    def optimizer_step(self, optimizer_id: int, *, learning_rate: float, record_time: float, dt: float) -> int:
        self._require(("optimizer_step",))
        return int(
            self.sim_client.optimizer_step(
                int(optimizer_id),
                float(learning_rate),
                float(record_time),
                float(dt),
            )
        )

