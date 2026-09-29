from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MappedSignal:
    """A mapped learning signal channel captured from runtime handles.

    This represents the common "gather + scatter-add" pattern:
    - Gather scalar values from `handles` (runtime variable handles)
    - Scatter-add them into a dense column space of size N using `dest` and `scale`

    `dest` and `scale` are aligned 1:1 with `handles`.
    """

    name: str
    handles: Sequence[int]
    dest: np.ndarray  # int32, shape (len(handles),)
    scale: np.ndarray  # float32, shape (len(handles),)

    def __post_init__(self) -> None:
        # Basic shape sanity; deeper validation is runtime/backend responsibility.
        if self.dest.ndim != 1 or self.scale.ndim != 1:
            raise ValueError(f"MappedSignal({self.name}): dest/scale must be 1D arrays")
        if int(self.dest.shape[0]) != len(self.handles) or int(self.scale.shape[0]) != len(self.handles):
            raise ValueError(
                f"MappedSignal({self.name}): size mismatch: "
                f"handles={len(self.handles)} dest={self.dest.shape[0]} scale={self.scale.shape[0]}"
            )
        if self.dest.dtype != np.int32:
            raise TypeError(f"MappedSignal({self.name}): dest must be int32 (got {self.dest.dtype})")
        if self.scale.dtype != np.float32:
            raise TypeError(f"MappedSignal({self.name}): scale must be float32 (got {self.scale.dtype})")


def empty_signal(name: str) -> MappedSignal:
    """Create an empty mapped signal channel."""

    return MappedSignal(
        name=name,
        handles=(),
        dest=np.zeros((0,), dtype=np.int32),
        scale=np.zeros((0,), dtype=np.float32),
    )


@dataclass(frozen=True)
class CaptureSpec:
    """Spec for a single forward pass + learning-signal capture.

    This is intentionally backend-oriented (uses resolved runtime handles).
    Higher-level tasks can build this from wrapper metadata later.

    Notes
    -----
    Current NEURONG backend requires:
    - `output_v_handles` non-empty
    - `signals['pure_i']` to be present and non-empty (used to infer N)
    """

    output_v_handles: Sequence[int]
    signals: Mapping[str, MappedSignal]
    tstop_ms: float
    k_mul: int
    percise: bool
    v_init: float


@dataclass(frozen=True)
class CapturedPack:
    """Capture result (CPU-visible output + GPU-cached learning signals)."""

    output_vs_tn: np.ndarray  # float32, shape (total_steps+1, n_output)
    k_mul: int
    percise: bool

    @property
    def total_steps(self) -> int:
        return int(self.output_vs_tn.shape[0]) - 1

    @property
    def n_output(self) -> int:
        return int(self.output_vs_tn.shape[1])

    @property
    def ksteps_total(self) -> int:
        return int(self.total_steps // int(self.k_mul))


@dataclass(frozen=True)
class ReplayGrads:
    """Replay result (gradients with respect to weights and optionally inputs)."""

    dw_out_n: np.ndarray  # float32, shape (N,)
    dx_lr_it: np.ndarray | None  # float32, shape (n_input, ksteps_total) if present


def replay_grads_from_cached_signals(
    manager,
    *,
    dLtdv_lr_ot: np.ndarray,
    poutput: np.ndarray,
    pre_of_col: np.ndarray,
    dt_ms: float,
    percise: bool,
    pinput: np.ndarray | None = None,
    grad_scale: float = 1.0,
    eps: float = 1e-6,
    grad_l2norm_threshold: float = 1e6,
    clip_strategy: int = 1,
    clip_check_every: int = 1,
) -> ReplayGrads:
    """Replay gradients from cached signals.

    This is the unified wrapper for the two backend entrypoints:
    - dw-only (when pinput is None or empty)
    - dw+dx (when pinput is provided and non-empty)
    """

    client = getattr(manager, "client", manager)

    dLtdv_lr_ot = np.asarray(dLtdv_lr_ot, dtype=np.float32, order="C")
    poutput = np.asarray(poutput, dtype=np.int32, order="C")
    pre_of_col = np.asarray(pre_of_col, dtype=np.int32, order="C")

    if dLtdv_lr_ot.ndim != 2:
        raise ValueError("replay_grads_from_cached_signals: dLtdv_lr_ot must be 2D")
    if poutput.ndim != 1 or pre_of_col.ndim != 1:
        raise ValueError("replay_grads_from_cached_signals: poutput/pre_of_col must be 1D")

    ksteps_total = int(dLtdv_lr_ot.shape[0])
    N = int(pre_of_col.shape[0])
    dw_out_n = np.zeros((N,), dtype=np.float32, order="C")

    if pinput is None:
        n_input = 0
    else:
        pinput = np.asarray(pinput, dtype=np.int32, order="C")
        if pinput.ndim != 1:
            raise ValueError("replay_grads_from_cached_signals: pinput must be 1D")
        n_input = int(pinput.shape[0])

    if n_input <= 0:
        rc = client.replay_compute_dw_from_cached_signals_into(
            dLtdv_lr_ot,
            poutput,
            pre_of_col,
            dw_out_n,
            float(dt_ms),
            bool(percise),
            float(grad_scale),
            float(eps),
            float(grad_l2norm_threshold),
            int(clip_strategy),
            int(clip_check_every),
        )
        if rc != 0:
            raise RuntimeError(f"replay_compute_dw_from_cached_signals_into failed (rc={rc})")
        return ReplayGrads(dw_out_n=dw_out_n, dx_lr_it=None)

    dx_lr_it = np.zeros((n_input, ksteps_total), dtype=np.float32, order="C")
    rc = client.replay_compute_dw_dx_from_cached_signals_into(
        dLtdv_lr_ot,
        poutput,
        pinput,
        pre_of_col,
        dw_out_n,
        dx_lr_it,
        float(dt_ms),
        bool(percise),
        float(grad_scale),
        float(eps),
        float(grad_l2norm_threshold),
        int(clip_strategy),
        int(clip_check_every),
    )
    if rc != 0:
        raise RuntimeError(f"replay_compute_dw_dx_from_cached_signals_into failed (rc={rc})")
    return ReplayGrads(dw_out_n=dw_out_n, dx_lr_it=dx_lr_it)


def replay_dw_from_cached_signals(
    manager,
    *,
    dLtdv_lr_ot: np.ndarray,
    poutput: np.ndarray,
    pre_of_col: np.ndarray,
    dt_ms: float,
    percise: bool,
    grad_scale: float = 1.0,
    eps: float = 1e-6,
    grad_l2norm_threshold: float = 1e6,
    clip_strategy: int = 1,
    clip_check_every: int = 1,
) -> np.ndarray:
    """Replay dv/dw (cached signals) and return only dw.

    Kept as a convenience wrapper for "no input training" tasks.
    """

    grads = replay_grads_from_cached_signals(
        manager,
        dLtdv_lr_ot=dLtdv_lr_ot,
        poutput=poutput,
        pre_of_col=pre_of_col,
        dt_ms=dt_ms,
        percise=percise,
        pinput=None,
        grad_scale=grad_scale,
        eps=eps,
        grad_l2norm_threshold=grad_l2norm_threshold,
        clip_strategy=clip_strategy,
        clip_check_every=clip_check_every,
    )
    return grads.dw_out_n


def capture_signals_cached(manager, spec: CaptureSpec, *, total_steps: int, dtype=np.float32) -> CapturedPack:
    """Run forward simulation and capture mapped signals into backend cache.

    Parameters
    ----------
    manager:
        NeurongManager (or anything exposing `.client` as the nanobind Sim wrapper).
    spec:
        Capture spec (resolved numeric handles and mapping).
    total_steps:
        Number of dt steps for the output buffer (output_vs has total_steps+1 rows).

    Returns
    -------
    CapturedPack
    """

    client = getattr(manager, "client", manager)

    n_output = len(spec.output_v_handles)
    if n_output <= 0:
        raise ValueError("capture_signals_cached: output_v_handles is empty")
    if total_steps <= 0:
        raise ValueError("capture_signals_cached: total_steps must be positive")
    if int(spec.k_mul) <= 0:
        raise ValueError("capture_signals_cached: k_mul must be positive")

    # The backend fills every element, so avoid the extra memset cost of np.zeros.
    output_vs_tn = np.empty((int(total_steps) + 1, int(n_output)), dtype=dtype, order="C")

    pure_i = spec.signals.get("pure_i") or empty_signal("pure_i")
    didv = spec.signals.get("didv") or empty_signal("didv")
    didvpre = spec.signals.get("didvpre") or empty_signal("didvpre")

    rc = client.simulate_and_capture_mapped_signals_cached(
        output_vs_tn,
        list(int(h) for h in spec.output_v_handles),
        list(int(h) for h in pure_i.handles),
        pure_i.dest,
        pure_i.scale,
        list(int(h) for h in didv.handles),
        didv.dest,
        didv.scale,
        list(int(h) for h in didvpre.handles),
        didvpre.dest,
        didvpre.scale,
        float(spec.tstop_ms),
        int(spec.k_mul),
        bool(spec.percise),
        float(spec.v_init),
    )
    if rc != 0:
        raise RuntimeError(f"simulate_and_capture_mapped_signals_cached failed (rc={rc})")

    return CapturedPack(output_vs_tn=output_vs_tn, k_mul=int(spec.k_mul), percise=bool(spec.percise))
