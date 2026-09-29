from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bundle import ExportBundle
from .signals import CaptureSpec, CapturedPack, ReplayGrads, capture_signals_cached, replay_grads_from_cached_signals
from .ops import LearningOps


@dataclass(frozen=True)
class RuntimeConfig:
    # None means "unspecified by caller"; runtime will decide according to manager explicit flags.
    device: str | None = None
    permute_type: int | None = None


class Runtime:
    """NEURONG runtime bound to an exported bundle.

    This is the "backend runtime" side of the framework: it loads and runs the
    exported model without relying on NEURON after export.
    """

    def __init__(self, bundle: ExportBundle, *, cfg: RuntimeConfig | None = None, manager: Any | None = None):
        self.bundle = bundle
        self.cfg = cfg or RuntimeConfig()

        from neurong_sim import NeurongManager  # type: ignore

        self.manager = manager or NeurongManager()
        self.ops = LearningOps.from_manager(self.manager)
        self._configure_learning_runtime_backend()

        self._loaded = False
        self._dt_ms: float | None = None
        self._v_init: float | None = None

    def _configure_learning_runtime_backend(self) -> None:
        """Learning runtime policy: GPU-only, with auto defaults when user did not explicitly configure."""
        recommended_device = "gpu"
        recommended_permute = 3

        cfg_device = self.cfg.device
        cfg_permute = self.cfg.permute_type

        mgr_device_explicit = (
            self.manager.is_device_explicitly_set()
            if hasattr(self.manager, "is_device_explicitly_set")
            else False
        )
        mgr_permute_explicit = (
            self.manager.is_permute_type_explicitly_set()
            if hasattr(self.manager, "is_permute_type_explicitly_set")
            else False
        )

        if cfg_device is not None:
            normalized = str(cfg_device).lower()
            if normalized != "gpu":
                raise ValueError(
                    f"neurong_learn runtime only supports GPU, but RuntimeConfig.device={cfg_device!r}"
                )
            target_device = normalized
            device_explicit = True
        elif mgr_device_explicit:
            current_device = str(getattr(self.manager, "device", "")).lower()
            if current_device != "gpu":
                raise ValueError(
                    f"neurong_learn runtime only supports GPU, but manager device is explicitly set to {current_device!r}"
                )
            target_device = current_device
            device_explicit = False
        else:
            print("[neurong_learn] device not explicitly set; using GPU for learning runtime.")
            target_device = recommended_device
            device_explicit = False
        self.manager.set_device(target_device, explicit=device_explicit)

        if cfg_permute is not None:
            target_permute = int(cfg_permute)
            permute_explicit = True
        elif mgr_permute_explicit:
            target_permute = int(getattr(self.manager, "permute_type", recommended_permute))
            permute_explicit = False
        elif not mgr_permute_explicit:
            print(f"[neurong_learn] permute_type not explicitly set; using recommended value {recommended_permute}.")
            target_permute = recommended_permute
            permute_explicit = False
        self.manager.set_permute_type(target_permute, explicit=permute_explicit)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def dt_ms(self) -> float:
        if self._dt_ms is None:
            raise RuntimeError("Runtime.dt_ms is unknown; export bundle has no neurong_config.json dt")
        return float(self._dt_ms)

    @property
    def v_init(self) -> float | None:
        return self._v_init

    def load(self) -> None:
        self.bundle.require_exists()
        # Prefer load_from_export if metadata exists; it falls back automatically.
        self.manager.load_from_export(self.bundle.export_dir)
        current_device = str(getattr(self.manager, "device", "")).lower()
        if current_device != "gpu":
            raise RuntimeError(
                f"neurong_learn runtime only supports GPU; load_from_export configured device={current_device!r}"
            )

        cfg = self.bundle.read_neurong_config() or {}
        if "dt" in cfg:
            self._dt_ms = float(cfg["dt"])
        if "v_init" in cfg:
            self._v_init = float(cfg["v_init"])

        self._loaded = True

    def capture_signals(self, spec: CaptureSpec, *, total_steps: int) -> CapturedPack:
        """Forward simulate and cache learning signals on the backend."""
        if not self.loaded:
            self.load()
        return capture_signals_cached(self.manager, spec, total_steps=int(total_steps))

    def replay_grads(
        self,
        *,
        dLtdv_lr_ot,
        poutput,
        pre_of_col,
        percise: bool,
        pinput=None,
        dt_ms: float | None = None,
        grad_scale: float = 1.0,
        eps: float = 1e-6,
        grad_l2norm_threshold: float = 1e6,
        clip_strategy: int = 1,
        clip_check_every: int = 1,
    ) -> ReplayGrads:
        """Replay gradients from cached signals (unified dw-only / dw+dx).

        `dt_ms` defaults to the value stored in the export bundle config.
        """
        if not self.loaded:
            self.load()
        dt_use = float(self.dt_ms if dt_ms is None else dt_ms)
        return replay_grads_from_cached_signals(
            self.manager,
            dLtdv_lr_ot=dLtdv_lr_ot,
            poutput=poutput,
            pre_of_col=pre_of_col,
            dt_ms=dt_use,
            percise=bool(percise),
            pinput=pinput,
            grad_scale=float(grad_scale),
            eps=float(eps),
            grad_l2norm_threshold=float(grad_l2norm_threshold),
            clip_strategy=int(clip_strategy),
            clip_check_every=int(clip_check_every),
        )
