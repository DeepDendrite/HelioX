from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Any

from .bundle import ExportBundle


@dataclass(frozen=True)
class CompileConfig:
    """Compilation/export configuration for NEURON -> NEURONG bundle."""

    dt: float
    v_init: float
    device: str = "gpu"  # "cpu"|"gpu" (passed to NeurongManager)
    permute_type: int = 3
    overwrite: bool = False


def compile_from_neuron(
    *,
    export_dir: str,
    build_fn: Callable[[], None],
    cfg: CompileConfig,
    register_wrappers_fn: Callable[[Any], None] | None = None,
    manager: Any | None = None,
) -> ExportBundle:
    """Build a model in NEURON and export an NEURONG-consumable bundle.

    This function is intentionally small and opinionated:
    - NEURON is the modeling frontend.
    - Export must happen after finitialize; do not parse/capture `_ref_* row=...`
      before export (row can change at finitialize).
    - The exported directory is the *artifact* (bundle) consumed by the backend.

    Parameters
    ----------
    export_dir:
        Output directory for the bundle.
    build_fn:
        User function that constructs the NEURON model (sections, cells, netcons...).
    cfg:
        Export/runtime config (dt, v_init, device, permute_type).
    register_wrappers_fn:
        Optional hook to register wrappers/handles on the NeurongManager before export.
        If omitted, `neurong_metadata.json` may be empty; that's fine for pure runtime use.
    manager:
        Optional `NeurongManager` instance. If omitted, a new one is created.

    Returns
    -------
    ExportBundle
    """
    export_dir = os.path.abspath(export_dir)
    if os.path.exists(export_dir) and os.listdir(export_dir):
        if not cfg.overwrite:
            raise FileExistsError(
                f"export_dir is not empty: {export_dir}. "
                "Set CompileConfig(overwrite=True) or choose a new directory."
            )

    os.makedirs(export_dir, exist_ok=True)

    # Import lazily: NEURON is only required at compile time.
    from neurong_wrapper import NeurongManager  # type: ignore

    if manager is None:
        manager = NeurongManager()

    manager.set_device(cfg.device)
    manager.set_permute_type(int(cfg.permute_type))

    # 1) Build NEURON model
    t0 = time.time()
    build_fn()

    # 2) (Optional) register wrappers/handles for metadata extraction
    if register_wrappers_fn is not None:
        register_wrappers_fn(manager)

    # 3) Export using the built-in enhanced exporter (includes finitialize + metadata + load_model).
    manager.enhanced_export_model(export_dir, dt=float(cfg.dt), v_init=float(cfg.v_init))

    bundle = ExportBundle(export_dir=export_dir)
    bundle.write_bundle_meta(
        {
            "version": "0.1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "compile_seconds": time.time() - t0,
            "compile_config": {
                "dt": float(cfg.dt),
                "v_init": float(cfg.v_init),
                "device": str(cfg.device),
                "permute_type": int(cfg.permute_type),
            },
        }
    )
    return bundle

