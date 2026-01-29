from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str
    name: str = "ckpt"

    def path(self) -> str:
        return os.path.join(self.output_dir, f"{self.name}.npz")


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Store python objects via numpy (pickle). Keep this internal to the framework.
    np.savez(path, **{k: np.asarray(v, dtype=object) for k, v in state.items()})


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        # unwrap 0-d object arrays
        if getattr(v, "dtype", None) == object and getattr(v, "shape", None) == ():
            out[k] = v.item()
        else:
            out[k] = v
    return out

