from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ExportBundle:
    """A NEURON-exported model bundle consumable by HELIOX.

    Bundle directory typically contains:
    - bbcore*.dat / globals.dat (NEURON export)
    - heliox_config.json (device/permute_type/dt/v_init)
    - heliox_metadata.json (wrapper metadata, optional)
    - heliox_bundle.json (framework-level metadata, optional)
    """

    export_dir: str

    def _path(self, name: str) -> str:
        return os.path.join(self.export_dir, name)

    def exists(self) -> bool:
        return os.path.isdir(self.export_dir)

    def require_exists(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"Export bundle directory not found: {self.export_dir}")

    def read_json_if_exists(self, name: str) -> Dict[str, Any] | None:
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def read_heliox_config(self) -> Dict[str, Any] | None:
        return self.read_json_if_exists("heliox_config.json")

    def read_heliox_metadata(self) -> Dict[str, Any] | None:
        return self.read_json_if_exists("heliox_metadata.json")

    def read_bundle_meta(self) -> Dict[str, Any] | None:
        return self.read_json_if_exists("heliox_bundle.json")

    def write_bundle_meta(self, meta: Dict[str, Any]) -> str:
        self.require_exists()
        path = self._path("heliox_bundle.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return path

