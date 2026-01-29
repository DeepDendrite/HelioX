from __future__ import annotations

import os
import sys
import tempfile


def _ensure_python_lib_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_lib = os.path.join(repo_root, "python_lib")
    if python_lib not in sys.path:
        sys.path.insert(0, python_lib)


def main() -> None:
    _ensure_python_lib_on_path()

    import heliox_learn

    # Import smoke: these should not require NEURON at import time.
    _ = heliox_learn.ExportBundle
    _ = heliox_learn.Runtime
    _ = heliox_learn.Trainer
    _ = heliox_learn.TaskSpec

    # CLI smoke: parse-only path.
    with tempfile.TemporaryDirectory(prefix="heliox_learn_smoke_") as d:
        code = heliox_learn.cli_main(["bundle-info", "--export-dir", d])
        assert code == 0


if __name__ == "__main__":
    main()
