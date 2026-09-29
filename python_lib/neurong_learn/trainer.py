from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

from .checkpoint import CheckpointConfig, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str
    resume: bool = False
    checkpoint_name: str = "ckpt"
    save_every: int = 1


class Trainer:
    """Generic training loop orchestration (minimal skeleton).

    This intentionally keeps logic small for the first milestone:
    - Task implements epoch behavior (forward/loss/update).
    - Trainer handles output_dir creation + basic timing + metric logging.
    """

    def __init__(self, *, runtime: Any, cfg: TrainerConfig):
        self.runtime = runtime
        self.cfg = cfg

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._ckpt = CheckpointConfig(output_dir=self.cfg.output_dir, name=self.cfg.checkpoint_name)

    def fit(self, task: Any, *, epochs: int) -> Dict[str, Any]:
        if not getattr(self.runtime, "loaded", False):
            self.runtime.load()

        start_epoch = 0
        if self.cfg.resume and os.path.exists(self._ckpt.path()):
            ckpt = load_checkpoint(self._ckpt.path())
            start_epoch = int(ckpt.get("start_epoch", 0))
            state = ckpt.get("state")
            history = list(ckpt.get("history", []))
        else:
            state = task.setup(self.runtime)
            history: list[dict[str, float]] = []

        for epoch in range(int(start_epoch), int(epochs)):
            t0 = time.time()
            state, metrics = task.run_epoch(self.runtime, state, epoch)
            metrics = dict(metrics)
            metrics.setdefault("epoch", float(epoch))
            metrics["epoch_seconds"] = time.time() - t0
            history.append(metrics)

            if self.cfg.save_every > 0 and ((epoch + 1) % int(self.cfg.save_every) == 0):
                save_checkpoint(
                    self._ckpt.path(),
                    {
                        "start_epoch": epoch + 1,
                        "history": history,
                        "state": state,
                        "task": getattr(task, "spec", None),
                    },
                )

        return {"history": history, "state": state}
