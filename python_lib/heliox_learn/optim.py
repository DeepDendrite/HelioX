from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class AdamState:
    """Minimal Adam state container for numpy arrays.

    This mirrors a common legacy checkpoint layout (m, v, beta_t) so it can be
    serialized in checkpoints and recovered deterministically.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-9
    m: Any = 0.0
    v: Any = 0.0
    beta1_t: float = 1.0
    beta2_t: float = 1.0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the bias-corrected Adam update direction for `grad`."""
        grad = np.asarray(grad)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

        m_hat = self.m / (1.0 - self.beta1_t)
        v_hat = self.v / (1.0 - self.beta2_t)
        return m_hat / (np.sqrt(v_hat) + self.eps)

    def to_legacy(self) -> Tuple[Any, Any, float, float]:
        """Export state as (m, v, beta1_t, beta2_t) for legacy checkpoints."""
        return self.m, self.v, float(self.beta1_t), float(self.beta2_t)

    @classmethod
    def from_legacy(
        cls,
        *,
        beta1: float,
        beta2: float,
        eps: float,
        m: Any,
        v: Any,
        beta1_t: float,
        beta2_t: float,
    ) -> "AdamState":
        st = cls(beta1=float(beta1), beta2=float(beta2), eps=float(eps))
        st.m = m
        st.v = v
        st.beta1_t = float(beta1_t)
        st.beta2_t = float(beta2_t)
        return st

    def state_dict(self) -> Dict[str, Any]:
        return {
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "m": self.m,
            "v": self.v,
            "beta1_t": float(self.beta1_t),
            "beta2_t": float(self.beta2_t),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.beta1 = float(d.get("beta1", self.beta1))
        self.beta2 = float(d.get("beta2", self.beta2))
        self.eps = float(d.get("eps", self.eps))
        self.m = d.get("m", self.m)
        self.v = d.get("v", self.v)
        self.beta1_t = float(d.get("beta1_t", self.beta1_t))
        self.beta2_t = float(d.get("beta2_t", self.beta2_t))
