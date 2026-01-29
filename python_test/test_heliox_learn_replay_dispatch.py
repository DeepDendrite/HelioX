from __future__ import annotations

import os
import sys

import numpy as np


def _ensure_python_lib_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_lib = os.path.join(repo_root, "python_lib")
    if python_lib not in sys.path:
        sys.path.insert(0, python_lib)


class _DummyClient:
    def __init__(self) -> None:
        self.called = []

    def replay_compute_dw_from_cached_signals_into(
        self,
        dLtdv_lr_ot,
        poutput,
        pre_of_col,
        dw_out_n,
        dt_ms,
        percise,
        grad_scale,
        eps,
        grad_l2norm_threshold,
        clip_strategy,
        clip_check_every,
    ) -> int:
        self.called.append("dw")
        dw_out_n[:] = 1.0
        return 0

    def replay_compute_dw_dx_from_cached_signals_into(
        self,
        dLtdv_lr_ot,
        poutput,
        pinput,
        pre_of_col,
        dw_out_n,
        dx_lr_it,
        dt_ms,
        percise,
        grad_scale,
        eps,
        grad_l2norm_threshold,
        clip_strategy,
        clip_check_every,
    ) -> int:
        self.called.append("dw_dx")
        dw_out_n[:] = 2.0
        dx_lr_it[:] = 3.0
        return 0


class _DummyManager:
    def __init__(self, client) -> None:
        self.client = client


def main() -> None:
    _ensure_python_lib_on_path()

    import heliox_learn as nl

    client = _DummyClient()
    mgr = _DummyManager(client)

    dLtdv = np.zeros((10, 4), dtype=np.float32)
    poutput = np.arange(4, dtype=np.int32)
    pre_of_col = np.arange(7, dtype=np.int32)

    grads = nl.replay_grads_from_cached_signals(
        mgr, dLtdv_lr_ot=dLtdv, poutput=poutput, pre_of_col=pre_of_col, dt_ms=0.1, percise=False, pinput=None
    )
    assert client.called == ["dw"]
    assert grads.dx_lr_it is None
    assert grads.dw_out_n.shape == (7,)
    assert float(grads.dw_out_n.max()) == 1.0

    client.called.clear()
    pinput = np.arange(3, dtype=np.int32)
    grads2 = nl.replay_grads_from_cached_signals(
        mgr, dLtdv_lr_ot=dLtdv, poutput=poutput, pre_of_col=pre_of_col, dt_ms=0.1, percise=False, pinput=pinput
    )
    assert client.called == ["dw_dx"]
    assert grads2.dx_lr_it is not None
    assert grads2.dw_out_n.shape == (7,)
    assert grads2.dx_lr_it.shape == (3, 10)
    assert float(grads2.dw_out_n.max()) == 2.0
    assert float(grads2.dx_lr_it.max()) == 3.0


if __name__ == "__main__":
    main()

