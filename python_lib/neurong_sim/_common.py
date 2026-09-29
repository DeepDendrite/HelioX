"""Internal shared utilities for neurong_sim (sim/runtime wrappers)."""

import os
import sys

# `neurong` is a local pybind module built into the repo `build/` directory.
# When running from a dev checkout (e.g. repo-local tests), it may not be installed
# into the active Python environment yet.
try:
    import neurong  # type: ignore
except ModuleNotFoundError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _BUILD = os.path.abspath(os.path.join(_HERE, "..", "..", "build"))
    if _BUILD not in sys.path:
        sys.path.insert(0, _BUILD)
    import neurong  # type: ignore
import re
import warnings
import atexit
import numpy as np
# 全局开关：控制包装器析构时是否显示警告
_WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = True

def _disable_wrapper_warnings():
    """禁用包装器析构警告（程序退出时调用）"""
    global _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED
    _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = False


def _try_parse_row_from_ref(ref):
    """
    从 NEURON 的 `_ref_*` 句柄字符串中尝试解析 `row=<int>`。

    说明：
    - 在 CoreNEURON/NEURONG 的语义里，很多“实例索引”与 NEURON 的 `node_index()` 并不总一致。
    - 对 segment/compartment 变量（如 v），NEURON 会在 data_handle 的字符串里带 `row=...`。
    - 本函数只做“尽力而为”的解析，失败时返回 None，由上层决定回退策略。
    """
    try:
        match = re.search(r"row=(\d+)", str(ref))
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

# 注册退出时的清理函数
atexit.register(_disable_wrapper_warnings)

basic_mech_func = ["baseattr", "get_loc", "has_loc", "hname", "hocobjptr", "loc", "same"]
