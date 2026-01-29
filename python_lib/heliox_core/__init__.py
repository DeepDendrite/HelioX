"""
heliox_core
============

HELIOX 的“仿真通用”Python 层入口（不包含学习/训练逻辑）。

设计目标
--------
- simulation-only 用户也能直接使用：导出/加载、Obj/Recorder/VecPlay 等 wrapper。
- 应用仓库（例如某个 demo / 下游工程）不需要自己重复实现：
  - `pc.nrnbbcore_write(...)` 的导出时序
  - row/index 的获取与重排坑
  - `_ref_*` 到 handle 的映射与批量记录/播放

说明
----
当前实现先作为“门面（facade）”对外暴露稳定 import 路径：
    `from heliox_core import HelioXManager`

底层暂时复用历史模块 `heliox_wrapper.py`，以保持向后兼容：
    `from heliox_wrapper import HelioXManager`

后续如果需要把 `heliox_wrapper.py` 拆分成多个文件，可以在不影响用户代码的前提下，
逐步把实现迁移到 `heliox_core` 内部。
"""

from __future__ import annotations

# Facade re-exports (keep API stable for simulation-only users)
from heliox_wrapper import (  # noqa: F401
    HelioXManager,
    ObjWrapper,
    MonitorWrapper,
    RecorderWrapper,
    VecPlayWrapper,
)

from .binding import (  # noqa: F401
    extract_mech_name,
    get_variable_handle,
    parse_row_from_ref,
    resolve_mech_and_row,
)

from .batch import (  # noqa: F401
    HandleBatch,
    VecPlayBatch,
)

__all__ = [
    "HelioXManager",
    "ObjWrapper",
    "MonitorWrapper",
    "RecorderWrapper",
    "VecPlayWrapper",
    "HandleBatch",
    "VecPlayBatch",
    "extract_mech_name",
    "get_variable_handle",
    "parse_row_from_ref",
    "resolve_mech_and_row",
]
