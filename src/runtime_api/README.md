# Runtime API（C++）设计说明（Phase 0）

本目录用于把 HELIOX 的“后端能力”从 `src/python_api/`（nanobind 绑定层）中拆出来，形成 **可复用的 C++ Runtime API**。

目标分两层：

1. **语言边界分层**
   - `src/runtime_api/`：纯 C++ API（不依赖 nanobind / Python 类型）
   - `src/python_api/`：仅做 Python 绑定与类型转换（ndarray/list → span/vector），不承载后端逻辑

2. **功能分层（你们说的 core vs learn）**
   - `src/runtime_api/core/`：仿真通用能力（仿真控制、变量读写、recording、输入/vecplay/batch、gap junction）
   - `src/runtime_api/learn/`：学习专用能力（optimizer、replay、grad clip、外部 grads、状态导入导出）

这份文档是 Phase 0：**定义 API 能力清单与归属表**，用于在大规模重构前先对齐边界与命名，避免“搬完才发现分类不符合预期”。

---

## 统一命名

对外（runtime_api）使用的主类名：

- `SimRuntime`：表示“可复用 runtime API 对象”，未来可以被：
  - Python binding（nanobind）调用
  - C++ CLI / benchmark 调用
  - RPC / 其他语言绑定调用

---

## API 归属表（按功能分组）

下表以当前 Python 暴露的 `heliox.Sim` 方法为参照（见 `src/python_api/PyBindings.cpp` / `src/python_api/SimWrapper.*`）。

### A. 仿真控制（core）

归属：`src/runtime_api/core/`

- `set_data_path(path)`
- `set_device(dev)`
- `set_permute_type(type)`
- `set_dt(dt)` / `get_dt()`
- `set_output_dir(dir)`
- `set_spike_output_enabled(bool)` / `is_spike_output_enabled()`
- `load_model()`
- `finitialize(v_init)`
- `run(tstop)` / `continue_run(runtime)` / `fadvance()`
- `get_t()`
- `flush_recorders()`

Python binding 层职责（`src/python_api`）：
- 只做参数类型转换与异常转换，不做额外流程编排。

### B. Recording / 变量读写（core）

归属：`src/runtime_api/core/`

- `add_monitor(mech, var, idx)` / `add_monitor_with_array(mech, var, idx, array_index)`
- `get_monitor_data(handle)` / `get_multiple_monitor_data(handles)`
- `set_variable_value(val, mech, var, idx)` / `set_variable_value_with_array(...)`
- `get_variable_value(mech, var, idx)` / `get_variable_value_with_array(...)`

Python binding 层职责：
- ndarray/list → span/vector 转换；不应包含 “monitor_to_handle” 等状态策略。

### C. 输入注入（core）

归属：`src/runtime_api/core/`

> 说明：输入（vecplay/netstim/vecstim/batch）不仅训练用，纯仿真也会使用，因此归 core。

- `register_netstim_batch(handle_triplets, ...)`
- `register_vecstim_batch(mech_indices, ...)`
- `set_input_batch_pixels(batch_id, pixels)`

### D. Gap Junction（core）

归属：`src/runtime_api/core/`

- `get_all_gap_junctions()`
- `get_gap_junction(...)`
- `add_gap_source(...)`
- `add_gap_target(...)`
- `clear_all_gap_junctions()`
- `get_next_available_sid()`

### E. Optimizer（learn）

归属：`src/runtime_api/learn/`

- `create_optimizer(optimizer_type)`
- `configure_optimizer(...)`
- `optimizer_add_param(...)` / `optimizer_add_param_batch(...)`
- `optimizer_step(...)` / `optimizer_step_with_inv_record_steps(...)`

外部梯度与状态（学习框架需要）：
- `optimizer_add_external_grads(...)`
- `optimizer_set_external_grads_f32(...)`
- `optimizer_clear_external_grads(...)`
- `optimizer_reset_state(...)`
- `optimizer_get_adam_state(...)`
- `optimizer_set_adam_state(...)`

Python binding 层职责：
- ndarray → span（例如 external grads），并把错误抛给 Python；不应在 binding 层维护 optimizer 的逻辑状态。

### F. Replay / LRDS 梯度（learn）

归属：`src/runtime_api/learn/`

- `simulate_and_replay_dw_dx(...)`
- `simulate_and_replay_dw_dx_streaming_into(...)`

Python binding 层职责：
- 只做 shape 检查与类型转换（ndarray → span/pointer），调用 learn API；不承载 ring buffer / clip 策略 / kernel 调度逻辑。

---

## Phase 1（接下来实施的范围）

Phase 1 目标是落地“骨架”与“类型收敛”，不改变现有行为：

1. 新增 `src/runtime_api/core/types.h` / `src/runtime_api/learn/replay_types.h` / `src/runtime_api/learn/optimizer_types.h`
2. 新增 `src/runtime_api/SimRuntime.h/.cpp`（先作为薄 wrapper，后续逐步把实现从 `SimWrapper` 迁移过来）
3. 保持 `src/python_api/` 现有导出 API 不变

验收标准：
- `cmake --build build --target heliox_py` 通过
- Python 侧 import `heliox` 不报错

