# HelioX：Spike 与执行流（CUDA streams）优化说明

本文档面向接手代码的同学，聚焦于 **(2) Spike 优化** 与 **(3) 执行流/依赖关系优化** 两块，帮助快速理解为什么这些优化能带来显著性能提升，以及如何验证“快且正确”。

> 说明：本仓库还有一套更大的优化基础设施（MechTemp/VarStruct 等 CPU/GPU 原生统一设计），属于另一个主题；本文不展开。

---

## 目录

- [背景：为什么 Spike/事件系统是瓶颈](#背景为什么-spike事件系统是瓶颈)
- [优化目标](#优化目标)
- [2) Spike 优化策略（已实现）](#2-spike-优化策略已实现)
  - [2.1 避免每步 device-wide 同步](#21-避免每步-device-wide-同步)
  - [2.2 “无 Spike” 快路径（4B 计数）](#22-无-spike-快路径4b-计数)
  - [2.3 只在必要时回拷 flags/index](#23-只在必要时回拷-flagsindex)
  - [2.4 减少 net_receive 空调用](#24-减少-net_receive-空调用)
  - [2.5 Spike profiling（用于论文/分析）](#25-spike-profiling用于论文分析)
- [3) 执行流与依赖（已实现）](#3-执行流与依赖已实现)
  - [3.1 为什么 non-blocking stream 会“快但可能错”](#31-为什么-non-blocking-stream-会快但可能错)
  - [3.2 用 event 保持语义但允许 overlap](#32-用-event-保持语义但允许-overlap)
  - [3.3 timestep 级别的 overlap 直觉图](#33-timestep-级别的-overlap-直觉图)
- [构建与运行建议](#构建与运行建议)
- [快速验证 checklist](#快速验证-checklist)

---

## 背景：为什么 Spike/事件系统是瓶颈

NEURON/固定步进（fixed-step）仿真里，事件（spike）通常表现为：

- 绝大多数时间步没有 spike（尤其网络稀疏、delay 较大、输入较少时）。
- 一旦出现 spike，需要触发 postsyn 的 `NET_RECEIVE`，更新一些突触状态，进而影响当步或后续步的电流/状态计算。

这类工作具有典型的“稀疏事件”特征：**99% 的时间步什么都不发生**，但如果实现方式不当，就会在每步都付出昂贵代价（尤其是同步）。

---

## 优化目标

我们的目标是：

1) **在没有 spike 的绝大多数时间步**，尽量只做极少量工作（最好是“4 字节计数 + 分支跳过”）。
2) 避免 `cudaDeviceSynchronize()` 这类 device-wide 同步进入热路径（20000 steps 的量级会被放大）。
3) 在保证 fixed-step 语义正确的前提下，让下一周期的一些工作能与上一周期的尾部工作 overlap（压榨 GPU pipeline）。

---

## 2) Spike 优化策略（已实现）

### 2.1 避免每步 device-wide 同步

核心原则：

- **不要在“每个 dt 一次”的热路径里使用 `cudaDeviceSynchronize()`**。
- 用 `cudaStreamNonBlocking + cudaEvent` 将同步粒度缩小到“只等待本次 spike detect 的 stream”，而不是等待整个 GPU。

实现要点：

- `PreSyn` 内部维护一个专用的 non-blocking CUDA stream（`spike_stream`）和 event（`spike_event`）
- spike detect kernel 在 `spike_stream` 上运行，结束后 record `spike_event`
- CPU 侧只等待这个 `spike_event`（而不是 device-wide）

相关代码：

- `src/spike/presyn.h`：`spike_stream / spike_event`
- `src/spike/presyn.cu`：`cuda_spike_send_async(...)`
- `src/spike/presyn.cpp`：`PreSyn::threshold_detect_gpu(...)`

**论文表述建议（可直接引用/改写）**

- 中文（方法/系统层面）：
  - “我们将 spike 检测从 device-wide 同步中解耦，使用 non-blocking CUDA stream 与事件（cudaEvent）构建显式依赖，从而把同步粒度从整卡同步缩小到单 stream 等待，避免每个时间步强制切断 GPU pipeline。”
- English (method/system phrasing):
  - “We decouple spike detection from device-wide synchronization by using a non-blocking CUDA stream with explicit event fences, reducing synchronization granularity from device-wide to stream-level waits and avoiding per-timestep pipeline stalls.”

### 2.2 “无 Spike” 快路径（4B 计数）

典型仿真里，绝大多数时间步没有 spike。为了利用这个稀疏性：

- GPU 侧只维护两个 device counter：`d_spk_num_real` / `d_spk_num_tot`
- 每步 detect 后只回传 2 个 `int` 到 host pinned 内存：
  - `spk_num_real`：真实 presyn spike 数（我们关心的关键分支依据）
  - `spk_num_tot`：保留 legacy 语义用

当 `spk_num_real == 0` 时：

- **不回拷 spk_flags**
- **不回拷 spk_idx_vec**
- 后续接收阶段也会尽量走“空路径”（见 2.4）

相关代码：

- `src/spike/presyn.cuh`：`cuda_spike_send_async(..., h_spk_num_real, ...)`
- `src/spike/presyn.cpp`：`if (spk_count == 0) return ...;`

**论文表述建议（可直接引用/改写）**

- 中文（稀疏事件/fast-path）：
  - “针对稀疏事件特征，我们采用两阶段事件流水线：第一阶段仅回传 4 字节 spike 计数作为 gating 信号；在计数为 0 的时间步完全跳过后续数据搬运与交付逻辑，从而把绝大多数 time step 的事件开销压缩到常数级。”
- English (sparsity-aware pipeline):
  - “Exploiting event sparsity, we implement a two-stage spike pipeline: a 4-byte counter is used as a gating signal, and when the counter is zero we skip all subsequent transfers and delivery logic, reducing per-timestep overhead to O(1) on spike-free steps.”

### 2.3 只在必要时回拷 flags/index

当 `spk_count > 0` 时才需要进一步处理：

- 将 `spk_idx_vec`（只回拷 `spk_count` 个）和 `spk_flags`（回拷 tot_len）从 GPU 回拷到 CPU pinned 内存
- 然后在 CPU 侧构造当前步的 `SpikeVector`（填 `spk_vec_bkp`）

这样保证：

- 稀疏步（无 spike）极轻量；
- spike 发生时再付出更大的搬运/处理成本。

**论文表述建议（可直接引用/改写）**

- 中文（条件回传/带宽节省）：
  - “我们将 GPU→CPU 的数据回传改为条件触发：只有检测到 spike 的时间步才回传必要的索引/flag 数据，避免在无 spike 的绝大多数时间步引入无意义的 PCIe/同步开销。”
- English (conditional transfers):
  - “We make GPU→CPU transfers conditional: indices/flags are copied back only on spike-active steps, avoiding unnecessary transfers and synchronization on the vast majority of spike-free timesteps.”

### 2.4 减少 net_receive 空调用

在没有任何到期事件的情况下，`postsyn->net_receive_gpu(t)` 的调用是纯浪费，尤其 postsyn 数量大时。

我们做了一个非常保守的优化：

- `post_spike_receive_gpu()` 会从每个 postsyn 的 priority_queue 中弹出“到期事件”，并设置 `receive_count`
- 只有 `receive_count > 0` 时才调用 `net_receive_gpu(t)`

相关代码：

- `src/spike/postsyn.h`：`PostSyn_trait::get_receive_count()`
- `src/simulate.cu`：`network_spike_receive_gpu()` 中的条件调用

**论文表述建议（可直接引用/改写）**

- 中文（事件就绪驱动）：
  - “我们将事件交付的控制流从 ‘每步遍历并调用’ 改为 ‘到期事件驱动（ready-event-driven）’：仅当 postsyn 队列中存在到期事件时才触发 `net_receive`，显著降低无事件步的控制流开销。”
- English (ready-event-driven delivery):
  - “We switch from ‘call net_receive every step’ to ‘ready-event-driven delivery’: net_receive is invoked only when there are due events, reducing control-flow overhead on event-free steps.”

### 2.5 Spike profiling（用于论文/分析）

为方便论文描述/实验分析，提供了一个尽量低开销的开关：

- 环境变量：`HELIOX_PROFILE_SPIKE=1`
- 输出：步数、发生 spike 的步比例、spike count 分布、`net_receive_gpu` 的 skip 比例等

开启示例：

```bash
HELIOX_PROFILE_SPIKE=1 ./build-release/heliox_exec -d ~/newhpc/500hpc_fortest/ -t 500 -m gpu
```

在 500hpc 样例上，一个典型统计输出会显示：

- 大多数步没有 presyn spike（接近 100%）
- `net_receive_gpu` 调用大多数被 skip（因为没有到期事件）

这类统计非常适合作为论文中“为什么这个优化有效”的依据：

> 稀疏事件：大量 time steps 无 spike → 4B 快路径 + 跳过逻辑 → 避免无意义同步与处理。

相关代码：

- `src/spike/presyn.h` / `src/spike/presyn.cpp`：直方图与计数
- `src/simulate.h` / `src/simulate.cpp` / `src/simulate.cu`：开关与 summary 输出

**论文表述建议（可直接引用/改写）**

- 中文（可复现性能论据）：
  - “我们提供可选的 spike profiling（运行时开关）以量化稀疏性：统计每步 presyn spike 数分布、无 spike 步比例、以及 `net_receive` 的跳过比例，为性能优化提供可复现实验依据。”
- English (reproducible characterization):
  - “We provide optional runtime spike profiling to quantify event sparsity (spike-count histogram, spike-free ratio, and net_receive skip ratio), enabling reproducible performance characterization.”

---

## 3) 执行流与依赖（已实现）

### 3.1 为什么 non-blocking stream 会“快但可能错”

CUDA 默认流（default stream）的隐式同步规则比较复杂。我们引入 non-blocking stream 的原因是为了并行，但会带来一个风险：

- 如果 spike detect 使用 non-blocking stream，它可能**绕过**默认流的隐式同步
- 这会导致它在某些情况下读到“上一周期尚未在 default stream 完成更新”的数据（典型例子：VecPlay 写入电压/参数的 kernel 在 default stream 上）

表现为：

- spike 时间发生 1 个 dt 量级的漂移
- 输出 spk.dat 与参考不一致

### 3.2 用 event 保持语义但允许 overlap

为了解决上述 race，同时保留并行空间，我们引入了一个 timestep 级别的“默认流完成事件”：

- `default_stream_step_done_event`：在每个 timestep 的 `last_part_gpu()` 末尾，在 **default stream** 上 record
- 下一 timestep 开始时，让每个 `PreSyn::spike_stream` 先 wait 这个 event，再开始 spike detect

含义：

- 保证 spike detect 不会抢跑 default-stream 的关键更新（例如 VecPlay）
- 同时由于这个 event 只代表 default stream，不会强制等待其它 mech streams，因此仍可能与上一周期的 state/ion 等多流工作 overlap

相关代码：

- `src/simulate.cu`：`default_stream_step_done_event`

**论文表述建议（可直接引用/改写）**

- 中文（显式依赖图/语义正确的异步执行）：
  - “我们将隐式默认流同步规则显式化：用 timestep 级别的事件栅栏（event fence）表达必要依赖，从而在保持 fixed-step 语义正确的前提下，允许 non-blocking streams 与其它机制流并发执行，实现更有效的延迟隐藏（latency hiding）。”
- English (explicit dependency management):
  - “We make implicit legacy default-stream ordering explicit via a timestep-level event fence, preserving fixed-step semantics while enabling concurrency across non-blocking streams and mechanism streams for effective latency hiding.”

### 3.3 timestep 级别的 overlap 直觉图

下图用“语义依赖”来描述我们想要的 overlap（示意，不代表完全精确的 kernel 级顺序）：

```
time →

step k:
  [ update v (default stream) ] [ last_part: state/ion on mech streams ... ]
                           \_____________________ potentially still running

step k+1:
  spike_detect (non-blocking stream)  <-- wait default_stream_step_done_event
  matrix init (streams 0/1)
  current/matrix/solve/update ...
```

关键点：

- spike_detect 只依赖电压（以及 default-stream 负责的 VecPlay 等更新），因此用 event 管住即可
- state/ion 这种“不会写 v”且走独立 streams 的工作，理论上可以与下一步 spike_detect overlap

**论文表述建议（可直接引用/改写）**

- 中文（软件流水线/跨步重叠）：
  - “我们采用跨时间步的软件流水线（software pipelining）：将下一步的 spike 前处理与上一时间步的 state 更新阶段进行重叠执行，并通过显式依赖管理确保不会引入时间步语义漂移。”
- English (cross-timestep software pipelining):
  - “We apply cross-timestep software pipelining by overlapping spike pre-processing for step k+1 with state updates of step k, while using explicit dependency management to prevent timestep semantic drift.”

---

## 构建与运行建议

### Release 构建建议

性能对比时建议使用 Release（避免 Debug 的 `-O0` 影响）：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

### CUDA 严格数学选项（可选）

默认关闭以保证性能（如果论文需要强调数值一致性再打开）：

- `ENABLE_STRICT_CUDA_MATH=OFF`（默认）
- `ENABLE_STRICT_CUDA_MATH=ON` 时启用：
  - `--fmad=false --prec-div=true --prec-sqrt=true`

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_STRICT_CUDA_MATH=ON
cmake --build build-release -j
```

---

## 快速验证 checklist

1) 性能（GPU 空闲、Release 构建）：

```bash
./build-release/heliox_exec -d ~/newhpc/500hpc_fortest/ -t 500 -m gpu
```

2) spike 输出正确性：

```bash
diff output/spk.dat ~/newhpc/500hpc_fortest/spike_output.txt
```

3) Spike profiling（用于论文支撑）：

```bash
HELIOX_PROFILE_SPIKE=1 ./build-release/heliox_exec -d ~/newhpc/500hpc_fortest/ -t 500 -m gpu
```
