# HelioX × Learning System Integration Notes

本文档面向两个场景：

1) **工程接手/维护**：解释 `hpc-net-learn`（学习系统）如何与 `heliox`（高性能仿真后端）交互，整体设计为什么“好用/快/可扩展”，以及关键的性能与灵活性来源。  
2) **论文素材库**：给写论文的同学提供可选的论述点（中英双语），方便按篇幅与定位取舍。

> 关联仓库：
> - 学习系统：`$HOME/hpc-net-learn`
> - 仿真后端：本仓库 `heliox`
> - 原始手工版（用于对比理解）：`$HOME/hpc_net_neuron_demo_org/hpc_net_neuron_demo`

---

## Part 1 — 工程视角：总体设计、优化与灵活性

### 1.1 核心思想：控制面 / 计算面分离（Python 只做 orchestration）

我们把系统拆成两层：

- **前端（NEURON / 学习系统）**：负责“建模与装配”（创建细胞、机制、`source_var/target_var` 路由、训练窗口参数等）。
- **后端（HelioX / 本仓库）**：负责“执行期热点”（GPU/CPU 仿真推进、批量输入注入、批量优化器更新、记录器等）。

学习循环不再把每个连接/每个参数的更新逻辑放在 Python；Python 的职责收敛为常数级的高层调用：

1) **一次性**导出+加载：`setup_and_load_model(export_path, dt, v_init)`  
2) 每个 step：`set_input_stimulus(values)` → `finitialize(v_init)` → `run(tstop)` → `optimizer_step(...)`

对应代码位置：

- 统一“冻结点”：`python_lib/heliox_sim/manager.py` 的 `setup_and_load_model()`  
- 学习系统的桥接封装：`$HOME/hpc-net-learn/hybrid_backend.py` 的 `initialize()` / `set_input_pixels()` / `optimizer_step()`

**为什么好：**

- 把高频的细粒度交互（逐个刺激器写入、逐个权重更新、逐条连接同步）从 Python 移除，避免 Python/FFI 调用碎片化成为瓶颈。
- 前端继续保留 NEURON 的生态与表达力（细胞形态、机制、并行路由），后端集中优化性能路径。

---

### 1.2 “两阶段协议”是关键：建模期 vs 执行期（late-binding）

系统对“什么时候可以拿到稳定索引/指针”有明确约束：

- **建模期**：你可以创建 wrapper（Obj/Recorder/VecPlay），但它们会进入 pending 状态。
- **导出+加载之后**：后端模型与索引稳定，wrapper 统一初始化并解析 row/handle；从此进入高性能访问路径。

对应代码位置：

- `python_lib/heliox_sim/manager.py`：导出 `nrnbbcore_write()` → 初始化 wrapper → `load_model()`  
- `python_lib/heliox_sim/wrappers.py`：优先从 `NEURON` 对象的 `_ref_*` 字符串解析 `row=...`，避免 `node_index()` 与导出后内部索引语义不一致导致的错绑。

**为什么好：**

- 避免“导出前缓存索引”导致的隐式 bug（最难排查的那类：运行正常但结果错/漂）。
- 用统一协议把“NEURON 世界的对象引用”安全地映射到“HelioX 世界的索引/handle”，把脆弱的约束变成显式 contract。

---

### 1.3 句柄（handle）与缓存：把动态对象 API 退化为近静态访问

一旦模型加载完成，wrapper 会尽量把变量访问走到 handle 路径：

- `get_handle(var_name, array_index)` 获取并缓存 handle
- 之后 `get_variable_by_handle(handle)` / `set_variable_by_handle(handle, value)` 快速访问

对应代码位置：

- `python_lib/heliox_sim/wrappers.py` 的 `ObjWrapper.get_handle()` / `get_var()` / `set_var()`

**为什么好：**

- 避免反复做字符串解析、机制查找、row 定位等昂贵操作。
- 让上层逻辑仍以“对象/属性”方式表达，但底层访问接近“数组索引/指针”性能模型。

**工程注意事项：**

- wrapper 必须保持引用（本项目明确在 docstring 中警告）。如果 wrapper 被 GC，后端可能无法自动回收资源并产生泄漏风险。

---

### 1.4 输入注入的批处理：从 O(N) Python 写入 → O(1) 后端下发

旧版手工实现通常会在 Python 里对每个像素/每个刺激器做赋值（例如 MNIST 784 维就 784 次写）。

HelioX 支持把输入刺激器注册为 batch：

- NetStim：批量拿到 `(interval, start, number)` 三元组 handle，后续一次性 `set_input_batch_pixels(batch_id, pixels)`  
  见 `python_lib/heliox_sim/manager.py` 的 `register_input_stimulators()` / `_create_input_batch()` / `set_input_stimulus()`
- VecStim：同理支持 batch（按 mech index 列表注册）

学习系统侧对应封装：`$HOME/hpc-net-learn/hybrid_backend.py` 的 `register_input_stimulators()` 与 `set_input_pixels()`

**为什么好：**

- 训练循环中最“高频、最碎片化”的输入更新变成一次调用，显著降低 Python→C++→GPU 的调用开销。
- batch 的参数（interval_scale/start_base/epsilon 等）可配置，便于做不同编码策略/鲁棒性实验。

---

### 1.5 优化器内置化：把 batch 梯度平均与权重广播放到后端

旧版手工实现常见模式是：

1) 每个 replica 保存/拉取 `delta_w` 到 numpy
2) Python 里求和取平均
3) 再把新权重写回每个 replica 的每条突触

HelioX 的做法是：

- 通过 handle 把 **weight** 与 **grad(acc_grad)** 直接注册给后端 optimizer
- 同一参数在多个 replica 的指针可作为一个“参数组”（batch）注册
- step 时在后端完成：
  - 对 batch 内 grad 求和/平均
  - 更新基准权重并广播到其他 replica
  - 清零 grad

对应代码位置：

- 学习系统参数注册：`$HOME/hpc-net-learn/hybrid_backend.py` 的 `_register_single_optimizer_param()`  
  - 从 wrapper 获取 `w` 与 `acc_grad` handle，再调用 `optimizer_add_param(_batch)`。
- 后端 optimizer 实现：`src/optimizer/optimizer.cpp`（CPU/GPU 路径都体现了 avg_grad + broadcast + zero grad 的语义）

**为什么好：**

- Python 不再参与大规模权重矩阵搬运与逐突触写回，训练 loop 更“像控制面”，热点更可控、更接近理论吞吐。
- 优化器策略（SGD/Momentum/Adam）成为后端可配置项；学习系统只需决定超参并调用 `configure_optimizer()`。

---

### 1.6 多副本 batch 训练的“延迟注册模式”：用接口设计获得 batch 语义

学习系统为了实现 mini-batch（replica-based），引入了一个很关键的桥接策略：**deferred registration / group mode**。

工作流程（概念）：

1) 构建多个 replica，但共享一个后端管理器对象  
2) 在构建期把“每个参数的 wrapper”先暂存起来  
3) 构建完成后再按 replica_count 把“同一参数位置”的 wrappers 分组成一个 batch param group  
4) 一次性注册给后端 optimizer

对应代码位置：

- 学习系统侧：`$HOME/hpc-net-learn/hybrid_backend.py` 的 `enable_multi_model_mode()` / `finalize_multi_model_mode()`

**为什么好：**

- 上层“复制模型”即可获得 batch 语义，不需要在 NEURON 侧重写建模逻辑，也不需要在 Python 侧手写“聚合+广播”的大循环。
- 对后端来说，batch param group 把并行化的机会显式暴露出来（更易写 GPU kernel、更易做性能优化）。

---

### 1.7 机制（Mech）拆分带来的灵活性：从“一个大状态机”到可组合算子

原始手工版常见问题是把 Softmax、误差传播、权重更新揉在一个机制里，并且把类别数写死为 10（例如 `u_0..u_9`、`s_0..s_9`、`tgt_0..tgt_9`）。

当前框架的策略是：

- `BP_Syn_FullyConnected`：负责前向电流与局部梯度累积（`acc_grad`）  
- `BP_Syn_SoftMax`：负责输出层的 softmax + CE 梯度产生  
- `BP_Syn_Aggregator`：负责隐藏层的梯度 fan-in 归约（sum）

**为什么好：**

- 更容易扩展（替换 loss、改变输出维度、引入新层/新机制），减少“改一处牵一发而动全身”的风险。
- 更容易对齐后端实现：每个机制可以在 HelioX 侧单独实现 CUDA/CPU 路径并验证一致性。

---

### 1.8 可复现性与数值契约：性能 vs 严格数学路径是显式开关

后端构建系统提供数值行为开关：

- 默认关闭 `fast-math`，并对 CPU 编译关闭 FMA contract（尽量贴近某些 NEURON 路径）。
- `ENABLE_STRICT_CUDA_MATH` 可选开启更严格的 CUDA 数学标志（会变慢，但更可控）。

对应代码位置：

- `CMakeLists.txt` 的 `-fno-fast-math` / `-ffp-contract=off` / `ENABLE_STRICT_CUDA_MATH`

**为什么好：**

- 工程上可调：debug/对齐阶段选严格；性能跑批阶段选默认性能。
- 论文上可讲“reproducibility knobs”：你在做系统优化时没有忽视数值一致性问题。

---

## Part 2 — 论文素材库（可选点，中英双语）

> 写法建议：每一点都可以当作一个“可独立引用的贡献/观察/设计选择”。你可以选其中 3–5 个拼成论文的 systems contribution。  
> 下面每一点先中文、后英文（按点分组，不交错）。

### 2.1 前端-后端分层：NEURON as Frontend, HelioX as Runtime Backend

中文：
我们将系统设计为前端-后端分层：NEURON 负责表达与构建高保真多分区神经元模型（包括形态、机制、并行路由），HelioX 负责加载导出后的模型工件并执行训练期间的热点计算。该分层允许在不牺牲建模表达力的前提下，对执行期进行针对性的 GPU/CPU 优化，并把 Python 降级为控制面，从而获得可扩展的训练吞吐。

English:
We adopt a frontend–backend architecture: NEURON serves as the modeling frontend for high-fidelity multi-compartment neuron networks (morphology, mechanisms, and parallel routing), while HelioX acts as the runtime backend that loads an exported model artifact and executes training-time hot paths. This separation preserves modeling expressiveness while enabling aggressive CPU/GPU optimizations in the execution phase, effectively relegating Python to a low-frequency control plane and improving scalable training throughput.

---

### 2.2 两阶段协议与 late-binding：显式冻结点保证正确性与可优化性

中文：
我们引入显式的“冻结点”（export+load）作为系统契约：建模期可以自由创建对象与 wrapper，但只有在导出并加载后才进行索引/句柄绑定。通过 late-binding，我们避免了导出前缓存索引导致的错绑与隐式错误，并使得后端可以在模型加载后获得稳定的数据布局与访问句柄，从而支持批处理与内核化优化。

English:
We enforce an explicit “freeze point” (export+load) as a system contract: objects and wrappers may be created during model construction, but index/handle binding is deferred until after export and backend loading. This late-binding strategy prevents subtle misbinding bugs caused by pre-export cached indices and enables the backend to operate on a stable data layout with stable handles, unlocking batched and kernel-level optimizations.

---

### 2.3 Handle caching：把动态对象接口变成可批处理的低开销访问路径

中文：
为降低跨语言与动态查找开销，我们采用基于 handle 的变量访问协议：后端在模型加载后为变量分配稳定 handle，wrapper 负责缓存 handle 并在后续通过 handle 进行快速读写。该设计将“对象/属性”风格接口退化为近似“数组索引/指针”的性能模型，使训练循环中的变量访问更接近批处理与向量化的运行时。

English:
To reduce cross-language overhead and dynamic lookups, we employ a handle-based access protocol: stable variable handles are obtained after model loading, cached by wrappers, and used for fast read/write operations. This turns an object/attribute-like interface into an almost array-index/pointer-like performance model, allowing training-time accesses to better match batched and vectorized runtime behavior.

---

### 2.4 Batched input injection：把 per-neuron 刺激更新变成一次调用

中文：
我们将输入刺激器注册为 batch，并提供一次性下发输入向量的接口，从而避免在 Python 中对每个输入神经元进行循环赋值（对高维输入如 MNIST 极其关键）。该设计显著降低训练循环中的小粒度 API 调用数量，使输入更新成为常数级开销，并易于探索不同编码策略（NetStim/VecStim 及其参数化）。

English:
We register input stimulators as batched groups and expose a single-call API to inject an input vector, avoiding per-input-neuron updates in Python—critical for high-dimensional inputs such as MNIST. This substantially reduces fine-grained API calls in the training loop, making input updates effectively constant-time and enabling flexible exploration of different encoding strategies (NetStim/VecStim and their parameterizations).

---

### 2.5 Runtime-native optimizer：后端完成 batch 梯度平均、权重更新与广播

中文：
我们将优化器下沉到仿真后端：学习系统只需将参数与梯度的 handle 注册给后端，后端在一次 step 中完成 batch 梯度归约、权重更新与跨副本广播，并清零梯度。该设计避免了 Python 端的权重矩阵搬运与逐突触写回，使训练计算更接近仿真内核，并为 GPU kernel 融合与并行化创造条件。

English:
We implement the optimizer natively in the simulation backend: the learning system registers parameter and gradient handles, and the backend performs batch gradient reduction, weight updates, and replica broadcast in a single step, followed by gradient reset. This eliminates Python-side weight tensor shuffling and per-synapse write-backs, aligning training computation with the simulation kernel and enabling GPU kernel fusion and parallelization opportunities.

---

### 2.6 Deferred registration for replica-based minibatch：以接口设计获得 batch 语义

中文：
针对 replica-based minibatch，我们提出 deferred registration：在构建多个副本网络时暂存每个参数的 wrapper，待所有副本构建完成后按参数位置分组注册为一个 batch 参数组。该策略使“复制模型”即可获得 minibatch 语义，而无需在前端建模逻辑中编码复杂的对齐规则，同时将 batch 并行机会显式交给后端执行。

English:
For replica-based minibatching, we introduce deferred registration: parameter wrappers are collected during replica construction and later grouped by parameter position to form batched parameter groups after all replicas are built. This lets minibatch semantics emerge naturally from model replication, without embedding complex alignment logic into the frontend modeling code, while explicitly exposing batch-parallel opportunities to the backend.

---

### 2.7 Modular mechanisms：从单一“大状态机机制”到可组合算子（更易扩展、更易验证）

中文：
与将 Softmax、误差传播与权重更新揉在单一机制并写死输出维度的手工实现不同，我们将学习相关机制拆分为可组合单元（全连接突触、Softmax/CE、梯度聚合器）。该拆分降低耦合、提升可维护性，并使每个机制能够独立在后端实现与验证，从而更容易扩展到不同输出维度、不同损失或新层类型。

English:
Unlike monolithic hand-crafted mechanisms that combine softmax, error propagation, and weight updates while hard-coding output dimensionality, we decompose learning-related logic into composable units (fully-connected synapse, softmax/CE, and gradient aggregator). This reduces coupling, improves maintainability, and allows each mechanism to be independently implemented and validated in the backend, facilitating extension to varying output dimensionalities, alternative losses, and new layer types.

---

### 2.8 Reproducibility knobs：性能与数值一致性的可切换契约

中文：
系统将“性能 vs 数值一致性”的折中显式化为构建/运行开关（例如禁用 fast-math、可选启用严格 CUDA 数学路径）。这种可切换契约既支持在调试/对齐阶段追求一致性，也支持在大规模实验中追求吞吐，使得性能优化不会以不可控的数值漂移为代价。

English:
We make the “performance vs numerical fidelity” trade-off explicit via build/runtime switches (e.g., disabling fast-math and optionally enabling strict CUDA math flags). This configurable contract supports fidelity-focused debugging and alignment, as well as throughput-focused large-scale experiments, ensuring performance optimization does not come at the cost of uncontrolled numerical drift.

---

## 附：写作时可用的一句话对比（旧手工版 → 新框架）

- 旧版（手工）：Python 负责输入更新、梯度收集/平均、权重写回，还需要把下游权重复制成上游的误差通道；机制也常被做成一个“状态机大杂烩”，输出维度写死。  
- 新版（框架）：Python 只做常数级控制面调用；输入与优化器均批处理化并内核化；梯度通过显式信号路由与聚合器在仿真图中流动；机制拆分为可组合算子以提高可扩展性与可验证性。

