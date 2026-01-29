# HelioX Python接口使用指南

HelioX是一个与NEURON兼容的高性能神经元模拟器，支持GPU/CPU加速。本文档介绍如何使用Python包装器（wrapper）来简化NEURON与HelioX之间的交互。

## Python库分层（推荐）

- **仿真通用层**：`heliox_core`（导出/加载、Obj/Recorder/VecPlay 等 wrapper）
- **学习训练层**：`heliox_learn`（训练任务/优化器/方法封装）
- **向后兼容**：历史入口 `heliox_wrapper` 仍可用，但新代码建议优先从 `heliox_core` 导入。

## 目录

1. [核心概念](#核心概念)
2. [快速入门](#快速入门)
3. [详细用法](#详细用法)
4. [高级功能](#高级功能)
5. [完整示例](#完整示例)
6. [常见问题](#常见问题)

---

## 核心概念

### 工作流程

HelioX的典型工作流程如下：

```
1. 在NEURON中建立模型（定义细胞、机制、刺激等）
2. 为每个细胞设置GID（必需，用于模型导出）
3. 创建wrapper对象（用于访问变量和记录数据）
4. 初始化HelioX并导出/加载模型
   ↓
   此时NEURON和HelioX模型分离成两个独立副本
   ↓
5. NEURON和HelioX可分别独立运行，互不影响
```

**重要概念：模型分离**

- **在初始化之前**：只有一个NEURON模型
- **调用`setup_and_load_model()`或`enhanced_export_model()`后**：
  - NEURON导出当前模型状态到文件
  - HelioX从文件加载模型
  - **两个模拟器拥有完全独立的模型副本**（初始状态相同）
- **在初始化之后**：
  - NEURON和HelioX可以分别运行
  - 修改一个不会影响另一个
  - 可以对比两个模拟器的结果来验证正确性

**小坑记录（row 索引/重排：一般用户无需关心）**

- **推荐做法**：尽量使用 `HelioXManager` 提供的 `ObjWrapper/Recorder/VecPlay`（以及 `heliox_core` 的绑定/批量 IO 辅助），它们会在**正确的时机**解析/绑定索引，从而自动规避 “row 重排” 的坑。
- **仅在手动 hack 时需要注意**：如果你绕过 wrapper，直接从 NEURON 的 `_ref_*` 字符串里解析 `row=...`，并自己调用底层 handle API（例如批量获取 handle、写临时扩展功能/性能路径），不要在 `setup_and_load_model()` / `enhanced_export_model()` 之前解析并缓存 `row`：`h.finitialize()` 以及导出/加载过程可能导致 row 重编号。
- **原则**：必须手工解析时，务必在 `setup_and_load_model()` / `enhanced_export_model()` 完成后（即 NEURON 已 finalize 并且 HELIOX 已加载）再解析并绑定。

### 三种Wrapper类型

1. **ObjWrapper**（对象包装器）
   - 用于**读写当前时刻的变量值**
   - 支持标量和数组变量
   - 适用于：参数控制、实时状态查询

2. **Recorder**（变量记录器，推荐使用）
   - 用于**记录变量的时间序列数据**
   - 每个时间步自动记录一次
   - 适用于：电压轨迹、电流轨迹等
   - API: `create_recorder()` ✓ 推荐
   - 兼容API: `create_monitor_wrapper()` (向后兼容，建议新代码使用`create_recorder`)

3. **VecPlayWrapper**（向量播放包装器）
   - 用于**动态播放时间序列数据**
   - 在运行过程中按时间表改变变量值
   - 适用于：复杂刺激模式、动态参数调整

---

## 快速入门

### 安装要求

```bash
pip install neuron matplotlib numpy
# 确保HelioX已正确编译和安装
```

### 最小示例

```python
from neuron import h
from heliox_core import HelioXManager  # 推荐（仿真通用层）

# 1. 创建NEURON模型
soma = h.Section(name='soma')
soma.L = 20
soma.diam = 20
soma.insert('hh')

iclamp = h.IClamp(soma(0.5))
iclamp.delay = 10
iclamp.dur = 100
iclamp.amp = 0.3

# 2. 设置GID（必需！用于模型导出）
pc = h.ParallelContext()
gid = 0
pc.set_gid2node(gid, pc.id())
nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
pc.cell(gid, nc)

# 3. 创建HelioX manager和wrapper
manager = HelioXManager()

# 创建记录器（记录时间序列）
v_recorder = manager.create_recorder(soma(0.5), "v")

# 创建对象包装器（读写变量）
iclamp_obj = manager.create_obj_wrapper(iclamp)

# 4. 导出NEURON模型并初始化HelioX
# 此时模型分离：NEURON和HelioX各有一个独立副本
manager.setup_and_load_model(
    export_path="./model_export",
    dt=0.025,
    v_init=-65
)

# 5. 运行HelioX模拟
manager.finitialize(-65)
manager.run(100)

# 6. 获取结果
voltage_data = v_recorder.data  # 获取电压轨迹
current_value = iclamp_obj.amp  # 读取当前电流值
iclamp_obj.amp = 0.5            # 修改电流值（只影响HelioX）
```

### 为什么需要设置GID？

NEURON的模型导出功能（`pc.nrnbbcore_write()`）需要每个细胞都有一个全局唯一标识符（GID）。即使你的模型中没有突触连接，也需要创建一个"虚拟"NetCon来设置GID：

```python
pc = h.ParallelContext()
gid = 0  # 给细胞分配GID
pc.set_gid2node(gid, pc.id())  # 将GID分配给当前进程

# 创建一个输出为None的NetCon（虚拟NetCon）
nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)

# 将细胞与GID关联
pc.cell(gid, nc)
```

对于多细胞模型，每个细胞需要不同的GID：
```python
for i, cell in enumerate(cells):
    pc.set_gid2node(i, pc.id())
    nc = h.NetCon(cell(0.5)._ref_v, None, sec=cell)
    pc.cell(i, nc)
```

---

## 详细用法

### 1. HelioXManager 管理类

HelioXManager是一个单例类，负责管理所有wrapper和HelioX实例。

```python
from heliox_core import HelioXManager  # 推荐

# 创建管理器（单例模式）
manager = HelioXManager()

# 配置设备和参数（可选）
manager.set_default_device("gpu")  # 或 "cpu"
manager.set_default_permute_type(3)
```

### 2. 创建Recorder（变量记录器）

**推荐API：`create_recorder()`**

```python
# 记录segment电压（默认变量为"v"）
v_rec = manager.create_recorder(soma(0.5))

# 记录机制变量
iclamp_rec = manager.create_recorder(iclamp, "amp")

# 记录数组变量的特定元素
array_rec = manager.create_recorder(syn, "weight", array_index=2)
```

**兼容API：`create_monitor_wrapper()`**
```python
# 向后兼容，功能与create_recorder完全相同
v_monitor = manager.create_monitor_wrapper(soma(0.5))
```

获取记录数据：
```python
# 运行模拟后
manager.run(100)

# 获取时间序列数据
voltage_trace = v_rec.data  # numpy数组
```

### 3. 创建ObjWrapper（对象包装器）

用于实时读写变量值（不记录历史）。

```python
# 创建包装器
iclamp_obj = manager.create_obj_wrapper(iclamp)
segment_obj = manager.create_obj_wrapper(soma(0.5))

# 读取变量
current_amp = iclamp_obj.amp
voltage = segment_obj.v

# 写入变量（标量）
iclamp_obj.amp = 0.5
iclamp_obj.delay = 20

# 数组变量访问
# 方法1: 使用get_var/set_var
value = obj.get_var("weights", array_index=0)
obj.set_var("weights", 1.5, array_index=0)

# 方法2: 批量操作整个数组
all_weights = obj.get_var_array("weights")  # 返回list
obj.set_var_array("weights", [1.0, 2.0, 3.0, 4.0])

# 查询数组长度
length = obj.get_var_array_length("weights")
```

### 4. 创建VecPlayWrapper（向量播放）

用于在模拟过程中动态改变变量值。

```python
# 创建VecPlay包装器
vecplay = manager.create_vecplay_wrapper(iclamp, "amp")

# 在setup_and_load_model之后使用
manager.setup_and_load_model("./export", dt=0.025, v_init=-65)

# 播放时间序列
tvec = [0, 10, 20, 30, 40]      # 时间点
yvec = [0, 0.1, 0.3, 0.2, 0]    # 对应的值
vecplay.play(tvec, yvec)

# 运行模拟（VecPlay会自动在指定时间点改变变量）
manager.finitialize(-65)
manager.run(50)

# 停止播放
vecplay.stop()
```

### 5. 模型导出和加载

#### 标准导出
```python
manager.setup_and_load_model(
    export_path="./model_data",
    dt=0.025,        # 时间步长
    v_init=-65       # 初始电压
)
```
此方法会：
1. 导出NEURON模型
2. 初始化所有wrapper
3. 加载HelioX模型

#### 增强导出（保存wrapper元数据）
```python
manager.enhanced_export_model(
    export_path="./model_data",
    dt=0.025,
    v_init=-65
)
```
生成文件：
```
model_data/
├── heliox_metadata.json   # Wrapper信息
├── heliox_config.json     # 配置参数
└── *.dat                   # NEURON模型数据
```

#### 从导出加载（无需NEURON）
```python
# 在新的Python会话中，不需要NEURON
manager = HelioXManager()
wrappers = manager.load_from_export("./model_data")

# 访问重建的wrapper
recorders = wrappers["monitors"]
obj_wrappers = wrappers["obj_wrappers"]
vecplay_wrappers = wrappers["vecplay_wrappers"]

# 使用wrapper
manager.finitialize(-65)
manager.run(100)
data = recorders[0].data
```

### 6. 运行模拟

```python
# 设置时间步长
manager.set_dt(0.025)

# 初始化
manager.finitialize(-65)

# 运行模拟
manager.run(100)  # 运行100ms

# 获取spike时间
spikes = manager.get_spk_by_gid(0)  # 获取GID=0的细胞的spike
```

---

## 高级功能

### 数组变量支持

HelioX完全支持NEURON中的数组变量（如突触权重数组）。

```python
# 假设机制有一个数组变量 "weights"，长度为5

# 方法1: 访问单个元素
obj = manager.create_obj_wrapper(synapse)
w0 = obj.get_var("weights", array_index=0)
obj.set_var("weights", 2.5, array_index=3)

# 方法2: 批量访问整个数组
all_weights = obj.get_var_array("weights")  # [w0, w1, w2, w3, w4]
obj.set_var_array("weights", [1.0, 1.5, 2.0, 2.5, 3.0])

# 方法3: 记录数组变量
rec0 = manager.create_recorder(synapse, "weights", array_index=0)
rec1 = manager.create_recorder(synapse, "weights", array_index=1)
# 运行后分别获取各元素的时间序列
```

### Gap Junction（间隙连接）

HelioX支持在模拟过程中动态创建间隙连接。

```python
from heliox_wrapper import GapJunctionInterface

# 获取Gap Junction接口（单例）
gap = GapJunctionInterface()

# 在setup_and_load_model之后创建间隙连接
manager.setup_and_load_model("./export", dt=0.025, v_init=-65)

# 创建间隙连接：cell0.v -> cell1.v
sid = 100  # 源ID（需要手动指定唯一ID）
gap.add_gap_source(sid, "global", "v", 0)  # 源：cell 0的电压
gap.add_gap_target(sid, "global", "v", 1)  # 目标：cell 1的电压

# 一个源可以有多个目标
gap.add_gap_target(sid, "global", "v", 2)  # 同时影响cell 2

# 查询间隙连接
info = gap.get_info(sid)
all_gaps = gap.list_all()

# 清空所有间隙连接
gap.clear_all()
```

### 调用机制函数

ObjWrapper可以调用机制对象的函数（如果有）。

```python
obj = manager.create_obj_wrapper(custom_mech)

# 调用机制的函数
result = obj.some_function(arg1, arg2)

# 支持的参数类型：
# - 基本类型：int, float, bool
# - 序列类型：list, numpy.ndarray（自动转换为list）
```

### 性能优化：Handle缓存

Wrapper内部使用handle缓存机制来加速重复访问：

```python
obj = manager.create_obj_wrapper(iclamp)

# 第一次访问：获取handle并缓存
value1 = obj.amp  # 慢（需要查找）

# 后续访问：使用缓存的handle
value2 = obj.amp  # 快（直接访问）
value3 = obj.amp  # 快
```

这使得在循环中频繁访问变量时性能大大提升。

---

## 完整示例

### 示例1: 对比NEURON和HelioX结果

这是验证HelioX正确性的标准方法。

```python
from neuron import h
from heliox_wrapper import HelioXManager
import matplotlib.pyplot as plt

# ========== 步骤1: 建立NEURON模型 ==========
soma = h.Section(name='soma')
soma.L = 20
soma.diam = 20
soma.insert('hh')

iclamp = h.IClamp(soma(0.5))
iclamp.delay = 10
iclamp.dur = 50
iclamp.amp = 0.3

# 设置GID
pc = h.ParallelContext()
pc.set_gid2node(0, pc.id())
nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
pc.cell(0, nc)

# NEURON记录
v_neuron = h.Vector()
v_neuron.record(soma(0.5)._ref_v)
t_neuron = h.Vector()
t_neuron.record(h._ref_t)

# ========== 步骤2: 创建HelioX wrapper ==========
manager = HelioXManager()
v_recorder = manager.create_recorder(soma(0.5), "v")

# ========== 步骤3: 初始化HelioX ==========
# 此时模型分离！NEURON和HelioX拥有独立的副本
manager.setup_and_load_model("./export", dt=0.025, v_init=-65)

# ========== 步骤4: 分别运行NEURON和HelioX ==========
# 运行NEURON
h.dt = 0.025
h.finitialize(-65)
h.continuerun(100)

# 运行HelioX
manager.finitialize(-65)
manager.run(100)

# ========== 步骤5: 对比结果 ==========
plt.figure(figsize=(12, 5))
plt.plot(t_neuron, v_neuron, label='NEURON', linewidth=2)
plt.plot(t_neuron, v_recorder.data, '--', label='HelioX', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.title('NEURON vs HelioX Comparison')
plt.savefig('comparison.png')
plt.show()
```

### 示例2: 使用VecPlay实现复杂刺激

```python
import numpy as np
from neuron import h
from heliox_wrapper import HelioXManager

# 创建模型
soma = h.Section(name='soma')
soma.L = 20
soma.diam = 20
soma.insert('hh')

iclamp = h.IClamp(soma(0.5))

# 设置GID
pc = h.ParallelContext()
pc.set_gid2node(0, pc.id())
nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
pc.cell(0, nc)

# 创建HelioX管理器
manager = HelioXManager()
v_rec = manager.create_recorder(soma(0.5), "v")
vecplay = manager.create_vecplay_wrapper(iclamp, "amp")

# 初始化
manager.setup_and_load_model("./export", dt=0.025, v_init=-65)

# 创建复杂刺激模式：正弦波调制
t = np.arange(0, 100, 1)
amp = 0.2 + 0.15 * np.sin(2 * np.pi * t / 20)
vecplay.play(t.tolist(), amp.tolist())

# 运行
manager.finitialize(-65)
manager.run(100)

# 获取结果
voltage = v_rec.data
```

### 示例3: 多细胞网络

```python
from neuron import h
from heliox_wrapper import HelioXManager

# 创建多个细胞
n_cells = 10
cells = []
iclamps = []
recorders = []

for i in range(n_cells):
    soma = h.Section(name=f'soma_{i}')
    soma.L = 20
    soma.diam = 20
    soma.insert('hh')
    cells.append(soma)

    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 10
    iclamp.dur = 50
    iclamp.amp = 0.1 * (i + 1)  # 不同的刺激强度
    iclamps.append(iclamp)

# 设置GID（每个细胞需要唯一GID）
pc = h.ParallelContext()
for i, soma in enumerate(cells):
    pc.set_gid2node(i, pc.id())
    nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
    pc.cell(i, nc)

# 创建HelioX wrapper
manager = HelioXManager()
for soma in cells:
    rec = manager.create_recorder(soma(0.5), "v")
    recorders.append(rec)

# 初始化和运行
manager.setup_and_load_model("./export", dt=0.025, v_init=-65)
manager.finitialize(-65)
manager.run(100)

# 获取所有细胞的电压
for i, rec in enumerate(recorders):
    voltage = rec.data
    print(f"Cell {i}: max voltage = {max(voltage):.2f} mV")
```

---

## 常见问题

### Q1: 为什么必须设置GID？

**A:** NEURON的模型导出功能（`pc.nrnbbcore_write()`）要求每个细胞都有GID。没有GID的细胞无法被导出，HelioX也就无法加载。即使你的模型没有突触连接，也需要创建虚拟NetCon来设置GID。

### Q2: 为什么wrapper必须保持引用？

**A:** HelioX的C++端资源管理尚未完全自动化。如果wrapper对象被Python垃圾回收，可能导致内存泄漏。因此必须保持wrapper的引用：

```python
# ✓ 正确：保持引用
self.v_rec = manager.create_recorder(soma(0.5))

# ✗ 错误：返回值被丢弃
manager.create_recorder(soma(0.5))
```

### Q3: NEURON和HelioX何时分离？

**A:** 调用`setup_and_load_model()`或`enhanced_export_model()`时。在此之前只有一个NEURON模型；调用后，NEURON和HelioX各有一个独立副本，可以分别运行。

### Q4: 如何验证HelioX结果正确性？

**A:** 标准方法是建立一次模型，初始化后NEURON和HelioX分别运行，然后对比结果：

```python
# 1. 建立模型（一次）
# ... 创建soma, iclamp等 ...

# 2. 初始化HelioX（模型分离）
manager.setup_and_load_model("./export")

# 3. NEURON运行
h.finitialize(-65)
h.continuerun(100)

# 4. HelioX运行
manager.finitialize(-65)
manager.run(100)

# 5. 对比结果
# ... 比较v_neuron和v_recorder.data ...
```

**注意：** 不要建立两次模型（一次给NEURON，一次给HelioX）。只需建立一次，初始化后它们就分离了。

### Q5: 如何关闭析构警告？

**A:** 程序正常退出时会自动禁用警告。如需手动控制：

```python
# 禁用警告
HelioXManager.disable_destructor_warnings()

# 重新启用
HelioXManager.enable_destructor_warnings()
```

### Q6: GPU模式失败怎么办？

**A:** 尝试使用CPU模式：

```python
manager = HelioXManager()
manager.set_default_device("cpu")
```

### Q7: 数组变量如何访问？

**A:** 三种方法：
```python
obj = manager.create_obj_wrapper(synapse)

# 方法1: 访问单个元素
value = obj.get_var("weights", array_index=2)
obj.set_var("weights", 3.5, array_index=2)

# 方法2: 批量访问
all_weights = obj.get_var_array("weights")
obj.set_var_array("weights", [1, 2, 3, 4, 5])

# 方法3: 记录特定元素
rec = manager.create_recorder(synapse, "weights", array_index=2)
```

### Q8: VecPlay和ObjWrapper有什么区别？

**A:**
- **VecPlay**: 按预定的时间表自动改变变量（如播放复杂刺激波形）
- **ObjWrapper**: 手动在任意时刻读写变量（如实时调整参数）

### Q9: 增强导出有什么用？

**A:** 增强导出会保存wrapper元数据，使得你可以在没有NEURON的环境中加载模型：

```python
# 在有NEURON的机器上导出
manager.enhanced_export_model("./model")

# 在没有NEURON的机器上加载
manager_new = HelioXManager()
wrappers = manager_new.load_from_export("./model")
# 直接使用，无需NEURON
```

---

## Learning API（实验性）

除了 wrapper（变量访问/记录）之外，本仓库还提供一个“学习框架”风格的实验性 API，目标是把 `NEURON(建模前端)` + `HELIOX(仿真/学习后端)` 的工作流封装成更像“调用框架 API”的形式（compile → runtime → trainer）。

- 入口目录：`python_lib/heliox_learn/`
- 目前阶段：优先固化导出/加载的 contract，并逐步把训练循环从应用脚本迁移到框架层。

> 注意：通常你不需要自己处理 `_ref_*` 的 `row=...`（wrapper 会处理）。只有当你直接使用底层 handle API 自己做绑定/批量访问时，导出/绑定的时序 contract 才需要你手动遵守：不要在 `setup_and_load_model()`/`enhanced_export_model()` 之前解析并缓存 `row`，因为 `h.finitialize()` 以及导出/加载可能导致 row 重编号。

---

## 相关文档

- **测试和示例**: 参见 `python_test/` 目录中的测试脚本
- **C++接口**: 参见 `src/pybinds/` 目录
- **问题反馈**: 提交issue到项目仓库

---

## 总结

HelioX通过Python wrapper提供了简洁的接口来：
1. ✓ 在NEURON中建模
2. ✓ 导出模型到HelioX
3. ✓ 使用GPU/CPU加速运行
4. ✓ 轻松获取和控制变量
5. ✓ 验证结果的正确性

**核心要点：**
- 每个细胞必须设置GID
- 在`setup_and_load_model()`时模型分离
- 保持wrapper引用
- 使用`create_recorder()`（推荐）而非`create_monitor_wrapper()`
- 对比NEURON和HelioX结果来验证正确性

祝你使用愉快！🚀
