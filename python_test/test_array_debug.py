#!/usr/bin/env python3
"""调试数组识别问题的简单脚本"""

import sys
sys.path.insert(0, '$HOME/heliox/python_lib')

from neuron import h

# 设置精确计算
h.usetable_hh = 0

# 创建简单模型
soma = h.Section(name='soma')
soma.nseg = 1
soma.diam = 10
soma.L = 10
soma.insert('hh')

# 设置GID
pc = h.ParallelContext()
gid = 1
pc.set_gid2node(gid, pc.id())
nc_dummy = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
pc.cell(gid, nc_dummy)

# 添加BP_Syn_SoftMax
softmax = h.BP_Syn_SoftMax(soma(0.5))
softmax.n_outputs = 10

# 检查属性
print("=== BP_Syn_SoftMax属性检查 ===")
print(f"tgt: {softmax.tgt}")
print(f"_ref_tgt: {softmax._ref_tgt}")

# 尝试获取数组元素引用
print("\n=== 尝试获取数组元素引用 ===")
try:
    ref_elem = softmax._ref_tgt[0]
    print(f"_ref_tgt[0]: {ref_elem}")
except Exception as e:
    print(f"错误: {e}")

# 设置一些值
print("\n=== 设置数组值 ===")
for i in range(10):
    softmax.tgt[i] = i * 0.1
    print(f"设置tgt[{i}] = {i * 0.1}")

# 使用HelioXManager初始化
print("\n=== 初始化HelioXManager ===")
from heliox_wrapper import HelioXManager

pc.setup_transfer()
pc.set_maxstep(10)

heliox_manager = HelioXManager()
heliox_manager.set_default_device("cpu")
heliox_manager.set_default_permute_type(0)

export_path = "./test_array_debug"
heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
print(f"模型导出到: {export_path}")

# 创建ObjWrapper
print("\n=== 创建ObjWrapper ===")
wrapper = heliox_manager.create_obj_wrapper(softmax)
print(f"初始化完成")
print(f"允许的变量: {wrapper._ObjWrapper__allowed_vars}")
print(f"数组变量: {wrapper._ObjWrapper__array_vars}")

# 检查是否可以访问
if 'tgt' in wrapper._ObjWrapper__allowed_vars:
    print("✅ tgt在允许的变量列表中")
else:
    print("❌ tgt不在允许的变量列表中")

# 清理
pc.done()