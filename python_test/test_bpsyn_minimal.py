#!/usr/bin/env python3
"""测试BPSYN数组访问功能（最小化版本）"""

import sys
sys.path.insert(0, '$HOME/heliox/python_lib')

from neuron import h
import numpy as np

# 设置精确计算
h.usetable_hh = 0

# 创建模型
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

# 设置数组值
print("设置数组值...")
for i in range(10):
    softmax.tgt[i] = 1.0 if i == 2 else 0.0  # one-hot编码，目标类别=2
    softmax.u[i] = i * 0.1  # 输入值

# 初始化HelioX
from heliox_wrapper import HelioXManager

pc.setup_transfer()
pc.set_maxstep(10)

heliox_manager = HelioXManager()
heliox_manager.set_default_device("cpu")
heliox_manager.set_default_permute_type(0)

export_path = "./test_bpsyn_minimal"
heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
print(f"模型导出到: {export_path}")

# 创建ObjWrapper
wrapper = heliox_manager.create_obj_wrapper(softmax)

# 测试数组访问
print("\n测试数组访问...")
try:
    # 读取单个元素
    print("读取tgt[2]:", wrapper.get_var("tgt", array_index=2))
    print("读取u[5]:", wrapper.get_var("u", array_index=5))

    # 设置单个元素
    wrapper.set_var("u", 1.5, array_index=3)
    print("设置u[3]=1.5后，读取u[3]:", wrapper.get_var("u", array_index=3))

    # 读取整个数组
    tgt_array = wrapper.get_var_array("tgt")
    u_array = wrapper.get_var_array("u")
    print(f"tgt数组: {tgt_array}")
    print(f"u数组: {u_array}")

    # 运行仿真
    print("\n运行仿真...")
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(1.0)

    # 读取输出
    s_array = wrapper.get_var_array("s")
    print(f"SoftMax输出: {s_array}")
    print(f"输出总和: {np.sum(s_array):.6f}")

    print("\n✅ 测试成功！")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

pc.done()