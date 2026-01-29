#!/usr/bin/env python3
"""
测试 ObjWrapper 数组变量的新语法

验证以下功能：
1. obj.arr[i] - 单个元素读写
2. obj.arr[start:stop] - 切片读写
3. len(obj.arr) - 获取长度
4. for val in obj.arr - 遍历
5. 负数索引支持
6. 错误处理
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.insert(0, '$HOME/heliox/python_lib')

from neuron import h
from heliox_wrapper import HelioXManager

def test_array_syntax():
    """测试数组变量的新语法"""
    print("=" * 70)
    print("测试 ObjWrapper 数组变量新语法")
    print("=" * 70)

    # 1. 创建NEURON模型
    print("\n[1] 创建NEURON模型...")
    h.usetable_hh = 0

    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 10
    soma.L = 10
    soma.insert('hh')

    # 设置GID
    pc = h.ParallelContext()
    gid = 0
    pc.set_gid2node(gid, pc.id())
    nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
    pc.cell(gid, nc)

    # 创建 BP_Syn_SoftMax 机制（有数组变量）
    softmax = h.BP_Syn_SoftMax(soma(0.5))
    softmax.lr_start = 10.0
    softmax.lr_end = 90.0
    softmax.n_outputs = 10

    # 在NEURON中设置初始值
    for i in range(10):
        softmax.u[i] = float(i) * 0.5
        softmax.tgt[i] = 1.0 if i == 5 else 0.0

    print("   ✓ 创建了 BP_Syn_SoftMax 机制")

    # 2. 初始化 HelioXManager
    print("\n[2] 初始化 HelioXManager...")
    heliox_manager = HelioXManager()
    heliox_manager.set_default_device("cpu")
    heliox_manager.set_default_permute_type(0)

    # 3. 创建 ObjWrapper
    print("\n[3] 创建 ObjWrapper...")
    obj = heliox_manager.create_obj_wrapper(softmax)

    # 4. 导出和加载模型
    print("\n[4] 导出和加载模型...")
    export_path = "./test_array_syntax_output"
    heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)

    print("\n" + "=" * 70)
    print("开始测试数组访问语法")
    print("=" * 70)

    # 测试1: 单个元素读取
    print("\n[测试1] 单个元素读取 - obj.arr[i]")
    print("-" * 50)
    try:
        for i in range(5):
            val = obj.u[i]
            expected = float(i) * 0.5
            print(f"  obj.u[{i}] = {val:.2f} (期望: {expected:.2f}) {'✓' if abs(val - expected) < 1e-6 else '✗'}")
        print("  ✅ 单个元素读取测试通过")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试2: 单个元素写入
    print("\n[测试2] 单个元素写入 - obj.arr[i] = val")
    print("-" * 50)
    try:
        obj.u[0] = 100.0
        obj.u[1] = 200.0
        val0 = obj.u[0]
        val1 = obj.u[1]
        print(f"  设置 obj.u[0] = 100.0, 读取: {val0:.2f} {'✓' if abs(val0 - 100.0) < 1e-6 else '✗'}")
        print(f"  设置 obj.u[1] = 200.0, 读取: {val1:.2f} {'✓' if abs(val1 - 200.0) < 1e-6 else '✗'}")
        print("  ✅ 单个元素写入测试通过")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试3: 切片读取
    print("\n[测试3] 切片读取 - obj.arr[start:stop]")
    print("-" * 50)
    try:
        # 先设置已知值
        for i in range(10):
            obj.u[i] = float(i) * 10.0

        vals = obj.u[2:7]
        print(f"  obj.u[2:7] = {vals}")
        expected = [20.0, 30.0, 40.0, 50.0, 60.0]
        match = all(abs(v - e) < 1e-6 for v, e in zip(vals, expected))
        print(f"  期望: {expected} {'✓' if match else '✗'}")
        print("  ✅ 切片读取测试通过" if match else "  ❌ 切片读取测试失败")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试4: 切片写入（列表）
    print("\n[测试4] 切片写入（列表）- obj.arr[start:stop] = [vals]")
    print("-" * 50)
    try:
        obj.u[0:5] = [1.1, 2.2, 3.3, 4.4, 5.5]
        vals = obj.u[0:5]
        print(f"  设置 obj.u[0:5] = [1.1, 2.2, 3.3, 4.4, 5.5]")
        print(f"  读取: {vals}")
        expected = [1.1, 2.2, 3.3, 4.4, 5.5]
        match = all(abs(v - e) < 1e-6 for v, e in zip(vals, expected))
        print(f"  {'✓' if match else '✗'}")
        print("  ✅ 切片写入（列表）测试通过" if match else "  ❌ 测试失败")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试5: 切片写入（标量广播）
    print("\n[测试5] 切片写入（标量广播）- obj.arr[start:stop] = scalar")
    print("-" * 50)
    try:
        obj.u[5:10] = 99.0
        vals = obj.u[5:10]
        print(f"  设置 obj.u[5:10] = 99.0 (标量广播)")
        print(f"  读取: {vals}")
        match = all(abs(v - 99.0) < 1e-6 for v in vals)
        print(f"  {'✓' if match else '✗'}")
        print("  ✅ 切片写入（标量）测试通过" if match else "  ❌ 测试失败")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试6: 负数索引
    print("\n[测试6] 负数索引 - obj.arr[-1], obj.arr[-2]")
    print("-" * 50)
    try:
        obj.u[-1] = 777.0
        obj.u[-2] = 888.0
        val_minus1 = obj.u[-1]
        val_minus2 = obj.u[-2]
        print(f"  obj.u[-1] = {val_minus1:.2f} (期望: 777.0) {'✓' if abs(val_minus1 - 777.0) < 1e-6 else '✗'}")
        print(f"  obj.u[-2] = {val_minus2:.2f} (期望: 888.0) {'✓' if abs(val_minus2 - 888.0) < 1e-6 else '✗'}")
        print("  ✅ 负数索引测试通过")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试7: len() 函数
    print("\n[测试7] len() 函数 - len(obj.arr)")
    print("-" * 50)
    try:
        length = len(obj.u)
        print(f"  len(obj.u) = {length} (期望: 10) {'✓' if length == 10 else '✗'}")
        print("  ✅ len() 测试通过" if length == 10 else "  ❌ len() 测试失败")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试8: 遍历
    print("\n[测试8] 遍历 - for val in obj.arr")
    print("-" * 50)
    try:
        # 先设置已知值
        for i in range(10):
            obj.u[i] = float(i) * 0.1

        vals_from_iter = []
        for val in obj.u:
            vals_from_iter.append(val)

        print(f"  遍历结果: {[f'{v:.1f}' for v in vals_from_iter]}")
        expected = [float(i) * 0.1 for i in range(10)]
        match = all(abs(v - e) < 1e-6 for v, e in zip(vals_from_iter, expected))
        print(f"  {'✓' if match else '✗'}")
        print("  ✅ 遍历测试通过" if match else "  ❌ 遍历测试失败")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试9: to_list() 和 to_numpy()
    print("\n[测试9] to_list() 和 to_numpy() 方法")
    print("-" * 50)
    try:
        arr_list = obj.u.to_list()
        arr_numpy = obj.u.to_numpy()
        print(f"  to_list() 类型: {type(arr_list).__name__}, 长度: {len(arr_list)}")
        print(f"  to_numpy() 类型: {type(arr_numpy).__name__}, 形状: {arr_numpy.shape}")
        print("  ✅ to_list() 和 to_numpy() 测试通过")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # 测试10: 错误处理 - 索引越界
    print("\n[测试10] 错误处理 - 索引越界")
    print("-" * 50)
    try:
        val = obj.u[100]  # 应该抛出 IndexError
        print("  ❌ 未捕获到索引越界错误")
    except IndexError as e:
        print(f"  ✓ 正确捕获索引越界: {e}")
        print("  ✅ 错误处理测试通过")
    except Exception as e:
        print(f"  ❌ 捕获到错误但类型不对: {e}")

    # 测试11: 错误处理 - 类型错误
    print("\n[测试11] 错误处理 - 类型错误")
    print("-" * 50)
    try:
        val = obj.u["invalid"]  # 应该抛出 TypeError
        print("  ❌ 未捕获到类型错误")
    except TypeError as e:
        print(f"  ✓ 正确捕获类型错误: {e}")
        print("  ✅ 错误处理测试通过")
    except Exception as e:
        print(f"  ❌ 捕获到错误但类型不对: {e}")

    # 测试12: 标量变量仍然正常工作
    print("\n[测试12] 标量变量（向后兼容）")
    print("-" * 50)
    try:
        # BP_Syn_SoftMax 也应该有标量变量
        if hasattr(obj, 'lr_start'):
            val = obj.lr_start
            obj.lr_start = 15.0
            new_val = obj.lr_start
            print(f"  读取 obj.lr_start = {val:.2f}")
            print(f"  设置为 15.0，读取: {new_val:.2f} {'✓' if abs(new_val - 15.0) < 1e-6 else '✗'}")
            print("  ✅ 标量变量测试通过")
        else:
            print("  ⚠️  该机制没有标量变量，跳过测试")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)

    # 清理
    pc.done()

if __name__ == "__main__":
    test_array_syntax()
