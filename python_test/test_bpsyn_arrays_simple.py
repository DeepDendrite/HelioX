#!/usr/bin/env python3
"""
简化的BPSYN数组访问测试
只测试核心功能，使用ObjWrapper API
"""

import sys
import numpy as np
from neuron import h

# 设置精确计算
h.usetable_hh = 0

# 添加heliox_wrapper路径
sys.path.insert(0, '$HOME/heliox/python_lib')
from heliox_wrapper import HelioXManager


def test_bpsyn_arrays(device_mode="cpu"):
    """测试BPSYN数组访问功能"""
    print(f"\n🎯 测试BPSYN数组访问 ({device_mode.upper()}模式)")
    print("=" * 60)

    try:
        # 1. 创建NEURON模型
        print("1. 创建NEURON模型...")
        soma1 = h.Section(name='soma1')
        soma1.nseg = 1
        soma1.diam = 10
        soma1.L = 10
        soma1.insert('hh')

        soma2 = h.Section(name='soma2')
        soma2.nseg = 1
        soma2.diam = 10
        soma2.L = 10
        soma2.insert('hh')

        # 设置GID
        pc = h.ParallelContext()
        gid1, gid2 = 0, 1
        pc.set_gid2node(gid1, pc.id())
        pc.set_gid2node(gid2, pc.id())
        nc1 = h.NetCon(soma1(0.5)._ref_v, None, sec=soma1)
        nc2 = h.NetCon(soma2(0.5)._ref_v, None, sec=soma2)
        pc.cell(gid1, nc1)
        pc.cell(gid2, nc2)

        # 添加BPSYN机制
        aggregator = h.BP_Syn_Aggregator(soma1(0.5))
        aggregator.n_outputs = 50

        softmax = h.BP_Syn_SoftMax(soma2(0.5))
        softmax.n_outputs = 10

        print("   ✓ 创建了BP_Syn_Aggregator和BP_Syn_SoftMax")

        # 2. 初始化HelioX（先导出模型）
        print("2. 初始化HelioX...")
        pc.setup_transfer()
        pc.set_maxstep(10)

        heliox_manager = HelioXManager()
        if device_mode == "cpu":
            heliox_manager.set_default_device("cpu")
            heliox_manager.set_default_permute_type(0)

        export_path = f"./test_bpsyn_arrays_{device_mode}"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        print(f"   ✓ 模型导入到: {export_path}")

        # 3. 创建wrapper
        print("3. 创建ObjWrapper...")
        aggregator_wrapper = heliox_manager.create_obj_wrapper(aggregator)
        softmax_wrapper = heliox_manager.create_obj_wrapper(softmax)
        print("   ✓ 创建了wrappers")

        # 4. 通过HelioX设置数组值（在导出后设置）
        print("4. 通过HelioX设置数组初始值...")
        test_grad_values = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
        for i in range(10):
            aggregator_wrapper.set_var("grad_from_output", test_grad_values[i], array_index=i)

        target_class = 2
        for i in range(10):
            softmax_wrapper.set_var("tgt", 1.0 if i == target_class else 0.0, array_index=i)

        test_u_values = [0.1, 0.5, 2.0, 0.8, 0.3, 0.2, 0.4, 0.6, 0.1, 0.3]
        for i in range(10):
            softmax_wrapper.set_var("u", test_u_values[i], array_index=i)

        print(f"   ✓ 设置了Aggregator前10个元素")
        print(f"   ✓ 设置了SoftMax数组（目标类别: {target_class}）")

        # 5. 测试数组元素访问
        print("\n5. 测试数组元素读取...")
        print("   📖 Aggregator grad_from_output:")
        all_passed = True
        for i in range(10):
            value = aggregator_wrapper.get_var("grad_from_output", array_index=i)
            expected = test_grad_values[i]
            error = abs(value - expected)
            status = "✅" if error < 1e-6 else "❌"
            print(f"      [{i}]: {value:.6f} (期望: {expected:.6f}) {status}")
            all_passed = all_passed and (error < 1e-6)

        print("\n   📖 SoftMax tgt:")
        for i in range(10):
            value = softmax_wrapper.get_var("tgt", array_index=i)
            expected = 1.0 if i == target_class else 0.0
            error = abs(value - expected)
            status = "✅" if error < 1e-6 else "❌"
            print(f"      [{i}]: {value:.6f} (期望: {expected:.6f}) {status}")
            all_passed = all_passed and (error < 1e-6)

        print("\n   📖 SoftMax u:")
        for i in range(10):
            value = softmax_wrapper.get_var("u", array_index=i)
            expected = test_u_values[i]
            error = abs(value - expected)
            status = "✅" if error < 1e-6 else "❌"
            print(f"      [{i}]: {value:.6f} (期望: {expected:.6f}) {status}")
            all_passed = all_passed and (error < 1e-6)

        # 6. 测试数组修改
        print("\n6. 测试数组修改...")
        new_val = 9.99
        softmax_wrapper.set_var("u", new_val, array_index=5)
        read_val = softmax_wrapper.get_var("u", array_index=5)
        error = abs(read_val - new_val)
        print(f"   设置 u[5]={new_val} -> 读取={read_val:.6f} (误差: {error:.6f}) {'✅' if error < 1e-6 else '❌'}")
        all_passed = all_passed and (error < 1e-6)

        # 7. 测试整个数组读取
        print("\n7. 测试整个数组读取...")
        grad_array = aggregator_wrapper.get_var_array("grad_from_output")
        u_array = softmax_wrapper.get_var_array("u")
        print(f"   grad_from_output长度: {len(grad_array)} (期望: 50) {'✅' if len(grad_array) == 50 else '❌'}")
        print(f"   u长度: {len(u_array)} (期望: 10) {'✅' if len(u_array) == 10 else '❌'}")
        all_passed = all_passed and (len(grad_array) == 50) and (len(u_array) == 10)

        # 8. 运行仿真并检查
        print("\n8. 运行仿真...")
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(100.0)

        s_array = softmax_wrapper.get_var_array("s")
        s_sum = np.sum(s_array)
        print(f"   SoftMax输出总和: {s_sum:.6f} (期望: 1.0) {'✅' if abs(s_sum - 1.0) < 1e-6 else '❌'}")
        print(f"   预测类别: {np.argmax(s_array)}")
        all_passed = all_passed and (abs(s_sum - 1.0) < 1e-6)

        # 清理
        pc.done()

        return all_passed

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 BPSYN数组访问简化测试")
    print("=" * 60)

    # CPU测试
    print("\n🖥️  CPU模式")
    cpu_passed = test_bpsyn_arrays("cpu")

    # GPU测试
    print("\n🎮 GPU模式")
    gpu_passed = test_bpsyn_arrays("gpu")

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试结果")
    print(f"CPU模式: {'✅ 通过' if cpu_passed else '❌ 失败'}")
    print(f"GPU模式: {'✅ 通过' if gpu_passed else '❌ 失败'}")
    print(f"整体: {'🎉 成功' if (cpu_passed and gpu_passed) else '⚠️  失败'}")

    sys.exit(0 if (cpu_passed and gpu_passed) else 1)


if __name__ == "__main__":
    main()
