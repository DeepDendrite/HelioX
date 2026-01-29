#!/usr/bin/env python3
"""
BPSYN数组访问功能简化测试脚本

这是一个简化版本的测试，用于快速验证BPSYN机制的基本数组访问功能。
适合初次测试和快速调试使用。
"""

import sys
import numpy as np
from neuron import h

# 设置精确计算
h.usetable_hh = 0

# 添加heliox_wrapper路径
sys.path.insert(0, '$HOME/heliox/python_lib')
from heliox_wrapper import HelioXManager


def simple_bpsyn_test(device_mode="cpu"):
    """简化的BPSYN数组测试"""
    print(f"🧪 简化BPSYN数组测试 ({device_mode.upper()}模式)")
    print("-" * 40)

    try:
        # 步骤1: 创建简单的NEURON模型
        print("1. 创建NEURON模型...")
        soma = h.Section(name='soma')
        soma.nseg = 1
        soma.diam = 10
        soma.L = 10
        soma.insert('hh')

        # 设置GID（重要！NEURON导出需要）
        pc = h.ParallelContext()
        gid = 1
        pc.set_gid2node(gid, pc.id())
        nc_dummy = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        pc.cell(gid, nc_dummy)

        # 添加BP_Syn_SoftMax机制（更容易测试）
        softmax = h.BP_Syn_SoftMax(soma(0.5))
        softmax.lr_start = 0.0
        softmax.lr_end = 100.0
        softmax.n_outputs = 10

        # 创建基本电流钳
        iclamp = h.IClamp(soma(0.5))
        iclamp.amp = 0
        iclamp.dur = 1e9
        iclamp.delay = 0

        print(f"   ✓ 创建了 {softmax.get_segment().sec.name()} 上的 BP_Syn_SoftMax")

        # 调试：查看BP_Syn_SoftMax的所有属性
        print("   🔍 BP_Syn_SoftMax的所有属性:")
        for attr in dir(softmax):
            if not attr.startswith('__'):
                print(f"      - {attr}")

        # 尝试直接访问数组
        print("   🔍 尝试访问数组属性:")
        try:
            print(f"      tgt: {softmax.tgt}")
        except Exception as e:
            print(f"      tgt 错误: {e}")
        try:
            print(f"      _ref_tgt: {softmax._ref_tgt}")
        except Exception as e:
            print(f"      _ref_tgt 错误: {e}")

        # 步骤2: 设置测试数据
        print("2. 设置测试数组值...")

        # 设置目标（one-hot，目标类别为2）
        target_class = 2
        for i in range(10):
            tgt_val = 1.0 if i == target_class else 0.0
            # setattr(softmax, f'tgt[{i}]', tgt_val)
            softmax.tgt[i] = tgt_val  # 直接赋值更可靠

        # 设置输入值
        test_u_values = [0.1, 0.5, 2.0, 0.8, 0.3, 0.2, 0.4, 0.6, 0.1, 0.3]
        for i in range(10):
            # setattr(softmax, f'u[{i}]', test_u_values[i])
            softmax.u[i] = test_u_values[i]  # 直接赋值更可靠

        print(f"   ✓ 目标类别: {target_class}")
        print(f"   ✓ 输入最大值在索引: {np.argmax(test_u_values)}")

        # 步骤3: 创建HelioX管理器
        print("3. 初始化HelioX...")

        pc.setup_transfer()
        pc.set_maxstep(10)

        heliox_manager = HelioXManager()
        if device_mode == "cpu":
            heliox_manager.set_default_device("cpu")
            heliox_manager.set_default_permute_type(0)

        export_path = f"./simple_bpsyn_test_{device_mode}"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        print(f"   ✓ 模型导入到: {export_path}")

        # 步骤4: 测试数组访问
        print("4. 测试数组访问...")

        # 创建ObjWrapper来访问变量
        softmax_wrapper = heliox_manager.create_obj_wrapper(softmax)

        # 测试单个元素读取
        print("   📖 读取tgt数组:")
        for i in range(5):  # 只显示前5个
            value = softmax_wrapper.get_var("tgt", array_index=i)
            expected = 1.0 if i == target_class else 0.0
            print(f"     tgt[{i}] = {value:.1f} (期望: {expected:.1f})")

        # 测试整个数组读取
        print("   📊 读取完整u数组:")
        u_array = softmax_wrapper.get_var_array("u")
        print(f"     u数组: {[f'{val:.2f}' for val in u_array]}")

        # 步骤5: 运行仿真并检查结果
        print("5. 运行仿真...")
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(25.0)  # 短时间仿真

        # 读取SoftMax输出
        s_array = softmax_wrapper.get_var_array("s")
        predicted_class = np.argmax(s_array)
        confidence = s_array[predicted_class]

        print("6. 检查仿真结果...")
        print(f"   SoftMax输出: {[f'{val:.3f}' for val in s_array]}")
        print(f"   预测类别: {predicted_class} (置信度: {confidence:.3f})")
        print(f"   目标类别: {target_class}")
        print(f"   输出总和: {np.sum(s_array):.6f}")

        # 简单验证
        sum_valid = abs(np.sum(s_array) - 1.0) < 1e-5
        print(f"   SoftMax性质: {'✅ 有效' if sum_valid else '❌ 无效'}")

        # 清理
        pc.done()

        return sum_valid

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 BPSYN数组访问简化测试")
    print("=" * 40)

    # CPU测试
    print("\n🖥️  CPU模式测试")
    cpu_success = simple_bpsyn_test("cpu")

    # GPU测试
    print("\n🎮 GPU模式测试")
    gpu_success = simple_bpsyn_test("gpu")

    # 总结
    print("\n" + "=" * 40)
    print("📋 测试总结")
    print(f"CPU模式: {'✅ 通过' if cpu_success else '❌ 失败'}")
    print(f"GPU模式: {'✅ 通过' if gpu_success else '❌ 失败'}")

    overall_success = cpu_success and gpu_success
    print(f"整体结果: {'🎉 成功' if overall_success else '⚠️  有问题'}")

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()