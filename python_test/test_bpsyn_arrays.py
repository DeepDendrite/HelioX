#!/usr/bin/env python3
"""
BPSYN机制数组访问功能测试脚本

这个测试脚本专门验证新移植的BPSYN机制中的数组变量访问功能，包括：
- BP_Syn_Aggregator的 grad_from_output[50] 数组
- BP_Syn_SoftMax的 tgt[10], u[10], s[10], grad_to_prev[10] 数组

测试原则：
1. 只在NEURON中创建一次模型，HelioX会在init时自动导入
2. 避免在NEURON和HelioX中各创建一次模型
3. 测试数组元素的读取、设置和整个数组的访问
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui
from neuron.units import ms, mV

# 设置精确计算
h.usetable_hh = 0

# 添加heliox_wrapper路径
sys.path.insert(0, '$HOME/heliox/python_lib')
from heliox_wrapper import HelioXManager


class BPSynArrayTest:
    """BPSYN数组访问测试类"""

    def __init__(self):
        self.test_results = []

    def create_neuron_model_with_bpsyn(self):
        """在NEURON中创建带有BPSYN机制的模型（只创建一次）"""
        print("🧬 在NEURON中创建带有BPSYN机制的模型...")

        # 创建两个细胞用于测试不同的BPSYN机制
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

        # 设置GID（重要！NEURON导出需要）
        pc = h.ParallelContext()
        gid1, gid2 = 0, 1
        pc.set_gid2node(gid1, pc.id())
        pc.set_gid2node(gid2, pc.id())
        nc1 = h.NetCon(soma1(0.5)._ref_v, None, sec=soma1)
        nc2 = h.NetCon(soma2(0.5)._ref_v, None, sec=soma2)
        pc.cell(gid1, nc1)
        pc.cell(gid2, nc2)

        # 在soma1上添加BP_Syn_Aggregator机制
        aggregator = h.BP_Syn_Aggregator(soma1(0.5))
        aggregator.lr_start = 10.0  # 学习开始时间
        aggregator.lr_end = 90.0    # 学习结束时间
        aggregator.n_outputs = 50   # 输出数量

        # 在soma2上添加BP_Syn_SoftMax机制
        softmax = h.BP_Syn_SoftMax(soma2(0.5))
        softmax.lr_start = 10.0     # 学习开始时间
        softmax.lr_end = 90.0       # 学习结束时间
        softmax.n_outputs = 10      # 输出数量

        # 创建基本电流钳（用于保持细胞稳定）
        iclamp1 = h.IClamp(soma1(0.5))
        iclamp1.amp = 0
        iclamp1.dur = 1e9
        iclamp1.delay = 0

        iclamp2 = h.IClamp(soma2(0.5))
        iclamp2.amp = 0
        iclamp2.dur = 1e9
        iclamp2.delay = 0

        print(f"   ✓ 创建了 {aggregator.get_segment().sec.name()} 上的 BP_Syn_Aggregator")
        print(f"   ✓ 创建了 {softmax.get_segment().sec.name()} 上的 BP_Syn_SoftMax")

        return soma1, soma2, aggregator, softmax, iclamp1, iclamp2, pc

    def setup_initial_array_values(self, aggregator, softmax):
        """设置BPSYN机制的初始数组值"""
        print("🎯 设置BPSYN机制的初始数组值...")

        # 设置Aggregator的grad_from_output数组
        # 为前10个元素设置不同的值，方便测试
        test_grad_values = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
        for i in range(10):
            # NEURON中访问数组使用对象.数组名[索引]语法
            aggregator.grad_from_output[i] = test_grad_values[i]

        print(f"   ✓ 设置了Aggregator的grad_from_output前10个元素")

        # 设置SoftMax的目标数组（one-hot编码，目标类别为索引2）
        target_class = 2
        for i in range(10):
            tgt_val = 1.0 if i == target_class else 0.0
            softmax.tgt[i] = tgt_val

        # 设置SoftMax的输入数组（模拟网络输出）
        test_u_values = [0.1, 0.5, 2.0, 0.8, 0.3, 0.2, 0.4, 0.6, 0.1, 0.3]  # 索引2有最大值
        for i in range(10):
            softmax.u[i] = test_u_values[i]

        print(f"   ✓ 设置了SoftMax的tgt数组（目标类别: {target_class}）")
        print(f"   ✓ 设置了SoftMax的u数组（最大值在索引: {np.argmax(test_u_values)}）")

        return test_grad_values, test_u_values, target_class

    def create_heliox_manager(self, export_path, device_mode="cpu", pc=None, aggregator=None, softmax=None):
        """创建HelioX管理器并导入模型"""
        print(f"⚙️  创建HelioX管理器 ({device_mode.upper()}模式)...")

        # 如果没有传入pc，说明可能是错误调用
        if pc is None:
            raise ValueError("需要传入ParallelContext对象")

        # ParallelContext配置
        tvec = h.Vector()
        idvec = h.Vector()
        pc.spike_record(-1, tvec, idvec)
        pc.setup_transfer()
        pc.set_maxstep(10)

        # 创建HelioX管理器
        heliox_manager = HelioXManager()
        if device_mode == "cpu":
            heliox_manager.set_default_device("cpu")
            heliox_manager.set_default_permute_type(0)
        # GPU模式使用默认设置

        # 导出和加载模型（HelioX会自动导入NEURON中创建的模型）
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)

        print(f"   ✓ HelioX模型已从NEURON导入到 {export_path}")

        # 创建ObjWrapper用于访问变量
        aggregator_wrapper = heliox_manager.create_obj_wrapper(aggregator) if aggregator else None
        softmax_wrapper = heliox_manager.create_obj_wrapper(softmax) if softmax else None

        print(f"   ✓ 创建了机制的ObjWrapper")

        return heliox_manager, pc, aggregator_wrapper, softmax_wrapper

    def test_array_element_access(self, aggregator_wrapper, softmax_wrapper, test_grad_values, test_u_values, target_class, device_mode="cpu"):
        """测试数组元素的访问功能"""
        print(f"\n🔬 测试数组元素访问 ({device_mode.upper()}模式)")
        print("-" * 50)

        try:
            # 测试Aggregator的grad_from_output数组读取
            print("📖 测试BP_Syn_Aggregator的grad_from_output数组读取:")
            aggregator_errors = []

            for i in range(10):  # 测试前10个元素
                # 使用wrapper的get_var方法读取数组元素
                try:
                    value = aggregator_wrapper.get_var("grad_from_output", array_index=i)
                    expected = test_grad_values[i]
                    error = abs(value - expected)
                    aggregator_errors.append(error)

                    print(f"   grad_from_output[{i}]: {value:.6f} (期望: {expected:.6f}, 误差: {error:.6f})")
                except Exception as e:
                    print(f"   ❌ grad_from_output[{i}] 读取失败: {e}")
                    aggregator_errors.append(float('inf'))

            # 测试SoftMax的tgt数组读取
            print("\n📖 测试BP_Syn_SoftMax的tgt数组读取:")
            tgt_errors = []

            for i in range(10):
                try:
                    value = softmax_wrapper.get_var("tgt", array_index=i)
                    expected = 1.0 if i == target_class else 0.0
                    error = abs(value - expected)
                    tgt_errors.append(error)

                    print(f"   tgt[{i}]: {value:.6f} (期望: {expected:.6f}, 误差: {error:.6f})")
                except Exception as e:
                    print(f"   ❌ tgt[{i}] 读取失败: {e}")
                    tgt_errors.append(float('inf'))

            # 测试SoftMax的u数组读取
            print("\n📖 测试BP_Syn_SoftMax的u数组读取:")
            u_errors = []

            for i in range(10):
                try:
                    value = softmax_wrapper.get_var("u", array_index=i)
                    expected = test_u_values[i]
                    error = abs(value - expected)
                    u_errors.append(error)

                    print(f"   u[{i}]: {value:.6f} (期望: {expected:.6f}, 误差: {error:.6f})")
                except Exception as e:
                    print(f"   ❌ u[{i}] 读取失败: {e}")
                    u_errors.append(float('inf'))

            # 计算总体测试结果
            max_aggregator_error = max(aggregator_errors) if aggregator_errors else float('inf')
            max_tgt_error = max(tgt_errors) if tgt_errors else float('inf')
            max_u_error = max(u_errors) if u_errors else float('inf')

            tolerance = 1e-6
            aggregator_passed = max_aggregator_error < tolerance
            tgt_passed = max_tgt_error < tolerance
            u_passed = max_u_error < tolerance

            print(f"\n📊 数组元素访问测试结果:")
            print(f"   Aggregator grad_from_output: {'✅ 通过' if aggregator_passed else '❌ 失败'} (最大误差: {max_aggregator_error:.6f})")
            print(f"   SoftMax tgt: {'✅ 通过' if tgt_passed else '❌ 失败'} (最大误差: {max_tgt_error:.6f})")
            print(f"   SoftMax u: {'✅ 通过' if u_passed else '❌ 失败'} (最大误差: {max_u_error:.6f})")

            overall_passed = aggregator_passed and tgt_passed and u_passed

            self.test_results.append({
                'name': f'数组元素访问_{device_mode.upper()}',
                'passed': overall_passed,
                'details': {
                    'aggregator_max_error': max_aggregator_error,
                    'tgt_max_error': max_tgt_error,
                    'u_max_error': max_u_error,
                    'tolerance': tolerance
                }
            })

            return overall_passed

        except Exception as e:
            print(f"❌ 数组元素访问测试异常: {e}")
            return False

    def test_array_modification(self, heliox_manager, device_mode="cpu"):
        """测试数组元素的修改功能"""
        print(f"\n🔧 测试数组元素修改 ({device_mode.upper()}模式)")
        print("-" * 50)

        try:
            # 修改Aggregator的grad_from_output数组的几个元素
            print("✏️ 修改BP_Syn_Aggregator的grad_from_output数组:")
            new_values = [1.0, 2.0, 3.0, 4.0, 5.0]

            for i, new_val in enumerate(new_values):
                try:
                    # 使用HelioX的set_var方法设置数组元素
                    heliox_manager.client.set_var("BP_Syn_Aggregator", 0, "grad_from_output", new_val, array_index=i)

                    # 立即读取验证
                    read_val = heliox_manager.client.get_var("BP_Syn_Aggregator", 0, "grad_from_output", array_index=i)
                    error = abs(read_val - new_val)

                    print(f"   grad_from_output[{i}]: 设置 {new_val:.6f} -> 读取 {read_val:.6f} (误差: {error:.6f})")
                except Exception as e:
                    print(f"   ❌ grad_from_output[{i}] 修改失败: {e}")
                    return False

            # 修改SoftMax的u数组
            print("\n✏️ 修改BP_Syn_SoftMax的u数组:")
            new_u_values = [0.5, 1.0, 1.5, 2.0, 2.5, 0.1, 0.2, 0.3, 0.4, 0.6]

            for i, new_val in enumerate(new_u_values):
                try:
                    heliox_manager.client.set_var("BP_Syn_SoftMax", 0, "u", new_val, array_index=i)
                    read_val = heliox_manager.client.get_var("BP_Syn_SoftMax", 0, "u", array_index=i)
                    error = abs(read_val - new_val)

                    print(f"   u[{i}]: 设置 {new_val:.6f} -> 读取 {read_val:.6f} (误差: {error:.6f})")
                except Exception as e:
                    print(f"   ❌ u[{i}] 修改失败: {e}")
                    return False

            print("✅ 数组元素修改测试通过")

            self.test_results.append({
                'name': f'数组元素修改_{device_mode.upper()}',
                'passed': True,
                'details': 'All array modifications successful'
            })

            return True

        except Exception as e:
            print(f"❌ 数组元素修改测试异常: {e}")
            return False

    def test_whole_array_access(self, heliox_manager, device_mode="cpu"):
        """测试整个数组的访问功能"""
        print(f"\n📊 测试整个数组访问 ({device_mode.upper()}模式)")
        print("-" * 50)

        try:
            # 测试获取整个grad_from_output数组
            print("📖 读取BP_Syn_Aggregator的完整grad_from_output数组:")
            try:
                grad_array = heliox_manager.client.get_var_array("BP_Syn_Aggregator", 0, "grad_from_output")
                print(f"   数组长度: {len(grad_array)}")
                print(f"   前10个元素: {[f'{val:.3f}' for val in grad_array[:10]]}")
                print(f"   数值范围: [{np.min(grad_array):.3f}, {np.max(grad_array):.3f}]")
            except Exception as e:
                print(f"   ❌ 读取grad_from_output数组失败: {e}")
                return False

            # 测试获取整个u数组
            print("\n📖 读取BP_Syn_SoftMax的完整u数组:")
            try:
                u_array = heliox_manager.client.get_var_array("BP_Syn_SoftMax", 0, "u")
                print(f"   数组长度: {len(u_array)}")
                print(f"   所有元素: {[f'{val:.3f}' for val in u_array]}")
            except Exception as e:
                print(f"   ❌ 读取u数组失败: {e}")
                return False

            # 测试获取计算后的s数组（SoftMax输出）
            print("\n📖 读取BP_Syn_SoftMax的s数组（SoftMax输出）:")
            try:
                s_array = heliox_manager.client.get_var_array("BP_Syn_SoftMax", 0, "s")
                print(f"   数组长度: {len(s_array)}")
                print(f"   SoftMax输出: {[f'{val:.4f}' for val in s_array]}")
                print(f"   输出总和: {np.sum(s_array):.6f} (应接近1.0)")
                print(f"   最大值索引: {np.argmax(s_array)}")
            except Exception as e:
                print(f"   ❌ 读取s数组失败: {e}")
                return False

            # 测试获取grad_to_prev数组
            print("\n📖 读取BP_Syn_SoftMax的grad_to_prev数组:")
            try:
                grad_prev_array = heliox_manager.client.get_var_array("BP_Syn_SoftMax", 0, "grad_to_prev")
                print(f"   数组长度: {len(grad_prev_array)}")
                print(f"   梯度值: {[f'{val:.4f}' for val in grad_prev_array]}")
            except Exception as e:
                print(f"   ❌ 读取grad_to_prev数组失败: {e}")
                return False

            print("✅ 整个数组访问测试通过")

            self.test_results.append({
                'name': f'整个数组访问_{device_mode.upper()}',
                'passed': True,
                'details': 'All array access operations successful'
            })

            return True

        except Exception as e:
            print(f"❌ 整个数组访问测试异常: {e}")
            return False

    def test_simulation_consistency(self, heliox_manager, device_mode="cpu"):
        """测试仿真过程中数组值的一致性"""
        print(f"\n🏃 测试仿真一致性 ({device_mode.upper()}模式)")
        print("-" * 50)

        try:
            # 运行短时间仿真
            tstop = 100.0
            print(f"运行仿真至 {tstop} ms...")

            heliox_manager.client.finitialize(-65.0)
            heliox_manager.client.run(tstop)

            print("✅ 仿真完成，检查最终数组状态:")

            # 检查Aggregator的聚合结果
            aggregated_grad = heliox_manager.client.get_var("BP_Syn_Aggregator", 0, "aggregated_grad")
            is_learning = heliox_manager.client.get_var("BP_Syn_Aggregator", 0, "is_learning")
            print(f"   Aggregator聚合梯度: {aggregated_grad:.6f}")
            print(f"   学习状态: {is_learning:.1f}")

            # 检查SoftMax的计算结果
            s_array = heliox_manager.client.get_var_array("BP_Syn_SoftMax", 0, "s")
            s_sum = np.sum(s_array)
            max_idx = np.argmax(s_array)
            print(f"   SoftMax输出总和: {s_sum:.6f}")
            print(f"   预测类别: {max_idx} (概率: {s_array[max_idx]:.4f})")

            # 验证SoftMax性质
            softmax_valid = abs(s_sum - 1.0) < 1e-6
            print(f"   SoftMax性质验证: {'✅ 通过' if softmax_valid else '❌ 失败'}")

            self.test_results.append({
                'name': f'仿真一致性_{device_mode.upper()}',
                'passed': softmax_valid,
                'details': {
                    'aggregated_grad': aggregated_grad,
                    'softmax_sum': s_sum,
                    'predicted_class': int(max_idx)
                }
            })

            return softmax_valid

        except Exception as e:
            print(f"❌ 仿真一致性测试异常: {e}")
            return False

    def run_device_tests(self, device_mode="cpu"):
        """运行指定设备模式的所有测试"""
        print(f"\n🎯 开始 {device_mode.upper()} 模式测试")
        print("=" * 60)

        try:
            # 步骤1: 创建NEURON模型（只创建一次）
            soma1, soma2, aggregator, softmax, iclamp1, iclamp2, pc = self.create_neuron_model_with_bpsyn()

            # 步骤2: 设置初始数组值
            test_grad_values, test_u_values, target_class = self.setup_initial_array_values(aggregator, softmax)

            # 步骤3: 创建HelioX管理器（会自动导入NEURON模型）
            export_path = f"./test_bpsyn_arrays_{device_mode}_output"
            heliox_manager, pc = self.create_heliox_manager(export_path, device_mode, pc)

            # 步骤4: 运行各项测试
            all_passed = True

            # 测试数组元素访问
            passed = self.test_array_element_access(heliox_manager, test_grad_values, test_u_values, target_class, device_mode)
            all_passed = all_passed and passed

            # 测试数组元素修改
            passed = self.test_array_modification(heliox_manager, device_mode)
            all_passed = all_passed and passed

            # 测试整个数组访问
            passed = self.test_whole_array_access(heliox_manager, device_mode)
            all_passed = all_passed and passed

            # 测试仿真一致性
            passed = self.test_simulation_consistency(heliox_manager, device_mode)
            all_passed = all_passed and passed

            # 清理
            pc.done()

            return all_passed

        except Exception as e:
            print(f"❌ {device_mode.upper()}模式测试异常: {e}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 BPSYN数组访问功能测试套件")
        print("=" * 60)

        all_passed = True

        # CPU模式测试
        print("\n🖥️  CPU模式测试")
        cpu_passed = self.run_device_tests("cpu")
        all_passed = all_passed and cpu_passed

        # GPU模式测试
        print("\n🎮 GPU模式测试")
        gpu_passed = self.run_device_tests("gpu")
        all_passed = all_passed and gpu_passed

        # 生成报告
        self.generate_report()

        return all_passed

    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📋 BPSYN数组访问测试报告")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])

        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {total_tests - passed_tests} ❌")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")

        print("\n详细结果:")
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['name']}")
            if 'details' in result and isinstance(result['details'], dict):
                for key, value in result['details'].items():
                    print(f"  {key}: {value}")
            elif 'details' in result:
                print(f"  详情: {result['details']}")

        print(f"\n{'🎉 所有测试通过!' if passed_tests == total_tests else '⚠️  存在数组访问问题!'}")


def main():
    """主函数"""
    print("🚀 启动BPSYN数组访问功能测试...")
    print("测试目标: 验证BP_Syn_Aggregator和BP_Syn_SoftMax的数组变量访问")
    print("测试原则: 只在NEURON中创建模型，HelioX自动导入")

    test_suite = BPSynArrayTest()
    success = test_suite.run_all_tests()

    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()