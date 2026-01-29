#!/usr/bin/env python3
"""
HelioX与NEURON原生Vector.play()兼容性测试

这个测试专门验证HelioX的动态VecPlay功能与NEURON原生Vector.play()的数据一致性。
测试包括CPU和GPU两种模式。
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

class NeuronCompatibilityTest:
    """NEURON兼容性测试类"""
    
    def __init__(self):
        self.test_results = []
        
    def create_neuron_model(self):
        """创建NEURON原生模型进行对比"""
        print("🧬 创建NEURON原生模型...")
        
        # 创建soma
        soma = h.Section(name='soma')
        soma.nseg = 1
        soma.diam = 10
        soma.L = 10
        soma.insert('hh')
        
        # 创建电流钳
        iclamp = h.IClamp(soma(0.5))
        iclamp.amp = 0
        iclamp.dur = 1e9
        iclamp.delay = 0
        
        return soma, iclamp
        
    def create_heliox_model(self, device_mode="cpu"):
        """创建HelioX模型"""
        print(f"⚙️  创建HelioX模型 ({device_mode.upper()}模式)...")
        
        # 创建ParallelContext
        pc = h.ParallelContext()
        
        # 创建soma
        soma = h.Section(name='soma_ng')
        soma.nseg = 1
        soma.diam = 10
        soma.L = 10
        soma.insert('hh')
        
        # 创建电流钳
        iclamp = h.IClamp(soma(0.5))
        iclamp.amp = 0
        iclamp.dur = 1e9
        iclamp.delay = 0
        
        # 注册到ParallelContext
        gid = 0
        pc.set_gid2node(gid, int(pc.id()))
        soma.push()
        pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
        h.pop_section()
        
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
        
        # 创建包装器
        vecplay = heliox_manager.create_vecplay_wrapper(iclamp, "amp")
        v_monitor = heliox_manager.create_monitor_wrapper(soma(0.5), "v")
        
        # 导出和加载模型
        export_path = f"./neuron_compatibility_{device_mode}_output"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        
        return soma, iclamp, vecplay, v_monitor, heliox_manager, pc
        
    def run_neuron_simulation(self, soma, iclamp, tvec, yvec, tstop=100.0):
        """运行NEURON原生仿真"""
        print("🔬 运行NEURON原生仿真...")
        
        # 创建Vector.play()
        neuron_tvec = h.Vector(tvec)
        neuron_yvec = h.Vector(yvec)
        neuron_tvec.play(iclamp._ref_amp, neuron_yvec, True)
        
        # 记录电压
        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        
        # 运行仿真
        h.finitialize(-65)
        h.continuerun(tstop)
        
        # 返回结果
        voltage = np.array(v_vec.as_numpy())
        time = np.array(t_vec.as_numpy())
        
        print(f"   NEURON仿真完成: {len(voltage)} 个数据点")
        print(f"   电压范围: {voltage.min():.2f} 到 {voltage.max():.2f} mV")
        
        return voltage, time
        
    def run_heliox_simulation(self, vecplay, v_monitor, heliox_manager, tvec, yvec, tstop=100.0):
        """运行HelioX仿真"""
        print("🚀 运行HelioX仿真...")
        
        # 设置VecPlay
        vecplay.play(tvec, yvec)
        
        # 运行仿真
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(tstop)
        
        # 获取结果
        voltage = np.array(v_monitor.get_data())
        time = np.arange(0, len(voltage) * 0.025, 0.025)
        
        print(f"   HelioX仿真完成: {len(voltage)} 个数据点")
        print(f"   电压范围: {voltage.min():.2f} 到 {voltage.max():.2f} mV")
        
        return voltage, time
        
    def compare_results(self, neuron_voltage, heliox_voltage, test_name, tolerance=0.001):
        """对比仿真结果"""
        print(f"\n📊 对比结果: {test_name}")
        
        # 确保长度一致
        min_len = min(len(neuron_voltage), len(heliox_voltage))
        neuron_v = neuron_voltage[:min_len]
        heliox_v = heliox_voltage[:min_len]
        
        # 计算差异
        diff = np.abs(neuron_v - heliox_v)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # 判断是否通过
        passed = max_diff < tolerance
        
        print(f"   数据点数: {min_len}")
        print(f"   最大差异: {max_diff:.6f} mV")
        print(f"   平均差异: {mean_diff:.6f} mV")
        print(f"   差异标准差: {std_diff:.6f} mV")
        print(f"   容差阈值: {tolerance:.6f} mV")
        print(f"   测试结果: {'✅ 通过' if passed else '❌ 失败'}")
        
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'tolerance': tolerance,
            'data_points': min_len
        })
        
        return passed, max_diff, diff
        
    def test_basic_stimulus(self, device_mode="cpu"):
        """测试基本刺激模式"""
        print(f"\n🎯 测试基本刺激模式 ({device_mode.upper()})")
        print("-" * 50)
        
        try:
            # 创建模型
            neuron_soma, neuron_iclamp = self.create_neuron_model()
            heliox_soma, heliox_iclamp, vecplay, v_monitor, heliox_manager, pc = self.create_heliox_model(device_mode)
            
            # 定义刺激模式
            tvec = [0, 25, 25, 50, 50, 100]
            yvec = [0, 0, 0.5, 0.5, 0, 0]
            
            # 运行NEURON仿真
            neuron_voltage, neuron_time = self.run_neuron_simulation(neuron_soma, neuron_iclamp, tvec, yvec)
            
            # 运行HelioX仿真
            heliox_voltage, heliox_time = self.run_heliox_simulation(vecplay, v_monitor, heliox_manager, tvec, yvec)
            
            # 对比结果
            passed, max_diff, diff = self.compare_results(
                neuron_voltage, heliox_voltage, 
                f"基本刺激_{device_mode.upper()}", 
                tolerance=0.001
            )
            
            # 清理
            pc.done()
            
            return passed, neuron_voltage, heliox_voltage, diff
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            return False, None, None, None
            
    def test_complex_stimulus(self, device_mode="cpu"):
        """测试复杂刺激模式"""
        print(f"\n🎯 测试复杂刺激模式 ({device_mode.upper()})")
        print("-" * 50)
        
        try:
            # 创建模型
            neuron_soma, neuron_iclamp = self.create_neuron_model()
            heliox_soma, heliox_iclamp, vecplay, v_monitor, heliox_manager, pc = self.create_heliox_model(device_mode)
            
            # 定义复杂刺激模式（双相脉冲）
            tvec = [0, 10, 10, 20, 20, 30, 30, 40, 40, 60, 60, 70, 70, 80, 80, 100]
            yvec = [0, 0, 0.8, 0.8, 0, 0, -0.3, -0.3, 0, 0, 0.5, 0.5, 0, 0, 0, 0]
            
            # 运行NEURON仿真
            neuron_voltage, neuron_time = self.run_neuron_simulation(neuron_soma, neuron_iclamp, tvec, yvec)
            
            # 运行HelioX仿真
            heliox_voltage, heliox_time = self.run_heliox_simulation(vecplay, v_monitor, heliox_manager, tvec, yvec)
            
            # 对比结果
            passed, max_diff, diff = self.compare_results(
                neuron_voltage, heliox_voltage, 
                f"复杂刺激_{device_mode.upper()}", 
                tolerance=0.001
            )
            
            # 清理
            pc.done()
            
            return passed, neuron_voltage, heliox_voltage, diff
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            return False, None, None, None
            
    def test_sine_wave_stimulus(self, device_mode="cpu"):
        """测试正弦波刺激"""
        print(f"\n🎯 测试正弦波刺激 ({device_mode.upper()})")
        print("-" * 50)
        
        try:
            # 创建模型
            neuron_soma, neuron_iclamp = self.create_neuron_model()
            heliox_soma, heliox_iclamp, vecplay, v_monitor, heliox_manager, pc = self.create_heliox_model(device_mode)
            
            # 定义正弦波刺激
            t_points = np.linspace(0, 100, 201)  # 每0.5ms一个点
            tvec = t_points.tolist()
            yvec = [0.3 * np.sin(2 * np.pi * t / 50.0) for t in t_points]  # 20Hz正弦波
            
            # 运行NEURON仿真
            neuron_voltage, neuron_time = self.run_neuron_simulation(neuron_soma, neuron_iclamp, tvec, yvec)
            
            # 运行HelioX仿真
            heliox_voltage, heliox_time = self.run_heliox_simulation(vecplay, v_monitor, heliox_manager, tvec, yvec)
            
            # 对比结果
            passed, max_diff, diff = self.compare_results(
                neuron_voltage, heliox_voltage, 
                f"正弦波刺激_{device_mode.upper()}", 
                tolerance=0.001
            )
            
            # 清理
            pc.done()
            
            return passed, neuron_voltage, heliox_voltage, diff
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            return False, None, None, None
            
    def plot_comparison(self, neuron_voltage, heliox_voltage, diff, test_name, device_mode):
        """绘制对比图"""
        
        min_len = min(len(neuron_voltage), len(heliox_voltage))
        time = np.arange(0, min_len * 0.025, 0.025)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 电压对比
        axes[0].plot(time, neuron_voltage[:min_len], 'b-', linewidth=2, label='NEURON原生')
        axes[0].plot(time, heliox_voltage[:min_len], 'r--', linewidth=2, label=f'HelioX {device_mode.upper()}')
        axes[0].set_ylabel('电压 (mV)')
        axes[0].set_title(f'{test_name} - 电压对比')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 差异图
        axes[1].plot(time, diff[:min_len], 'g-', linewidth=1)
        axes[1].set_ylabel('电压差异 (mV)')
        axes[1].set_title(f'电压差异 (最大: {np.max(diff):.6f} mV)')
        axes[1].grid(True, alpha=0.3)
        
        # 差异直方图
        axes[2].hist(diff[:min_len], bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('电压差异 (mV)')
        axes[2].set_ylabel('频次')
        axes[2].set_title('差异分布直方图')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'neuron_compatibility_{test_name.replace(" ", "_")}_{device_mode}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 对比图已保存: {filename}")
        plt.close()
        
    def run_all_tests(self):
        """运行所有兼容性测试"""
        print("🧪 NEURON兼容性测试套件")
        print("=" * 60)
        
        all_passed = True
        
        # CPU模式测试
        for test_func, test_name in [
            (self.test_basic_stimulus, "基本刺激"),
            (self.test_complex_stimulus, "复杂刺激"),
            (self.test_sine_wave_stimulus, "正弦波刺激")
        ]:
            passed, neuron_v, heliox_v, diff = test_func("cpu")
            # 无论是否通过都生成对比图，方便调试
            if neuron_v is not None and heliox_v is not None:
                self.plot_comparison(neuron_v, heliox_v, diff, test_name, "cpu")
            all_passed = all_passed and passed
            
        # GPU模式测试
        for test_func, test_name in [
            (self.test_basic_stimulus, "基本刺激"),
            (self.test_complex_stimulus, "复杂刺激"),
            (self.test_sine_wave_stimulus, "正弦波刺激")
        ]:
            passed, neuron_v, heliox_v, diff = test_func("gpu")
            # 无论是否通过都生成对比图，方便调试
            if neuron_v is not None and heliox_v is not None:
                self.plot_comparison(neuron_v, heliox_v, diff, test_name, "gpu")
            all_passed = all_passed and passed
            
        # 生成报告
        self.generate_report()
        
        return all_passed
        
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📋 NEURON兼容性测试报告")
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
            print(f"  最大差异: {result['max_diff']:.6f} mV")
            print(f"  平均差异: {result['mean_diff']:.6f} mV")
            print(f"  数据点数: {result['data_points']}")
            
        print(f"\n{'🎉 所有测试通过!' if passed_tests == total_tests else '⚠️  存在兼容性问题!'}")

def main():
    """主函数"""
    test_suite = NeuronCompatibilityTest()
    success = test_suite.run_all_tests()
    
    # 返回状态码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()