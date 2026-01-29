#!/usr/bin/env python3
"""
VecPlay正确性测试

正确的兼容性测试方法：
1. 独立创建NEURON模型，用Vector.play()测试
2. 独立创建HelioX模型，用VecPlay测试  
3. 使用相同的刺激参数，对比最终的电压结果
4. 测试CPU和GPU两种模式
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui
from neuron.units import ms, mV

# 设置精确计算
h.usetable_hh = 0

# 添加heliox_wrapper路径
sys.path.insert(0, '$HOME/heliox/python_lib')
from heliox_wrapper import HelioXManager

class VecPlayCorrectnessTest:
    """VecPlay正确性测试类"""
    
    def __init__(self):
        self.test_results = []
        
    def run_neuron_simulation(self, stim_tvec, stim_yvec, tstop=100.0):
        """运行NEURON原生仿真（独立session）"""
        print("🧬 NEURON原生仿真...")
        
        # 清理之前的NEURON状态
        for sec in h.allsec():
            h.delete_section(sec=sec)
        
        # 创建新的NEURON模型
        soma = h.Section(name='soma_neuron')
        soma.nseg = 1
        soma.diam = 10
        soma.L = 10
        soma.insert('hh')
        
        iclamp = h.IClamp(soma(0.5))
        iclamp.amp = 0
        iclamp.dur = 1e9
        iclamp.delay = 0
        
        # 创建Vector.play()
        neuron_tvec = h.Vector(stim_tvec)
        neuron_yvec = h.Vector(stim_yvec)
        neuron_tvec.play(iclamp._ref_amp, neuron_yvec, True)
        
        # 记录电压
        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        
        # 设置参数
        h.dt = 0.025
        h.steps_per_ms = 1.0 / h.dt
        
        # 运行仿真
        h.finitialize(-65)
        h.continuerun(tstop)
        
        # 获取结果
        voltage = np.array(v_vec.as_numpy())
        time = np.array(t_vec.as_numpy())
        
        print(f"   NEURON结果: {len(voltage)} 点, 电压范围: {voltage.min():.2f}~{voltage.max():.2f} mV")
        
        return voltage, time
        
    def run_heliox_simulation(self, stim_tvec, stim_yvec, device_mode="cpu", tstop=100.0):
        """运行HelioX仿真（独立session）"""
        print(f"🚀 HelioX仿真 ({device_mode.upper()}模式)...")
        
        # 清理之前的NEURON状态
        for sec in h.allsec():
            h.delete_section(sec=sec)
        
        # 创建ParallelContext
        pc = h.ParallelContext()
        
        # 创建新的NEURON模型（用于导出到HelioX）
        soma = h.Section(name='soma_heliox')
        soma.nseg = 1
        soma.diam = 10
        soma.L = 10
        soma.insert('hh')
        
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
        export_path = f"./vecplay_correctness_{device_mode}_output"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        
        # 设置VecPlay
        vecplay.play(stim_tvec, stim_yvec)
        
        # 运行仿真
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(tstop)
        
        # 获取结果
        voltage = np.array(v_monitor.get_data())
        time = np.arange(0, len(voltage) * 0.025, 0.025)
        
        print(f"   HelioX结果: {len(voltage)} 点, 电压范围: {voltage.min():.2f}~{voltage.max():.2f} mV")
        
        # 清理
        pc.done()
        
        return voltage, time
        
    def compare_results(self, neuron_voltage, heliox_voltage, test_name, tolerance=0.001):
        """对比仿真结果"""
        print(f"\n📊 结果对比: {test_name}")
        
        # 确保长度一致
        min_len = min(len(neuron_voltage), len(heliox_voltage))
        neuron_v = neuron_voltage[:min_len]
        heliox_v = heliox_voltage[:min_len]
        
        # 计算差异
        diff = np.abs(neuron_v - heliox_v)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # 计算相对误差
        neuron_range = np.max(neuron_v) - np.min(neuron_v)
        relative_error = max_diff / neuron_range if neuron_range > 0 else 0
        
        # 判断是否通过
        passed = max_diff < tolerance
        
        print(f"   数据点数: {min_len}")
        print(f"   最大差异: {max_diff:.6f} mV")
        print(f"   平均差异: {mean_diff:.6f} mV")
        print(f"   差异标准差: {std_diff:.6f} mV")
        print(f"   相对误差: {relative_error:.6f}")
        print(f"   容差阈值: {tolerance:.6f} mV")
        print(f"   测试结果: {'✅ 通过' if passed else '❌ 失败'}")
        
        # 记录结果
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'relative_error': relative_error,
            'tolerance': tolerance
        })
        
        return passed, diff
        
    def plot_comparison(self, neuron_voltage, heliox_voltage, neuron_time, heliox_time, 
                       diff, test_name, stim_tvec, stim_yvec):
        """绘制详细对比图"""
        
        min_len = min(len(neuron_voltage), len(heliox_voltage))
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 完整电压对比
        axes[0].plot(neuron_time[:min_len], neuron_voltage[:min_len], 'b-', linewidth=2, label='NEURON原生', alpha=0.8)
        axes[0].plot(heliox_time[:min_len], heliox_voltage[:min_len], 'r--', linewidth=2, label='HelioX', alpha=0.8)
        axes[0].set_ylabel('电压 (mV)')
        axes[0].set_title(f'{test_name} - 完整电压对比')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 差异图
        axes[1].plot(neuron_time[:min_len], diff[:min_len], 'g-', linewidth=1)
        axes[1].set_ylabel('电压差异 (mV)')
        axes[1].set_title(f'电压差异 (最大: {np.max(diff):.6f} mV)')
        axes[1].grid(True, alpha=0.3)
        
        # 刺激参数
        axes[2].plot(stim_tvec, stim_yvec, 'ko-', linewidth=2, markersize=6)
        axes[2].set_ylabel('电流 (nA)')
        axes[2].set_title('刺激参数')
        axes[2].grid(True, alpha=0.3)
        
        # 局部放大（前50ms）
        mask = neuron_time[:min_len] <= 50
        if np.any(mask):
            axes[3].plot(neuron_time[:min_len][mask], neuron_voltage[:min_len][mask], 'b-', linewidth=2, label='NEURON')
            axes[3].plot(heliox_time[:min_len][mask], heliox_voltage[:min_len][mask], 'r--', linewidth=2, label='HelioX')
            axes[3].set_xlabel('时间 (ms)')
            axes[3].set_ylabel('电压 (mV)')
            axes[3].set_title('前50ms放大图')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'vecplay_correctness_{test_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 对比图已保存: {filename}")
        plt.close()
        
    def test_basic_pulse(self, device_mode="cpu"):
        """测试基本脉冲刺激"""
        print(f"\n🎯 测试基本脉冲刺激 ({device_mode.upper()})")
        print("-" * 60)
        
        # 定义刺激参数
        stim_tvec = [0, 25, 25, 50, 50, 100]
        stim_yvec = [0, 0, 0.5, 0.5, 0, 0]
        
        # 分别运行仿真
        neuron_v, neuron_t = self.run_neuron_simulation(stim_tvec, stim_yvec)
        heliox_v, heliox_t = self.run_heliox_simulation(stim_tvec, stim_yvec, device_mode)
        
        # 对比结果
        passed, diff = self.compare_results(neuron_v, heliox_v, f"基本脉冲_{device_mode.upper()}")
        
        # 绘制对比图
        self.plot_comparison(neuron_v, heliox_v, neuron_t, heliox_t, diff, 
                           f"基本脉冲_{device_mode.upper()}", stim_tvec, stim_yvec)
        
        return passed
        
    def test_biphasic_pulse(self, device_mode="cpu"):
        """测试双相脉冲刺激"""
        print(f"\n🎯 测试双相脉冲刺激 ({device_mode.upper()})")
        print("-" * 60)
        
        # 定义双相脉冲
        stim_tvec = [0, 20, 20, 30, 30, 40, 40, 50, 50, 100]
        stim_yvec = [0, 0, 0.8, 0.8, 0, 0, -0.3, -0.3, 0, 0]
        
        # 分别运行仿真
        neuron_v, neuron_t = self.run_neuron_simulation(stim_tvec, stim_yvec)
        heliox_v, heliox_t = self.run_heliox_simulation(stim_tvec, stim_yvec, device_mode)
        
        # 对比结果
        passed, diff = self.compare_results(neuron_v, heliox_v, f"双相脉冲_{device_mode.upper()}")
        
        # 绘制对比图
        self.plot_comparison(neuron_v, heliox_v, neuron_t, heliox_t, diff, 
                           f"双相脉冲_{device_mode.upper()}", stim_tvec, stim_yvec)
        
        return passed
        
    def test_ramp_stimulus(self, device_mode="cpu"):
        """测试斜坡刺激"""
        print(f"\n🎯 测试斜坡刺激 ({device_mode.upper()})")
        print("-" * 60)
        
        # 定义斜坡刺激
        stim_tvec = [0, 20, 60, 80, 100]
        stim_yvec = [0, 0, 1.0, 0, 0]  # 线性上升下降
        
        # 分别运行仿真
        neuron_v, neuron_t = self.run_neuron_simulation(stim_tvec, stim_yvec)
        heliox_v, heliox_t = self.run_heliox_simulation(stim_tvec, stim_yvec, device_mode)
        
        # 对比结果
        passed, diff = self.compare_results(neuron_v, heliox_v, f"斜坡刺激_{device_mode.upper()}")
        
        # 绘制对比图
        self.plot_comparison(neuron_v, heliox_v, neuron_t, heliox_t, diff, 
                           f"斜坡刺激_{device_mode.upper()}", stim_tvec, stim_yvec)
        
        return passed
        
    def test_high_frequency_stimulus(self, device_mode="cpu"):
        """测试高频刺激"""
        print(f"\n🎯 测试高频刺激 ({device_mode.upper()})")
        print("-" * 60)
        
        # 定义高频脉冲序列（10Hz）
        stim_tvec = []
        stim_yvec = []
        
        for i in range(10):  # 10个脉冲
            t_start = 10 + i * 10
            t_end = t_start + 2
            stim_tvec.extend([t_start, t_end])
            stim_yvec.extend([0.3, 0.3])
            if i < 9:  # 除了最后一个
                stim_tvec.extend([t_end, t_start + 10])
                stim_yvec.extend([0, 0])
        
        # 添加结束点
        stim_tvec.extend([100])
        stim_yvec.extend([0])
        
        # 分别运行仿真
        neuron_v, neuron_t = self.run_neuron_simulation(stim_tvec, stim_yvec)
        heliox_v, heliox_t = self.run_heliox_simulation(stim_tvec, stim_yvec, device_mode)
        
        # 对比结果
        passed, diff = self.compare_results(neuron_v, heliox_v, f"高频刺激_{device_mode.upper()}")
        
        # 绘制对比图
        self.plot_comparison(neuron_v, heliox_v, neuron_t, heliox_t, diff, 
                           f"高频刺激_{device_mode.upper()}", stim_tvec, stim_yvec)
        
        return passed
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 VecPlay正确性测试套件")
        print("=" * 80)
        
        all_passed = True
        
        # 测试列表
        test_cases = [
            (self.test_basic_pulse, "基本脉冲"),
            (self.test_biphasic_pulse, "双相脉冲"),
            (self.test_ramp_stimulus, "斜坡刺激"),
            (self.test_high_frequency_stimulus, "高频刺激")
        ]
        
        # CPU模式测试
        print("\n" + "="*40 + " CPU模式测试 " + "="*40)
        for test_func, test_name in test_cases:
            passed = test_func("cpu")
            all_passed = all_passed and passed
            
        # GPU模式测试
        print("\n" + "="*40 + " GPU模式测试 " + "="*40)
        for test_func, test_name in test_cases:
            passed = test_func("gpu")
            all_passed = all_passed and passed
            
        # 生成总报告
        self.generate_final_report()
        
        return all_passed
        
    def generate_final_report(self):
        """生成最终测试报告"""
        print("\n" + "=" * 80)
        print("📋 VecPlay正确性测试最终报告")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {total_tests - passed_tests} ❌")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n详细结果:")
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['name']}")
            print(f"    最大差异: {result['max_diff']:.6f} mV")
            print(f"    平均差异: {result['mean_diff']:.6f} mV")
            print(f"    相对误差: {result['relative_error']:.6f}")
            
        if passed_tests == total_tests:
            print(f"\n🎉 所有测试通过！VecPlay功能与NEURON完全兼容！")
        else:
            print(f"\n⚠️  存在{total_tests - passed_tests}个测试失败，需要检查兼容性问题。")

def main():
    """主函数"""
    test_suite = VecPlayCorrectnessTest()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()