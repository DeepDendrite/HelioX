#!/usr/bin/env python3
"""
基于旧版notebook的正确VecPlay测试
使用新的HelioXManager API，但参考旧版本的正确参数设置
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

class VecPlayReferenceTest:
    """基于旧版本的VecPlay参考测试"""
    
    def __init__(self):
        self.cells = []
        
    def create_neuron_reference_model(self, gid=0):
        """创建参考NEURON模型（基于旧版本）"""
        print("🧬 创建NEURON参考模型...")
        
        # 清理之前的模型
        for sec in h.allsec():
            h.delete_section(sec=sec)
        
        # 创建ParallelContext
        pc = h.ParallelContext()
        pc.set_gid2node(gid, int(pc.id()))
        
        # 创建soma（参考旧版本参数）
        soma = h.Section(name='soma')
        soma.nseg = 1
        soma.diam = 1  # 旧版本用的是1，不是10
        soma.L = 1     # 旧版本用的是1，不是10
        soma.insert('hh')
        
        # 创建电流钳
        stim = h.IClamp(soma(0.5))
        stim.amp = 0
        stim.dur = 1e9
        stim.delay = 0
        
        # 创建简化的刺激模式（基于旧版本但简化）
        ramps_t = h.Vector()
        ramps_i = h.Vector()
        
        # 简化的刺激模式：参考旧版本的电流幅度
        stimulus_times = [0, 25, 25, 50, 50, 100]
        stimulus_currents = [0, 0, 0.015, 0.015, 0, 0]  # 使用旧版本量级的电流(~0.015 nA)
        
        for t, i in zip(stimulus_times, stimulus_currents):
            ramps_t.append(t)
            ramps_i.append(i)
        
        print(f"NEURON刺激参数:")
        print(f"  时间: {stimulus_times}")
        print(f"  电流: {stimulus_currents}")
        print(f"  soma: diam={soma.diam}, L={soma.L}")
        
        # 重要：使用旧版本的play参数（最后一个参数是1，表示连续插值）
        ramps_i.play(stim._ref_amp, ramps_t, 1)
        
        # 记录电压
        vvec = h.Vector()
        tvec = h.Vector()
        amp_vec = h.Vector()  # 记录电流
        
        vvec.record(soma(0.5)._ref_v)
        tvec.record(h._ref_t)
        amp_vec.record(stim._ref_amp)
        
        # 注册到PC
        soma.push()
        pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
        h.pop_section()
        
        # PC设置
        spike_tvec = h.Vector()
        spike_idvec = h.Vector()
        pc.spike_record(-1, spike_tvec, spike_idvec)
        pc.setup_transfer()
        pc.set_maxstep(10)
        
        # 运行仿真
        h.dt = 0.025
        h.finitialize(-65)
        h.continuerun(100)
        
        # 获取结果
        voltage = np.array(vvec.as_numpy())
        time = np.array(tvec.as_numpy())
        current = np.array(amp_vec.as_numpy())
        
        print(f"NEURON结果:")
        print(f"  电压范围: {voltage.min():.2f} ~ {voltage.max():.2f} mV")
        print(f"  电流范围: {current.min():.6f} ~ {current.max():.6f} nA")
        
        # 检查电流是否正确
        print(f"电流检查:")
        for i, t in enumerate([0, 25, 50, 75, 100]):
            idx = int(t / h.dt)
            if idx < len(current):
                print(f"  t={t:3.0f}ms: I={current[idx]:.6f} nA")
        
        pc.done()
        
        return voltage, time, current, stimulus_times, stimulus_currents
        
    def create_heliox_model(self, stimulus_times, stimulus_currents, device_mode="cpu"):
        """创建HelioX模型"""
        print(f"\n🚀 创建HelioX模型 ({device_mode.upper()}模式)...")
        
        # 清理之前的模型
        for sec in h.allsec():
            h.delete_section(sec=sec)
        
        # 创建ParallelContext
        pc = h.ParallelContext()
        
        # 创建相同的soma（与NEURON参考模型一致）
        soma = h.Section(name='soma_ng')
        soma.nseg = 1
        soma.diam = 1  # 与NEURON一致
        soma.L = 1     # 与NEURON一致
        soma.insert('hh')
        
        # 创建电流钳
        stim = h.IClamp(soma(0.5))
        stim.amp = 0
        stim.dur = 1e9
        stim.delay = 0
        
        # 注册到PC
        gid = 0
        pc.set_gid2node(gid, int(pc.id()))
        soma.push()
        pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
        h.pop_section()
        
        spike_tvec = h.Vector()
        spike_idvec = h.Vector()
        pc.spike_record(-1, spike_tvec, spike_idvec)
        pc.setup_transfer()
        pc.set_maxstep(10)
        
        # 创建HelioX管理器
        heliox_manager = HelioXManager()
        if device_mode == "cpu":
            heliox_manager.set_default_device("cpu")
            heliox_manager.set_default_permute_type(0)
        # GPU使用默认设置
        
        # 创建包装器
        vecplay = heliox_manager.create_vecplay_wrapper(stim, "amp")
        v_monitor = heliox_manager.create_monitor_wrapper(soma(0.5), "v")
        
        print(f"HelioX刺激参数:")
        print(f"  时间: {stimulus_times}")
        print(f"  电流: {stimulus_currents}")
        print(f"  soma: diam={soma.diam}, L={soma.L}")
        
        # 导出和加载模型
        export_path = f"./vecplay_reference_{device_mode}_output"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        
        # 设置VecPlay（使用与NEURON完全相同的参数）
        vecplay.play(stimulus_times, stimulus_currents)
        
        # 运行仿真
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(100.0)
        
        # 获取结果
        voltage = np.array(v_monitor.get_data())
        time = np.arange(0, len(voltage) * 0.025, 0.025)
        
        print(f"HelioX结果:")
        print(f"  电压范围: {voltage.min():.2f} ~ {voltage.max():.2f} mV")
        
        pc.done()
        
        return voltage, time
        
    def compare_and_plot(self, neuron_v, neuron_t, neuron_i, heliox_v, heliox_t, 
                        stimulus_times, stimulus_currents, device_mode):
        """对比并绘制结果"""
        
        # 确保长度一致
        min_len = min(len(neuron_v), len(heliox_v))
        neuron_voltage = neuron_v[:min_len]
        heliox_voltage = heliox_v[:min_len]
        time = neuron_t[:min_len]
        
        # 计算差异
        diff = np.abs(neuron_voltage - heliox_voltage)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n📊 结果对比 ({device_mode.upper()}模式):")
        print(f"  数据点数: {min_len}")
        print(f"  最大差异: {max_diff:.6f} mV")
        print(f"  平均差异: {mean_diff:.6f} mV")
        print(f"  测试结果: {'✅ 通过' if max_diff < 0.001 else '❌ 失败'}")
        
        # 绘制对比图
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 电压对比
        axes[0].plot(time, neuron_voltage, 'b-', linewidth=2, label='NEURON原生', alpha=0.8)
        axes[0].plot(time, heliox_voltage, 'r--', linewidth=2, label=f'HelioX {device_mode.upper()}', alpha=0.8)
        axes[0].set_ylabel('电压 (mV)')
        axes[0].set_title(f'电压对比 ({device_mode.upper()}模式)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 差异图
        axes[1].plot(time, diff, 'g-', linewidth=1)
        axes[1].set_ylabel('电压差异 (mV)')
        axes[1].set_title(f'电压差异 (最大: {max_diff:.6f} mV)')
        axes[1].grid(True, alpha=0.3)
        
        # 电流图
        axes[2].plot(time, neuron_i[:min_len], 'orange', linewidth=2, label='NEURON电流')
        axes[2].plot(stimulus_times, stimulus_currents, 'ko-', markersize=6, label='设定电流')
        axes[2].set_ylabel('电流 (nA)')
        axes[2].set_title('NEURON电流验证')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 局部放大（前60ms）
        mask = time <= 60
        axes[3].plot(time[mask], neuron_voltage[mask], 'b-', linewidth=2, label='NEURON')
        axes[3].plot(time[mask], heliox_voltage[mask], 'r--', linewidth=2, label=f'HelioX {device_mode.upper()}')
        axes[3].set_xlabel('时间 (ms)')
        axes[3].set_ylabel('电压 (mV)')
        axes[3].set_title('前60ms放大对比')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'vecplay_reference_comparison_{device_mode}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 对比图已保存: {filename}")
        plt.close()
        
        return max_diff < 0.001
        
    def run_test(self, device_mode="cpu"):
        """运行单个测试"""
        print(f"\n🎯 VecPlay参考测试 ({device_mode.upper()}模式)")
        print("=" * 60)
        
        # 运行NEURON参考仿真
        neuron_v, neuron_t, neuron_i, stim_times, stim_currents = self.create_neuron_reference_model()
        
        # 检查NEURON电流是否正常
        if neuron_i.max() > 1.0 or neuron_i.min() < -1.0:
            print(f"❌ NEURON电流异常: {neuron_i.min():.6f} ~ {neuron_i.max():.6f} nA")
            return False
            
        # 运行HelioX仿真
        heliox_v, heliox_t = self.create_heliox_model(stim_times, stim_currents, device_mode)
        
        # 对比结果
        passed = self.compare_and_plot(neuron_v, neuron_t, neuron_i, heliox_v, heliox_t, 
                                     stim_times, stim_currents, device_mode)
        
        return passed
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 VecPlay参考测试套件（基于旧版本notebook）")
        print("=" * 80)
        
        # CPU测试
        cpu_passed = self.run_test("cpu")
        
        # GPU测试  
        gpu_passed = self.run_test("gpu")
        
        # 总结
        print(f"\n🎯 测试总结:")
        print(f"  CPU模式: {'✅ 通过' if cpu_passed else '❌ 失败'}")
        print(f"  GPU模式: {'✅ 通过' if gpu_passed else '❌ 失败'}")
        
        if cpu_passed and gpu_passed:
            print(f"\n🎉 所有测试通过！VecPlay功能与NEURON完全兼容！")
        else:
            print(f"\n⚠️  存在兼容性问题，需要进一步调试。")
        
        return cpu_passed and gpu_passed

def main():
    """主函数"""
    test_suite = VecPlayReferenceTest()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()