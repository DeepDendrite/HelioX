#!/usr/bin/env python3
"""
修正的NEURON Vector.play()测试
基于旧notebook的正确用法
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui

# 设置精确计算
h.usetable_hh = 0

def test_correct_neuron_vecplay():
    """用正确的方法测试NEURON Vector.play()"""
    print("🔍 正确的NEURON Vector.play()测试")
    print("=" * 50)
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建模型（使用旧notebook的参数）
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 1
    soma.L = 1
    soma.insert('hh')
    
    stim = h.IClamp(soma(0.5))
    stim.amp = 0
    stim.dur = 1e9
    stim.delay = 0
    
    # 关键修正：使用旧notebook的正确方法
    ramps_t = h.Vector()
    ramps_i = h.Vector()
    
    # 简单的刺激模式（模仿旧notebook）
    stimulus_times = [0, 25, 25, 50, 50, 100]
    stimulus_currents = [0, 0, 0.015, 0.015, 0, 0]
    
    # 用append方法构建向量（而不是构造函数）
    for t, i in zip(stimulus_times, stimulus_currents):
        ramps_t.append(t)
        ramps_i.append(i)
    
    print(f"刺激模式: {stimulus_times} -> {stimulus_currents}")
    
    # 关键修正：正确的play方法
    # 旧notebook用法：self.ramps_i.play(self.stim._ref_amp, self.ramps_t, 1)
    # 电流向量.play(目标变量, 时间向量, 插值模式)
    ramps_i.play(stim._ref_amp, ramps_t, 1)
    
    # 记录
    v_vec = h.Vector()
    i_vec = h.Vector()
    t_vec = h.Vector()
    
    v_vec.record(soma(0.5)._ref_v)
    i_vec.record(stim._ref_amp)
    t_vec.record(h._ref_t)
    
    # 运行（使用旧notebook的设置）
    h.dt = 0.025
    h.finitialize(-65)
    h.continuerun(100)
    
    # 结果
    voltage = np.array(v_vec.as_numpy())
    current = np.array(i_vec.as_numpy())
    time = np.array(t_vec.as_numpy())
    
    print(f"结果:")
    print(f"  电压范围: {voltage.min():.2f} ~ {voltage.max():.2f} mV")
    print(f"  电流范围: {current.min():.6f} ~ {current.max():.6f} nA")
    
    # 检查关键时间点
    print(f"关键时间点检查:")
    for check_time in [0, 25, 37.5, 50, 75, 100]:
        idx = int(check_time / h.dt)
        if idx < len(current):
            expected = 0.015 if 25 <= check_time <= 50 else 0
            print(f"  t={check_time:5.1f}ms: I={current[idx]:.6f} nA (期望: {expected:.3f})")
    
    # 判断是否成功
    # 检查电流是否在正确的时间段内
    current_25_50 = current[int(25/h.dt):int(50/h.dt)]
    current_correct = np.all(np.abs(current_25_50 - 0.015) < 0.001)
    
    voltage_reasonable = voltage.max() < 100 and voltage.max() > -50
    
    success = current_correct and voltage_reasonable
    print(f"正确的Vector.play(): {'✅ 成功' if success else '❌ 失败'}")
    
    return success, voltage, current, time

def test_dual_iclamp_correct():
    """用正确方法测试双IClamp"""
    print("\n🔍 正确的双IClamp测试")
    print("=" * 50)
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建模型
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 1
    soma.L = 1
    soma.insert('hh')
    
    # 两个IClamp
    stim1 = h.IClamp(soma(0.5))
    stim1.amp = 0
    stim1.dur = 1e9
    stim1.delay = 0
    
    stim2 = h.IClamp(soma(0.5))
    stim2.amp = 0
    stim2.dur = 1e9
    stim2.delay = 0
    
    # 正确构建向量
    ramps_t1 = h.Vector()
    ramps_i1 = h.Vector()
    ramps_t2 = h.Vector()
    ramps_i2 = h.Vector()
    
    # 非重叠刺激
    times1 = [0, 20, 20, 40, 40, 100]
    currents1 = [0, 0, 0.010, 0.010, 0, 0]
    
    times2 = [0, 60, 60, 80, 80, 100]
    currents2 = [0, 0, 0.005, 0.005, 0, 0]
    
    # append方法构建
    for t, i in zip(times1, currents1):
        ramps_t1.append(t)
        ramps_i1.append(i)
        
    for t, i in zip(times2, currents2):
        ramps_t2.append(t)
        ramps_i2.append(i)
    
    print(f"IClamp1: {currents1}")
    print(f"IClamp2: {currents2}")
    
    # 正确的play方法
    ramps_i1.play(stim1._ref_amp, ramps_t1, 1)
    ramps_i2.play(stim2._ref_amp, ramps_t2, 1)
    
    # 记录
    v_vec = h.Vector()
    i1_vec = h.Vector()
    i2_vec = h.Vector()
    
    v_vec.record(soma(0.5)._ref_v)
    i1_vec.record(stim1._ref_amp)
    i2_vec.record(stim2._ref_amp)
    
    # 运行
    h.dt = 0.025
    h.finitialize(-65)
    h.continuerun(100)
    
    # 结果
    voltage = np.array(v_vec.as_numpy())
    current1 = np.array(i1_vec.as_numpy())
    current2 = np.array(i2_vec.as_numpy())
    
    print(f"结果:")
    print(f"  电压范围: {voltage.min():.2f} ~ {voltage.max():.2f} mV")
    print(f"  电流1: {current1.min():.6f} ~ {current1.max():.6f} nA")
    print(f"  电流2: {current2.min():.6f} ~ {current2.max():.6f} nA")
    
    # 检查关键时间点
    print(f"关键时间点检查:")
    for t in [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        idx = int(t / h.dt)
        if idx < len(current1):
            print(f"  t={t:3.0f}ms: I1={current1[idx]:.6f}, I2={current2[idx]:.6f} nA")
    
    # 判断成功
    i1_correct = np.all(np.abs(current1[int(20/h.dt):int(40/h.dt)] - 0.010) < 0.001)
    i2_correct = np.all(np.abs(current2[int(60/h.dt):int(80/h.dt)] - 0.005) < 0.001)
    voltage_ok = voltage.max() < 100
    
    success = i1_correct and i2_correct and voltage_ok
    print(f"正确的双IClamp: {'✅ 成功' if success else '❌ 失败'}")
    
    return success, voltage, current1, current2

def plot_correct_results(single_data, dual_data):
    """绘制正确结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 单个IClamp
    if single_data[0]:
        voltage, current, time = single_data[1:]
        
        axes[0,0].plot(time, voltage, 'b-', linewidth=2)
        axes[0,0].set_title('正确的单IClamp - 电压')
        axes[0,0].set_ylabel('电压 (mV)')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(time, current, 'orange', linewidth=2)
        axes[0,1].set_title('正确的单IClamp - 电流')
        axes[0,1].set_ylabel('电流 (nA)')
        axes[0,1].grid(True, alpha=0.3)
    
    # 双IClamp
    if dual_data[0]:
        voltage, current1, current2 = dual_data[1:]
        time = np.arange(0, len(voltage) * 0.025, 0.025)
        
        axes[1,0].plot(time, voltage, 'g-', linewidth=2)
        axes[1,0].set_title('正确的双IClamp - 电压')
        axes[1,0].set_xlabel('时间 (ms)')
        axes[1,0].set_ylabel('电压 (mV)')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(time, current1, 'orange', linewidth=2, label='IClamp1')
        axes[1,1].plot(time, current2, 'purple', linewidth=2, label='IClamp2')
        axes[1,1].set_title('正确的双IClamp - 电流')
        axes[1,1].set_xlabel('时间 (ms)')
        axes[1,1].set_ylabel('电流 (nA)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correct_neuron_vecplay.png', dpi=300, bbox_inches='tight')
    print("📈 正确结果图已保存: correct_neuron_vecplay.png")
    plt.close()

def main():
    """主函数"""
    print("🧪 正确的NEURON Vector.play()测试")
    print("=" * 60)
    
    # 测试单个IClamp
    single_result = test_correct_neuron_vecplay()
    
    # 测试双IClamp
    dual_result = test_dual_iclamp_correct()
    
    # 绘制结果
    plot_correct_results(single_result, dual_result)
    
    # 总结
    print(f"\n🎯 修正后的测试结果:")
    print(f"  单IClamp: {'✅ 成功' if single_result[0] else '❌ 失败'}")
    print(f"  双IClamp: {'✅ 成功' if dual_result[0] else '❌ 失败'}")
    
    if single_result[0] and dual_result[0]:
        print(f"\n🎉 NEURON Vector.play()修正成功！现在可以正确进行HelioX对比了。")
        return True
    else:
        print(f"\n❌ 仍有问题需要解决。")
        return False

if __name__ == "__main__":
    main()