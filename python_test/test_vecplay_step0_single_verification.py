#!/usr/bin/env python3
"""
VecPlay验证测试 - 步骤0: 单个IClamp验证
确保单个Vector.play()正常工作，然后逐步添加复杂度
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui

# 设置精确计算
h.usetable_hh = 0

# 添加heliox_wrapper路径
sys.path.insert(0, '$HOME/heliox/python_lib')
from heliox_wrapper import HelioXManager

def test_single_vector_play():
    """测试单个Vector.play()是否正常"""
    print("🔍 验证单个Vector.play()...")
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建最简单的模型
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 1
    soma.L = 1
    soma.insert('hh')
    
    iclamp = h.IClamp(soma(0.5))
    iclamp.amp = 0
    iclamp.dur = 1e9
    iclamp.delay = 0
    
    # 简单刺激
    times = [0, 25, 25, 50, 50, 100]
    currents = [0, 0, 0.015, 0.015, 0, 0]
    
    print(f"设定刺激: {times} -> {currents}")
    
    # Vector.play()
    tvec = h.Vector(times)
    ivec = h.Vector(currents)
    tvec.play(iclamp._ref_amp, ivec, 1)
    
    # 记录
    v_vec = h.Vector()
    i_vec = h.Vector()
    t_vec = h.Vector()
    
    v_vec.record(soma(0.5)._ref_v)
    i_vec.record(iclamp._ref_amp)
    t_vec.record(h._ref_t)
    
    # 运行
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
    
    # 检查关键时间点的电流
    print(f"关键时间点电流检查:")
    for check_time in [0, 25, 37.5, 50, 75, 100]:
        idx = int(check_time / h.dt)
        if idx < len(current):
            print(f"  t={check_time:5.1f}ms: I={current[idx]:.6f} nA")
    
    # 判断是否正常
    current_ok = current.max() < 1.0 and abs(current.max() - 0.015) < 0.001
    voltage_ok = voltage.max() < 100
    
    print(f"单个Vector.play(): {'✅ 正常' if current_ok and voltage_ok else '❌ 异常'}")
    
    return current_ok and voltage_ok, voltage, current, time

def test_two_separate_iclamps():
    """测试两个完全独立的IClamp（不同细胞）"""
    print("\n🔍 验证两个独立IClamp...")
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建两个完全独立的细胞
    soma1 = h.Section(name='soma1')
    soma1.nseg = 1
    soma1.diam = 1
    soma1.L = 1
    soma1.insert('hh')
    
    soma2 = h.Section(name='soma2')
    soma2.nseg = 1
    soma2.diam = 1
    soma2.L = 1
    soma2.insert('hh')
    
    # 两个独立的IClamp
    iclamp1 = h.IClamp(soma1(0.5))
    iclamp1.amp = 0
    iclamp1.dur = 1e9
    iclamp1.delay = 0
    
    iclamp2 = h.IClamp(soma2(0.5))
    iclamp2.amp = 0
    iclamp2.dur = 1e9
    iclamp2.delay = 0
    
    # 简单的不重叠刺激
    times1 = [0, 25, 25, 50, 50, 100]
    currents1 = [0, 0, 0.010, 0.010, 0, 0]
    
    times2 = [0, 50, 50, 75, 75, 100]
    currents2 = [0, 0, 0.005, 0.005, 0, 0]
    
    print(f"IClamp1刺激: {currents1}")
    print(f"IClamp2刺激: {currents2}")
    
    # Vector.play()
    tvec1 = h.Vector(times1)
    ivec1 = h.Vector(currents1)
    tvec1.play(iclamp1._ref_amp, ivec1, 1)
    
    tvec2 = h.Vector(times2)
    ivec2 = h.Vector(currents2)
    tvec2.play(iclamp2._ref_amp, ivec2, 1)
    
    # 记录
    v1_vec = h.Vector()
    v2_vec = h.Vector()
    i1_vec = h.Vector()
    i2_vec = h.Vector()
    
    v1_vec.record(soma1(0.5)._ref_v)
    v2_vec.record(soma2(0.5)._ref_v)
    i1_vec.record(iclamp1._ref_amp)
    i2_vec.record(iclamp2._ref_amp)
    
    # 运行
    h.dt = 0.025
    h.finitialize(-65)
    h.continuerun(100)
    
    # 结果
    voltage1 = np.array(v1_vec.as_numpy())
    voltage2 = np.array(v2_vec.as_numpy())
    current1 = np.array(i1_vec.as_numpy())
    current2 = np.array(i2_vec.as_numpy())
    
    print(f"结果:")
    print(f"  细胞1电压: {voltage1.min():.2f} ~ {voltage1.max():.2f} mV")
    print(f"  细胞2电压: {voltage2.min():.2f} ~ {voltage2.max():.2f} mV")
    print(f"  电流1: {current1.min():.6f} ~ {current1.max():.6f} nA")
    print(f"  电流2: {current2.min():.6f} ~ {current2.max():.6f} nA")
    
    # 判断
    current1_ok = current1.max() < 1.0 and abs(current1.max() - 0.010) < 0.002
    current2_ok = current2.max() < 1.0 and abs(current2.max() - 0.005) < 0.002
    voltage_ok = voltage1.max() < 100 and voltage2.max() < 100
    
    print(f"两个独立IClamp: {'✅ 正常' if current1_ok and current2_ok and voltage_ok else '❌ 异常'}")
    
    return current1_ok and current2_ok and voltage_ok, voltage1, voltage2, current1, current2

def test_same_cell_two_iclamps():
    """测试同一个细胞的两个IClamp"""
    print("\n🔍 验证同细胞双IClamp...")
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 单个细胞
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 1
    soma.L = 1
    soma.insert('hh')
    
    # 两个IClamp在同一个细胞
    iclamp1 = h.IClamp(soma(0.5))
    iclamp1.amp = 0
    iclamp1.dur = 1e9
    iclamp1.delay = 0
    
    iclamp2 = h.IClamp(soma(0.5))  # 相同位置
    iclamp2.amp = 0
    iclamp2.dur = 1e9
    iclamp2.delay = 0
    
    # 非常保守的刺激 - 绝对不重叠
    times1 = [0, 20, 20, 40, 40, 100]
    currents1 = [0, 0, 0.008, 0.008, 0, 0]  # 更小的电流
    
    times2 = [0, 60, 60, 80, 80, 100]
    currents2 = [0, 0, 0.004, 0.004, 0, 0]  # 更小的电流
    
    print(f"IClamp1刺激: {currents1}")
    print(f"IClamp2刺激: {currents2}")
    
    # Vector.play()
    tvec1 = h.Vector(times1)
    ivec1 = h.Vector(currents1)
    tvec1.play(iclamp1._ref_amp, ivec1, 1)
    
    tvec2 = h.Vector(times2)
    ivec2 = h.Vector(currents2)
    tvec2.play(iclamp2._ref_amp, ivec2, 1)
    
    # 记录
    v_vec = h.Vector()
    i1_vec = h.Vector()
    i2_vec = h.Vector()
    
    v_vec.record(soma(0.5)._ref_v)
    i1_vec.record(iclamp1._ref_amp)
    i2_vec.record(iclamp2._ref_amp)
    
    # 运行
    h.dt = 0.025
    h.finitialize(-65)
    h.continuerun(100)
    
    # 结果
    voltage = np.array(v_vec.as_numpy())
    current1 = np.array(i1_vec.as_numpy())
    current2 = np.array(i2_vec.as_numpy())
    
    print(f"结果:")
    print(f"  电压: {voltage.min():.2f} ~ {voltage.max():.2f} mV")
    print(f"  电流1: {current1.min():.6f} ~ {current1.max():.6f} nA")
    print(f"  电流2: {current2.min():.6f} ~ {current2.max():.6f} nA")
    
    # 检查关键时间点
    print("关键时间点检查:")
    for t in [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        idx = int(t / h.dt)
        if idx < len(current1):
            print(f"  t={t:3.0f}ms: I1={current1[idx]:.6f}, I2={current2[idx]:.6f} nA")
    
    # 判断
    current1_ok = current1.max() < 1.0
    current2_ok = current2.max() < 1.0  
    voltage_ok = voltage.max() < 100
    
    result = current1_ok and current2_ok and voltage_ok
    print(f"同细胞双IClamp: {'✅ 正常' if result else '❌ 异常'}")
    
    return result, voltage, current1, current2

def plot_verification_results(single_data, separate_data, same_cell_data):
    """绘制验证结果"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 单个Vector.play()
    if single_data[0]:
        voltage, current, time = single_data[1:]
        axes[0,0].plot(time, voltage, 'b-', linewidth=2)
        axes[0,0].set_title('单个Vector.play() - 电压')
        axes[0,0].set_ylabel('电压 (mV)')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(time, current, 'orange', linewidth=2)
        axes[0,1].set_title('单个Vector.play() - 电流')
        axes[0,1].set_ylabel('电流 (nA)')
        axes[0,1].grid(True, alpha=0.3)
    
    # 两个独立IClamp
    if separate_data[0]:
        v1, v2, i1, i2 = separate_data[1:]
        time = np.arange(0, len(v1) * 0.025, 0.025)
        
        axes[1,0].plot(time, v1, 'b-', linewidth=2, label='细胞1')
        axes[1,0].plot(time, v2, 'r-', linewidth=2, label='细胞2')
        axes[1,0].set_title('两个独立IClamp - 电压')
        axes[1,0].set_ylabel('电压 (mV)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(time, i1, 'orange', linewidth=2, label='电流1')
        axes[1,1].plot(time, i2, 'purple', linewidth=2, label='电流2')
        axes[1,1].set_title('两个独立IClamp - 电流')
        axes[1,1].set_ylabel('电流 (nA)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # 同细胞双IClamp
    if same_cell_data[0]:
        voltage, current1, current2 = same_cell_data[1:]
        time = np.arange(0, len(voltage) * 0.025, 0.025)
        
        axes[2,0].plot(time, voltage, 'g-', linewidth=2)
        axes[2,0].set_title('同细胞双IClamp - 电压')
        axes[2,0].set_xlabel('时间 (ms)')
        axes[2,0].set_ylabel('电压 (mV)')
        axes[2,0].grid(True, alpha=0.3)
        
        axes[2,1].plot(time, current1, 'orange', linewidth=2, label='IClamp1')
        axes[2,1].plot(time, current2, 'purple', linewidth=2, label='IClamp2')
        axes[2,1].set_title('同细胞双IClamp - 电流')
        axes[2,1].set_xlabel('时间 (ms)')
        axes[2,1].set_ylabel('电流 (nA)')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vecplay_verification_steps.png', dpi=300, bbox_inches='tight')
    print("📈 验证图已保存: vecplay_verification_steps.png")
    plt.close()

def main():
    """主函数 - 逐步验证"""
    print("🧪 VecPlay逐步验证测试")
    print("=" * 60)
    
    # 步骤1: 单个Vector.play()
    single_result = test_single_vector_play()
    
    # 步骤2: 两个独立IClamp
    separate_result = test_two_separate_iclamps()
    
    # 步骤3: 同细胞双IClamp
    same_cell_result = test_same_cell_two_iclamps()
    
    # 绘制结果
    plot_verification_results(single_result, separate_result, same_cell_result)
    
    # 总结
    print(f"\n🎯 验证总结:")
    print(f"  单个Vector.play(): {'✅ 正常' if single_result[0] else '❌ 异常'}")
    print(f"  两个独立IClamp: {'✅ 正常' if separate_result[0] else '❌ 异常'}")
    print(f"  同细胞双IClamp: {'✅ 正常' if same_cell_result[0] else '❌ 异常'}")
    
    if all([single_result[0], separate_result[0], same_cell_result[0]]):
        print(f"\n🎉 所有NEURON基准测试通过！可以继续HelioX对比测试。")
        return True
    else:
        print(f"\n❌ NEURON基准测试失败！需要先解决NEURON Vector.play()问题。")
        return False

if __name__ == "__main__":
    main()