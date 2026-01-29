#!/usr/bin/env python3
"""
VecPlay逐步测试 - 步骤1: 单细胞双IClamp
先确保这个最基本的场景能工作
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

def test_dual_iclamp_simple():
    """最简单的双IClamp测试"""
    print("🎯 单细胞双IClamp测试")
    print("=" * 50)
    
    # === NEURON参考模型 ===
    print("🧬 NEURON参考模型...")
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建模型 - 使用成功的参数
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 1  # 成功的参数
    soma.L = 1     # 成功的参数
    soma.insert('hh')
    
    # 创建两个IClamp - 但用非常简单的设置
    iclamp1 = h.IClamp(soma(0.5))  # 都放在中间，避免位置问题
    iclamp1.amp = 0
    iclamp1.dur = 1e9
    iclamp1.delay = 0
    
    iclamp2 = h.IClamp(soma(0.5))  # 相同位置
    iclamp2.amp = 0
    iclamp2.dur = 1e9
    iclamp2.delay = 0
    
    # 非常简单的刺激 - 不重叠，避免冲突
    # IClamp1: 前半段
    times1 = [0, 25, 25, 50, 50, 100]
    currents1 = [0, 0, 0.010, 0.010, 0, 0]  # 小电流
    
    # IClamp2: 后半段  
    times2 = [0, 50, 50, 75, 75, 100]
    currents2 = [0, 0, 0.005, 0.005, 0, 0]  # 更小的电流
    
    print(f"IClamp1刺激: {times1} -> {currents1}")
    print(f"IClamp2刺激: {times2} -> {currents2}")
    
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
    
    # 获取结果
    neuron_v = np.array(v_vec.as_numpy())
    neuron_i1 = np.array(i1_vec.as_numpy())
    neuron_i2 = np.array(i2_vec.as_numpy())
    
    print(f"NEURON结果:")
    print(f"  电压范围: {neuron_v.min():.2f} ~ {neuron_v.max():.2f} mV")
    print(f"  IClamp1电流: {neuron_i1.min():.6f} ~ {neuron_i1.max():.6f} nA")
    print(f"  IClamp2电流: {neuron_i2.min():.6f} ~ {neuron_i2.max():.6f} nA")
    
    # 检查电流是否异常
    if neuron_i1.max() > 1.0 or neuron_i2.max() > 1.0:
        print("❌ NEURON电流异常！")
        return False
    
    if neuron_v.max() > 100:
        print("❌ NEURON电压异常！")
        return False
    
    # === HelioX模型 ===
    print("\n🚀 HelioX模型...")
    
    # 清理
    for sec in h.allsec():
        h.delete_section(sec=sec)
    
    # 创建ParallelContext
    pc = h.ParallelContext()
    
    # 创建相同模型
    soma_ng = h.Section(name='soma_ng')
    soma_ng.nseg = 1
    soma_ng.diam = 1
    soma_ng.L = 1
    soma_ng.insert('hh')
    
    iclamp1_ng = h.IClamp(soma_ng(0.5))
    iclamp1_ng.amp = 0
    iclamp1_ng.dur = 1e9
    iclamp1_ng.delay = 0
    
    iclamp2_ng = h.IClamp(soma_ng(0.5))
    iclamp2_ng.amp = 0
    iclamp2_ng.dur = 1e9
    iclamp2_ng.delay = 0
    
    # PC设置
    gid = 0
    pc.set_gid2node(gid, int(pc.id()))
    soma_ng.push()
    pc.cell(gid, h.NetCon(soma_ng(0.5)._ref_v, None))
    h.pop_section()
    
    spike_tvec = h.Vector()
    spike_idvec = h.Vector()
    pc.spike_record(-1, spike_tvec, spike_idvec)
    pc.setup_transfer()
    pc.set_maxstep(10)
    
    # 创建HelioX管理器
    heliox_manager = HelioXManager()
    heliox_manager.set_default_device("cpu")
    heliox_manager.set_default_permute_type(0)
    
    # 创建VecPlay包装器
    print("创建VecPlay包装器...")
    try:
        vecplay1 = heliox_manager.create_vecplay_wrapper(iclamp1_ng, "amp")
        print(f"VecPlay1创建成功: {vecplay1.get_info()}")
        
        vecplay2 = heliox_manager.create_vecplay_wrapper(iclamp2_ng, "amp")
        print(f"VecPlay2创建成功: {vecplay2.get_info()}")
        
        v_monitor = heliox_manager.create_monitor_wrapper(soma_ng(0.5), "v")
        print("Monitor创建成功")
        
    except Exception as e:
        print(f"❌ 包装器创建失败: {e}")
        pc.done()
        return False
    
    # 导出和加载
    print("导出和加载模型...")
    try:
        export_path = "./test_dual_iclamp_output"
        heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
        print("模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        pc.done()
        return False
    
    # 设置VecPlay
    print("设置VecPlay...")
    try:
        vecplay1.play(times1, currents1)
        print("VecPlay1设置成功")
        
        vecplay2.play(times2, currents2)
        print("VecPlay2设置成功")
    except Exception as e:
        print(f"❌ VecPlay设置失败: {e}")
        pc.done()
        return False
    
    # 运行仿真
    print("运行仿真...")
    try:
        heliox_manager.client.finitialize(-65.0)
        heliox_manager.client.run(100.0)
        
        heliox_v = np.array(v_monitor.get_data())
        print("仿真完成")
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        pc.done()
        return False
    
    print(f"HelioX结果:")
    print(f"  电压范围: {heliox_v.min():.2f} ~ {heliox_v.max():.2f} mV")
    
    # 对比结果
    print("\n📊 结果对比...")
    min_len = min(len(neuron_v), len(heliox_v))
    diff = np.abs(neuron_v[:min_len] - heliox_v[:min_len])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"最大差异: {max_diff:.6f} mV")
    print(f"平均差异: {mean_diff:.6f} mV")
    
    # 绘制对比图
    plot_comparison(neuron_v, heliox_v, neuron_i1, neuron_i2, min_len)
    
    # 清理
    pc.done()
    
    # 判断结果
    if max_diff < 0.001:
        print("✅ 双IClamp测试通过！")
        return True
    else:
        print(f"❌ 双IClamp测试失败！差异: {max_diff:.6f} mV")
        return False

def plot_comparison(neuron_v, heliox_v, neuron_i1, neuron_i2, min_len):
    """绘制对比图"""
    
    time = np.arange(0, min_len * 0.025, 0.025)
    diff = neuron_v[:min_len] - heliox_v[:min_len]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 电压对比
    axes[0].plot(time, neuron_v[:min_len], 'b-', linewidth=2, label='NEURON')
    axes[0].plot(time, heliox_v[:min_len], 'r--', linewidth=2, label='HelioX')
    axes[0].set_ylabel('电压 (mV)')
    axes[0].set_title('双IClamp电压对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 电流验证
    axes[1].plot(time, neuron_i1[:min_len], 'orange', linewidth=2, label='IClamp1')
    axes[1].plot(time, neuron_i2[:min_len], 'purple', linewidth=2, label='IClamp2')
    axes[1].set_ylabel('电流 (nA)')
    axes[1].set_title('电流验证')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 差异图
    axes[2].plot(time, diff, 'g-', linewidth=1)
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('电压差异 (mV)')
    axes[2].set_title(f'电压差异 (最大: {np.max(np.abs(diff)):.6f} mV)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_dual_iclamp_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 对比图已保存: test_dual_iclamp_comparison.png")
    plt.close()

def main():
    """主函数"""
    print("🧪 VecPlay逐步测试 - 步骤1")
    print("=" * 60)
    
    success = test_dual_iclamp_simple()
    
    if success:
        print("\n🎉 步骤1成功！可以继续下一步测试。")
    else:
        print("\n❌ 步骤1失败！需要先解决基础问题。")
    
    return success

if __name__ == "__main__":
    main()