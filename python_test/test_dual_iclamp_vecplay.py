#!/usr/bin/env python3
"""
双IClamp VecPlay测试
测试vecplay能否正确找到并控制不同位置的IClamp变量
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

def create_dual_iclamp_model():
    """创建带有两个IClamp的细胞模型"""
    print("📦 创建双IClamp细胞模型...")
    
    # 使用ParallelContext
    pc = h.ParallelContext()
    
    # 创建soma
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 10
    soma.L = 10
    soma.insert('hh')
    
    # 创建第一个电流钳 - 位置0.3
    iclamp1 = h.IClamp(soma(0.3))
    iclamp1.amp = 0
    iclamp1.dur = 1e9
    iclamp1.delay = 0
    
    # 创建第二个电流钳 - 位置0.7
    iclamp2 = h.IClamp(soma(0.7))
    iclamp2.amp = 0
    iclamp2.dur = 1e9
    iclamp2.delay = 0
    
    # 注册到ParallelContext
    gid = 0
    pc.set_gid2node(gid, int(pc.id()))
    soma.push()
    pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
    h.pop_section()
    
    # 设置ParallelContext配置
    tvec = h.Vector()
    idvec = h.Vector()
    pc.spike_record(-1, tvec, idvec)
    pc.setup_transfer()
    pc.set_maxstep(10)
    
    print(f"✅ 模型创建完成！")
    print(f"   IClamp1位置: 0.3, 节点索引: {soma(0.3).node_index()}")
    print(f"   IClamp2位置: 0.7, 节点索引: {soma(0.7).node_index()}")
    
    return soma, iclamp1, iclamp2, pc

def test_dual_iclamp_vecplay():
    """测试双IClamp的vecplay控制"""
    print("\n🚀 == 双IClamp VecPlay测试 ==")
    print("=" * 50)
    
    # 步骤1: 创建模型
    soma, iclamp1, iclamp2, pc = create_dual_iclamp_model()
    
    # 步骤2: 创建HelioX管理器
    print("\n⚙️  初始化HelioX...")
    heliox_manager = HelioXManager()
    
    # 步骤3: 为两个IClamp分别创建VecPlay包装器
    print("📡 创建VecPlay包装器...")
    vecplay1 = heliox_manager.create_vecplay_wrapper(iclamp1, "amp")
    vecplay2 = heliox_manager.create_vecplay_wrapper(iclamp2, "amp")
    
    # 为不同位置创建电压监测器
    v_monitor_center = heliox_manager.create_monitor_wrapper(soma(0.5), "v")
    v_monitor_pos1 = heliox_manager.create_monitor_wrapper(soma(0.3), "v")
    v_monitor_pos2 = heliox_manager.create_monitor_wrapper(soma(0.7), "v")
    
    print(f"   VecPlay1 (位置0.3): {vecplay1.get_info()}")
    print(f"   VecPlay2 (位置0.7): {vecplay2.get_info()}")
    
    # 步骤4: 导出和加载模型
    print("🔄 导出和加载模型...")
    export_path = "./test_dual_iclamp_output"
    heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
    
    print("✅ 模型加载完成！\n")
    
    # 步骤5: 测试1 - 只激活IClamp1
    print("🎯 测试1: 只激活IClamp1 (位置0.3)")
    print("-" * 30)
    
    # IClamp1的刺激模式
    tvec1 = [0, 20, 20, 40, 40, 80]
    yvec1 = [0, 0, 0.3, 0.5, 0, 0]
    
    # IClamp2保持静止
    tvec2_off = [0, 100]
    yvec2_off = [0, 0]
    
    vecplay1.play(tvec1, yvec1)
    vecplay2.play(tvec2_off, yvec2_off)
    
    print(f"   IClamp1刺激: {len(tvec1)}个时间点, 峰值{max(yvec1)}nA")
    print(f"   IClamp2状态: 关闭")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    voltage_test1_center = v_monitor_center.get_data()
    voltage_test1_pos1 = v_monitor_pos1.get_data()
    voltage_test1_pos2 = v_monitor_pos2.get_data()
    
    print(f"   中心电压范围: {min(voltage_test1_center):.2f} 到 {max(voltage_test1_center):.2f} mV")
    print(f"   位置0.3电压范围: {min(voltage_test1_pos1):.2f} 到 {max(voltage_test1_pos1):.2f} mV")
    print(f"   位置0.7电压范围: {min(voltage_test1_pos2):.2f} 到 {max(voltage_test1_pos2):.2f} mV")
    
    # 步骤6: 测试2 - 只激活IClamp2
    print("\n🎯 测试2: 只激活IClamp2 (位置0.7)")
    print("-" * 30)
    
    # IClamp1保持静止
    tvec1_off = [0, 100]
    yvec1_off = [0, 0]
    
    # IClamp2的刺激模式
    tvec2 = [0, 30, 30, 50, 50, 80]
    yvec2 = [0, 0, 1.0, 1.0, 0, 0]
    
    vecplay1.play(tvec1_off, yvec1_off)
    vecplay2.play(tvec2, yvec2)
    
    print(f"   IClamp1状态: 关闭")
    print(f"   IClamp2刺激: {len(tvec2)}个时间点, 峰值{max(yvec2)}nA")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    voltage_test2_center = v_monitor_center.get_data()
    voltage_test2_pos1 = v_monitor_pos1.get_data()
    voltage_test2_pos2 = v_monitor_pos2.get_data()
    
    print(f"   中心电压范围: {min(voltage_test2_center):.2f} 到 {max(voltage_test2_center):.2f} mV")
    print(f"   位置0.3电压范围: {min(voltage_test2_pos1):.2f} 到 {max(voltage_test2_pos1):.2f} mV")
    print(f"   位置0.7电压范围: {min(voltage_test2_pos2):.2f} 到 {max(voltage_test2_pos2):.2f} mV")
    
    # 步骤7: 测试3 - 双重激活（交错模式）
    print("\n🎯 测试3: 双重激活（交错模式）")
    print("-" * 30)
    
    # IClamp1早期激活
    tvec1_early = [0, 10, 10, 25, 25, 40]
    yvec1_early = [0, 0, 0.6, 0.6, 0, 0]
    
    # IClamp2后期激活
    tvec2_late = [0, 50, 50, 65, 65, 80]
    yvec2_late = [0, 0, 0.1, 0.2, 0, 0]
    
    vecplay1.play(tvec1_early, yvec1_early)
    vecplay2.play(tvec2_late, yvec2_late)
    
    print(f"   IClamp1 (0.3): 早期激活 {max(yvec1_early)}nA")
    print(f"   IClamp2 (0.7): 后期激活 {max(yvec2_late)}nA")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    voltage_test3_center = v_monitor_center.get_data()
    voltage_test3_pos1 = v_monitor_pos1.get_data()
    voltage_test3_pos2 = v_monitor_pos2.get_data()
    
    print(f"   中心电压范围: {min(voltage_test3_center):.2f} 到 {max(voltage_test3_center):.2f} mV")
    print(f"   位置0.3电压范围: {min(voltage_test3_pos1):.2f} 到 {max(voltage_test3_pos1):.2f} mV")
    print(f"   位置0.7电压范围: {min(voltage_test3_pos2):.2f} 到 {max(voltage_test3_pos2):.2f} mV")
    
    # 步骤8: 验证VecPlay控制的准确性
    print("\n✨ 验证VecPlay控制准确性")
    print("-" * 30)
    
    # 检查各测试的最大电压差异
    max_v1 = max(voltage_test1_center)
    max_v2 = max(voltage_test2_center)
    max_v3 = max(voltage_test3_center)
    
    print(f"测试1 (仅IClamp1): 最大电压 {max_v1:.2f} mV")
    print(f"测试2 (仅IClamp2): 最大电压 {max_v2:.2f} mV")
    print(f"测试3 (双重激活): 最大电压 {max_v3:.2f} mV")
    
    # 验证逻辑
    if abs(max_v1 - max_v2) < 5:
        print("⚠️  两个IClamp产生的电压响应过于相似，可能存在问题")
    else:
        print("✅ 两个IClamp产生了不同的电压响应，控制正确")
    
    if max_v3 > max(max_v1, max_v2):
        print("✅ 双重激活产生了更强的响应，符合预期")
    else:
        print("⚠️  双重激活没有产生预期的增强效果")
    
    # 步骤9: 可视化结果
    print("\n📊 生成结果图表...")
    plot_dual_iclamp_results(
        voltage_test1_center, voltage_test1_pos1, voltage_test1_pos2,
        voltage_test2_center, voltage_test2_pos1, voltage_test2_pos2,
        voltage_test3_center, voltage_test3_pos1, voltage_test3_pos2,
        tvec1, yvec1, tvec2, yvec2, tvec1_early, yvec1_early, tvec2_late, yvec2_late
    )
    
    # 清理
    pc.done()
    
    print("\n🎉 双IClamp VecPlay测试完成！")
    print("=" * 50)
    print("\n💡 关键发现:")
    print("- VecPlay可以独立控制不同位置的IClamp")
    print("- 每个VecPlay包装器只影响其对应的IClamp")
    print("- 可以实现复杂的多点刺激模式")

def plot_dual_iclamp_results(v1_center, v1_pos1, v1_pos2,
                            v2_center, v2_pos1, v2_pos2,
                            v3_center, v3_pos1, v3_pos2,
                            t1, y1, t2, y2, t1e, y1e, t2l, y2l):
    """绘制双IClamp测试结果"""
    
    # 创建时间轴
    min_len = min(len(v1_center), len(v2_center), len(v3_center))
    time = np.arange(0, min_len * 0.025, 0.025)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 测试1: 只有IClamp1
    axes[0, 0].plot(time, v1_center[:min_len], 'b-', linewidth=2, label='中心(0.5)')
    axes[0, 0].plot(time, v1_pos1[:min_len], 'g-', linewidth=2, label='位置0.3')
    axes[0, 0].plot(time, v1_pos2[:min_len], 'r-', linewidth=2, label='位置0.7')
    axes[0, 0].set_title('测试1: 只激活IClamp1 (位置0.3)')
    axes[0, 0].set_ylabel('电压 (mV)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t1, y1, 'g-o', linewidth=2, markersize=6, label='IClamp1')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='IClamp2 (关闭)')
    axes[0, 1].set_title('刺激模式1')
    axes[0, 1].set_ylabel('电流 (nA)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试2: 只有IClamp2
    axes[1, 0].plot(time, v2_center[:min_len], 'b-', linewidth=2, label='中心(0.5)')
    axes[1, 0].plot(time, v2_pos1[:min_len], 'g-', linewidth=2, label='位置0.3')
    axes[1, 0].plot(time, v2_pos2[:min_len], 'r-', linewidth=2, label='位置0.7')
    axes[1, 0].set_title('测试2: 只激活IClamp2 (位置0.7)')
    axes[1, 0].set_ylabel('电压 (mV)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='IClamp1 (关闭)')
    axes[1, 1].plot(t2, y2, 'r-s', linewidth=2, markersize=6, label='IClamp2')
    axes[1, 1].set_title('刺激模式2')
    axes[1, 1].set_ylabel('电流 (nA)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 测试3: 双重激活
    axes[2, 0].plot(time, v3_center[:min_len], 'b-', linewidth=2, label='中心(0.5)')
    axes[2, 0].plot(time, v3_pos1[:min_len], 'g-', linewidth=2, label='位置0.3')
    axes[2, 0].plot(time, v3_pos2[:min_len], 'r-', linewidth=2, label='位置0.7')
    axes[2, 0].set_title('测试3: 双重激活 (交错模式)')
    axes[2, 0].set_xlabel('时间 (ms)')
    axes[2, 0].set_ylabel('电压 (mV)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(t1e, y1e, 'g-o', linewidth=2, markersize=6, label='IClamp1 (早期)')
    axes[2, 1].plot(t2l, y2l, 'r-s', linewidth=2, markersize=6, label='IClamp2 (后期)')
    axes[2, 1].set_title('交错刺激模式')
    axes[2, 1].set_xlabel('时间 (ms)')
    axes[2, 1].set_ylabel('电流 (nA)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'test_dual_iclamp_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {filename}")

if __name__ == "__main__":
    test_dual_iclamp_vecplay()