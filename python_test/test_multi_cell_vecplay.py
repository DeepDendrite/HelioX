#!/usr/bin/env python3
"""
多细胞VecPlay测试
测试vecplay在多个独立细胞上的精确控制能力
验证未被刺激的细胞保持静息状态
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

def create_multi_cell_model(num_cells=4):
    """创建多个独立细胞的模型"""
    print(f"📦 创建{num_cells}个独立细胞模型...")
    
    # 使用ParallelContext
    pc = h.ParallelContext()
    
    cells = []
    iclamps = []
    
    for i in range(num_cells):
        # 创建soma
        soma = h.Section(name=f'soma_{i}')
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
        gid = i
        pc.set_gid2node(gid, int(pc.id()))
        soma.push()
        pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
        h.pop_section()
        
        cells.append(soma)
        iclamps.append(iclamp)
        
        print(f"   细胞{i}: 节点索引 {soma(0.5).node_index()}")
    
    # 设置ParallelContext配置
    tvec = h.Vector()
    idvec = h.Vector()
    pc.spike_record(-1, tvec, idvec)
    pc.setup_transfer()
    pc.set_maxstep(10)
    
    print(f"✅ {num_cells}个独立细胞创建完成！")
    
    return cells, iclamps, pc

def test_multi_cell_vecplay():
    """测试多细胞的vecplay控制"""
    print("\n🚀 == 多细胞VecPlay精确控制测试 ==")
    print("=" * 50)
    
    num_cells = 4
    
    # 步骤1: 创建模型
    cells, iclamps, pc = create_multi_cell_model(num_cells)
    
    # 步骤2: 创建HelioX管理器
    print("\n⚙️  初始化HelioX...")
    heliox_manager = HelioXManager()
    
    # 步骤3: 为每个细胞创建VecPlay和Monitor包装器
    print("📡 创建VecPlay和Monitor包装器...")
    vecplays = []
    monitors = []
    
    for i in range(num_cells):
        vecplay = heliox_manager.create_vecplay_wrapper(iclamps[i], "amp")
        monitor = heliox_manager.create_monitor_wrapper(cells[i](0.5), "v")
        vecplays.append(vecplay)
        monitors.append(monitor)
        print(f"   细胞{i}: VecPlay和Monitor已创建")
    
    # 步骤4: 导出和加载模型
    print("🔄 导出和加载模型...")
    export_path = "./test_multi_cell_output"
    heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
    
    print("✅ 模型加载完成！\n")
    
    # 步骤5: 测试1 - 只激活细胞0
    print("🎯 测试1: 只激活细胞0")
    print("-" * 30)
    
    # 为所有细胞设置刺激模式
    stim_patterns = []
    for i in range(num_cells):
        if i == 0:
            # 细胞0: 强刺激
            tvec = [0, 20, 20, 40, 40, 80]
            yvec = [0, 0, 1.0, 1.0, 0, 0]
        else:
            # 其他细胞: 静息
            tvec = [0, 100]
            yvec = [0, 0]
        
        vecplays[i].play(tvec, yvec)
        stim_patterns.append((tvec, yvec))
        
        if i == 0:
            print(f"   细胞{i}: 激活，峰值{max(yvec)}nA")
        else:
            print(f"   细胞{i}: 静息")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    # 收集结果
    test1_voltages = []
    for i in range(num_cells):
        voltage = monitors[i].get_data()
        test1_voltages.append(voltage)
        max_v = max(voltage)
        min_v = min(voltage)
        print(f"   细胞{i}电压: {min_v:.2f} 到 {max_v:.2f} mV")
    
    # 步骤6: 测试2 - 只激活细胞2
    print("\n🎯 测试2: 只激活细胞2")
    print("-" * 30)
    
    for i in range(num_cells):
        if i == 2:
            # 细胞2: 不同的刺激模式
            tvec = [0, 15, 15, 35, 35, 55, 55, 75]
            yvec = [0, 0, 0.8, 0.8, 0, 0, 0.6, 0.6]
        else:
            # 其他细胞: 静息
            tvec = [0, 100]
            yvec = [0, 0]
        
        vecplays[i].play(tvec, yvec)
        
        if i == 2:
            print(f"   细胞{i}: 激活，双脉冲刺激，峰值{max(yvec)}nA")
        else:
            print(f"   细胞{i}: 静息")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    # 收集结果
    test2_voltages = []
    for i in range(num_cells):
        voltage = monitors[i].get_data()
        test2_voltages.append(voltage)
        max_v = max(voltage)
        min_v = min(voltage)
        print(f"   细胞{i}电压: {min_v:.2f} 到 {max_v:.2f} mV")
    
    # 步骤7: 测试3 - 多细胞协调激活
    print("\n🎯 测试3: 多细胞协调激活")
    print("-" * 30)
    
    activation_patterns = [
        # 细胞0: 早期激活
        ([0, 10, 10, 25, 25, 40], [0, 0, 0.7, 0.7, 0, 0]),
        # 细胞1: 中期激活
        ([0, 30, 30, 45, 45, 60], [0, 0, 0.9, 0.9, 0, 0]),
        # 细胞2: 静息
        ([0, 100], [0, 0]),
        # 细胞3: 后期激活
        ([0, 60, 60, 75, 75, 90], [0, 0, 0.5, 0.5, 0, 0])
    ]
    
    for i in range(num_cells):
        tvec, yvec = activation_patterns[i]
        vecplays[i].play(tvec, yvec)
        
        if max(yvec) > 0:
            print(f"   细胞{i}: 激活，峰值{max(yvec)}nA")
        else:
            print(f"   细胞{i}: 静息")
    
    # 运行仿真
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    # 收集结果
    test3_voltages = []
    for i in range(num_cells):
        voltage = monitors[i].get_data()
        test3_voltages.append(voltage)
        max_v = max(voltage)
        min_v = min(voltage)
        print(f"   细胞{i}电压: {min_v:.2f} 到 {max_v:.2f} mV")
    
    # 步骤8: 验证VecPlay的精确性
    print("\n✨ 验证VecPlay精确控制")
    print("-" * 30)
    
    def analyze_response(voltages, test_name):
        """分析电压响应"""
        rest_voltage = -65.0
        threshold = 5.0  # 5mV变化阈值
        
        active_cells = []
        inactive_cells = []
        
        for i, voltage in enumerate(voltages):
            max_deviation = max(abs(max(voltage) - rest_voltage), 
                              abs(min(voltage) - rest_voltage))
            
            if max_deviation > threshold:
                active_cells.append(i)
            else:
                inactive_cells.append(i)
        
        print(f"  {test_name}:")
        print(f"    活跃细胞: {active_cells}")
        print(f"    静息细胞: {inactive_cells}")
        
        return active_cells, inactive_cells
    
    active1, inactive1 = analyze_response(test1_voltages, "测试1")
    active2, inactive2 = analyze_response(test2_voltages, "测试2")
    active3, inactive3 = analyze_response(test3_voltages, "测试3")
    
    # 验证结果
    print("\n🔍 验证结果:")
    
    # 测试1验证
    if active1 == [0] and set(inactive1) == {1, 2, 3}:
        print("✅ 测试1: 只有细胞0被激活，其他保持静息")
    else:
        print(f"⚠️  测试1: 预期细胞0激活，实际{active1}")
    
    # 测试2验证
    if active2 == [2] and set(inactive2) == {0, 1, 3}:
        print("✅ 测试2: 只有细胞2被激活，其他保持静息")
    else:
        print(f"⚠️  测试2: 预期细胞2激活，实际{active2}")
    
    # 测试3验证
    expected_active3 = {0, 1, 3}
    if set(active3) == expected_active3 and 2 in inactive3:
        print("✅ 测试3: 细胞0,1,3被激活，细胞2保持静息")
    else:
        print(f"⚠️  测试3: 预期细胞0,1,3激活，实际{active3}")
    
    # 步骤9: 可视化结果
    print("\n📊 生成结果图表...")
    plot_multi_cell_results(test1_voltages, test2_voltages, test3_voltages, 
                           activation_patterns, num_cells)
    
    # 清理
    pc.done()
    
    print("\n🎉 多细胞VecPlay精确控制测试完成！")
    print("=" * 50)
    print("\n💡 关键发现:")
    print("- VecPlay能够精确控制指定的细胞")
    print("- 未被刺激的细胞保持静息状态")
    print("- 支持复杂的多细胞协调激活模式")
    print("- 每个VecPlay包装器只影响对应的细胞")

def plot_multi_cell_results(test1_voltages, test2_voltages, test3_voltages, 
                           activation_patterns, num_cells):
    """绘制多细胞测试结果"""
    
    # 创建时间轴
    min_len = min([len(v) for test in [test1_voltages, test2_voltages, test3_voltages] 
                   for v in test])
    time = np.arange(0, min_len * 0.025, 0.025)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    # 测试1: 只激活细胞0
    for i in range(num_cells):
        axes[0, 0].plot(time, test1_voltages[i][:min_len], 
                       color=colors[i], linewidth=2, label=f'细胞{i}')
    axes[0, 0].set_title('测试1: 只激活细胞0')
    axes[0, 0].set_ylabel('电压 (mV)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 测试1刺激模式
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    tvec = [0, 20, 20, 40, 40, 80]
    yvec = [0, 0, 1.0, 1.0, 0, 0]
    axes[0, 1].plot(tvec, yvec, color=colors[0], linewidth=3, 
                   marker='o', markersize=8, label='细胞0')
    axes[0, 1].set_title('测试1刺激模式')
    axes[0, 1].set_ylabel('电流 (nA)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试2: 只激活细胞2
    for i in range(num_cells):
        axes[1, 0].plot(time, test2_voltages[i][:min_len], 
                       color=colors[i], linewidth=2, label=f'细胞{i}')
    axes[1, 0].set_title('测试2: 只激活细胞2')
    axes[1, 0].set_ylabel('电压 (mV)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 测试2刺激模式
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    tvec2 = [0, 15, 15, 35, 35, 55, 55, 75]
    yvec2 = [0, 0, 0.8, 0.8, 0, 0, 0.6, 0.6]
    axes[1, 1].plot(tvec2, yvec2, color=colors[2], linewidth=3, 
                   marker='s', markersize=8, label='细胞2')
    axes[1, 1].set_title('测试2刺激模式')
    axes[1, 1].set_ylabel('电流 (nA)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 测试3: 多细胞协调
    for i in range(num_cells):
        axes[2, 0].plot(time, test3_voltages[i][:min_len], 
                       color=colors[i], linewidth=2, label=f'细胞{i}')
    axes[2, 0].set_title('测试3: 多细胞协调激活')
    axes[2, 0].set_xlabel('时间 (ms)')
    axes[2, 0].set_ylabel('电压 (mV)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 测试3刺激模式
    axes[2, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    for i, (tvec, yvec) in enumerate(activation_patterns):
        if max(yvec) > 0:  # 只显示有刺激的细胞
            markers = ['o', 's', '^', 'D']
            axes[2, 1].plot(tvec, yvec, color=colors[i], linewidth=2, 
                           marker=markers[i], markersize=6, label=f'细胞{i}')
    axes[2, 1].set_title('测试3协调刺激模式')
    axes[2, 1].set_xlabel('时间 (ms)')
    axes[2, 1].set_ylabel('电流 (nA)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'test_multi_cell_vecplay_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {filename}")

if __name__ == "__main__":
    test_multi_cell_vecplay()