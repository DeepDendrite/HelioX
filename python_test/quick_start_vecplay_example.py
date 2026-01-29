#!/usr/bin/env python3
"""
HelioX动态VecPlay快速入门示例

这个示例展示了如何使用HelioX的动态VecPlay功能来控制仿真中的刺激参数。
VecPlay允许你在仿真过程中动态修改电流、电压等参数，非常适合参数扫描和优化算法。
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

def create_simple_cell_model():
    """创建简单的细胞模型"""
    print("📦 创建细胞模型...")
    
    # 使用ParallelContext（HelioX导出需要）
    pc = h.ParallelContext()
    
    # 创建soma
    soma = h.Section(name='soma')
    soma.nseg = 1
    soma.diam = 10
    soma.L = 10
    soma.insert('hh')  # Hodgkin-Huxley通道
    
    # 创建电流钳
    iclamp = h.IClamp(soma(0.5))
    iclamp.amp = 0      # 初始电流为0
    iclamp.dur = 1e9    # 持续整个仿真
    iclamp.delay = 0    # 无延迟
    
    # 注册到ParallelContext（必须的步骤）
    gid = 0
    pc.set_gid2node(gid, int(pc.id()))
    soma.push()
    pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
    h.pop_section()
    
    # 设置必要的ParallelContext配置
    tvec = h.Vector()
    idvec = h.Vector()
    pc.spike_record(-1, tvec, idvec)
    pc.setup_transfer()
    pc.set_maxstep(10)
    
    print(f"✅ 模型创建完成！soma节点索引: {soma(0.5).node_index()}")
    
    return soma, iclamp, pc

def demonstrate_basic_usage():
    """演示基本用法"""
    print("\n🚀 == HelioX动态VecPlay快速入门 ==")
    print("=" * 50)
    
    # 步骤1: 创建模型
    soma, iclamp, pc = create_simple_cell_model()
    
    # 步骤2: 创建HelioX管理器
    print("\n⚙️  初始化HelioX...")
    heliox_manager = HelioXManager()
    # heliox_manager.set_default_device("cpu")     # 使用CPU（调试方便）
    # heliox_manager.set_default_permute_type(0)   # CPU排列类型
    
    # 步骤3: 创建包装器
    print("📡 创建包装器...")
    # VecPlay包装器 - 用于动态控制电流
    vecplay = heliox_manager.create_vecplay_wrapper(iclamp, "amp")
    # Monitor包装器 - 用于记录电压
    v_monitor = heliox_manager.create_monitor_wrapper(soma(0.5), "v")
    
    # 步骤4: 导出和加载模型
    print("🔄 导出和加载模型...")
    export_path = "./vecplay_quickstart_output"
    heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
    
    print("✅ 模型加载完成！\n")
    
    # 步骤5: 演示VecPlay基本功能
    print("🎯 演示VecPlay基本功能")
    print("-" * 30)
    
    # 5.1 设置初始刺激
    print("1️⃣  设置初始刺激模式...")
    tvec1 = [0, 25, 25, 50, 50, 100]  # 时间点 (ms)
    yvec1 = [0, 0, 0.5, 0.5, 0, 0]   # 电流值 (nA)
    
    vecplay.play(tvec1, yvec1)
    print(f"   刺激设置: {len(tvec1)} 个时间点")
    print(f"   VecPlay状态: {vecplay.get_info()}")
    
    # 5.2 运行第一次仿真
    print("\n2️⃣  运行第一次仿真...")
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    voltage1 = v_monitor.get_data()
    print(f"   仿真完成: {len(voltage1)} 个数据点")
    print(f"   电压范围: {min(voltage1):.2f} 到 {max(voltage1):.2f} mV")
    
    # 5.3 动态修改刺激
    print("\n3️⃣  动态修改刺激模式...")
    tvec2 = [0, 20, 20, 40, 40, 60, 60, 80, 80, 100]  # 更多时间点
    yvec2 = [0, 0, 1.0, 1.0, 0, 0, -0.3, -0.3, 0, 0]  # 更强的双相刺激
    
    vecplay.play(tvec2, yvec2)
    print(f"   新刺激设置: {len(tvec2)} 个时间点")
    print(f"   更强的双相刺激: 正向 {max(yvec2)} nA, 负向 {min(yvec2)} nA")
    
    # 5.4 运行第二次仿真
    print("\n4️⃣  重新运行仿真...")
    heliox_manager.client.finitialize(-65.0)
    heliox_manager.client.run(100.0)
    
    voltage2 = v_monitor.get_data()
    print(f"   仿真完成: {len(voltage2)} 个数据点")
    print(f"   电压范围: {min(voltage2):.2f} 到 {max(voltage2):.2f} mV")
    
    # 5.5 验证修改是否生效
    voltage_change = abs(max(voltage2) - max(voltage1))
    print(f"\n✨ 电压变化: {voltage_change:.2f} mV")
    if voltage_change > 10:  # 如果电压变化超过10mV
        print("✅ 动态修改成功！刺激变化产生了显著的神经元响应。")
    else:
        print("⚠️  动态修改可能没有生效，电压变化较小。")
    
    # 步骤6: 演示其他VecPlay功能
    print("\n🔧 演示其他VecPlay功能")
    print("-" * 30)
    
    # 6.1 查询状态
    print("状态查询:")
    print(f"  - 是否在播放: {vecplay.is_playing()}")
    print(f"  - 详细信息: {vecplay.get_info()}")
    
    # 6.2 停止VecPlay
    print("\n停止VecPlay:")
    vecplay.stop()
    print(f"  - 停止后状态: {vecplay.is_playing()}")
    
    # 6.3 重新启动
    print("\n重新启动VecPlay:")
    vecplay.play(tvec1, yvec1)  # 使用第一个刺激模式
    print(f"  - 重启后状态: {vecplay.is_playing()}")
    
    # 步骤7: 可视化结果
    print("\n📊 生成结果图表...")
    plot_results(voltage1, voltage2, tvec1, yvec1, tvec2, yvec2)
    
    # 清理
    pc.done()
    
    print("\n🎉 快速入门演示完成！")
    print("=" * 50)
    print("\n💡 小贴士:")
    print("- VecPlay可以随时修改，支持不同长度的时间序列")
    print("- 适用于参数扫描、优化算法、自适应刺激等场景")
    print("- 支持任何机制的任何变量（不只是电流）")
    print("- 可以控制多个变量，每个变量用一个VecPlayWrapper")

def plot_results(voltage1, voltage2, tvec1, yvec1, tvec2, yvec2):
    """绘制结果对比图"""
    
    # 创建时间轴
    time1 = np.arange(0, len(voltage1) * 0.025, 0.025)
    time2 = np.arange(0, len(voltage2) * 0.025, 0.025)
    min_len = min(len(voltage1), len(voltage2))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 电压对比
    axes[0].plot(time1[:min_len], voltage1[:min_len], 'b-', linewidth=2, label='第一次仿真（弱刺激）')
    axes[0].plot(time2[:min_len], voltage2[:min_len], 'r-', linewidth=2, label='第二次仿真（强刺激）')
    axes[0].set_ylabel('电压 (mV)')
    axes[0].set_title('HelioX动态VecPlay演示 - 电压响应对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 刺激模式对比
    axes[1].plot(tvec1, yvec1, 'b-o', linewidth=2, markersize=6, label='第一次刺激模式')
    axes[1].plot(tvec2, yvec2, 'r-s', linewidth=2, markersize=6, label='第二次刺激模式')
    axes[1].set_xlabel('时间 (ms)')
    axes[1].set_ylabel('电流 (nA)')
    axes[1].set_title('刺激模式对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'vecplay_quickstart_demo.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {filename}")

if __name__ == "__main__":
    demonstrate_basic_usage()