#!/usr/bin/env python3
"""
Gap Junction GPU模式测试
=========================

测试Gap Junction在GPU模式下的功能。
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from neuron import h, coreneuron
from heliox_wrapper import HelioXManager, GapJunctionInterface

# 配置NEURON
h.usetable_hh = 0
h.load_file("nrngui.hoc")
h.cvode.cache_efficient(1)
coreneuron.enable = True

def test_gap_junction_gpu():
    """测试GPU模式下的Gap Junction"""
    print("\n" + "="*60)
    print("Gap Junction GPU模式测试")
    print("="*60)
    
    # 创建两个细胞
    cells = []
    for i in range(2):
        soma = h.Section(name=f'soma_{i}')
        soma.L = 50
        soma.diam = 50
        soma.insert('hh')
        soma.Ra = 100
        cells.append({'soma': soma, 'gid': i})
    
    # 添加电流刺激到第一个细胞
    stim = h.IClamp(cells[0]['soma'](0.5))
    stim.delay = 10
    stim.dur = 50
    stim.amp = 0.5
    
    h.celsius = 6.3
    
    # 设置ParallelContext
    pc = h.ParallelContext()
    for i, cell in enumerate(cells):
        pc.set_gid2node(i, pc.id())
        nc = h.NetCon(cell['soma'](0.5)._ref_v, None, sec=cell['soma'])
        pc.cell(i, nc)
    
    # 初始化HelioXManager - GPU模式
    manager = HelioXManager()
    manager.device = "gpu"  # 设置为GPU模式
    manager.permute_type = -1  # GPU模式使用默认排列
    
    print("\n1. 设置为GPU模式")
    print(f"   Device: {manager.device}")
    print(f"   Permute type: {manager.permute_type}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "gap_gpu_test")
        
        # 创建监控器
        monitors = []
        for i, cell in enumerate(cells):
            monitor = manager.create_monitor_wrapper(cell['soma'](0.5), "v")
            monitors.append(monitor)
        
        # 导出模型
        print("\n2. 导出模型...")
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = 0.025
        h.finitialize(-65)
        pc.nrnbbcore_write(export_path)
        
        # 加载到HELIOX (GPU模式)
        manager.set_data_path(export_path)
        manager.set_device("gpu")
        manager.set_permute_type(-1)
        manager.load_model()
        
        print("   ✅ 模型已加载到GPU")
        
        # 获取监控器索引
        monitor_indices = []
        for i, monitor in enumerate(monitors):
            monitor._initialize()
            idx = monitor._node_or_mech_idx
            monitor_indices.append(idx)
            print(f"   Cell {i} 索引: {idx}")
        
        # 获取Gap Junction接口
        gap = GapJunctionInterface()
        
        print("\n3. 创建Gap Junctions (GPU模式)...")
        
        # Cell 0 -> Cell 1
        sid1 = gap.create("", "v", monitor_indices[0], sid=200)
        success1 = gap.add_target(sid1, "", "v", monitor_indices[1])
        
        # Cell 1 -> Cell 0 (双向)
        sid2 = gap.create("", "v", monitor_indices[1], sid=201)
        success2 = gap.add_target(sid2, "", "v", monitor_indices[0])
        
        if success1 and success2:
            print(f"   ✅ Gap Junctions创建成功")
            print(f"      - Gap {sid1}: Cell 0 -> Cell 1")
            print(f"      - Gap {sid2}: Cell 1 -> Cell 0")
        else:
            print("   ❌ Gap Junction创建失败")
        
        # 运行仿真
        print("\n4. 运行GPU仿真...")
        manager.finitialize(-65)
        manager.run(100)
        
        # 获取结果
        v_gpu = [m.data for m in monitors]
        
        print("\n5. GPU模式结果:")
        print(f"   - Cell 0 最大电压: {np.max(v_gpu[0]):.2f} mV")
        print(f"   - Cell 1 最大电压: {np.max(v_gpu[1]):.2f} mV")
        
        # 判断Gap是否生效
        if np.max(v_gpu[1]) > -60:
            print("\n✅ GPU模式Gap Junction传输成功！")
            gpu_success = True
        else:
            print("\n❌ GPU模式Gap Junction传输失败")
            gpu_success = False
        
        return gpu_success, v_gpu

def test_cpu_vs_gpu():
    """对比CPU和GPU模式的Gap Junction"""
    print("\n" + "🔬"*30)
    print(" Gap Junction CPU vs GPU 对比测试 ")
    print("🔬"*30)
    
    # 首先测试GPU模式
    gpu_success, v_gpu = test_gap_junction_gpu()
    
    # 清理并重新创建网络测试CPU模式
    h.quit()
    from neuron import h
    h.load_file("nrngui.hoc")
    h.cvode.cache_efficient(1)
    coreneuron.enable = True
    
    print("\n" + "="*60)
    print("Gap Junction CPU模式测试")
    print("="*60)
    
    # 创建相同的网络
    cells = []
    for i in range(2):
        soma = h.Section(name=f'soma_{i}')
        soma.L = 50
        soma.diam = 50
        soma.insert('hh')
        soma.Ra = 100
        cells.append({'soma': soma, 'gid': i})
    
    stim = h.IClamp(cells[0]['soma'](0.5))
    stim.delay = 10
    stim.dur = 50
    stim.amp = 0.5
    
    h.celsius = 6.3
    
    pc = h.ParallelContext()
    for i, cell in enumerate(cells):
        pc.set_gid2node(i, pc.id())
        nc = h.NetCon(cell['soma'](0.5)._ref_v, None, sec=cell['soma'])
        pc.cell(i, nc)
    
    # CPU模式
    manager = HelioXManager()
    manager.device = "cpu"
    manager.permute_type = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "gap_cpu_test")
        
        monitors = []
        for i, cell in enumerate(cells):
            monitor = manager.create_monitor_wrapper(cell['soma'](0.5), "v")
            monitors.append(monitor)
        
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = 0.025
        h.finitialize(-65)
        pc.nrnbbcore_write(export_path)
        
        manager.set_data_path(export_path)
        manager.set_device("cpu")
        manager.set_permute_type(0)
        manager.load_model()
        
        monitor_indices = []
        for monitor in monitors:
            monitor._initialize()
            monitor_indices.append(monitor._node_or_mech_idx)
        
        gap = GapJunctionInterface()
        
        # 创建相同的Gap Junctions
        gap.create("", "v", monitor_indices[0], sid=200)
        gap.add_target(200, "", "v", monitor_indices[1])
        gap.create("", "v", monitor_indices[1], sid=201)
        gap.add_target(201, "", "v", monitor_indices[0])
        
        manager.finitialize(-65)
        manager.run(100)
        
        v_cpu = [m.data for m in monitors]
        
        print(f"\nCPU模式结果:")
        print(f"   - Cell 0 最大电压: {np.max(v_cpu[0]):.2f} mV")
        print(f"   - Cell 1 最大电压: {np.max(v_cpu[1]):.2f} mV")
    
    # 对比结果
    print("\n" + "="*60)
    print("CPU vs GPU 对比分析")
    print("="*60)
    
    min_len = min(len(v_cpu[0]), len(v_gpu[0]))
    
    for i in range(2):
        cpu_data = v_cpu[i][:min_len]
        gpu_data = v_gpu[i][:min_len]
        
        max_diff = np.max(np.abs(cpu_data - gpu_data))
        mean_diff = np.mean(np.abs(cpu_data - gpu_data))
        
        print(f"\nCell {i}:")
        print(f"   CPU峰值: {np.max(cpu_data):.4f} mV")
        print(f"   GPU峰值: {np.max(gpu_data):.4f} mV")
        print(f"   最大差异: {max_diff:.6f} mV")
        print(f"   平均差异: {mean_diff:.6f} mV")
    
    # 绘图
    time_points = np.arange(min_len) * 0.025
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cell 0
    ax = axes[0, 0]
    ax.plot(time_points, v_cpu[0][:min_len], 'b-', label='CPU', linewidth=2)
    ax.plot(time_points, v_gpu[0][:min_len], 'r--', label='GPU', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Cell 0 (Stimulated)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cell 1
    ax = axes[0, 1]
    ax.plot(time_points, v_cpu[1][:min_len], 'b-', label='CPU', linewidth=2)
    ax.plot(time_points, v_gpu[1][:min_len], 'r--', label='GPU', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Cell 1 (Gap-connected)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 差异图
    ax = axes[1, 0]
    diff0 = np.array(v_cpu[0][:min_len]) - np.array(v_gpu[0][:min_len])
    ax.plot(time_points, diff0, 'g-', alpha=0.7)
    ax.set_ylabel('CPU - GPU (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Cell 0 Difference')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    diff1 = np.array(v_cpu[1][:min_len]) - np.array(v_gpu[1][:min_len])
    ax.plot(time_points, diff1, 'm-', alpha=0.7)
    ax.set_ylabel('CPU - GPU (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Cell 1 Difference')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gap Junction: CPU vs GPU Comparison')
    plt.tight_layout()
    plt.savefig('gap_cpu_vs_gpu.png', dpi=150)
    print("\n📊 对比图已保存为 gap_cpu_vs_gpu.png")
    
    # 总结
    max_overall_diff = max(np.max(np.abs(v_cpu[0][:min_len] - v_gpu[0][:min_len])),
                           np.max(np.abs(v_cpu[1][:min_len] - v_gpu[1][:min_len])))
    
    print("\n" + "="*60)
    if max_overall_diff < 0.01:
        print("✅ CPU和GPU模式结果完全一致！")
    elif max_overall_diff < 0.1:
        print("✅ CPU和GPU模式结果基本一致（数值误差范围内）")
    else:
        print(f"⚠️ CPU和GPU模式存在差异 (最大 {max_overall_diff:.4f} mV)")
    
    return max_overall_diff < 0.1

if __name__ == "__main__":
    try:
        # 先测试单独的GPU模式
        print("📋 首先测试GPU模式...")
        gpu_works, _ = test_gap_junction_gpu()
        
        if gpu_works:
            print("\n📋 现在对比CPU和GPU模式...")
            success = test_cpu_vs_gpu()
            
            if success:
                print("\n🎉 Gap Junction GPU支持实现成功！")
            else:
                print("\n⚠️ GPU支持已实现但存在差异")
        else:
            print("\n❌ GPU模式Gap Junction未能正常工作")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()