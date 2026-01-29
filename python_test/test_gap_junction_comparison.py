#!/usr/bin/env python3
"""
Gap Junction NEURON vs HELIOX 真正的对比测试
==============================================

1. 创建一个基础网络（无Gap Junction）
2. 导出到HELIOX（HELIOX获得独立副本）
3. 分别在NEURON和HELIOX中添加相同的Gap Junction
4. 独立运行并对比结果
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
manager = HelioXManager()
manager.set_device("gpu")
# manager.set_permute_type(3)

def main():
    """主测试函数"""
    print("\n" + "🔬"*30)
    print(" Gap Junction - NEURON vs HELIOX 真正对比测试 ")
    print("🔬"*30)
    
    # ========================================
    # 步骤1: 创建基础网络（无Gap）
    # ========================================
    print("\n" + "="*60)
    print("步骤1: 创建基础网络（无Gap Junction）")
    print("="*60)
    
    cells = []
    for i in range(2):
        soma = h.Section(name=f'soma_{i}')
        soma.L = 50
        soma.diam = 50
        soma.insert('hh')
        soma.Ra = 100
        cells.append({'soma': soma, 'gid': i})
    
    # 只给第一个细胞添加刺激
    stim = h.IClamp(cells[0]['soma'](0.5))
    stim.delay = 10
    stim.dur = 50
    stim.amp = 0.5
    
    h.celsius = 6.3
    
    print(f"✅ 创建了{len(cells)}个细胞")
    print(f"   - Cell 0: 有电流刺激")
    print(f"   - Cell 1: 无刺激")
    print(f"   - 此时没有Gap Junction连接")
    
    # 检查节点索引
    print("\n节点索引:")
    for i, cell in enumerate(cells):
        idx = cell['soma'](0.5).node_index()
        print(f"   - Cell {i}: node_index = {idx}")
    
    # ========================================
    # 步骤2: 准备导出到HELIOX
    # ========================================
    print("\n" + "="*60)
    print("步骤2: 导出模型到HELIOX（创建独立副本）")
    print("="*60)
    
    # 设置ParallelContext
    pc = h.ParallelContext()
    for i, cell in enumerate(cells):
        pc.set_gid2node(i, pc.id())
        nc = h.NetCon(cell['soma'](0.5)._ref_v, None, sec=cell['soma'])
        pc.cell(i, nc)
    
    # 初始化HelioXManager
    
    # 创建监控器（在导出前创建）
    heliox_monitors = []
    heliox_indices = []
    for i, cell in enumerate(cells):
        monitor = manager.create_monitor_wrapper(cell['soma'](0.5), "v")
        heliox_monitors.append(monitor)
    
    # 导出模型
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "gap_comparison")
        
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = 0.025
        h.finitialize(-65)
        pc.nrnbbcore_write(export_path)
        
        # 加载到HELIOX
        manager.set_data_path(export_path)
        manager.load_model()
        
        print("✅ 模型已导出并加载到HELIOX")
        print("   此时HELIOX有了一个独立的副本，与NEURON分离")
        
        # 获取HELIOX中的索引
        for i, monitor in enumerate(heliox_monitors):
            monitor._initialize()
            idx = monitor._node_or_mech_idx
            heliox_indices.append(idx)
            print(f"   - HELIOX Cell {i} 使用索引: {idx}")
        
        # ========================================
        # 步骤3: 在NEURON中添加Gap Junction
        # ========================================
        print("\n" + "="*60)
        print("步骤3: 在NEURON中添加Gap Junction")
        print("="*60)
        
        # 创建新的ParallelContext用于Gap Junction
        pc_gap = h.ParallelContext()
        
        # Cell 0 -> Cell 1
        pc_gap.source_var(cells[0]['soma'](0.5)._ref_v, 0, sec=cells[0]['soma'])
        pc_gap.target_var(cells[1]['soma'](0.5)._ref_v, 0)
        
        # # Cell 1 -> Cell 0 (双向)
        # pc_gap.source_var(cells[1]['soma'](0.5)._ref_v, 1, sec=cells[1]['soma'])
        # pc_gap.target_var(cells[0]['soma'](0.5)._ref_v, 1)
        
        pc_gap.setup_transfer()
        print("✅ NEURON Gap Junction创建完成（双向连接）")
        
        # ========================================
        # 步骤4: 在HELIOX中添加相同的Gap Junction
        # ========================================
        print("\n" + "="*60)
        print("步骤4: 在HELIOX中添加相同的Gap Junction")
        print("="*60)
        
        gap = GapJunctionInterface()
        
        # Cell 0 -> Cell 1 (添加源和目标)
        sid1 = 100  # 指定sid
        result1 = gap.add_gap_source(sid1, "global", "v", heliox_indices[0])
        if result1 >= 0:  # 成功时返回sid（>=0）
            success1 = gap.add_gap_target(sid1, "global", "v", heliox_indices[1])  # 返回True/False
        else:
            success1 = False
            
        # # Cell 1 -> Cell 0 (双向，添加源和目标)
        # sid2 = 101  # 指定sid
        # result2 = gap.add_gap_source(sid2, "global", "v", heliox_indices[1])
        # if result2 >= 0:  # 成功时返回sid（>=0）
        #     success2 = gap.add_gap_target(sid2, "global", "v", heliox_indices[0])  # 返回True/False
        # else:
        #     success2 = False
        
        # if success1 and success2:
        #     print(f"✅ HELIOX Gap Junction创建完成（双向连接）")
        #     print(f"   - Gap 1: Cell 0 (idx={heliox_indices[0]}) -> Cell 1 (idx={heliox_indices[1]})")
        #     print(f"   - Gap 2: Cell 1 (idx={heliox_indices[1]}) -> Cell 0 (idx={heliox_indices[0]})")
        # else:
        #     print("❌ HELIOX Gap Junction创建失败")
        
        # ========================================
        # 步骤5: 并行运行NEURON和HELIOX仿真
        # ========================================
        print("\n" + "="*60)
        print("步骤5: 并行运行仿真")
        print("="*60)
        
        # NEURON记录
        v_neuron_rec = []
        for cell in cells:
            vec = h.Vector()
            vec.record(cell['soma'](0.5)._ref_v)
            v_neuron_rec.append(vec)
        
        # 运行NEURON
        print("\n运行NEURON仿真...")
        h.dt = 0.025
        h.finitialize(-65)
        h.continuerun(100)
        v_neuron = [np.array(v) for v in v_neuron_rec]
        print(f"✅ NEURON完成")
        print(f"   - Cell 0 最大电压: {np.max(v_neuron[0]):.2f} mV")
        print(f"   - Cell 1 最大电压: {np.max(v_neuron[1]):.2f} mV")
        
        # 运行HELIOX
        print("\n运行HELIOX仿真...")
        manager.finitialize(-65)
        manager.run(100)
        v_heliox = [m.data for m in heliox_monitors]
        print(f"✅ HELIOX完成")
        print(f"   - Cell 0 最大电压: {np.max(v_heliox[0]):.2f} mV")
        print(f"   - Cell 1 最大电压: {np.max(v_heliox[1]):.2f} mV")
        
        # ========================================
        # 步骤6: 结果对比分析
        # ========================================
        print("\n" + "="*60)
        print("步骤6: 结果对比分析")
        print("="*60)
        
        # 确保长度一致
        min_len = min(len(v_neuron[0]), len(v_heliox[0]))
        
        # 计算统计
        print("\n详细对比:")
        max_diffs = []
        for i in range(2):
            v_n = v_neuron[i][:min_len]
            v_ng = v_heliox[i][:min_len]
            
            max_diff = np.max(np.abs(v_n - v_ng))
            mean_diff = np.mean(np.abs(v_n - v_ng))
            max_diffs.append(max_diff)
            
            print(f"\nCell {i}:")
            print(f"   峰值电压:")
            print(f"      - NEURON:  {np.max(v_n):.4f} mV")
            print(f"      - HELIOX: {np.max(v_ng):.4f} mV")
            print(f"   差异分析:")
            print(f"      - 最大差异: {max_diff:.4f} mV")
            print(f"      - 平均差异: {mean_diff:.4f} mV")
        
        # 检查Gap Junction效果
        print("\nGap Junction效果验证:")
        neuron_gap_works = np.max(v_neuron[1]) > -60
        heliox_gap_works = np.max(v_heliox[1]) > -60
        
        print(f"   NEURON:  {'✅ Gap传输成功' if neuron_gap_works else '❌ Gap传输失败'}")
        print(f"   HELIOX: {'✅ Gap传输成功' if heliox_gap_works else '❌ Gap传输失败'}")
        
        # ========================================
        # 步骤7: 绘图
        # ========================================
        time_points = np.arange(min_len) * 0.025
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cell 0对比
        ax = axes[0, 0]
        ax.plot(time_points, v_neuron[0][:min_len], 'b-', label='NEURON', linewidth=2)
        ax.plot(time_points, v_heliox[0][:min_len], 'r--', label='HELIOX', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Cell 0 (Stimulated)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cell 1对比
        ax = axes[0, 1]
        ax.plot(time_points, v_neuron[1][:min_len], 'b-', label='NEURON', linewidth=2)
        ax.plot(time_points, v_heliox[1][:min_len], 'r--', label='HELIOX', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Cell 1 (Gap-connected)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cell 0差异
        ax = axes[1, 0]
        diff0 = v_neuron[0][:min_len] - v_heliox[0][:min_len]
        ax.plot(time_points, diff0, 'g-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Difference (mV)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Cell 0: NEURON - HELIOX')
        ax.grid(True, alpha=0.3)
        
        # Cell 1差异
        ax = axes[1, 1]
        diff1 = v_neuron[1][:min_len] - v_heliox[1][:min_len]
        ax.plot(time_points, diff1, 'm-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Difference (mV)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Cell 1: NEURON - HELIOX')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Gap Junction: NEURON vs HELIOX (独立动态创建)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gap_comparison_true.png', dpi=150)
        print("\n📊 对比图已保存为 gap_comparison_true.png")
        
        # ========================================
        # 最终判定
        # ========================================
        print("\n" + "="*60)
        print("最终结论")
        print("="*60)
        
        if max(max_diffs) < 0.01:
            print("✅ 完美匹配！NEURON和HELIOX的Gap Junction实现完全一致")
        elif max(max_diffs) < 0.1:
            print("✅ 高度一致！差异在数值误差范围内")
        elif max(max_diffs) < 1.0:
            print("✅ 基本一致！Gap Junction功能正常")
        else:
            print(f"⚠️ 存在差异 (最大 {max(max_diffs):.2f} mV)，需要进一步调查")
        
        if neuron_gap_works and heliox_gap_works:
            print("✅ 两个系统的Gap Junction传输都成功")
        else:
            print("❌ Gap Junction传输有问题")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()