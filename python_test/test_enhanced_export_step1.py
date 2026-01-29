#!/usr/bin/env python3
"""
增强版导出功能测试 - 步骤1：导出模型
=====================================

此脚本演示如何使用增强版导出功能，将NEURON模型和wrapper元数据一起导出。
导出后的模型可以在没有NEURON的环境中加载（见test_enhanced_export_step2.py）。

使用方法：
1. 首先运行此脚本导出模型：python test_enhanced_export_step1.py
2. 然后运行step2脚本加载模型：python test_enhanced_export_step2.py
"""

import os
import sys
import numpy as np
from neuron import h, coreneuron
from heliox_wrapper import HelioXManager

# 配置NEURON
h.usetable_hh = 0
h.load_file("nrngui.hoc")
h.cvode.cache_efficient(1)
coreneuron.enable = True

def create_test_model():
    """创建一个包含多种wrapper类型的测试模型"""
    print("🔬 创建测试模型...")
    
    # 创建两个细胞
    cells = []
    for i in range(2):
        soma = h.Section(name=f'soma_{i}')
        soma.L = 30
        soma.diam = 30
        soma.insert('hh')
        
        dend = h.Section(name=f'dend_{i}')
        dend.L = 100
        dend.diam = 2
        dend.insert('pas')
        dend.connect(soma(1))
        
        cells.append({'soma': soma, 'dend': dend})
    
    # 添加突触和刺激
    iclamps = []
    synapses = []
    vecstims = []
    
    for i, cell in enumerate(cells):
        # IClamp (电流钳)
        iclamp = h.IClamp(cell['soma'](0.5))
        iclamp.delay = 10 + i * 5
        iclamp.dur = 50
        iclamp.amp = 0.1 + i * 0.05
        iclamps.append(iclamp)
        
        # ExpSyn (突触)
        syn = h.ExpSyn(cell['soma'](0.5))
        syn.tau = 5.0
        synapses.append(syn)
        
        # VecStim (用于突触输入)
        vstim = h.VecStim()
        vstim.play(h.Vector([20.0 + i*10, 40.0 + i*10, 60.0 + i*10]))
        vecstims.append(vstim)
    
    # 设置网络连接（用于NEURON导出）
    pc = h.ParallelContext()
    netcons = []
    for i in range(len(cells)):
        gid = i
        pc.set_gid2node(gid, pc.id())
        nc = h.NetCon(vecstims[i], synapses[i])
        nc.threshold = 0.1
        nc.weight[0] = 0.2
        pc.cell(gid, nc)
        netcons.append(nc)
    
    h.celsius = 6.3
    
    return cells, iclamps, synapses, vecstims, pc

def test_enhanced_export():
    """测试增强版导出功能"""
    print("\n" + "="*60)
    print("🚀 增强版导出功能测试 - 步骤1：导出模型")
    print("="*60 + "\n")
    
    # 1. 创建模型
    cells, iclamps, synapses, vecstims, pc = create_test_model()
    print("✅ 模型创建完成")
    print(f"  - {len(cells)} 个细胞")
    print(f"  - {len(iclamps)} 个IClamp")
    print(f"  - {len(synapses)} 个突触")
    print(f"  - {len(vecstims)} 个VecStim")
    
    # 2. 初始化HelioXManager
    print("\n📦 初始化HelioXManager...")
    manager = HelioXManager()
    manager.device = "cpu"  # 使用CPU模式
    manager.permute_type = 0
    
    # 3. 创建各种类型的wrapper
    print("\n🔧 创建wrapper...")
    
    # 监控器wrapper（记录电压）
    monitors = []
    for i, cell in enumerate(cells):
        soma_v_monitor = manager.create_monitor_wrapper(cell['soma'](0.5), "v")
        dend_v_monitor = manager.create_monitor_wrapper(cell['dend'](0.5), "v")
        monitors.extend([soma_v_monitor, dend_v_monitor])
    print(f"  ✅ 创建了 {len(monitors)} 个MonitorWrapper")
    
    # 对象wrapper（控制参数）
    obj_wrappers = []
    for iclamp in iclamps:
        obj_wrapper = manager.create_obj_wrapper(iclamp)
        obj_wrappers.append(obj_wrapper)
    for vstim in vecstims:
        obj_wrapper = manager.create_obj_wrapper(vstim)
        obj_wrappers.append(obj_wrapper)
    print(f"  ✅ 创建了 {len(obj_wrappers)} 个ObjWrapper")
    
    # VecPlay wrapper（时变控制）
    vecplay_wrappers = []
    for iclamp in iclamps:
        vecplay = manager.create_vecplay_wrapper(iclamp, "amp")
        vecplay_wrappers.append(vecplay)
    print(f"  ✅ 创建了 {len(vecplay_wrappers)} 个VecPlayWrapper")
    
    # 4. 使用增强版导出
    export_path = "./enhanced_export_test_model"
    print(f"\n💾 导出模型到: {export_path}")
    
    # 这是关键调用！使用enhanced_export_model而不是setup_and_load_model
    manager.enhanced_export_model(
        export_path=export_path,
        dt=0.025,
        v_init=-65.0
    )
    
    print("✅ 模型导出完成！")
    
    # 5. 检查导出的文件
    print("\n📂 检查导出的文件:")
    if os.path.exists(export_path):
        files = os.listdir(export_path)
        dat_files = [f for f in files if f.endswith('.dat')]
        json_files = [f for f in files if f.endswith('.json')]
        
        print(f"  - {len(dat_files)} 个.dat文件（NEURON模型）")
        print(f"  - {len(json_files)} 个.json文件（元数据）")
        
        if "heliox_metadata.json" in files:
            print("  ✅ heliox_metadata.json 存在")
        if "heliox_config.json" in files:
            print("  ✅ heliox_config.json 存在")
        
        # 显示元数据内容摘要
        import json
        metadata_path = os.path.join(export_path, "heliox_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print("\n📊 元数据摘要:")
                print(f"  - 版本: {metadata.get('version', 'N/A')}")
                print(f"  - 导出时间: {metadata.get('export_timestamp', 'N/A')}")
                wrappers = metadata.get('wrappers', {})
                print(f"  - Monitors: {len(wrappers.get('monitors', []))} 个")
                print(f"  - ObjWrappers: {len(wrappers.get('obj_wrappers', []))} 个")
                print(f"  - VecPlayWrappers: {len(wrappers.get('vecplay_wrappers', []))} 个")
    
    print("\n" + "="*60)
    print("✅ 步骤1完成！")
    print("📌 请运行 test_enhanced_export_step2.py 来测试导入功能")
    print("="*60)

if __name__ == "__main__":
    test_enhanced_export()