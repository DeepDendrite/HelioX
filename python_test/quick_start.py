#!/usr/bin/env python3
"""
HelioXWrapper 快速入门示例
==========================

最简单的使用方式演示，适合快速上手。
"""

from neuron import h
from heliox_wrapper import HelioXManager
import numpy as np
import tempfile
import os

def quick_start():
    """5分钟快速入门"""
    
    print("🚀 HelioXWrapper 快速入门")
    print("=" * 40)
    
    # 步骤1: 创建简单的NEURON模型
    print("📦 步骤1: 创建NEURON模型")
    soma = h.Section(name='soma')
    soma.L = soma.diam = 30
    soma.insert('hh')
    
    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 50
    iclamp.dur = 100  
    iclamp.amp = 0.2
    
    # 重要：设置GID用于导出
    pc = h.ParallelContext()
    gid = 0
    pc.set_gid2node(gid, pc.id())
    nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
    pc.cell(gid, nc)
    
    v = h.Vector()
    v.record(soma(0.5)._ref_v)
    
    h.dt = 0.025
    pc.set_maxstep(5)
    pc.setup_transfer()
    h.finitialize(-65)
    pc.psolve(200)
    v = v.to_python()
    
    print("  ✅ 创建了单室神经元 + 电流钳")
    
    # 步骤2: 初始化包装器管理器
    print("\n🎯 步骤2: 初始化HelioXManager")
    manager = HelioXManager()
    
    # 步骤3: 创建包装器
    print("\n🔧 步骤3: 创建包装器")
    v_monitor = manager.create_monitor_wrapper(soma(0.5), "v")  # 电压监控
    iclamp_ctrl = manager.create_obj_wrapper(iclamp)       # 电流控制
    
    print("  ✅ 创建了电压监控器和电流控制器")
    
    # 步骤4: 导出并加载模型
    print("\n⚙️  步骤4: 导出NEURON模型到HelioX")
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "model")
        # manager.device = "cpu"
        # manager.permute_type = 0
        manager.setup_and_load_model(export_path, dt=0.025, v_init=-65)
        
        # 步骤5: 修改参数并运行仿真
        print("\n🎮 步骤5: 控制参数并运行仿真")
        print(f"  原始电流: {iclamp_ctrl.amp:.2f} nA")
        
        # iclamp_ctrl.amp = 0.3  # 修改电流强度
        # print(f"  修改电流: {iclamp_ctrl.amp:.2f} nA")
        
        manager.finitialize(-65)
        manager.run(200)  # 运行200ms
        
        import matplotlib.pyplot as plt
        plt.plot(v, label='NEURON')
        plt.plot(v_monitor.data, label='HelioX', linestyle='--')
        plt.legend()
        plt.savefig("voltage_neuron.png")
        
        # 步骤6: 获取结果
        print("\n📊 步骤6: 获取仿真结果")
        voltage = v_monitor.data
        print(f"  电压数据: {len(voltage)} 个点")
        print(f"  电压范围: {np.min(voltage):.1f} ~ {np.max(voltage):.1f} mV")
        
        # 检查是否产生了动作电位
        if np.max(voltage) > 0:
            print("  🎉 成功产生动作电位!")
        else:
            print("  😴 未产生动作电位")
            
        # 获取gid 0的spk时间戳
        spk_vec = manager.get_spk_by_gid(0)
        print("gid 0 spk_vec: ", spk_vec)
    
    print("\n✅ 快速入门完成!")

if __name__ == "__main__":
    quick_start()
