#!/usr/bin/env python3
"""
HelioXWrapper 快速入门示例
==========================

最简单的使用方式演示，适合快速上手。
"""

from neuron import h, coreneuron
from heliox_wrapper import HelioXManager
import numpy as np
import tempfile
import os
h.usetable_hh = 0
h.load_file("nrngui.hoc")
h.cvode.cache_efficient(1)
coreneuron.enable = True

def quick_start():
    """5分钟快速入门"""
    
    
    print("🚀 HelioXWrapper 快速入门")
    print("=" * 40)
    
    # 步骤1: 创建简单的NEURON模型
    print("📦 步骤1: 创建NEURON模型")
    soma = h.Section(name='soma')
    soma.L = soma.diam = 30
    soma.insert('hh')
    h.celsius = 6.3
    
    stim = h.VecStim()
    stim.play(h.Vector([10.0, 20.0, 30.0]))
    
    syn = h.ExpSyn(soma(0.5))
    
    # 重要：设置GID用于导出
    pc = h.ParallelContext()
    gid = 0
    pc.set_gid2node(gid, pc.id())
    nc = h.NetCon(stim, syn)
    nc.threshold = 0.1
    nc.weight[0] = 0.3
    pc.cell(gid, nc)
    
    v_out = h.Vector()
    v_out.record(soma(0.5)._ref_v)
    
    h.dt = 0.025
    pc.set_maxstep(5)
    pc.setup_transfer()
    h.finitialize(-65)
    pc.psolve(200)
    
    v_out = v_out.to_python()
    
    print("  ✅ 创建了单室神经元 + 电流钳")
    
    # 步骤2: 初始化包装器管理器
    print("\n🎯 步骤2: 初始化HelioXManager")
    manager = HelioXManager()
    #manager.permute_type = 0
    #manager.device = "cpu"  # 使用CPU设备
    
    # 步骤3: 创建包装器
    print("\n🔧 步骤3: 创建包装器")
    v_monitor = manager.create_monitor_wrapper(soma(0.5), "v")  # 电压监控
    stim_ctrl = manager.create_obj_wrapper(stim)       # 电流控制
    
    print("  ✅ 创建了电压监控器和电流控制器")
    
    # 步骤4: 导出并加载模型
    print("\n⚙️  步骤4: 导出NEURON模型到HelioX")
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "model")
        manager.setup_and_load_model(export_path, dt=0.025, v_init=-65)
        manager.finitialize(-65)
        manager.run(200)  # 运行200ms
        
        # 步骤5: 获取结果
        print("\n📊 步骤5: 获取仿真结果")
        voltage = v_monitor.data
        print(f"  电压数据: {len(voltage)} 个点")
        print(f"  电压范围: {np.min(voltage):.1f} ~ {np.max(voltage):.1f} mV")
        
        # 检查是否产生了动作电位
        if np.max(voltage) > 0:
            print("  🎉 成功产生动作电位!")
        else:
            print("  😴 未产生动作电位")
            
        # plot
        import matplotlib.pyplot as plt
        # 计算差距
        max_diff = np.max(np.abs(np.array(v_out) - np.array(voltage)))
        max_diff_index = np.argmax(np.abs(np.array(v_out) - np.array(voltage)))
        
        print(f"  max diff: {max_diff:.6f} mV")
        print(f"  max diff index: {max_diff_index}")
        print(f"  voltage[max diff index]: {voltage[max_diff_index]}")
        print(f"  v_out[max diff index]: {np.array(v_out)[max_diff_index]}")
        
        import matplotlib.pyplot as plt
        plt.plot(v_out, label='NEURON')
        plt.plot(voltage, label='HelioX', linestyle='--')
        plt.legend()
        plt.savefig("voltage_neuron.png")
        plt.close()
        
        # 绘制电压差值
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(np.array(v_out) - np.array(voltage)), label='diff')
        plt.legend()
        plt.savefig("voltage_neuron_diff.png")
        plt.close()
        
        # 重新play
        stim_ctrl.play([10.0, 20.0, 30.0, 40.0])
        manager.finitialize(-65)
        manager.run(200)  # 运行200ms
        voltage = v_monitor.data
        
        import matplotlib.pyplot as plt
        plt.plot(v_out, label='NEURON')
        plt.plot(voltage, label='HelioX', linestyle='--')
        plt.legend()
        plt.savefig("voltage_neuron_replay.png")
        plt.close()
        
    
    print("\n✅ 快速入门完成!")

if __name__ == "__main__":
    quick_start()
