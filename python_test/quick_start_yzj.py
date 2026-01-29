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
    
    stim = h.VecStim()
    stim.play(h.Vector([5.0]))
    run_time = 10
    
    syn = h.Exp2Syn(soma(0.5))
    
    # 重要：设置GID用于导出
    pc = h.ParallelContext()
    gid = 1
    pc.set_gid2node(gid, pc.id())
    nc = h.NetCon(stim, syn)
    nc.threshold = 0.1
    nc.weight[0] = 0.3
    nc.delay = 0.0
    pc.cell(gid, nc)
    # nc_log = h.NetCon(soma(0.5)._ref_v, None)
    # pc.cell(gid, nc_log)
    # 
    state_m = h.Vector().record(soma(0.5).hh._ref_m)
    state_h = h.Vector().record(soma(0.5).hh._ref_h)
    state_n = h.Vector().record(soma(0.5).hh._ref_n)
    
    v_out = h.Vector()
    v_out.record(soma(0.5)._ref_v)
    
    h.dt = 0.025
    pc.set_maxstep(5)
    pc.setup_transfer()
    h.finitialize(-65)
    pc.psolve(run_time)
    
    v_out = v_out.to_python()
    
    import matplotlib.pyplot as plt
    plt.plot(v_out, label='NEURON')
    plt.savefig("voltage_neuron.png")
    
    print("  ✅ 创建了单室神经元 + 电流钳")
    
    # 步骤2: 初始化包装器管理器
    print("\n🎯 步骤2: 初始化HelioXManager")
    manager = HelioXManager()
    manager.device = "cpu"  # 设置为CPU设备
    manager.permute_type = 0
    
    # 步骤3: 创建包装器
    print("\n🔧 步骤3: 创建包装器")
    m_monitor = manager.create_monitor_wrapper(soma(0.5).hh, "m")
    h_monitor = manager.create_monitor_wrapper(soma(0.5).hh, "h")
    n_monitor = manager.create_monitor_wrapper(soma(0.5).hh, "n")
    
    v_monitor = manager.create_monitor_wrapper(soma(0.5), "v")  # 电压监控
    stim_ctrl = manager.create_obj_wrapper(stim)       # 电流控制
    
    print("  ✅ 创建了电压监控器和电流控制器")
    
    # 步骤4: 导出并加载模型
    print("\n⚙️  步骤4: 导出NEURON模型到HelioX")
    with tempfile.TemporaryDirectory() as temp_dir:
        # export_path = os.path.join(temp_dir, "model")
        export_path = os.path.join("yzj_model")
        manager.setup_and_load_model(export_path, dt=0.025, v_init=-65)
        # stim_ctrl.play([1.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        manager.finitialize(-65)
        manager.run(run_time)  # 运行30ms

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
        plt.plot(voltage, label='HelioX', linestyle='--')
        plt.plot(max_diff_index, 
                voltage[max_diff_index],
                'ro', label=f'max diff: {max_diff:.6f}mV')
        plt.legend()
        plt.savefig("voltage_neuron.png")
        plt.close()
        # 检查m, h, n
        plt.plot(m_monitor.data, label='m', linestyle='--')
        plt.plot(h_monitor.data, label='h', linestyle='--')
        plt.plot(n_monitor.data, label='n', linestyle='--')
        plt.plot(list(state_m), label='ref_m')
        plt.plot(list(state_h), label='ref_h')
        plt.plot(list(state_n), label='ref_n')
        plt.legend()
        plt.savefig("state_neuron.png")
        plt.close()
        
        # 获取gid 0的spk时间戳
        spk_vec = manager.get_spk_by_gid(0)
        print("gid 0 spk_vec: ", spk_vec)
    
    print("\n✅ 快速入门完成!")

if __name__ == "__main__":
    quick_start()