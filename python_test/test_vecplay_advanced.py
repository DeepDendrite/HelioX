#!/usr/bin/env python3
"""
VecPlay高级测试
测试多变量控制、多细胞、复杂索引等场景
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

class VecPlayAdvancedTest:
    """VecPlay高级测试类"""
    
    def __init__(self):
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        """记录测试结果"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status} {test_name}: {details}")
        
    def test_multiple_iclamps_single_cell(self, device_mode="cpu"):
        """测试1: 单细胞多个IClamp控制"""
        print(f"\n🎯 测试1: 单细胞多个IClamp ({device_mode.upper()})")
        print("-" * 60)
        
        try:
            # === NEURON参考模型 ===
            print("🧬 NEURON参考模型（双IClamp）...")
            
            # 清理
            for sec in h.allsec():
                h.delete_section(sec=sec)
            
            # 创建模型
            soma = h.Section(name='soma')
            soma.nseg = 1
            soma.diam = 1
            soma.L = 1
            soma.insert('hh')
            
            # 创建两个IClamp
            iclamp1 = h.IClamp(soma(0.3))  # 位置0.3
            iclamp1.amp = 0
            iclamp1.dur = 1e9
            iclamp1.delay = 0
            
            iclamp2 = h.IClamp(soma(0.7))  # 位置0.7
            iclamp2.amp = 0
            iclamp2.dur = 1e9
            iclamp2.delay = 0
            
            # 不同的刺激模式
            # IClamp1: 早期刺激
            times1 = [0, 20, 20, 40, 40, 100]
            currents1 = [0, 0, 0.01, 0.01, 0, 0]
            
            # IClamp2: 晚期刺激
            times2 = [0, 60, 60, 80, 80, 100]
            currents2 = [0, 0, 0.012, 0.012, 0, 0]
            
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
            
            neuron_v = np.array(v_vec.as_numpy())
            neuron_i1 = np.array(i1_vec.as_numpy())
            neuron_i2 = np.array(i2_vec.as_numpy())
            
            print(f"   NEURON: 电压范围 {neuron_v.min():.2f}~{neuron_v.max():.2f} mV")
            print(f"   IClamp1电流: {neuron_i1.min():.6f}~{neuron_i1.max():.6f} nA")
            print(f"   IClamp2电流: {neuron_i2.min():.6f}~{neuron_i2.max():.6f} nA")
            
            # === HelioX模型 ===
            print("\n🚀 HelioX模型（双IClamp）...")
            
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
            
            iclamp1_ng = h.IClamp(soma_ng(0.3))
            iclamp1_ng.amp = 0
            iclamp1_ng.dur = 1e9
            iclamp1_ng.delay = 0
            
            iclamp2_ng = h.IClamp(soma_ng(0.7))
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
            if device_mode == "cpu":
                heliox_manager.set_default_device("cpu")
                heliox_manager.set_default_permute_type(0)
            
            # 创建两个VecPlay包装器
            vecplay1 = heliox_manager.create_vecplay_wrapper(iclamp1_ng, "amp")
            vecplay2 = heliox_manager.create_vecplay_wrapper(iclamp2_ng, "amp")
            v_monitor = heliox_manager.create_monitor_wrapper(soma_ng(0.5), "v")
            
            print(f"   VecPlay1信息: {vecplay1.get_info()}")
            print(f"   VecPlay2信息: {vecplay2.get_info()}")
            
            # 导出和加载
            export_path = f"./test1_multi_iclamp_{device_mode}_output"
            heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
            
            # 设置VecPlay
            vecplay1.play(times1, currents1)
            vecplay2.play(times2, currents2)
            
            # 运行
            heliox_manager.client.finitialize(-65.0)
            heliox_manager.client.run(100.0)
            
            heliox_v = np.array(v_monitor.get_data())
            
            print(f"   HelioX: 电压范围 {heliox_v.min():.2f}~{heliox_v.max():.2f} mV")
            
            # 对比
            min_len = min(len(neuron_v), len(heliox_v))
            diff = np.abs(neuron_v[:min_len] - heliox_v[:min_len])
            max_diff = np.max(diff)
            
            passed = max_diff < 0.001
            self.log_test(f"多IClamp_{device_mode}", passed, f"最大差异: {max_diff:.6f} mV")
            
            pc.done()
            
            return passed, neuron_v, heliox_v, neuron_i1, neuron_i2
            
        except Exception as e:
            self.log_test(f"多IClamp_{device_mode}", False, f"异常: {str(e)}")
            return False, None, None, None, None
            
    def test_multiple_cells(self, device_mode="cpu"):
        """测试2: 多细胞VecPlay控制"""
        print(f"\n🎯 测试2: 多细胞控制 ({device_mode.upper()})")
        print("-" * 60)
        
        try:
            ncells = 3  # 3个细胞
            
            # === NEURON参考模型 ===
            print(f"🧬 NEURON参考模型（{ncells}个细胞）...")
            
            # 清理
            for sec in h.allsec():
                h.delete_section(sec=sec)
            
            # 创建多个细胞
            cells = []
            for i in range(ncells):
                soma = h.Section(name=f'soma_{i}')
                soma.nseg = 1
                soma.diam = 1
                soma.L = 1
                soma.insert('hh')
                
                iclamp = h.IClamp(soma(0.5))
                iclamp.amp = 0
                iclamp.dur = 1e9
                iclamp.delay = 0
                
                # 不同的刺激模式
                delay = i * 20  # 每个细胞延迟20ms
                times = [0, 10+delay, 10+delay, 30+delay, 30+delay, 100]
                currents = [0, 0, 0.01+i*0.002, 0.01+i*0.002, 0, 0]  # 递增的电流
                
                tvec = h.Vector(times)
                ivec = h.Vector(currents)
                tvec.play(iclamp._ref_amp, ivec, 1)
                
                # 记录电压
                v_vec = h.Vector()
                v_vec.record(soma(0.5)._ref_v)
                
                cells.append({
                    'soma': soma,
                    'iclamp': iclamp,
                    'v_vec': v_vec,
                    'times': times,
                    'currents': currents
                })
            
            # 运行
            h.dt = 0.025
            h.finitialize(-65)
            h.continuerun(100)
            
            # 收集结果
            neuron_voltages = []
            for i, cell in enumerate(cells):
                voltage = np.array(cell['v_vec'].as_numpy())
                neuron_voltages.append(voltage)
                print(f"   NEURON细胞{i}: 电压范围 {voltage.min():.2f}~{voltage.max():.2f} mV")
            
            # === HelioX模型 ===
            print(f"\n🚀 HelioX模型（{ncells}个细胞）...")
            
            # 清理
            for sec in h.allsec():
                h.delete_section(sec=sec)
            
            # 创建ParallelContext
            pc = h.ParallelContext()
            
            # 创建多个细胞
            ng_cells = []
            for i in range(ncells):
                soma = h.Section(name=f'soma_ng_{i}')
                soma.nseg = 1
                soma.diam = 1
                soma.L = 1
                soma.insert('hh')
                
                iclamp = h.IClamp(soma(0.5))
                iclamp.amp = 0
                iclamp.dur = 1e9
                iclamp.delay = 0
                
                # 注册到PC
                gid = i
                pc.set_gid2node(gid, int(pc.id()))
                soma.push()
                pc.cell(gid, h.NetCon(soma(0.5)._ref_v, None))
                h.pop_section()
                
                ng_cells.append({
                    'soma': soma,
                    'iclamp': iclamp,
                    'times': cells[i]['times'],
                    'currents': cells[i]['currents']
                })
            
            # PC设置
            spike_tvec = h.Vector()
            spike_idvec = h.Vector()
            pc.spike_record(-1, spike_tvec, spike_idvec)
            pc.setup_transfer()
            pc.set_maxstep(10)
            
            # 创建HelioX管理器
            heliox_manager = HelioXManager()
            if device_mode == "cpu":
                heliox_manager.set_default_device("cpu")
                heliox_manager.set_default_permute_type(0)
            
            # 为每个细胞创建VecPlay和Monitor
            vecplays = []
            monitors = []
            for i, cell in enumerate(ng_cells):
                vecplay = heliox_manager.create_vecplay_wrapper(cell['iclamp'], "amp")
                monitor = heliox_manager.create_monitor_wrapper(cell['soma'](0.5), "v")
                vecplays.append(vecplay)
                monitors.append(monitor)
                
                print(f"   细胞{i} VecPlay: {vecplay.get_info()}")
            
            # 导出和加载
            export_path = f"./test2_multi_cell_{device_mode}_output"
            heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
            
            # 设置所有VecPlay
            for i, (vecplay, cell) in enumerate(zip(vecplays, ng_cells)):
                vecplay.play(cell['times'], cell['currents'])
            
            # 运行
            heliox_manager.client.finitialize(-65.0)
            heliox_manager.client.run(100.0)
            
            # 收集结果
            heliox_voltages = []
            for i, monitor in enumerate(monitors):
                voltage = np.array(monitor.get_data())
                heliox_voltages.append(voltage)
                print(f"   HelioX细胞{i}: 电压范围 {voltage.min():.2f}~{voltage.max():.2f} mV")
            
            # 对比所有细胞
            max_diffs = []
            for i in range(ncells):
                min_len = min(len(neuron_voltages[i]), len(heliox_voltages[i]))
                diff = np.abs(neuron_voltages[i][:min_len] - heliox_voltages[i][:min_len])
                max_diff = np.max(diff)
                max_diffs.append(max_diff)
                print(f"   细胞{i}最大差异: {max_diff:.6f} mV")
            
            overall_max_diff = max(max_diffs)
            passed = overall_max_diff < 0.001
            self.log_test(f"多细胞_{device_mode}", passed, f"最大差异: {overall_max_diff:.6f} mV")
            
            pc.done()
            
            return passed, neuron_voltages, heliox_voltages
            
        except Exception as e:
            self.log_test(f"多细胞_{device_mode}", False, f"异常: {str(e)}")
            return False, None, None
            
    def test_different_mechanisms(self, device_mode="cpu"):
        """测试3: 不同机制的变量控制"""
        print(f"\n🎯 测试3: 不同机制变量控制 ({device_mode.upper()})")
        print("-" * 60)
        
        try:
            # === NEURON参考模型 ===
            print("🧬 NEURON参考模型（IClamp + VClamp）...")
            
            # 清理
            for sec in h.allsec():
                h.delete_section(sec=sec)
            
            # 创建模型
            soma = h.Section(name='soma')
            soma.nseg = 1
            soma.diam = 1
            soma.L = 1
            soma.insert('hh')
            
            # 创建IClamp和VClamp
            iclamp = h.IClamp(soma(0.3))
            iclamp.amp = 0
            iclamp.dur = 1e9
            iclamp.delay = 0
            
            vclamp = h.VClamp(soma(0.7))
            vclamp.amp[0] = -65  # 初始箝位电压
            vclamp.dur[0] = 1e9
            vclamp.gain = 1e5
            vclamp.rstim = 1
            vclamp.tau1 = 0.1
            vclamp.tau2 = 0
            
            # IClamp刺激
            i_times = [0, 20, 20, 40, 40, 100]
            i_currents = [0, 0, 0.01, 0.01, 0, 0]
            
            # VClamp电压变化
            v_times = [0, 60, 60, 80, 80, 100]
            v_voltages = [-65, -65, -40, -40, -65, -65]  # 去极化
            
            # Vector.play()
            i_tvec = h.Vector(i_times)
            i_ivec = h.Vector(i_currents)
            i_tvec.play(iclamp._ref_amp, i_ivec, 1)
            
            v_tvec = h.Vector(v_times)
            v_vvec = h.Vector(v_voltages)
            v_tvec.play(vclamp._ref_amp[0], v_vvec, 1)
            
            # 记录
            soma_v = h.Vector()
            iclamp_i = h.Vector()
            vclamp_v = h.Vector()
            
            soma_v.record(soma(0.5)._ref_v)
            iclamp_i.record(iclamp._ref_amp)
            vclamp_v.record(vclamp._ref_amp[0])
            
            # 运行
            h.dt = 0.025
            h.finitialize(-65)
            h.continuerun(100)
            
            neuron_v = np.array(soma_v.as_numpy())
            neuron_i = np.array(iclamp_i.as_numpy())
            neuron_vc = np.array(vclamp_v.as_numpy())
            
            print(f"   NEURON: 电压范围 {neuron_v.min():.2f}~{neuron_v.max():.2f} mV")
            print(f"   IClamp电流: {neuron_i.min():.6f}~{neuron_i.max():.6f} nA")
            print(f"   VClamp电压: {neuron_vc.min():.2f}~{neuron_vc.max():.2f} mV")
            
            # === HelioX模型 ===
            print("\n🚀 HelioX模型（IClamp + VClamp）...")
            
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
            
            iclamp_ng = h.IClamp(soma_ng(0.3))
            iclamp_ng.amp = 0
            iclamp_ng.dur = 1e9
            iclamp_ng.delay = 0
            
            vclamp_ng = h.VClamp(soma_ng(0.7))
            vclamp_ng.amp[0] = -65
            vclamp_ng.dur[0] = 1e9
            vclamp_ng.gain = 1e5
            vclamp_ng.rstim = 1
            vclamp_ng.tau1 = 0.1
            vclamp_ng.tau2 = 0
            
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
            if device_mode == "cpu":
                heliox_manager.set_default_device("cpu")
                heliox_manager.set_default_permute_type(0)
            
            # 创建不同机制的VecPlay
            iclamp_vecplay = heliox_manager.create_vecplay_wrapper(iclamp_ng, "amp")
            vclamp_vecplay = heliox_manager.create_vecplay_wrapper(vclamp_ng, "amp[0]")  # 注意VClamp的变量名
            v_monitor = heliox_manager.create_monitor_wrapper(soma_ng(0.5), "v")
            
            print(f"   IClamp VecPlay: {iclamp_vecplay.get_info()}")
            print(f"   VClamp VecPlay: {vclamp_vecplay.get_info()}")
            
            # 导出和加载
            export_path = f"./test3_diff_mech_{device_mode}_output"
            heliox_manager.setup_and_load_model(export_path, dt=0.025, v_init=-65.0)
            
            # 设置VecPlay
            iclamp_vecplay.play(i_times, i_currents)
            vclamp_vecplay.play(v_times, v_voltages)
            
            # 运行
            heliox_manager.client.finitialize(-65.0)
            heliox_manager.client.run(100.0)
            
            heliox_v = np.array(v_monitor.get_data())
            
            print(f"   HelioX: 电压范围 {heliox_v.min():.2f}~{heliox_v.max():.2f} mV")
            
            # 对比
            min_len = min(len(neuron_v), len(heliox_v))
            diff = np.abs(neuron_v[:min_len] - heliox_v[:min_len])
            max_diff = np.max(diff)
            
            passed = max_diff < 0.001
            self.log_test(f"不同机制_{device_mode}", passed, f"最大差异: {max_diff:.6f} mV")
            
            pc.done()
            
            return passed, neuron_v, heliox_v
            
        except Exception as e:
            self.log_test(f"不同机制_{device_mode}", False, f"异常: {str(e)}")
            return False, None, None
            
    def plot_multi_test_results(self, test1_data, test2_data, test3_data, device_mode):
        """绘制多测试结果对比"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 测试1: 多IClamp
        if test1_data[1] is not None:
            neuron_v, heliox_v, neuron_i1, neuron_i2 = test1_data[1:]
            min_len = min(len(neuron_v), len(heliox_v))
            time = np.arange(0, min_len * 0.025, 0.025)
            
            axes[0,0].plot(time, neuron_v[:min_len], 'b-', label='NEURON', linewidth=2)
            axes[0,0].plot(time, heliox_v[:min_len], 'r--', label='HelioX', linewidth=2)
            axes[0,0].set_title(f'测试1: 多IClamp电压对比 ({device_mode})')
            axes[0,0].set_ylabel('电压 (mV)')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 电流图
            axes[0,1].plot(time, neuron_i1[:min_len], 'orange', label='IClamp1', linewidth=2)
            axes[0,1].plot(time, neuron_i2[:min_len], 'purple', label='IClamp2', linewidth=2)
            axes[0,1].set_title('电流控制验证')
            axes[0,1].set_ylabel('电流 (nA)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 测试2: 多细胞
        if test2_data[1] is not None:
            neuron_voltages, heliox_voltages = test2_data[1:]
            ncells = len(neuron_voltages)
            
            for i in range(min(ncells, 3)):  # 最多显示3个细胞
                min_len = min(len(neuron_voltages[i]), len(heliox_voltages[i]))
                time = np.arange(0, min_len * 0.025, 0.025)
                
                if i == 0:
                    axes[1,0].plot(time, neuron_voltages[i][:min_len], 'b-', label='NEURON', linewidth=2)
                    axes[1,0].plot(time, heliox_voltages[i][:min_len], 'r--', label='HelioX', linewidth=2)
                else:
                    axes[1,0].plot(time, neuron_voltages[i][:min_len], 'b-', alpha=0.6, linewidth=1)
                    axes[1,0].plot(time, heliox_voltages[i][:min_len], 'r--', alpha=0.6, linewidth=1)
            
            axes[1,0].set_title(f'测试2: 多细胞电压对比 ({device_mode})')
            axes[1,0].set_ylabel('电压 (mV)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # 峰值电压对比
            peak_neuron = [np.max(v) for v in neuron_voltages]
            peak_heliox = [np.max(v) for v in heliox_voltages]
            x = range(ncells)
            
            axes[1,1].bar([i-0.2 for i in x], peak_neuron, 0.4, label='NEURON', alpha=0.7)
            axes[1,1].bar([i+0.2 for i in x], peak_heliox, 0.4, label='HelioX', alpha=0.7)
            axes[1,1].set_title('各细胞峰值电压对比')
            axes[1,1].set_xlabel('细胞编号')
            axes[1,1].set_ylabel('峰值电压 (mV)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 测试3: 不同机制
        if test3_data[1] is not None:
            neuron_v, heliox_v = test3_data[1:]
            min_len = min(len(neuron_v), len(heliox_v))
            time = np.arange(0, min_len * 0.025, 0.025)
            diff = neuron_v[:min_len] - heliox_v[:min_len]
            
            axes[2,0].plot(time, neuron_v[:min_len], 'b-', label='NEURON', linewidth=2)
            axes[2,0].plot(time, heliox_v[:min_len], 'r--', label='HelioX', linewidth=2)
            axes[2,0].set_title(f'测试3: 不同机制电压对比 ({device_mode})')
            axes[2,0].set_xlabel('时间 (ms)')
            axes[2,0].set_ylabel('电压 (mV)')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
            
            # 差异图
            axes[2,1].plot(time, diff, 'g-', linewidth=1)
            axes[2,1].set_title(f'电压差异 (最大: {np.max(np.abs(diff)):.6f} mV)')
            axes[2,1].set_xlabel('时间 (ms)')
            axes[2,1].set_ylabel('电压差异 (mV)')
            axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'vecplay_advanced_tests_{device_mode}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 高级测试图已保存: {filename}")
        plt.close()
        
    def run_all_tests(self):
        """运行所有高级测试"""
        print("🧪 VecPlay高级测试套件")
        print("=" * 80)
        
        all_passed = True
        
        for device_mode in ["cpu", "gpu"]:
            print(f"\n{'='*20} {device_mode.upper()}模式测试 {'='*20}")
            
            # 运行三个测试
            test1_result = self.test_multiple_iclamps_single_cell(device_mode)
            test2_result = self.test_multiple_cells(device_mode)
            test3_result = self.test_different_mechanisms(device_mode)
            
            # 绘制结果
            self.plot_multi_test_results(test1_result, test2_result, test3_result, device_mode)
            
            # 更新总体结果
            all_passed = all_passed and test1_result[0] and test2_result[0] and test3_result[0]
        
        # 生成最终报告
        self.generate_final_report()
        
        return all_passed
        
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 80)
        print("📋 VecPlay高级测试最终报告")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {total_tests - passed_tests} ❌")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n详细结果:")
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['name']}: {result['details']}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 所有高级测试通过！VecPlay支持复杂场景：")
            print(f"  ✅ 单细胞多变量控制")
            print(f"  ✅ 多细胞独立控制")  
            print(f"  ✅ 不同机制变量控制")
            print(f"  ✅ CPU和GPU模式完全兼容")
        else:
            print(f"\n⚠️  存在{total_tests - passed_tests}个高级功能问题。")

def main():
    """主函数"""
    test_suite = VecPlayAdvancedTest()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()