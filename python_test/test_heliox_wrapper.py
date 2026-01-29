#!/usr/bin/env python3
"""
HelioXWrapper 测试和使用范例
============================

这个文件展示了如何使用 heliox_wrapper 来：
1. 创建简单的NEURON模型
2. 使用包装器管理NEURON和HelioX的交互
3. 运行仿真并获取数据
4. 可视化结果

依赖:
- neuron
- heliox
- matplotlib
- numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from heliox_wrapper import HelioXManager
import os
import tempfile

# 全局GID计数器，避免重复分配
_GLOBAL_GID_COUNTER = 0

def get_next_gid():
    """获取下一个可用的GID"""
    global _GLOBAL_GID_COUNTER
    gid = _GLOBAL_GID_COUNTER
    _GLOBAL_GID_COUNTER += 1
    return gid

def reset_neuron():
    """重置NEURON环境，清理所有GID"""
    h.finitialize()
    pc = h.ParallelContext()
    pc.gid_clear()  # 清理所有GID
    print("  🧹 NEURON环境已重置")

class SimpleCell:
    """简单的单室神经元模型"""
    
    def __init__(self, gid=None):
        """创建简单的神经元模型"""
        if gid is None:
            gid = get_next_gid()
        self.gid = gid
        
        # 创建soma
        self.soma = h.Section(name='soma')
        self.soma.L = 30          # 长度 30 μm
        self.soma.diam = 30       # 直径 30 μm
        self.soma.Ra = 100        # 轴向阻抗
        self.soma.cm = 1          # 膜电容
        
        # 插入机制
        self.soma.insert('hh')    # Hodgkin-Huxley 机制
        
        # 创建电流钳刺激
        self.iclamp = h.IClamp(self.soma(0.5))
        self.iclamp.delay = 50    # 延迟 50 ms
        self.iclamp.dur = 100     # 持续 100 ms  
        self.iclamp.amp = 0.1     # 电流 0.1 nA
        
        # 设置ParallelContext和GID (重要！)
        self.pc = h.ParallelContext()
        self.pc.set_gid2node(self.gid, self.pc.id())  # 将GID分配给当前节点
        self.nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)  # 创建NetCon用于spike检测
        self.pc.cell(self.gid, self.nc)  # 将细胞与GID关联
        
        print(f"SimpleCell created (GID={self.gid}):")
        print(f"  Soma: L={self.soma.L} μm, diam={self.soma.diam} μm")
        print(f"  IClamp: delay={self.iclamp.delay} ms, dur={self.iclamp.dur} ms, amp={self.iclamp.amp} nA")


def test_heliox_wrapper():
    """测试 heliox_wrapper 的完整流程"""
    print("=" * 60)
    print("HelioXWrapper 测试开始")
    print("=" * 60)
    
    # 0. 重置NEURON环境
    print("0. 重置NEURON环境...")
    reset_neuron()
    
    # 1. 创建NEURON模型
    print("\n1. 创建NEURON模型...")
    cell = SimpleCell()
    
    # 2. 初始化 HelioXManager
    print("\n2. 初始化 HelioXManager...")
    heliox_manager = HelioXManager()
    heliox_manager.set_default_device("gpu")  # 或 "cpu"
    heliox_manager.set_default_permute_type(3)
    
    # 3. 创建包装器（延迟初始化）
    print("\n3. 创建包装器...")
    
    # 电压监控器 - 记录时间序列
    v_monitor = heliox_manager.create_monitor_wrapper(cell.soma(0.5), "v")
    print("  创建电压监控器")
    
    # 电压对象包装器 - 读写当前时刻的值
    v_obj = heliox_manager.create_obj_wrapper(cell.soma(0.5))
    print("  创建电压对象包装器")
    
    # IClamp监控器 - 记录电流时间序列
    iclamp_monitor = heliox_manager.create_monitor_wrapper(cell.iclamp, "amp")
    print("  创建IClamp监控器")
    
    # IClamp对象包装器 - 读写电流参数
    iclamp_obj = heliox_manager.create_obj_wrapper(cell.iclamp)
    print("  创建IClamp对象包装器")
    
    # 4. 模型导出和加载
    print("\n4. 导出NEURON模型并加载到HelioX...")
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "model")
        
        try:
            heliox_manager.setup_and_load_model(
                export_path=export_path,
                dt=0.025,      # 时间步长 0.025 ms
                v_init=-65.0   # 初始电压 -65 mV
            )
            
            # 5. 测试对象包装器的读写功能
            print("\n5. 测试对象包装器...")
            
            # 读取初始电压
            initial_voltage = v_obj.v
            print(f"  初始电压: {initial_voltage:.2f} mV")
            
            # 读取IClamp参数
            print(f"  IClamp参数:")
            print(f"    amp: {iclamp_obj.amp:.3f} nA")
            print(f"    delay: {iclamp_obj.delay:.1f} ms") 
            print(f"    dur: {iclamp_obj.dur:.1f} ms")
            
            # 修改刺激强度
            print("\n  修改刺激强度从 0.1 nA 到 0.2 nA")
            iclamp_obj.amp = 0.2
            print(f"    修改后 amp: {iclamp_obj.amp:.3f} nA")
            
            # 6. 运行仿真
            print("\n6. 运行仿真...")
            runtime = 200.0  # 仿真 200 ms
            
            heliox_manager.finitialize(-65.0)
            print(f"  开始仿真 {runtime} ms...")
            heliox_manager.run(runtime)
            print("  仿真完成")
            
            # 7. 获取仿真数据
            print("\n7. 获取仿真数据...")
            
            # 获取电压时间序列
            voltage_data = v_monitor.data
            print(f"  电压数据长度: {len(voltage_data)} 个时间点")
            print(f"  电压范围: {np.min(voltage_data):.2f} ~ {np.max(voltage_data):.2f} mV")
            
            # 获取电流时间序列
            current_data = iclamp_monitor.data
            print(f"  电流数据长度: {len(current_data)} 个时间点")
            print(f"  电流范围: {np.min(current_data):.3f} ~ {np.max(current_data):.3f} nA")
            
            # 8. 可视化结果
            print("\n8. 绘制结果...")
            plot_results(voltage_data, current_data, runtime, heliox_manager.dt)
            
            # 9. 测试实时数据读取
            print("\n9. 测试实时数据读取...")
            print("  运行短时仿真并实时读取电压...")
            
            heliox_manager.finitialize(-65.0)
            for i in range(5):
                heliox_manager.run(10.0)  # 每次运行10ms
                current_v = v_obj.v
                print(f"    t={10*(i+1)} ms: v = {current_v:.2f} mV")
            
            print("\n✅ 所有测试通过!")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("HelioXWrapper 测试完成")
    print("=" * 60)


def plot_results(voltage_data, current_data, runtime, dt):
    """绘制仿真结果"""
    # 创建时间轴
    time = np.arange(len(voltage_data)) * dt
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制电压
    ax1.plot(time, voltage_data, 'b-', linewidth=2, label='Membrane Voltage')
    ax1.set_ylabel('Voltage (mV)', fontsize=12)
    ax1.set_title('Neuron Simulation Results (HelioX)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制电流
    ax2.plot(time, current_data, 'r-', linewidth=2, label='Injected Current')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Current (nA)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 设置x轴范围
    for ax in [ax1, ax2]:
        ax.set_xlim(0, runtime)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = "heliox_wrapper_test_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  结果图片已保存: {output_file}")
    
    # 显示图片
    plt.show()


def test_error_handling():
    """测试错误处理和边界情况"""
    print("\n" + "=" * 40)
    print("测试错误处理...")
    print("=" * 40)
    
    # 重置环境
    reset_neuron()
    
    heliox_manager = HelioXManager()
    cell = SimpleCell()  # 使用新的GID
    
    # 测试未初始化的包装器访问
    print("\n1. 测试未初始化的包装器访问...")
    v_monitor = heliox_manager.create_monitor_wrapper(cell.soma(0.5))
    
    try:
        data = v_monitor.data  # 应该抛出异常
        print("  ❌ 应该抛出异常但没有")
    except RuntimeError as e:
        print(f"  ✅ 正确抛出异常: {e}")
    
    # 测试析构函数警告控制
    print("\n2. 测试析构函数警告控制...")
    print("  禁用析构警告...")
    HelioXManager.disable_destructor_warnings()
    
    # 创建一个临时包装器（会被自动回收）
    temp_wrapper = heliox_manager.create_monitor_wrapper(cell.soma(0.5))
    del temp_wrapper  # 手动删除，不应该有警告
    
    print("  重新启用析构警告...")
    HelioXManager.enable_destructor_warnings()


def demo_advanced_usage():
    """演示高级用法"""
    print("\n" + "=" * 40)
    print("高级用法演示...")
    print("=" * 40)
    
    # 重置环境
    reset_neuron()
    
    # 创建多个细胞
    print("\n1. 创建多个神经元...")
    cells = []
    for i in range(3):
        cell = SimpleCell()  # 自动分配新的GID
        cells.append(cell)
    
    heliox_manager = HelioXManager()
    
    # 为每个细胞创建包装器
    monitors = []
    obj_wrappers = []
    
    for i, cell in enumerate(cells):
        # 每个细胞不同的刺激强度
        cell.iclamp.amp = 0.1 * (i + 1)
        
        v_monitor = heliox_manager.create_monitor_wrapper(cell.soma(0.5))
        iclamp_obj = heliox_manager.create_obj_wrapper(cell.iclamp)
        
        monitors.append(v_monitor)
        obj_wrappers.append(iclamp_obj)
        
        print(f"  细胞 {i+1} (GID={cell.gid}): 刺激强度 {cell.iclamp.amp:.1f} nA")
    
    print("\n2. 批量操作演示...")
    
    # 展示包装器的可用变量
    print("  IClamp 可用变量:")
    available_vars = obj_wrappers[0].get_available_vars()
    for var in available_vars:
        print(f"    - {var}")
    
    print("\n✅ 高级用法演示完成")


if __name__ == "__main__":
    """主函数"""
    try:
        # 基本功能测试
        test_heliox_wrapper()
        
        # 错误处理测试
        test_error_handling()
        
        # 高级用法演示
        demo_advanced_usage()
        
        print("\n🎉 所有测试和演示完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理警告
        HelioXManager.disable_destructor_warnings()
        print("\n🧹 清理完成")
