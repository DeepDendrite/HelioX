#!/usr/bin/env python3
"""
ObjWrapper性能测试
================

测试obj_wrapper读变量和写变量的性能。
测试场景：
1. 读变量性能测试（segment.v 和 IClamp.amp）
2. 写变量性能测试（segment.v 和 IClamp.amp）
3. 批量读写性能测试
"""

from neuron import h
from heliox_wrapper import HelioXManager
import numpy as np
import tempfile
import os
import time
import matplotlib.pyplot as plt

def create_test_model():
    """创建测试用的NEURON模型"""
    print("创建测试模型...")
    
    # 创建多个神经元以增加测试负载
    somas = []
    iclamps = []
    
    for i in range(10):  # 创建10个神经元
        soma = h.Section(name=f'soma_{i}')
        soma.L = soma.diam = 30
        soma.insert('hh')
        somas.append(soma)
        
        iclamp = h.IClamp(soma(0.5))
        iclamp.delay = 50
        iclamp.dur = 100  
        iclamp.amp = 0.2
        iclamps.append(iclamp)
    
    # 设置GID
    pc = h.ParallelContext()
    for i, soma in enumerate(somas):
        gid = i
        pc.set_gid2node(gid, pc.id())
        nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        pc.cell(gid, nc)
    
    return somas, iclamps

def benchmark_read_performance(manager, v_wrappers, iclamp_wrappers, iterations=10000):
    """测试读变量性能"""
    print(f"\n🔍 读变量性能测试 ({iterations}次迭代)")
    print("=" * 50)
    
    # 预热
    for _ in range(100):
        for v_wrapper in v_wrappers:
            _ = v_wrapper.v
        for iclamp_wrapper in iclamp_wrappers:
            _ = iclamp_wrapper.amp
    
    # 测试segment.v读取性能
    start_time = time.time()
    for _ in range(iterations):
        for v_wrapper in v_wrappers:
            _ = v_wrapper.v
    v_read_time = time.time() - start_time
    
    # 测试IClamp.amp读取性能
    start_time = time.time()
    for _ in range(iterations):
        for iclamp_wrapper in iclamp_wrappers:
            _ = iclamp_wrapper.amp
    amp_read_time = time.time() - start_time
    
    # 计算性能指标
    v_ops_per_sec = (iterations * len(v_wrappers)) / v_read_time
    amp_ops_per_sec = (iterations * len(iclamp_wrappers)) / amp_read_time
    
    print(f"Segment.v 读取:")
    print(f"  总时间: {v_read_time:.4f}s")
    print(f"  操作数: {iterations * len(v_wrappers)}")
    print(f"  性能: {v_ops_per_sec:.0f} ops/s")
    print(f"  平均延迟: {v_read_time * 1000000 / (iterations * len(v_wrappers)):.2f} μs/op")
    
    print(f"\nIClamp.amp 读取:")
    print(f"  总时间: {amp_read_time:.4f}s")
    print(f"  操作数: {iterations * len(iclamp_wrappers)}")
    print(f"  性能: {amp_ops_per_sec:.0f} ops/s")
    print(f"  平均延迟: {amp_read_time * 1000000 / (iterations * len(iclamp_wrappers)):.2f} μs/op")
    
    return {
        'v_read_time': v_read_time,
        'amp_read_time': amp_read_time,
        'v_ops_per_sec': v_ops_per_sec,
        'amp_ops_per_sec': amp_ops_per_sec
    }

def benchmark_write_performance(manager, v_wrappers, iclamp_wrappers, iterations=10000):
    """测试写变量性能"""
    print(f"\n✍️  写变量性能测试 ({iterations}次迭代)")
    print("=" * 50)
    
    # 预热
    for _ in range(100):
        for iclamp_wrapper in iclamp_wrappers:
            iclamp_wrapper.amp = 0.1
    
    # 测试IClamp.amp写入性能（segment.v通常是只读的）
    values = np.random.uniform(0.1, 0.3, iterations)
    
    start_time = time.time()
    for i in range(iterations):
        for j, iclamp_wrapper in enumerate(iclamp_wrappers):
            iclamp_wrapper.amp = values[i] + j * 0.01  # 给每个iclamp稍微不同的值
    amp_write_time = time.time() - start_time
    
    # 计算性能指标
    amp_write_ops_per_sec = (iterations * len(iclamp_wrappers)) / amp_write_time
    
    print(f"IClamp.amp 写入:")
    print(f"  总时间: {amp_write_time:.4f}s")
    print(f"  操作数: {iterations * len(iclamp_wrappers)}")
    print(f"  性能: {amp_write_ops_per_sec:.0f} ops/s")
    print(f"  平均延迟: {amp_write_time * 1000000 / (iterations * len(iclamp_wrappers)):.2f} μs/op")
    
    return {
        'amp_write_time': amp_write_time,
        'amp_write_ops_per_sec': amp_write_ops_per_sec
    }

def benchmark_mixed_operations(manager, v_wrappers, iclamp_wrappers, iterations=5000):
    """测试混合读写操作性能"""
    print(f"\n🔄 混合读写操作测试 ({iterations}次迭代)")
    print("=" * 50)
    
    values = np.random.uniform(0.1, 0.3, iterations)
    
    start_time = time.time()
    for i in range(iterations):
        # 读取所有电压值
        voltages = [v_wrapper.v for v_wrapper in v_wrappers]
        
        # 根据电压调整电流
        for j, iclamp_wrapper in enumerate(iclamp_wrappers):
            if j < len(voltages):
                # 简单的反馈控制逻辑
                if voltages[j] < -60:
                    iclamp_wrapper.amp = values[i] + 0.1
                else:
                    iclamp_wrapper.amp = values[i] - 0.1
    
    mixed_time = time.time() - start_time
    total_ops = iterations * (len(v_wrappers) + len(iclamp_wrappers))
    mixed_ops_per_sec = total_ops / mixed_time
    
    print(f"混合操作 (读电压 + 写电流):")
    print(f"  总时间: {mixed_time:.4f}s")
    print(f"  操作数: {total_ops}")
    print(f"  性能: {mixed_ops_per_sec:.0f} ops/s")
    print(f"  平均延迟: {mixed_time * 1000000 / total_ops:.2f} μs/op")
    
    return {
        'mixed_time': mixed_time,
        'mixed_ops_per_sec': mixed_ops_per_sec
    }

def benchmark_scaling_test(manager, somas, iclamps):
    """测试不同对象数量下的性能扩展性"""
    print(f"\n📈 性能扩展性测试")
    print("=" * 50)
    
    object_counts = [1, 2, 5, 10]
    read_results = []
    write_results = []
    
    for count in object_counts:
        if count > len(somas):
            break
            
        # 创建子集包装器
        subset_v_wrappers = [manager.create_obj_wrapper(soma(0.5)) for soma in somas[:count]]
        subset_iclamp_wrappers = [manager.create_obj_wrapper(iclamp) for iclamp in iclamps[:count]]
        
        # 初始化包装器（如果需要）
        for wrapper in subset_v_wrappers + subset_iclamp_wrappers:
            if hasattr(wrapper, '_initialize'):
                wrapper._initialize()
        
        # 测试读性能
        iterations = 2000
        start_time = time.time()
        for _ in range(iterations):
            for v_wrapper in subset_v_wrappers:
                _ = v_wrapper.v
        read_time = time.time() - start_time
        read_ops_per_sec = (iterations * count) / read_time
        
        # 测试写性能
        start_time = time.time()
        for _ in range(iterations):
            for iclamp_wrapper in subset_iclamp_wrappers:
                iclamp_wrapper.amp = 0.2
        write_time = time.time() - start_time
        write_ops_per_sec = (iterations * count) / write_time
        
        read_results.append(read_ops_per_sec)
        write_results.append(write_ops_per_sec)
        
        print(f"{count:2d}个对象 - 读: {read_ops_per_sec:8.0f} ops/s, 写: {write_ops_per_sec:8.0f} ops/s")
    
    return object_counts, read_results, write_results

def plot_results(results, scaling_data):
    """绘制性能测试结果"""
    print("\n📊 生成性能图表...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 读写性能对比
    operations = ['Segment.v\n读取', 'IClamp.amp\n读取', 'IClamp.amp\n写入', '混合操作']
    performance = [
        results['v_ops_per_sec'],
        results['amp_ops_per_sec'],
        results['amp_write_ops_per_sec'],
        results['mixed_ops_per_sec']
    ]
    
    bars = ax1.bar(operations, performance, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax1.set_ylabel('操作/秒')
    ax1.set_title('ObjWrapper操作性能对比')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, perf in zip(bars, performance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(performance)*0.01,
                f'{perf:.0f}', ha='center', va='bottom')
    
    # 2. 延迟对比
    latencies = [
        results['v_read_time'] * 1000000 / (10000 * 10),  # μs per operation
        results['amp_read_time'] * 1000000 / (10000 * 10),
        results['amp_write_time'] * 1000000 / (10000 * 10),
        results['mixed_time'] * 1000000 / (5000 * 20)
    ]
    
    bars = ax2.bar(operations, latencies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax2.set_ylabel('微秒/操作')
    ax2.set_title('平均操作延迟')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(latencies)*0.01,
                f'{lat:.2f}', ha='center', va='bottom')
    
    # 3. 扩展性测试
    object_counts, read_results, write_results = scaling_data
    ax3.plot(object_counts, read_results, 'o-', label='读操作', linewidth=2, markersize=8)
    ax3.plot(object_counts, write_results, 's-', label='写操作', linewidth=2, markersize=8)
    ax3.set_xlabel('对象数量')
    ax3.set_ylabel('操作/秒')
    ax3.set_title('性能扩展性')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 时间分布
    times = [
        results['v_read_time'],
        results['amp_read_time'],
        results['amp_write_time'],
        results['mixed_time']
    ]
    
    bars = ax4.bar(operations, times, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax4.set_ylabel('总时间 (秒)')
    ax4.set_title('操作总时间')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{t:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('$HOME/heliox/python_lib/obj_wrapper_performance.png', dpi=300, bbox_inches='tight')
    print("  📁 性能图表已保存: obj_wrapper_performance.png")

def performance_test():
    """主性能测试函数"""
    print("🚀 ObjWrapper性能测试")
    print("=" * 60)
    
    # 创建测试模型
    somas, iclamps = create_test_model()
    
    # 初始化管理器
    print("\n初始化HelioXManager...")
    manager = HelioXManager()
    #manager.device = "cpu"
    #manager.permute_type = 0
    
    # 创建包装器
    print("创建对象包装器...")
    v_wrappers = [manager.create_obj_wrapper(soma(0.5)) for soma in somas]
    iclamp_wrappers = [manager.create_obj_wrapper(iclamp) for iclamp in iclamps]
    
    # 导出并加载模型
    print("导出并加载模型...")
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "perf_test_model")
        manager.setup_and_load_model(export_path, dt=0.025, v_init=-65)
        
        # 运行一步仿真确保所有包装器正常工作
        manager.finitialize(-65)
        manager.run(1)  # 运行1ms
        
        print(f"\n✅ 模型初始化完成，共{len(v_wrappers)}个电压包装器，{len(iclamp_wrappers)}个电流包装器")
        
        # 运行性能测试
        read_results = benchmark_read_performance(manager, v_wrappers, iclamp_wrappers)
        write_results = benchmark_write_performance(manager, v_wrappers, iclamp_wrappers)
        mixed_results = benchmark_mixed_operations(manager, v_wrappers, iclamp_wrappers)
        scaling_data = benchmark_scaling_test(manager, somas, iclamps)
        
        # 合并结果
        results = {**read_results, **write_results, **mixed_results}
        
        # 生成报告
        print(f"\n📋 性能测试总结")
        print("=" * 60)
        print(f"测试配置:")
        print(f"  神经元数量: {len(somas)}")
        print(f"  包装器数量: {len(v_wrappers) + len(iclamp_wrappers)}")
        print(f"  测试迭代: 读写10000次, 混合5000次")
        
        print(f"\n最佳性能:")
        print(f"  最快读操作: {max(results['v_ops_per_sec'], results['amp_ops_per_sec']):.0f} ops/s")
        print(f"  写操作性能: {results['amp_write_ops_per_sec']:.0f} ops/s")
        print(f"  混合操作性能: {results['mixed_ops_per_sec']:.0f} ops/s")
        
        # 绘制图表
        plot_results(results, scaling_data)
        
    print("\n✅ 性能测试完成!")

if __name__ == "__main__":
    performance_test()
