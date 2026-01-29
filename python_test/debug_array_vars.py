#!/usr/bin/env python3
"""调试脚本：检查BPSYN机制的数组变量如何在NEURON中暴露"""

from neuron import h
h.load_file("stdrun.hoc")

# 创建测试细胞和机制
soma = h.Section(name='soma')
soma.L = soma.diam = 10
soma.insert('hh')

# 创建BP_Syn_SoftMax
softmax = h.BP_Syn_SoftMax(soma(0.5))

print("=== BP_Syn_SoftMax 对象分析 ===")
print(f"对象: {softmax}")
print(f"类型: {type(softmax)}")

# 列出所有属性
print("\n所有属性:")
attrs = dir(softmax)
for attr in sorted(attrs):
    if not attr.startswith('__'):
        print(f"  {attr}")

# 检查数组相关的属性
print("\n数组相关属性:")
for var in ['tgt', 'u', 's', 'grad_to_prev']:
    print(f"\n检查 {var}:")
    # 直接访问
    if hasattr(softmax, var):
        print(f"  直接访问 softmax.{var} 存在")
        try:
            # 尝试获取数组元素
            for i in range(3):
                val = getattr(softmax, f'{var}[{i}]')
                print(f"    {var}[{i}] = {val}")
        except:
            pass

    # _ref_ 形式
    ref_name = f'_ref_{var}'
    if hasattr(softmax, ref_name):
        ref = getattr(softmax, ref_name)
        print(f"  {ref_name}: {ref}")

    # 尝试数组形式的_ref_
    for i in range(3):
        ref_array_name = f'_ref_{var}[{i}]'
        if hasattr(softmax, ref_array_name):
            print(f"  {ref_array_name} 存在")

# 检查如何访问数组
print("\n数组访问测试:")
# 设置值
softmax.tgt[0] = 1.0
softmax.tgt[1] = 0.0
print(f"设置后 tgt[0] = {softmax.tgt[0]}")
print(f"设置后 tgt[1] = {softmax.tgt[1]}")

# 获取_ref_
print("\n尝试获取_ref_tgt:")
try:
    ref_tgt = softmax._ref_tgt
    print(f"_ref_tgt = {ref_tgt}")
    print(f"_ref_tgt[0] = {ref_tgt[0]}")
    print(f"_ref_tgt[1] = {ref_tgt[1]}")
except Exception as e:
    print(f"错误: {e}")