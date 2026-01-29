#!/usr/bin/env python3
"""
Gap Junction Usage Example
=========================

This example demonstrates how to use the Gap Junction Interface
in a real neuronal simulation context.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from heliox_wrapper import GapJunctionInterface, HelioXManager
except ImportError as e:
    print(f"Error importing heliox_wrapper: {e}")
    print("Make sure heliox module is compiled with Gap Junction support")
    sys.exit(1)

def basic_gap_junction_example():
    """
    基本的Gap Junction使用示例
    
    注意：此示例需要在加载了真实NEURON模型后运行
    """
    print("=== Basic Gap Junction Example ===")
    
    # 获取Gap Junction接口实例（单例）
    gap_interface = GapJunctionInterface()
    
    try:
        # 示例1：连接两个segment的电压
        # 创建gap junction: segment 0 -> segment 1
        sid1 = gap_interface.create(
            src_mech="global",   # segment电压
            src_var="v",         # 电压变量
            src_idx=0            # 第一个segment (node_index=0)
        )
        print(f"Created gap junction with SID: {sid1}")
        
        # 为这个gap junction添加目标
        success = gap_interface.add_target(
            sid=sid1,
            tgt_mech="global",   # 目标也是segment电压
            tgt_var="v",         # 电压变量
            tgt_idx=1            # 第二个segment (node_index=1)
        )
        print(f"Added target to gap junction: {'success' if success else 'failed'}")
        
        # 示例2：连接IClamp机制
        # 创建gap junction: IClamp[0].amp -> IClamp[1].amp
        sid2 = gap_interface.create(
            src_mech="IClamp",   # IClamp机制
            src_var="amp",       # 电流幅度
            src_idx=0,           # 第一个IClamp实例
            sid=10               # 手动指定SID
        )
        print(f"Created IClamp gap junction with SID: {sid2}")
        
        gap_interface.add_target(
            sid=sid2,
            tgt_mech="IClamp",
            tgt_var="amp",
            tgt_idx=1            # 第二个IClamp实例
        )
        
        # 获取gap junction信息
        gap_info = gap_interface.get_info(sid1)
        print(f"Gap junction {sid1} info: {gap_info}")
        
        # 列出所有gap junctions
        all_gaps = gap_interface.list_all()
        print(f"All gap junctions: {all_gaps}")
        
        # 删除gap junction
        success = gap_interface.remove(sid2)
        print(f"Removed gap junction {sid2}: {'success' if success else 'failed'}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if heliox module doesn't have Gap Junction support compiled in")

def batch_operations_example():
    """
    批量操作示例 - 用于创建大量gap junctions时提高性能
    """
    print("\n=== Batch Operations Example ===")
    
    gap_interface = GapJunctionInterface()
    
    try:
        # 使用批量操作上下文管理器
        with gap_interface.batch_context():
            print("Starting batch operations...")
            
            # 批量创建gap junctions
            # 例如：创建链状连接 0->1->2->3->...
            sids = []
            for i in range(10):  # 假设有10个segments
                sid = gap_interface.create(
                    src_mech="global",
                    src_var="v", 
                    src_idx=i
                )
                if sid != -1:
                    sids.append(sid)
                    
                    # 添加到下一个segment的连接
                    if i < 9:  # 不是最后一个
                        gap_interface.add_target(
                            sid=sid,
                            tgt_mech="global",
                            tgt_var="v",
                            tgt_idx=i+1
                        )
            
            print(f"Created {len(sids)} gap junctions in batch mode")
        
        print("Batch operations completed")
        
    except Exception as e:
        print(f"Batch operation error: {e}")

def integration_with_heliox_manager():
    """
    与HelioXManager集成的完整示例
    """
    print("\n=== Integration with HelioXManager ===")
    
    # 这个示例展示如何在完整的heliox工作流中使用Gap Junction
    print("Complete workflow example:")
    print("""
    # 1. 设置NEURON模型并导出
    from neuron import h
    from heliox_wrapper import HelioXManager, GapJunctionInterface
    
    # ... 创建NEURON模型 ...
    
    # 2. 导出并加载到heliox
    heliox_manager = HelioXManager()
    heliox_manager.setup_and_load_model("path/to/export", dt=0.05, v_init=-62.5)
    
    # 3. 创建Gap Junctions
    gap_interface = GapJunctionInterface()
    
    # 连接相邻的segments
    for i in range(num_segments - 1):
        sid = gap_interface.create("global", "v", i)
        gap_interface.add_target(sid, "global", "v", i+1)
    
    # 4. 运行仿真
    heliox_manager.finitialize(-62.5)
    heliox_manager.run(300)
    
    # 5. 分析结果 - gap junctions会影响电压传播
    """)

def advanced_usage_example():
    """
    高级用法示例
    """
    print("\n=== Advanced Usage Example ===")
    
    gap_interface = GapJunctionInterface()
    
    print("Advanced features:")
    print("- Multiple targets per source")
    print("- Different mechanism types")
    print("- SID management")
    print("""
    # 一个源连接多个目标
    sid = gap_interface.create("global", "v", 0)  # 源: segment 0
    gap_interface.add_target(sid, "global", "v", 1)     # 目标1: segment 1
    gap_interface.add_target(sid, "global", "v", 2)     # 目标2: segment 2
    gap_interface.add_target(sid, "IClamp", "amp", 0)   # 目标3: IClamp[0].amp
    
    # 复杂的连接模式
    # 星形连接：中心节点连接到所有外围节点
    center_sid = gap_interface.create("global", "v", center_node)
    for peripheral_node in peripheral_nodes:
        gap_interface.add_target(center_sid, "global", "v", peripheral_node)
    
    # 双向连接：A->B 和 B->A
    sid_ab = gap_interface.create("global", "v", node_a)
    gap_interface.add_target(sid_ab, "global", "v", node_b)
    
    sid_ba = gap_interface.create("global", "v", node_b)  
    gap_interface.add_target(sid_ba, "global", "v", node_a)
    """)

def error_handling_examples():
    """
    错误处理示例
    """
    print("\n=== Error Handling Examples ===")
    
    gap_interface = GapJunctionInterface()
    
    print("Common error scenarios:")
    print("1. Creating gap junction before model is loaded")
    print("2. Invalid mechanism or variable names")
    print("3. Out-of-range indices")
    print("4. Duplicate SIDs")
    print("""
    try:
        # 这些操作可能失败
        sid = gap_interface.create("invalid_mech", "v", 0)
        if sid == -1:
            print("Failed to create gap junction")
        
        success = gap_interface.add_target(999, "global", "v", 0)  # 不存在的SID
        if not success:
            print("Failed to add target")
            
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    """)

def main():
    print("Gap Junction Usage Examples")
    print("=" * 50)
    print("This demonstrates the Gap Junction Interface usage.")
    print("Note: Some examples require a compiled heliox module with Gap Junction support.\n")
    
    basic_gap_junction_example()
    batch_operations_example()
    integration_with_heliox_manager()
    advanced_usage_example()
    error_handling_examples()
    
    print("\n" + "=" * 50)
    print("Key Benefits of Gap Junction Interface:")
    print("✓ Single source of truth - all data stored in C++")
    print("✓ Singleton pattern prevents multiple instances") 
    print("✓ Pure interface layer - no state in Python")
    print("✓ Batch operations for performance")
    print("✓ Comprehensive error handling")
    print("✓ Integration with existing heliox workflow")
    print("✓ Support for both segments and mechanisms")

if __name__ == "__main__":
    main()