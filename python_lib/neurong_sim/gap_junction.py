from __future__ import annotations

from ._common import *  # noqa: F403
from .manager import NeurongManager
from .wrappers import ObjWrapper

class GapJunctionInterface:
    """
    Gap Junction Interface - 单例模式的间隙连接管理接口
    
    这是一个纯接口层，所有数据都存储在C++端，Python端只是调用接口。
    使用单例模式确保全局只有一个实例。
    
    功能说明:
    --------
    - 间隙连接是单向的（源 → 目标）
    - 一个源可以有多个目标
    - 使用sid（源ID）来标识间隙连接
    - 支持批量操作以提高性能
    - 复用现有的变量指针解析逻辑
    - 支持compartment voltage和mechanism variables
    
    使用方法:
    --------
    
    1. 获取单例实例:
    ```python
    gap_interface = GapJunctionInterface()
    # 或者
    gap_interface = GapJunctionInterface.get_instance()
    ```
    
    2. 创建间隙连接:
    ```python
    # 创建新的间隙连接源
    sid = 100  # 指定sid
    gap_interface.add_gap_source(sid, "global", "v", 0)  # 源: segment voltage
    # 或
    gap_interface.add_gap_source(sid, "IClamp", "amp", 1)  # 源: IClamp amplitude
    ```
    
    3. 添加目标:
    ```python
    gap_interface.add_gap_target(sid, "global", "v", 1)      # 目标: 另一个segment
    gap_interface.add_gap_target(sid, "IClamp", "amp", 2)    # 目标: 另一个IClamp
    ```
    
    4. 查询和管理:
    ```python
    # 列出所有间隙连接
    all_gaps = gap_interface.list_all()
    
    # 获取特定间隙连接信息
    gap_info = gap_interface.get_info(sid)
    
    # 清空所有间隙连接
    gap_interface.clear_all()
    ```
    
    注意事项:
    --------
    - 必须在模型加载后使用
    - sid必须唯一，需要手动指定
    - 变量名和索引必须在模型中存在
    - 只支持添加和全部清空，不支持删除单个连接
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GapJunctionInterface, cls).__new__(cls)
            # 获取NeurongManager的客户端
            neurong_manager = NeurongManager()
            cls._client = neurong_manager.client
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        return cls()
    
    def add_gap_source(self, sid, src_mech, src_var, src_idx):
        """
        添加间隙连接源
        
        Parameters:
        -----------
        sid : int
            指定的源ID
        src_mech : str
            源机制名称（如""或"global"表示segment，"IClamp"表示机制）
        src_var : str
            源变量名称（如"v"表示电压，"amp"表示电流）
        src_idx : int
            源索引（segment的node_index或mechanism的实例索引）
            
        Returns:
        --------
        int
            成功时返回sid（>=0），失败时返回-1
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available. Make sure to initialize NeurongManager first.")
        
        result = self._client.add_gap_source(sid, src_mech, src_var, src_idx)
        if result < 0:  # 失败时返回-1
            print(f"Failed to add gap source with sid {sid}")
        
        return result
    
    def add_gap_target(self, sid, tgt_mech, tgt_var, tgt_idx):
        """
        为间隙连接添加目标
        
        Parameters:
        -----------
        sid : int
            源ID
        tgt_mech : str
            目标机制名称
        tgt_var : str
            目标变量名称
        tgt_idx : int
            目标索引
            
        Returns:
        --------
        bool
            True表示成功，False表示失败
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")
        
        result = self._client.add_gap_target(sid, tgt_mech, tgt_var, tgt_idx)
        return result == 0
    
    def clear_all(self):
        """
        清空所有间隙连接
        
        Returns:
        --------
        bool
            True表示成功，False表示失败
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")
        
        result = self._client.clear_all_gap_junctions()
        return result == 0

    def finalize(self):
        """
        显式完成间隙连接路由构建。

        普通运行路径不需要手动调用：SimWrapper 会在 finitialize/run/
        continue_run/fadvance 前 lazy finalize dirty routes。这个方法保留给测试
        和需要立即检查路由构建错误的调用方。
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")

        return self._client.finalize_gap_junctions() == 0
    
    def get_info(self, sid):
        """
        获取特定间隙连接信息
        
        Parameters:
        -----------
        sid : int
            源ID
            
        Returns:
        --------
        dict
            包含间隙连接信息的字典，如果不存在则返回空字典
            格式: {
                'source': {'mech_name': str, 'var_name': str, 'idx': int},
                'targets': [{'mech_name': str, 'var_name': str, 'idx': int}, ...]
            }
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")
        
        return self._client.get_gap_junction(sid)
    
    def list_all(self):
        """
        获取所有间隙连接信息
        
        Returns:
        --------
        dict
            所有间隙连接的映射，键为sid，值为gap信息
            格式: {
                sid1: {
                    'source': {'mech_name': str, 'var_name': str, 'idx': int},
                    'targets': [{'mech_name': str, 'var_name': str, 'idx': int}, ...]
                },
                sid2: {...},
                ...
            }
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")
        
        return self._client.get_all_gap_junctions()
    
    def get_next_available_sid(self):
        """
        获取下一个可用的源ID
        
        Returns:
        --------
        int
            下一个可用的sid
        """
        if self._client is None:
            raise RuntimeError("NeurongManager client not available.")
        
        return self._client.get_next_available_sid()

    def get_variable_handle(self, mech_name, var_name, node_or_mech_idx, array_index=0):
        """获取变量handle（支持数组索引）"""
        from neurong_core.binding import VariableDescriptor, get_variable_handle_from_descriptor

        descriptor = VariableDescriptor(
            str(mech_name),
            str(var_name),
            int(node_or_mech_idx),
            int(array_index),
        )
        return get_variable_handle_from_descriptor(self._client, descriptor)

    def get_monitor_handle(self, mech_name, var_name, node_or_mech_idx, array_index=0):
        """获取监控器handle（支持数组索引）"""
        return self._client.get_monitor_handle_with_array(mech_name, var_name, node_or_mech_idx, array_index)
