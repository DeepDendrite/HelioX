from __future__ import annotations

from . import _common as _common
from ._common import *  # noqa: F403
from neurong_core.binding import (
    VariableDescriptor,
    extract_mech_name,
    get_variable_handle_from_descriptor,
    resolve_variable_descriptor,
)

class MonitorWrapper:
    """
    监控器包装类 - 用于记录时间序列数据

    ⚠️ 重要: 此对象必须保持引用，不要在创建后丢弃！
    """
    def __init__(self, obj, var_name, client, array_index=0):
        self.obj = obj  # NEURON对象 (segment 或 mech)
        self.var_name = var_name  # 变量名
        self.array_index = array_index  # 数组索引
        self.__client = client
        self.monitor_id = -1  # 初始化为-1表示pending
        self.initialized = False
        self._mech_name = None
        self._node_or_mech_idx = None
        self.handle_kind = "history"
    
    def _initialize(self):
        """初始化监控器（在模型导出后调用）"""
        if self.initialized:
            return

        desc = resolve_variable_descriptor(self.obj, self.var_name, self.array_index)
        self._mech_name = desc.mech
        self._node_or_mech_idx = desc.index
        
        # 添加监控器（此时返回-1表示pending）
        self.monitor_id = self.__client.add_monitor_with_array(self._mech_name, self.var_name, self._node_or_mech_idx, self.array_index)
        self.initialized = True

    def get_descriptor(self) -> VariableDescriptor:
        """Return the resolved backend descriptor for this monitor target."""
        if not self.initialized:
            self._initialize()
        return VariableDescriptor(
            str(self._mech_name),
            str(self.var_name),
            int(self._node_or_mech_idx),
            int(self.array_index),
        )

    def get_index(self) -> int:
        """Return the descriptor index for this monitor target."""
        return int(self.get_descriptor().index)
    
    def _extract_mech_name(self, mech):
        """从NEURON对象中提取机制名称"""
        # 方法1：从对象的hname()方法中提取（如果有的话）
        try:
            hname = mech.hname()
            if '[' in hname:
                return hname.split('[')[0]
            else:
                return hname
        except (AttributeError, TypeError):
            # 没有 hname() 方法，尝试其他方法
            pass
        
        # 方法2：从repr字符串中提取
        repr_str = repr(mech)
        match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\[', repr_str)
        if match:
            return match.group(1)
        else:
            return repr_str
        
        # 方法3：从对象类型字符串中提取
        obj_str = str(type(mech))
        match = re.search(r"'([A-Za-z][A-Za-z0-9_]*)'", obj_str)
        if match:
            return match.group(1)
        
        # 如果都失败了，抛出异常
        raise ValueError(f"Cannot extract mechanism name from {mech}")
    
    def get_handle(self) -> int:
        """Return the resolved persistent recorder handle."""
        if not self.initialized:
            self._initialize()
        
        if self.monitor_id == -1:
            real_handle = self.__client.get_monitor_handle_with_array(self._mech_name, self.var_name, self._node_or_mech_idx, self.array_index)
            if real_handle != -1:
                self.monitor_id = real_handle
            else:
                raise RuntimeError(f"Cannot get monitor handle for {self._mech_name}.{self.var_name}[{self.array_index}]. "
                                 f"Make sure the model is loaded and the monitor was properly registered.")
        return int(self.monitor_id)

    def get_data(self):
        """获取监控数据"""
        self.get_handle()

        return self.__client.get_monitor_data(self.monitor_id)
    
    @property
    def data(self):
        """属性方式获取监控数据"""
        return self.get_data()

    def __del__(self):
        """析构函数：提醒用户包装器被回收"""
        if (_common._WRAPPER_DESTRUCTOR_WARNINGS_ENABLED and 
            hasattr(self, 'monitor_id') and self.monitor_id is not None):
            warnings.warn(
                f"MonitorWrapper for {self._mech_name}.{self.var_name} is being garbage collected. "
                f"This may cause memory leaks in neurong. Consider keeping a reference to this wrapper.",
                ResourceWarning,
                stacklevel=2
            )


# RecorderWrapper 是 MonitorWrapper 的别名（新命名）
RecorderWrapper = MonitorWrapper


class WindowRecorderWrapper(MonitorWrapper):
    """Segment-local training observer; it does not keep a full history."""

    def __init__(self, obj, var_name, client, array_index=0):
        super().__init__(obj, var_name, client, array_index)
        self.handle_kind = "window"
        self.window_observer_id = -1

    def _initialize(self):
        if self.initialized:
            return
        desc = resolve_variable_descriptor(self.obj, self.var_name, self.array_index)
        self._mech_name = desc.mech
        self._node_or_mech_idx = desc.index
        self.window_observer_id = self._MonitorWrapper__client.add_window_observer_with_array(
            self._mech_name,
            self.var_name,
            self._node_or_mech_idx,
            self.array_index,
        )
        self.initialized = True

    def get_handle(self) -> int:
        if not self.initialized:
            self._initialize()
        if self.window_observer_id == -1:
            real_handle = self._MonitorWrapper__client.get_window_observer_handle_with_array(
                self._mech_name,
                self.var_name,
                self._node_or_mech_idx,
                self.array_index,
            )
            if real_handle == -1:
                raise RuntimeError(
                    f"Cannot get window observer handle for {self._mech_name}.{self.var_name}[{self.array_index}]. "
                    "Make sure the model is loaded and the observer was properly registered."
                )
            self.window_observer_id = real_handle
        return int(self.window_observer_id)

    @property
    def monitor_id(self):
        return self.window_observer_id

    @monitor_id.setter
    def monitor_id(self, value):
        self.window_observer_id = int(value)

    def get_data(self):
        raise RuntimeError("WindowRecorderWrapper only exposes data through SegmentRunner windows.")


class ArrayVariableProxy:
    """
    数组变量代理类 - 提供自然的数组访问语法

    这个类包装了数组变量的访问，让用户可以使用自然的 Python 语法：
    - obj.arr[i] 读取数组元素
    - obj.arr[i] = val 设置数组元素
    - obj.arr[start:stop] 切片访问
    - len(obj.arr) 获取数组长度
    - for val in obj.arr: 遍历数组

    ⚠️ 注意：此对象由 ObjWrapper 自动创建，用户不应直接实例化
    """

    def __init__(self, wrapper, var_name, array_length):
        """
        初始化数组变量代理

        Parameters:
        -----------
        wrapper : ObjWrapper
            父对象包装器
        var_name : str
            数组变量名
        array_length : int
            数组长度
        """
        self._wrapper = wrapper
        self._var_name = var_name
        self._array_length = array_length

    def __getitem__(self, index):
        """
        支持下标和切片读取

        Examples:
        ---------
        val = obj.arr[5]           # 读取单个元素
        vals = obj.arr[0:10]       # 切片读取
        vals = obj.arr[::2]        # 步长切片
        """
        if isinstance(index, slice):
            # 切片访问：arr[start:stop:step]
            start, stop, step = index.indices(self._array_length)
            return [self._wrapper.get_var(self._var_name, i)
                    for i in range(start, stop, step)]
        else:
            # 单个元素访问：arr[i]
            if not isinstance(index, int):
                raise TypeError(f"Array indices must be integers, not {type(index).__name__}")
            if index < 0:
                # 支持负数索引
                index = self._array_length + index
            if index < 0 or index >= self._array_length:
                raise IndexError(f"Array index {index} out of range [0, {self._array_length})")
            return self._wrapper.get_var(self._var_name, index)

    def __setitem__(self, index, value):
        """
        支持下标和切片赋值

        Examples:
        ---------
        obj.arr[5] = 1.0                    # 设置单个元素
        obj.arr[0:10] = [1,2,3,4,5,6,7,8,9,10]  # 切片赋值（列表）
        obj.arr[0:10] = 1.0                 # 切片广播赋值（标量）
        """
        if isinstance(index, slice):
            # 切片赋值
            start, stop, step = index.indices(self._array_length)
            indices = list(range(start, stop, step))

            if hasattr(value, '__iter__') and not isinstance(value, str):
                # 赋值列表/数组
                values = list(value)
                if len(values) != len(indices):
                    raise ValueError(f"Cannot assign {len(values)} values to {len(indices)} elements")
                for i, v in zip(indices, values):
                    self._wrapper.set_var(self._var_name, v, i)
            else:
                # 广播标量
                for i in indices:
                    self._wrapper.set_var(self._var_name, value, i)
        else:
            # 单个元素赋值
            if not isinstance(index, int):
                raise TypeError(f"Array indices must be integers, not {type(index).__name__}")
            if index < 0:
                # 支持负数索引
                index = self._array_length + index
            if index < 0 or index >= self._array_length:
                raise IndexError(f"Array index {index} out of range [0, {self._array_length})")
            self._wrapper.set_var(self._var_name, value, index)

    def __len__(self):
        """支持 len(obj.arr)"""
        return self._array_length

    def __iter__(self):
        """
        支持迭代

        Example:
        --------
        for val in obj.arr:
            print(val)
        """
        for i in range(self._array_length):
            yield self._wrapper.get_var(self._var_name, i)

    def __repr__(self):
        """字符串表示"""
        return f"ArrayVariableProxy({self._var_name}[{self._array_length}])"

    def to_list(self):
        """
        将整个数组转换为 Python 列表

        Returns:
        --------
        list
            包含所有数组元素的列表
        """
        return list(self)

    def to_numpy(self):
        """
        将整个数组转换为 NumPy 数组

        Returns:
        --------
        numpy.ndarray
            包含所有数组元素的 NumPy 数组
        """
        return np.array(self.to_list())


class ObjWrapper:
    """
    对象包装器类 - 用于读写当前时刻的变量值

    支持标量和数组变量的自然访问：

    标量变量:
    ---------
    val = obj.amp          # 读取
    obj.amp = 0.5          # 设置

    数组变量:
    ---------
    val = obj.arr[5]              # 读取单个元素
    obj.arr[5] = 1.0              # 设置单个元素
    vals = obj.arr[0:10]          # 切片读取
    obj.arr[0:10] = [1,2,3,...]   # 切片设置
    for val in obj.arr:           # 遍历
        print(val)
    length = len(obj.arr)         # 获取长度

    ⚠️ 重要: 此对象必须保持引用，不要在创建后丢弃！
    """
    def __init__(self, obj, client, var_names=None):
        self.__obj = obj  # NEURON对象 (segment 或 mech)
        self.__client = client
        self.__allowed_vars = set()
        self.__mech_name = None
        self.__is_segment = False
        self.initialized = False
        self.__var_names_hint = tuple(var_names) if var_names is not None else None
        # 数组变量支持
        self.__array_vars = {}  # 变量名 -> 数组长度
        self.__var_handles = {}  # (var_name, array_index) -> handle
    
    def _initialize(self):
        """初始化对象包装器（在模型导出后调用）"""
        if self.initialized:
            return
        
        # 判断对象类型
        if hasattr(self.__obj, 'node_index'):
            # Segment 对象 (如 soma(0.5))
            self.__is_segment = True
            self.__allowed_vars.add("v")
            desc = resolve_variable_descriptor(self.__obj, "v")
            self.__mech_name = desc.mech
            self.__row = desc.index
            self._create_property("v")
        else:
            # Mech 对象 (如 IClamp)
            self.__is_segment = False
            self.__mech_name = extract_mech_name(self.__obj)

            names_to_scan = self.__var_names_hint
            if names_to_scan is None:
                names_to_scan = dir(self.__obj)

            # 遍历所有可能的变量，或只解析调用方声明的变量白名单。
            for name in names_to_scan:
                if name.startswith('_'):
                    continue

                # 先检查是否是方法
                if self.__var_names_hint is None:
                    try:
                        func = getattr(self.__obj, name)

                        # 特殊处理：HocObject数组也是callable，但我们要将其作为变量处理
                        func_str = str(func)
                        if callable(func) and "[?]" not in func_str:
                            if name in basic_mech_func:
                                continue
                            self._create_method(name)
                            continue
                        elif "[?]" in func_str:
                            # 继续处理作为数组变量
                            pass
                    except Exception as e:
                        pass

                # 检查_ref_属性（包括数组）
                ref_name = f"_ref_{name}"
                try:
                    ref = getattr(self.__obj, ref_name)
                    ref_str = str(ref)

                    # 检查是否是数组变量
                    if "incomplete pointer to hoc array" in ref_str:
                        # 数组变量处理
                        # 尝试获取数组第一个元素来提取row信息
                        try:
                            ref_elem = ref[0]  # 获取数组第一个元素
                            # 从 "data_handle<double>{cont=BP_Syn_SoftMax tgt[0/10] row=0/1 val=1}" 提取
                            row = _common._try_parse_row_from_ref(ref_elem)
                            if row is not None:
                                self.__row = row
                                self.__allowed_vars.add(name)
                                # 提取数组长度
                                elem_str = str(ref_elem)
                                array_match = re.search(r'\[(\d+)/(\d+)\]', elem_str)
                                if array_match:
                                    array_length = int(array_match.group(2))
                                    self.__array_vars[name] = array_length
                                self._create_property(name)
                        except (IndexError, AttributeError, KeyError):
                            # 数组访问失败，可能不是有效的数组变量
                            pass
                    else:
                        # 标量变量处理（原有逻辑）
                        row = _common._try_parse_row_from_ref(ref)
                        if row is not None:
                            self.__row = row
                            self.__allowed_vars.add(name)
                            self._create_property(name)
                except (AttributeError, TypeError):
                    # 没有_ref_属性，跳过（不是NEURON变量）
                    pass

            # 检查是否成功提取到 row（mech_idx）
            # Mech 对象必须有 row，因为访问变量或调用方法都需要 row
            if not hasattr(self, '_ObjWrapper__row') or self.__row is None:
                raise RuntimeError(
                    f"Failed to initialize ObjWrapper for mechanism '{self.__mech_name}'.\n"
                    f"  Reason: No RANGE variables found (no _ref_ attributes).\n"
                    f"  The mechanism needs 'row' parameter for variable access or method calls.\n"
                    f"  Solution: Add a RANGE variable to the MOD file (e.g., 'RANGE dummy_var' or 'RANGE mech_id').\n"
                    f"  Object: {repr(self.__obj)}\n"
                    f"  Available attributes: {[attr for attr in dir(self.__obj) if not attr.startswith('_')]}"
                )

        self.initialized = True

    def _extract_mech_name(self, mech):
        """从NEURON对象中提取机制名称"""
        # 方法1：从对象的hname()方法中提取（如果有的话）
        try:
            hname = mech.hname()
            if '[' in hname:
                return hname.split('[')[0]
            return hname
        except (AttributeError, TypeError):
            # 没有 hname() 方法，尝试其他方法
            pass

        # 方法2：从repr字符串中提取
        repr_str = repr(mech)
        match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\[', repr_str)
        if match:
            return match.group(1)

        # 方法3：从对象类型字符串中提取
        obj_str = str(type(mech))
        match = re.search(r"'([A-Za-z][A-Za-z0-9_]*)'", obj_str)
        if match:
            return match.group(1)

        # 如果都失败了，抛出异常
        raise ValueError(f"Cannot extract mechanism name from {mech}")

    def _create_property(self, name):
        """
        为变量创建属性访问器

        对于标量变量，创建 getter 和 setter
        对于数组变量，只创建 getter，返回 ArrayVariableProxy 对象
        """
        # 检查是否是数组变量
        is_array = name in self.__array_vars

        if is_array:
            # 数组变量：返回 ArrayVariableProxy
            def array_getter(self):
                array_length = self._ObjWrapper__array_vars[name]
                return ArrayVariableProxy(self, name, array_length)

            # 数组变量不支持直接赋值（使用 arr[:] = vals 或 arr[i] = val 代替）
            setattr(self.__class__, name, property(array_getter))
        else:
            # 标量变量：原有逻辑
            def scalar_getter(self):
                # 延迟获取handle
                if not hasattr(self, '_ObjWrapper__var_handles'):
                    self._ObjWrapper__var_handles = {}

                handle_key = (name, 0)  # 标量使用 (name, 0) 作为key
                if handle_key not in self._ObjWrapper__var_handles:
                    # 第一次访问时获取handle
                    descriptor = VariableDescriptor(
                        self._ObjWrapper__mech_name,
                        name,
                        int(self._ObjWrapper__row),
                        0,
                    )
                    try:
                        handle = get_variable_handle_from_descriptor(self._ObjWrapper__client, descriptor)
                    except RuntimeError:
                        # 如果获取失败，回退到原始方法
                        return self._ObjWrapper__client.get_variable_value(
                            self._ObjWrapper__mech_name, name, self._ObjWrapper__row
                        )
                    self._ObjWrapper__var_handles[handle_key] = handle

                # 使用缓存的handle快速访问
                return self._ObjWrapper__client.get_variable_by_handle(
                    self._ObjWrapper__var_handles[handle_key]
                )

            def scalar_setter(self, value):
                # 延迟获取handle
                if not hasattr(self, '_ObjWrapper__var_handles'):
                    self._ObjWrapper__var_handles = {}

                handle_key = (name, 0)  # 标量使用 (name, 0) 作为key
                if handle_key not in self._ObjWrapper__var_handles:
                    # 第一次访问时获取handle
                    descriptor = VariableDescriptor(
                        self._ObjWrapper__mech_name,
                        name,
                        int(self._ObjWrapper__row),
                        0,
                    )
                    try:
                        handle = get_variable_handle_from_descriptor(self._ObjWrapper__client, descriptor)
                    except RuntimeError:
                        # 如果获取失败，回退到原始方法
                        self._ObjWrapper__client.set_variable_value(
                            value, self._ObjWrapper__mech_name, name, self._ObjWrapper__row
                        )
                        return
                    self._ObjWrapper__var_handles[handle_key] = handle

                # 使用缓存的handle快速设置
                self._ObjWrapper__client.set_variable_by_handle(
                    self._ObjWrapper__var_handles[handle_key], value
                )

            setattr(self.__class__, name, property(scalar_getter, scalar_setter))
    
    def _create_method(self, name):
        def method(self, *args):
            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], list):
                    continue
                # numpy数组转list
                if isinstance(args[i], np.ndarray):
                    args[i] = args[i].tolist()
                # pandas Series/DataFrame转list 
                elif 'pandas' in str(type(args[i])):
                    args[i] = args[i].values.tolist()
                # 其他序列类型转list
                elif isinstance(args[i], (tuple, set)):
                    args[i] = list(args[i])
                # 复数转实部
                elif isinstance(args[i], complex):
                    args[i] = args[i].real
                # 确保数值类型
                elif isinstance(args[i], (bool, int, float)):
                    continue
                # 其他类型抛出异常
                else:
                    raise TypeError(f"Argument type {type(args[i])} not supported")
            args = [self.__row] + args
            return self.__client.call_mech_func(self.__mech_name, name, *args)

        setattr(self.__class__, name, method)

    def __setattr__(self, name, value):
        # 允许内部属性
        if name.startswith('_ObjWrapper__'):
            super().__setattr__(name, value)
        elif hasattr(self, '_ObjWrapper__allowed_vars') and name in self.__allowed_vars:
            # 调用属性setter
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        # 只允许访问有效的变量
        if hasattr(self, '_ObjWrapper__allowed_vars') and name in self.__allowed_vars:
            # 使用缓存的handle
            if hasattr(self, '_ObjWrapper__var_handles') and name in self.__var_handles:
                return self.__client.get_variable_by_handle(self.__var_handles[name])
            else:
                # 兜底：如果没有handle，使用原始方法
                return self.__client.get_variable_value(self.__mech_name, name, self.__row)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_mech_name(self):
        """获取机制名称"""
        return self.__mech_name
    
    def get_row_id(self, var_name):
        """Return the descriptor index for this object variable."""
        _ = var_name
        return self.__row
    
    def get_node_index(self):
        """返回当前包装对象在机制中的索引"""
        if not self.initialized:
            raise RuntimeError("Object wrapper not initialized. Call setup_and_load_model() first.")
        return self.__row
    
    def get_available_vars(self):
        """获取所有可用的变量名"""
        return list(self.__allowed_vars)

    def get_var_array_length(self, var_name):
        """获取变量的数组长度，如果不是数组返回1"""
        return self.__array_vars.get(var_name, 1)

    def get_handle(self, var_name, array_index=0):
        """获取变量对应的handle（支持数组索引）"""
        if not self.initialized:
            raise RuntimeError("Object wrapper not initialized. Call setup_and_load_model() first.")

        if var_name not in self.__allowed_vars:
            raise AttributeError(f"Variable '{var_name}' not available in {self.__mech_name}")

        array_length = self.__array_vars.get(var_name, 1)
        if array_index >= array_length:
            raise IndexError(f"Array index {array_index} out of range for variable '{var_name}' (length: {array_length})")

        handle_key = (var_name, array_index)
        if handle_key not in self.__var_handles:
            descriptor = VariableDescriptor(self.__mech_name, var_name, int(self.__row), int(array_index))
            handle = get_variable_handle_from_descriptor(self.__client, descriptor)
            self.__var_handles[handle_key] = handle

        return self.__var_handles[handle_key]

    def get_var(self, var_name, array_index=0):
        """获取变量值（支持数组索引）"""
        if not self.initialized:
            raise RuntimeError("Object wrapper not initialized. Call setup_and_load_model() first.")

        if var_name not in self.__allowed_vars:
            raise AttributeError(f"Variable '{var_name}' not available in {self.__mech_name}")

        # 检查数组索引范围
        array_length = self.__array_vars.get(var_name, 1)
        if array_index >= array_length:
            raise IndexError(f"Array index {array_index} out of range for variable '{var_name}' (length: {array_length})")

        # 延迟获取handle
        handle_key = (var_name, array_index)
        if handle_key not in self.__var_handles:
            descriptor = VariableDescriptor(self.__mech_name, var_name, int(self.__row), int(array_index))
            try:
                handle = get_variable_handle_from_descriptor(self.__client, descriptor)
            except RuntimeError:
                # 如果获取失败，回退到原始方法
                return self.__client.get_variable_value_with_array(self.__mech_name, var_name, self.__row, array_index)
            self.__var_handles[handle_key] = handle

        # 使用缓存的handle快速访问
        return self.__client.get_variable_by_handle(self.__var_handles[handle_key])

    def set_var(self, var_name, value, array_index=0):
        """设置变量值（支持数组索引）"""
        if not self.initialized:
            raise RuntimeError("Object wrapper not initialized. Call setup_and_load_model() first.")

        if var_name not in self.__allowed_vars:
            raise AttributeError(f"Variable '{var_name}' not available in {self.__mech_name}")

        # 检查数组索引范围
        array_length = self.__array_vars.get(var_name, 1)
        if array_index >= array_length:
            raise IndexError(f"Array index {array_index} out of range for variable '{var_name}' (length: {array_length})")

        # 延迟获取handle
        handle_key = (var_name, array_index)
        if handle_key not in self.__var_handles:
            descriptor = VariableDescriptor(self.__mech_name, var_name, int(self.__row), int(array_index))
            try:
                handle = get_variable_handle_from_descriptor(self.__client, descriptor)
            except RuntimeError:
                # 如果获取失败，回退到原始方法
                self.__client.set_variable_value_with_array(value, self.__mech_name, var_name, self.__row, array_index)
                return
            self.__var_handles[handle_key] = handle

        # 使用缓存的handle快速设置
        self.__client.set_variable_by_handle(self.__var_handles[handle_key], value)

    def get_var_array(self, var_name):
        """获取整个数组变量的值"""
        if var_name not in self.__allowed_vars:
            raise AttributeError(f"Variable '{var_name}' not available in {self.__mech_name}")

        array_length = self.__array_vars.get(var_name, 1)
        return [self.get_var(var_name, i) for i in range(array_length)]

    def set_var_array(self, var_name, values):
        """设置整个数组变量的值"""
        if var_name not in self.__allowed_vars:
            raise AttributeError(f"Variable '{var_name}' not available in {self.__mech_name}")

        array_length = self.__array_vars.get(var_name, 1)
        if len(values) != array_length:
            raise ValueError(f"Array length mismatch: expected {array_length}, got {len(values)}")

        for i, value in enumerate(values):
            self.set_var(var_name, value, i)

    def __del__(self):
        """析构函数：提醒用户包装器被回收"""
        if (_common._WRAPPER_DESTRUCTOR_WARNINGS_ENABLED and 
            hasattr(self, '_ObjWrapper__mech_name') and self.__mech_name is not None):
            warnings.warn(
                f"ObjWrapper for {self.__mech_name} is being garbage collected. "
                f"This may cause memory leaks in neurong. Consider keeping a reference to this wrapper.",
                ResourceWarning,
                stacklevel=2
            )


class VecPlayWrapper:
    """
    VecPlay包装器类 - 用于动态播放时间序列数据
    
    ⚠️ 重要: 此对象必须保持引用，不要在创建后丢弃！
    """
    
    def __init__(self, obj, var_name, client):
        """
        初始化VecPlay包装器
        
        Parameters:
        -----------
        obj : NEURON object
            要控制的NEURON对象 (mech 或 segment)
        var_name : str
            要控制的变量名
        client : neurong.Sim
            neurong客户端实例
        """
        self.obj = obj
        self.var_name = var_name
        self.__client = client
        self.initialized = False
        self._mech_name = None
        self._instance_id = None
        self.vecplay_key = None
    
    def _initialize(self):
        """初始化VecPlay包装器（在模型导出后调用）"""
        if self.initialized:
            return

        desc = resolve_variable_descriptor(self.obj, self.var_name)
        self._mech_name = desc.mech
        self._instance_id = desc.index
        
        self.initialized = True
    
    def _extract_mech_name(self, mech):
        """从NEURON对象中提取机制名称"""
        # 方法1：从对象的hname()方法中提取（如果有的话）
        try:
            hname = mech.hname()
            if '[' in hname:
                return hname.split('[')[0]
            return hname
        except (AttributeError, TypeError):
            # 没有 hname() 方法，尝试其他方法
            pass

        # 方法2：从repr字符串中提取
        repr_str = repr(mech)
        match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\[', repr_str)
        if match:
            return match.group(1)

        # 方法3：从对象类型字符串中提取
        obj_str = str(type(mech))
        match = re.search(r"'([A-Za-z][A-Za-z0-9_]*)'", obj_str)
        if match:
            return match.group(1)

        # 如果都失败了，抛出异常
        raise ValueError(f"Cannot extract mechanism name from {mech}")
    
    def play(self, tvec, yvec):
        """
        播放时间序列数据
        
        Parameters:
        -----------
        tvec : list or array
            时间向量
        yvec : list or array
            对应的值向量
        """
        if not self.initialized:
            raise RuntimeError("VecPlay wrapper not initialized. Call setup_and_load_model() first.")
        
        # 检查长度
        if len(tvec) != len(yvec):
            raise ValueError("tvec and yvec must have the same length")
        
        if len(tvec) == 0:
            raise ValueError("tvec and yvec cannot be empty")
        
        # 转换为列表（确保兼容性）
        tvec_list = list(tvec)
        yvec_list = list(yvec)
        
        # 如果是第一次调用，添加VecPlay
        if not self.__client.has_vecplay(self._mech_name, self.var_name, self._instance_id):
            result = self.__client.add_vecplay(self._mech_name, self.var_name, self._instance_id, 
                                              tvec_list, yvec_list)
            if result != 0:
                raise RuntimeError(f"Failed to add vecplay for {self._mech_name}.{self.var_name}[{self._instance_id}]")
        else:
            # 更新现有的VecPlay
            result = self.__client.update_vecplay(self._mech_name, self.var_name, self._instance_id, 
                                                 tvec_list, yvec_list)
            if result != 0:
                raise RuntimeError(f"Failed to update vecplay for {self._mech_name}.{self.var_name}[{self._instance_id}]")
            # print(f"Updated vecplay for {self._mech_name}.{self.var_name}[{self._instance_id}] with {len(tvec)} points")
    
    def stop(self):
        """停止VecPlay（删除）"""
        if not self.initialized:
            raise RuntimeError("VecPlay wrapper not initialized. Call setup_and_load_model() first.")
        
        if self.__client.has_vecplay(self._mech_name, self.var_name, self._instance_id):
            result = self.__client.remove_vecplay(self._mech_name, self.var_name, self._instance_id)
            if result != 0:
                raise RuntimeError(f"Failed to remove vecplay for {self._mech_name}.{self.var_name}[{self._instance_id}]")
    
    def is_playing(self):
        """检查是否正在播放"""
        if not self.initialized:
            return False
        return self.__client.has_vecplay(self._mech_name, self.var_name, self._instance_id)
    
    def get_info(self):
        """获取VecPlay信息"""
        if not self.initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "mech_name": self._mech_name,
            "var_name": self.var_name,
            "instance_id": self._instance_id,
            "is_playing": self.is_playing()
        }

    def __del__(self):
        """析构函数：提醒用户包装器被回收"""
        if (_common._WRAPPER_DESTRUCTOR_WARNINGS_ENABLED and 
            hasattr(self, '_mech_name') and self._mech_name is not None):
            warnings.warn(
                f"VecPlayWrapper for {self._mech_name}.{self.var_name} is being garbage collected. "
                f"This may cause memory leaks in neurong. Consider keeping a reference to this wrapper.",
                ResourceWarning,
                stacklevel=2
            )
