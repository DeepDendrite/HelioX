from __future__ import annotations

from ._common import *  # noqa: F403
from .wrappers import ObjWrapper, MonitorWrapper, RecorderWrapper, VecPlayWrapper

class HelioXManager:
    """HelioX模拟器的单例管理类"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HelioXManager, cls).__new__(cls)
            cls._client = heliox.Sim()
            cls._client.set_spike_output_enabled(False)
            cls._instance.pending_monitor_wrappers = []  # 待初始化的监控器包装器
            cls._instance.pending_obj_wrappers = []      # 待初始化的对象包装器
            cls._instance.pending_vecplay_wrappers = []  # 待初始化的VecPlay包装器
            # Registry to track all created wrappers for export functionality
            cls._instance._all_monitor_wrappers = []     # 所有监控器包装器
            cls._instance._all_obj_wrappers = []         # 所有对象包装器
            cls._instance._all_vecplay_wrappers = []     # 所有VecPlay包装器
            # 设置默认参数
            cls._instance.device = "gpu"
            cls._instance.permute_type = 3
            cls._instance._dt = 0.05  # 默认时间步长
            cls._instance._model_loaded = False  # 模型加载状态
            cls._instance._optimizer_id = None
            cls._instance._optimizer_type = None
            cls._instance._pending_input_batches = []
            cls._instance._input_batches = {}
            cls._instance._active_input_mode = None
            cls._instance._netstim_default = {
                "interval_scale": 5.0,
                "start_base": 9.0,
                "epsilon": 0.01,
                "number": 100.0,
            }
            cls._instance._vecstim_default = {
                "spike_scale": 5.0,
                "start_base": 9.0,
                "epsilon": 0.01,
                "spike_count": 20,
            }
        return cls._instance
    
    @property
    def client(self):
        """获取heliox客户端"""
        return self._client
    
    @property
    def dt(self):
        """获取时间步长"""
        return self._client.get_dt()
    
    @dt.setter
    def dt(self, value):
        """设置时间步长"""
        self._client.set_dt(value)
    
    def set_default_device(self, device="gpu"):
        """设置默认设备"""
        self.device = device
    
    def set_default_permute_type(self, permute_type=3):
        """设置默认排列类型"""
        self.permute_type = permute_type
    
    def setup_and_load_model(self, export_path, dt=0.05, v_init=-62.5):
        """
        统一处理NEURON模型导出和heliox模型加载
        
        Parameters:
        - export_path: 模型导出路径
        - dt: 时间步长
        - v_init: 初始电压
        """
        from neuron import h
        
        # 1. NEURON模型导出
        # 确保ParallelContext正确设置
        pc = h.ParallelContext()
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = dt
        h.finitialize(v_init)
        
        # 导出模型
        pc.nrnbbcore_write(export_path)
        print(f"NEURON model exported to: {export_path}")
        
        # 2. 配置heliox
        self.set_data_path(export_path)
        self.set_device(self.device)
        self.set_permute_type(self.permute_type)
        
        # 3. 初始化所有对象包装器
        self._initialize_obj_wrappers()
        
        # 4. 初始化所有监控器包装器
        self._initialize_monitor_wrappers()
        
        # 5. 初始化所有VecPlay包装器
        self._initialize_vecplay_wrappers()
        
        # 6. 加载heliox模型
        self.load_model()
        print("HelioX model loaded successfully")
        self._initialize_input_batches()
    
    def create_obj_wrapper(self, obj):
        """
        创建对象包装器（延迟初始化）- 用于读写当前时刻的变量值
        
        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，heliox暂不支持自动资源回收。
        
        正确用法:
            wrapper = heliox_manager.create_obj_wrapper(iclamp)
            wrapper = heliox_manager.create_obj_wrapper(soma(0.5))
            
        错误用法:
            heliox_manager.create_obj_wrapper(iclamp)  # 返回值被丢弃！
        
        Parameters:
        -----------
        obj : NEURON object
            要包装的NEURON对象 (mech 或 segment)
            
        Returns:
        --------
        ObjWrapper
            对象包装器，必须保持引用
        """
        wrapper = ObjWrapper(obj, self._client)
        
        # Add to registry for export functionality
        self._all_obj_wrappers.append(wrapper)
        
        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
            # print(f"Immediately initialized obj wrapper: {wrapper.get_mech_name()}")
        else:
            # 否则添加到待初始化列表
            self.pending_obj_wrappers.append(wrapper)
        
        return wrapper
    
    def create_recorder(self, obj, var_name="v", array_index=0):
        """
        创建变量记录器（延迟初始化）- 用于记录时间序列数据

        这是推荐的新API，取代旧的create_monitor_wrapper。

        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，heliox暂不支持自动资源回收。

        正确用法:
            recorder = heliox_manager.create_recorder(soma(0.5))
            recorder = heliox_manager.create_recorder(iclamp, "amp")
            recorder_array = heliox_manager.create_recorder(iclamp, "amp", 1)  # 数组索引1

        错误用法:
            heliox_manager.create_recorder(soma(0.5))  # 返回值被丢弃！

        Parameters:
        -----------
        obj : NEURON object
            要记录的NEURON对象 (segment 或 mech)
        var_name : str, optional
            要记录的变量名，默认为"v"
        array_index : int, optional
            数组索引，默认为0

        Returns:
        --------
        RecorderWrapper
            变量记录器对象，必须保持引用
        """
        wrapper = RecorderWrapper(obj, var_name, self._client, array_index)

        # Add to registry for export functionality
        self._all_monitor_wrappers.append(wrapper)  # 内部仍使用monitor列表

        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
            print(f"Immediately initialized recorder for {wrapper._mech_name}.{wrapper.var_name}")
        else:
            # 否则添加到待初始化列表
            self.pending_monitor_wrappers.append(wrapper)

        return wrapper

    def create_monitor_wrapper(self, obj, var_name="v", array_index=0):
        """
        创建监控器包装器（兼容旧API）- 用于记录时间序列数据

        ⚠️ 已弃用: 推荐使用 create_recorder()，此方法仅为向后兼容保留

        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，heliox暂不支持自动资源回收。

        正确用法:
            monitor = heliox_manager.create_monitor_wrapper(soma(0.5))
            monitor = heliox_manager.create_monitor_wrapper(iclamp, "amp")
            monitor_array = heliox_manager.create_monitor_wrapper(iclamp, "amp", 1)  # 数组索引1

        错误用法:
            heliox_manager.create_monitor_wrapper(soma(0.5))  # 返回值被丢弃！

        Parameters:
        -----------
        obj : NEURON object
            要监控的NEURON对象 (segment 或 mech)
        var_name : str, optional
            要监控的变量名，默认为"v"
        array_index : int, optional
            数组索引，默认为0

        Returns:
        --------
        RecorderWrapper (兼容MonitorWrapper)
            监控器包装器对象，必须保持引用
        """
        # 直接调用新API
        return self.create_recorder(obj, var_name, array_index)
    
    def create_vecplay_wrapper(self, obj, var_name="amp"):
        """
        创建VecPlay包装器（延迟初始化）- 用于动态播放时间序列数据
        
        ⚠️ 重要: 必须保持对返回的包装器的引用！
        不要丢弃返回值，heliox暂不支持自动资源回收。
        
        正确用法:
            vecplay = heliox_manager.create_vecplay_wrapper(iclamp, "amp")
            vecplay = heliox_manager.create_vecplay_wrapper(soma(0.5), "v")
            
        错误用法:
            heliox_manager.create_vecplay_wrapper(iclamp)  # 返回值被丢弃！
        
        Parameters:
        -----------
        obj : NEURON object
            要控制的NEURON对象 (mech 或 segment)
        var_name : str, optional
            要控制的变量名，默认为"amp"
            
        Returns:
        --------
        VecPlayWrapper
            VecPlay包装器对象，必须保持引用
        """
        wrapper = VecPlayWrapper(obj, var_name, self._client)
        
        # Add to registry for export functionality
        self._all_vecplay_wrappers.append(wrapper)
        
        # 如果模型已经加载，立即初始化
        if hasattr(self, '_model_loaded') and self._model_loaded:
            wrapper._initialize()
            print(f"Immediately initialized vecplay wrapper for {wrapper._mech_name}.{wrapper.var_name}")
        else:
            # 否则添加到待初始化列表
            self.pending_vecplay_wrappers.append(wrapper)
        
        return wrapper

    def _initialize_obj_wrappers(self):
        """初始化所有对象包装器"""
        for wrapper in self.pending_obj_wrappers:
            wrapper._initialize()
            # print(f"Initialized obj wrapper: {wrapper.get_mech_name()}")

        # 清空待初始化列表
        self.pending_obj_wrappers.clear()
    
    def _initialize_monitor_wrappers(self):
        """初始化所有监控器包装器"""
        for wrapper in self.pending_monitor_wrappers:
            wrapper._initialize()
            # print(f"Initialized monitor wrapper for {wrapper._mech_name}.{wrapper.var_name}")

        # 清空待初始化列表
        self.pending_monitor_wrappers.clear()

    def _initialize_vecplay_wrappers(self):
        """初始化所有VecPlay包装器"""
        for wrapper in self.pending_vecplay_wrappers:
            wrapper._initialize()
            print(f"Initialized vecplay wrapper for {wrapper._mech_name}.{wrapper.var_name}")

        # 清空待初始化列表
        self.pending_vecplay_wrappers.clear()

    def configure_input_stimulus_defaults(self, mode="netstim", **params):
        """配置输入刺激的默认参数"""
        mode = (mode or "netstim").lower()
        if mode == "netstim":
            target = self._netstim_default
        elif mode == "vecstim":
            target = self._vecstim_default
        else:
            raise ValueError(f"Unsupported input stimulus mode: {mode}")

        for key, value in params.items():
            if value is None:
                continue
            target[key] = value

    def register_input_stimulators(self, wrappers, mode="netstim", config=None):
        """
        注册输入刺激包装器，供HelioX批量更新

        wrappers: 包装器列表
        mode: "netstim" 或 "vecstim"
        config: 覆盖默认参数的字典
        """
        if not wrappers:
            return
        mode = (mode or "netstim").lower()
        valid_wrappers = [w for w in wrappers if w is not None]
        if not valid_wrappers:
            return

        record = {
            "mode": mode,
            "wrappers": valid_wrappers,
            "config": dict(config) if config else None,
        }

        if self._model_loaded:
            self._create_input_batch(record)
        else:
            self._pending_input_batches.append(record)

    def _initialize_input_batches(self):
        """在模型加载完成后初始化所有待注册的输入刺激批次"""
        if not self._pending_input_batches:
            return
        for record in self._pending_input_batches:
            self._create_input_batch(record)
        self._pending_input_batches.clear()

    def _create_input_batch(self, record):
        mode = record["mode"]
        wrappers = record["wrappers"]
        if not wrappers:
            return

        if mode == "netstim":
            params = dict(self._netstim_default)
        elif mode == "vecstim":
            params = dict(self._vecstim_default)
        else:
            raise ValueError(f"Unsupported input stimulus mode: {mode}")

        if record["config"]:
            params.update(record["config"])

        if mode == "netstim":
            handle_triplets = []
            for wrapper in wrappers:
                interval_handle = wrapper.get_handle("interval")
                start_handle = wrapper.get_handle("start")
                number_handle = wrapper.get_handle("number")
                handle_triplets.append((interval_handle, start_handle, number_handle))

            batch_id = self._client.register_netstim_batch(
                handle_triplets,
                params.get("interval_scale", 5.0),
                params.get("start_base", 9.0),
                params.get("epsilon", 0.01),
                params.get("number", 100.0),
            )
        elif mode == "vecstim":
            mech_indices = [wrapper.get_node_index() for wrapper in wrappers]
            batch_id = self._client.register_vecstim_batch(
                mech_indices,
                params.get("spike_scale", 5.0),
                params.get("start_base", 9.0),
                params.get("epsilon", 0.01),
                int(params.get("spike_count", 20)),
            )
        else:
            return

        if batch_id < 0:
            raise RuntimeError(f"Failed to register input stimulus batch for mode {mode}")

        self._input_batches[mode] = {
            "batch_id": batch_id,
            "size": len(wrappers),
            "config": params,
        }
        self._active_input_mode = mode

    def set_input_stimulus(self, values, mode=None):
        """
        批量设置输入刺激。返回True表示已由HelioX处理。
        """
        if not self._model_loaded or not self._input_batches:
            return False

        mode = (mode or self._active_input_mode or "netstim").lower()
        if mode not in self._input_batches:
            return False

        batch_info = self._input_batches[mode]
        pixels = np.asarray(values, dtype=np.float64).reshape(-1)
        if pixels.size != batch_info["size"]:
            raise ValueError(
                f"Input size mismatch for {mode}: expected {batch_info['size']}, got {pixels.size}"
            )
        self._client.set_input_batch_pixels(batch_info["batch_id"], pixels)
        return True

    def set_data_path(self, path):
        """设置数据路径"""
        self._client.set_data_path(path)
    
    def set_device(self, device):
        """设置设备"""
        self._client.set_device(device)
    
    def set_permute_type(self, permute_type):
        """设置排列类型"""
        self._client.set_permute_type(permute_type)

    def enable_spike_output(self):
        """开启spk文件输出"""
        self._client.set_spike_output_enabled(True)

    def disable_spike_output(self):
        """关闭spk文件输出"""
        self._client.set_spike_output_enabled(False)

    def is_spike_output_enabled(self):
        """查询spk文件输出状态"""
        return self._client.is_spike_output_enabled()
    
    def load_model(self):
        """加载模型"""
        self._client.load_model()
        self._model_loaded = True
    
    def add_monitor(self, monitor_type, variable, node_idx, array_index=0):
        """添加监控器"""
        return self._client.add_monitor_with_array(monitor_type, variable, node_idx, array_index)
    
    def get_monitor_data(self, monitor_id):
        """获取监控数据"""
        return self._client.get_monitor_data(monitor_id)
    
    def set_dt(self, dt):
        """设置时间步长（兼容旧接口）"""
        self.dt = dt
    
    def finitialize(self, v_init):
        """初始化"""
        self._client.finitialize(v_init)
    
    def run(self, runtime):
        """运行仿真"""
        self._client.run(runtime)

    def continue_run(self, runtime):
        """继续运行仿真（不重置t，从当前t开始推进runtime时间）"""
        self._client.continue_run(runtime)

    def fadvance(self):
        """单步推进一次（推进一个dt）"""
        self._client.fadvance()

    def get_t(self):
        """获取当前仿真时间t"""
        return self._client.get_t()

    def flush_recorders(self):
        """将record缓冲区刷入输出缓冲（step模式可用）"""
        return self._client.flush_recorders()
    
    def get_variable_value(self, mech_name, var_name, row_id, array_index=0):
        """获取变量值"""
        return self._client.get_variable_value_with_array(mech_name, var_name, row_id, array_index)

    def set_variable_value(self, value, mech_name, var_name, row_id, array_index=0):
        """设置变量值"""
        self._client.set_variable_value_with_array(value, mech_name, var_name, row_id, array_index)

    def get_variable_array(self, mech_name, var_name, row_id, array_length):
        """获取整个数组变量的值"""
        return [self._client.get_variable_value_with_array(mech_name, var_name, row_id, i)
                for i in range(array_length)]

    def get_variables_by_handles(self, handles):
        """批量读取变量值"""
        if not handles:
            return []
        handle_list = [int(h) for h in handles]
        return self._client.get_variables_by_handles(handle_list)

    def get_variables_by_handles_f32(self, handles):
        """
        批量读取变量值（float32 numpy array）。

        性能目标：避免 GPU 模式下逐个 handle 的小额 GPU->CPU 拷贝，以及 Python list 装箱开销。
        若底层不支持，则回退到 `get_variables_by_handles()` 再转 numpy。
        """
        if not handles:
            return np.zeros((0,), dtype=np.float32)
        handle_list = [int(h) for h in handles]
        if hasattr(self._client, "get_variables_by_handles_f32_into"):
            out = np.empty((len(handle_list),), dtype=np.float32)
            self._client.get_variables_by_handles_f32_into(handle_list, out)
            return out
        vals = self._client.get_variables_by_handles(handle_list)
        return np.asarray(vals, dtype=np.float32)

    def set_variables_by_handles(self, handles, values):
        """批量写入变量值"""
        if not handles:
            return 0
        handle_list = [int(h) for h in handles]
        value_list = [float(v) for v in values]
        return self._client.set_variables_by_handles(handle_list, value_list)

    def create_optimizer(self, optimizer_type="sgd"):
        """创建优化器并返回ID"""
        if not self._model_loaded:
            raise RuntimeError("Model must be loaded before creating optimizer")
        optimizer_id = self._client.create_optimizer(optimizer_type)
        if optimizer_id < 0:
            raise RuntimeError(f"Failed to create optimizer of type {optimizer_type}")
        self._optimizer_id = optimizer_id
        self._optimizer_type = optimizer_type.lower()
        return optimizer_id

    def optimizer_add_param(self, optimizer_id, weight_handle, grad_handle, impedance):
        """向优化器注册参数"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.optimizer_add_param(optimizer_id, weight_handle, grad_handle, impedance)
        if result < 0:
            raise RuntimeError(f"Failed to register optimizer param (weight_handle={weight_handle}, grad_handle={grad_handle})")
        return result

    def optimizer_add_param_batch(self, optimizer_id, weight_handles, grad_handles, impedance):
        """批量向优化器注册一组参数（用于batch训练）"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        if len(weight_handles) != len(grad_handles):
            raise ValueError("weight_handles and grad_handles must have the same length")
        if len(weight_handles) == 0:
            raise ValueError("optimizer_add_param_batch requires at least one handle")

        weight_list = [int(h) for h in weight_handles]
        grad_list = [int(h) for h in grad_handles]

        result = self._client.optimizer_add_param_batch(optimizer_id, weight_list, grad_list, impedance)
        if result < 0:
            raise RuntimeError("Failed to register batch optimizer params")
        return result

    def configure_optimizer(self, optimizer_id, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """配置优化器超参数"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.configure_optimizer(optimizer_id, momentum, beta1, beta2, epsilon)
        if result < 0:
            raise RuntimeError("Failed to configure optimizer")
        return result

    def optimizer_step(self, optimizer_id, learning_rate, record_time, dt):
        """执行优化器更新"""
        if optimizer_id < 0:
            raise ValueError("optimizer_id must be non-negative")
        result = self._client.optimizer_step(optimizer_id, learning_rate, record_time, dt)
        if result < 0:
            raise RuntimeError("Optimizer step failed")
        return result
    
    def get_spk_by_gid(self, gid):
        """获取某个gid对应的spk时间戳"""
        return self._client.get_spk_by_gid(gid)

    def enhanced_export_model(self, export_path, dt=0.05, v_init=-62.5):
        """
        Enhanced export functionality that saves both NEURON model and wrapper metadata
        
        This method:
        1. Exports NEURON model using existing functionality
        2. Initializes all wrappers to extract metadata
        3. Saves wrapper metadata and configuration to JSON files
        
        Parameters:
        -----------
        export_path : str
            Directory path for export
        dt : float, optional
            Time step (default: 0.05)
        v_init : float, optional
            Initial voltage (default: -62.5)
            
        Files created:
        --------------
        - heliox_metadata.json: Wrapper information
        - heliox_config.json: Configuration (device, permute_type, dt, v_init)
        - *.dat files: NEURON exported data
        """
        import heliox_export
        from neuron import h
        
        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)
        
        # 1. Standard NEURON model export
        pc = h.ParallelContext()
        pc.setup_transfer()
        pc.set_maxstep(10)
        h.dt = dt
        h.finitialize(v_init)
        
        # Export model
        pc.nrnbbcore_write(export_path)
        print(f"NEURON model exported to: {export_path}")
        
        # 2. Configure heliox for metadata extraction
        self.set_data_path(export_path)
        self.set_device(self.device)
        self.set_permute_type(self.permute_type)
        
        # 3. Initialize all wrappers to extract metadata
        self._initialize_obj_wrappers()
        self._initialize_monitor_wrappers()
        self._initialize_vecplay_wrappers()
        
        # 4. Collect all wrapper metadata
        metadata = {
            "version": "1.0",
            "export_timestamp": heliox_export.datetime.datetime.now().isoformat(),
            "wrappers": {
                "monitors": [],
                "obj_wrappers": [],
                "vecplay_wrappers": []
            }
        }
        
        # Serialize all monitor wrappers (both pending and previously initialized)
        all_monitors = list(self.pending_monitor_wrappers) + [w for w in self._all_monitor_wrappers if w not in self.pending_monitor_wrappers]
        for wrapper in all_monitors:
            if wrapper.initialized:
                metadata["wrappers"]["monitors"].append(
                    heliox_export.serialize_monitor_wrapper(wrapper)
                )
        
        # Serialize all object wrappers
        all_obj_wrappers = list(self.pending_obj_wrappers) + [w for w in self._all_obj_wrappers if w not in self.pending_obj_wrappers]
        for wrapper in all_obj_wrappers:
            if wrapper.initialized:
                metadata["wrappers"]["obj_wrappers"].append(
                    heliox_export.serialize_obj_wrapper(wrapper)
                )
        
        # Serialize all VecPlay wrappers
        all_vecplay_wrappers = list(self.pending_vecplay_wrappers) + [w for w in self._all_vecplay_wrappers if w not in self.pending_vecplay_wrappers]
        for wrapper in all_vecplay_wrappers:
            if wrapper.initialized:
                metadata["wrappers"]["vecplay_wrappers"].append(
                    heliox_export.serialize_vecplay_wrapper(wrapper)
                )
        
        # 5. Save metadata to file
        metadata_path = os.path.join(export_path, "heliox_metadata.json")
        heliox_export.save_metadata_to_file(metadata, metadata_path)
        print(f"Wrapper metadata saved to: {metadata_path}")
        print(f"  - {len(metadata['wrappers']['monitors'])} monitor wrappers")
        print(f"  - {len(metadata['wrappers']['obj_wrappers'])} object wrappers")
        print(f"  - {len(metadata['wrappers']['vecplay_wrappers'])} VecPlay wrappers")
        
        # 6. Save configuration
        config = {
            "device": self.device,
            "permute_type": self.permute_type,
            "dt": dt,
            "v_init": v_init
        }
        config_path = os.path.join(export_path, "heliox_config.json")
        heliox_export.save_config_to_file(config, config_path)
        print(f"Configuration saved to: {config_path}")
        
        # 7. Load heliox model
        self.load_model()
        print("Enhanced export completed successfully")

    def load_from_export(self, export_path):
        """
        Load model and recreate wrappers from exported metadata
        
        This method allows loading models without NEURON by:
        1. Loading configuration from JSON
        2. Loading wrapper metadata from JSON
        3. Creating standalone wrappers
        4. Loading heliox model
        
        Parameters:
        -----------
        export_path : str
            Directory containing exported model and metadata
            
        Returns:
        --------
        dict
            Dictionary containing recreated wrappers organized by type:
            {
                'monitors': [list of MonitorWrapper objects],
                'obj_wrappers': [list of ObjWrapper objects], 
                'vecplay_wrappers': [list of VecPlayWrapper objects]
            }
        """
        import heliox_export
        # Check for metadata files
        metadata_path = os.path.join(export_path, "heliox_metadata.json")
        config_path = os.path.join(export_path, "heliox_config.json")
        
        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata found at {metadata_path}")
            print("Falling back to standard load behavior")
            # Fallback to existing behavior
            self.set_data_path(export_path)
            self.set_device(self.device)
            self.set_permute_type(self.permute_type)
            self.load_model()
            return {"monitors": [], "obj_wrappers": [], "vecplay_wrappers": []}
        
        # Load metadata and configuration
        metadata = heliox_export.load_metadata_from_file(metadata_path)
        if not heliox_export.validate_metadata(metadata):
            raise ValueError(f"Invalid metadata format in {metadata_path}")
        print(f"Loaded metadata from: {metadata_path}")

        # Load configuration if available
        if os.path.exists(config_path):
            config = heliox_export.load_config_from_file(config_path)
            if heliox_export.validate_config(config):
                self.set_device(config["device"])
                self.set_permute_type(config["permute_type"])
                print(f"Loaded configuration from: {config_path}")
            else:
                raise ValueError(f"Invalid configuration format in {config_path}")
        else:
            print("Warning: No configuration file found, using current settings")
        
        # Configure heliox
        self.set_data_path(export_path)
        
        # Load heliox model first
        self.load_model()
        print("HelioX model loaded successfully")
        
        # Recreate wrappers from metadata
        recreated_wrappers = {
            "monitors": [],
            "obj_wrappers": [],
            "vecplay_wrappers": []
        }

        # Recreate monitor wrappers
        for monitor_data in metadata["wrappers"]["monitors"]:
            wrapper = heliox_export.create_standalone_monitor_wrapper(monitor_data, self._client)
            # Register the monitor with heliox
            monitor_id = self._client.add_monitor(
                wrapper._mech_name,
                wrapper.var_name,
                wrapper._node_or_mech_idx
            )
            wrapper.monitor_id = monitor_id
            recreated_wrappers["monitors"].append(wrapper)

        # Recreate object wrappers
        for obj_data in metadata["wrappers"]["obj_wrappers"]:
            wrapper = heliox_export.create_standalone_obj_wrapper(obj_data, self._client)
            recreated_wrappers["obj_wrappers"].append(wrapper)

        # Recreate VecPlay wrappers
        for vecplay_data in metadata["wrappers"]["vecplay_wrappers"]:
            wrapper = heliox_export.create_standalone_vecplay_wrapper(vecplay_data, self._client)
            recreated_wrappers["vecplay_wrappers"].append(wrapper)

        print(f"Recreated {len(recreated_wrappers['monitors'])} monitor wrappers")
        print(f"Recreated {len(recreated_wrappers['obj_wrappers'])} object wrappers")
        print(f"Recreated {len(recreated_wrappers['vecplay_wrappers'])} VecPlay wrappers")
        
        return recreated_wrappers

    @classmethod
    def disable_destructor_warnings(cls):
        """手动禁用包装器析构警告"""
        global _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED
        _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = False
        print("Wrapper destructor warnings disabled")
    
    @classmethod  
    def enable_destructor_warnings(cls):
        """手动启用包装器析构警告"""
        global _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED
        _WRAPPER_DESTRUCTOR_WARNINGS_ENABLED = True
        print("Wrapper destructor warnings enabled")

