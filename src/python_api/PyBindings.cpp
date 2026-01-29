#include "SimWrapper.h"
#include <nanobind/nanobind.h>
#include <span>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

NB_MODULE(heliox, m) {
    m.doc() = "神经元模拟器Python绑定";
    
    nb::class_<SimWrapper>(m, "Sim")
        .def(nb::init<>())
        .def("set_data_path", &SimWrapper::set_data_path, "设置数据路径")
        .def("set_device", &SimWrapper::set_device, "设置计算设备")
        .def("load_model", &SimWrapper::load_model, "加载模型")
        .def("finitialize", &SimWrapper::finitialize, "初始化模拟")
        .def("run", &SimWrapper::run, "运行模拟")
        .def("continue_run", &SimWrapper::continue_run, "继续运行模拟（不重置t）")
        .def("fadvance", &SimWrapper::fadvance, "单步推进一次（推进一个dt）")
        .def("get_t", &SimWrapper::get_t, "获取当前仿真时间t")
        .def("flush_recorders", &SimWrapper::flush_recorders, "将record缓冲区刷入输出缓冲（step模式可用）")
        .def("set_spike_output_enabled", &SimWrapper::set_spike_output_enabled, "设置是否输出spk文件")
        .def("is_spike_output_enabled", &SimWrapper::is_spike_output_enabled, "获取spk文件输出状态")
        .def("set_permute_type", &SimWrapper::set_permute_type, "设置排列类型")
        .def("add_monitor", &SimWrapper::add_monitor, "添加监视点")
        .def("add_monitor_with_array", &SimWrapper::add_monitor_with_array, "添加监视点（支持数组索引）")
        .def("set_dt", &SimWrapper::set_dt, "设置时间步长")
        .def("get_dt", &SimWrapper::get_dt, "获取时间步长")
        .def("set_output_dir", &SimWrapper::set_output_dir, "设置输出目录") 
        .def("get_monitor_data", &SimWrapper::get_monitor_data, "获取监视点数据")
        .def("get_multiple_monitor_data", &SimWrapper::get_multiple_monitor_data, "获取多个监视点的数据")
        .def("set_variable_value", &SimWrapper::set_variable_value, "设置变量值")
        .def("set_variable_value_with_array", &SimWrapper::set_variable_value_with_array, "设置变量值（支持数组索引）")
        .def("get_variable_value", &SimWrapper::get_variable_value, "获取变量值")
        .def("get_variable_value_with_array", &SimWrapper::get_variable_value_with_array, "获取变量值（支持数组索引）")
        .def("create_optimizer", &SimWrapper::create_optimizer, "创建优化器并返回ID")
        .def("optimizer_add_param", &SimWrapper::optimizer_add_param, "向优化器注册参数")
        .def("optimizer_add_param_batch", &SimWrapper::optimizer_add_param_batch, "批量向优化器注册参数")
        .def("configure_optimizer", &SimWrapper::configure_optimizer, "配置优化器超参数",
             nb::arg("optimizer_id"), nb::arg("momentum") = 0.9, nb::arg("beta1") = 0.9,
             nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8)
        .def("optimizer_step", &SimWrapper::optimizer_step, "执行优化器更新")
        .def("optimizer_step_with_inv_record_steps",
             &SimWrapper::optimizer_step_with_inv_record_steps,
             "执行优化器更新（直接指定 inv_record_steps，避免 record_time/dt 缩放）")
        .def("optimizer_add_external_grads",
             &SimWrapper::optimizer_add_external_grads,
             "向优化器注册外部梯度槽位（每个weight_handle对应一个独立参数；batch_size=1）")
        .def("optimizer_set_external_grads_f32",
             &SimWrapper::optimizer_set_external_grads_f32,
             "设置外部梯度（float32数组，自动转换为double并同步到GPU）")
        .def("optimizer_clear_external_grads",
             &SimWrapper::optimizer_clear_external_grads,
             "清空外部梯度（置0，并同步到GPU）")
        .def("optimizer_reset_state", &SimWrapper::optimizer_reset_state, "重置优化器内部状态（例如Adam动量）")
        .def("optimizer_get_adam_state",
             &SimWrapper::optimizer_get_adam_state,
             "获取Adam状态 (step, m, v, beta1, beta2, epsilon)")
        .def("optimizer_set_adam_state",
             &SimWrapper::optimizer_set_adam_state,
             "设置Adam状态 (optimizer_id, step, m, v, beta1, beta2, epsilon)")
        .def("call_mech_func", &SimWrapper::call_mech_func, "调用自定义函数")
        // .def("vecevent_play", &SimWrapper::vecevent_play, "VecEvent播放")
        .def("set_user_mod_num", &SimWrapper::set_user_mod_num, "设置用户模块数量")
        .def("get_spk_by_gid", &SimWrapper::get_spk_by_gid, "获取某个gid对应的spk时间戳",
            nb::rv_policy::reference_internal)
        .def("get_monitor_handle", &SimWrapper::get_monitor_handle, "获取监视点handle")
        .def("get_monitor_handle_with_array", &SimWrapper::get_monitor_handle_with_array, "获取监视点handle（支持数组索引）")
        // VecPlay相关方法
        .def("add_vecplay", &SimWrapper::add_vecplay, "添加VecPlay")
        .def("update_vecplay", &SimWrapper::update_vecplay, "更新VecPlay")
        .def("remove_vecplay", &SimWrapper::remove_vecplay, "删除VecPlay")
        .def("has_vecplay", &SimWrapper::has_vecplay, "检查VecPlay是否存在")
        .def("get_all_vecplay_keys", &SimWrapper::get_all_vecplay_keys, "获取所有VecPlay键")
        // 变量handle相关方法
        .def("get_variable_handle", &SimWrapper::get_variable_handle, "获取变量的快速访问handle")
        .def("get_variable_handle_with_array", &SimWrapper::get_variable_handle_with_array, "获取变量的快速访问handle（支持数组索引）")
        .def("get_variable_by_handle", &SimWrapper::get_variable_by_handle, "通过handle获取变量值")
        .def("get_variables_by_handles", &SimWrapper::get_variables_by_handles, "批量通过handle获取变量值")
        .def("get_variables_by_handles_f32_into", &SimWrapper::get_variables_by_handles_f32_into, "批量通过handle获取变量值（float32，写入预分配buffer）")
        .def("set_variable_by_handle", &SimWrapper::set_variable_by_handle, "通过handle设置变量值")
        .def("set_variables_by_handles", &SimWrapper::set_variables_by_handles, "批量通过handle设置变量值")
        // Training/learning helper fast paths
        .def("set_dense_blocks_f32", &SimWrapper::set_dense_blocks_f32, "上传分块矩阵到GPU（可选）")
        .def("clear_dense_blocks", &SimWrapper::clear_dense_blocks, "清理分块矩阵GPU缓存")
        .def("simulate_output_vs_into",
             &SimWrapper::simulate_output_vs_into,
             "运行仿真并采集输出v(t)（写入预分配buffer）",
             nb::arg("output_vs_tn"),
             nb::arg("output_v_handles"),
             nb::arg("tstop_ms"),
             nb::arg("v_init"))
        .def("simulate_and_capture_mapped_signals_into",
             &SimWrapper::simulate_and_capture_mapped_signals_into,
             "运行仿真并采集训练信号（写入预分配buffer）",
             nb::arg("output_vs_tn"),
             nb::arg("it_lr_tn"),
             nb::arg("ditdv_lr_tn"),
             nb::arg("ditdvpre_lr_tn"),
             nb::arg("output_v_handles"),
             nb::arg("pure_i_handles"),
             nb::arg("pure_i_dest"),
             nb::arg("pure_i_scale"),
             nb::arg("didv_handles"),
             nb::arg("didv_dest"),
             nb::arg("didv_scale"),
             nb::arg("didvpre_handles"),
             nb::arg("didvpre_dest"),
             nb::arg("didvpre_scale"),
             nb::arg("tstop_ms"),
             nb::arg("k_mul"),
             nb::arg("percise"),
             nb::arg("v_init"))
        .def("simulate_and_capture_mapped_signals_cached",
             &SimWrapper::simulate_and_capture_mapped_signals_cached,
             "运行仿真并采集训练信号（缓存到GPU；仅输出output_vs）",
             nb::arg("output_vs_tn"),
             nb::arg("output_v_handles"),
             nb::arg("pure_i_handles"),
             nb::arg("pure_i_dest"),
             nb::arg("pure_i_scale"),
             nb::arg("didv_handles"),
             nb::arg("didv_dest"),
             nb::arg("didv_scale"),
             nb::arg("didvpre_handles"),
             nb::arg("didvpre_dest"),
             nb::arg("didvpre_scale"),
             nb::arg("tstop_ms"),
             nb::arg("k_mul"),
             nb::arg("percise"),
             nb::arg("v_init"))
        .def("replay_compute_dw_dx_from_signals_into",
             &SimWrapper::replay_compute_dw_dx_from_signals_into,
             "Replay dv/dw 并直接输出 dw/dx（写入预分配buffer）",
             nb::arg("it_lr_nt"),
             nb::arg("ditdv_lr_nt"),
             nb::arg("ditdvpre_lr_nt"),
             nb::arg("dLtdv_lr_to"),
             nb::arg("poutput"),
             nb::arg("pinput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dx_lr_it"),
             nb::arg("dt_ms"),
             nb::arg("percise"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        .def("replay_compute_dw_dx_from_cached_signals_into",
             &SimWrapper::replay_compute_dw_dx_from_cached_signals_into,
             "Replay dv/dw（使用GPU缓存的signal矩阵），直接输出 dw/dx（写入预分配buffer）",
             nb::arg("dLtdv_lr_ot"),
             nb::arg("poutput"),
             nb::arg("pinput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dx_lr_it"),
             nb::arg("dt_ms"),
             nb::arg("percise"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        .def("replay_compute_dw_from_cached_signals_into",
             &SimWrapper::replay_compute_dw_from_cached_signals_into,
             "Replay dv/dw（使用GPU缓存的signal矩阵），只输出 dw（不计算 dx）",
             nb::arg("dLtdv_lr_ot"),
             nb::arg("poutput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dt_ms"),
             nb::arg("percise"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        .def("simulate_and_replay_dw_dx_streaming_into",
             &SimWrapper::simulate_and_replay_dw_dx_streaming_into,
             "流式Replay：只保留LR信号ring buffer，直接输出dw/dx（写入预分配buffer）",
             nb::arg("dLtdv_lr_ot"),
             nb::arg("poutput"),
             nb::arg("pinput"),
             nb::arg("pre_of_col"),
             nb::arg("dw_out_n"),
             nb::arg("dx_lr_it"),
             nb::arg("pure_i_handles"),
             nb::arg("pure_i_dest"),
             nb::arg("pure_i_scale"),
             nb::arg("didv_handles"),
             nb::arg("didv_dest"),
             nb::arg("didv_scale"),
             nb::arg("didvpre_handles"),
             nb::arg("didvpre_dest"),
             nb::arg("didvpre_scale"),
             nb::arg("tstop_ms"),
             nb::arg("k_mul"),
             nb::arg("percise"),
             nb::arg("v_init"),
             nb::arg("dt_ms"),
             nb::arg("grad_scale") = 1.0,
             nb::arg("eps") = 1e-6,
             nb::arg("grad_l2norm_threshold") = 1e6,
             nb::arg("clip_strategy") = 1,
             nb::arg("clip_check_every") = 1)
        // Gap Junction相关方法
        .def("get_all_gap_junctions", &SimWrapper::get_all_gap_junctions, "获取所有间隙连接信息")
        .def("get_gap_junction", &SimWrapper::get_gap_junction, "获取特定间隙连接信息")
        .def("add_gap_source", &SimWrapper::add_gap_source, "添加间隙连接源")
        .def("add_gap_target", &SimWrapper::add_gap_target, "为间隙连接添加目标")
        .def("clear_all_gap_junctions", &SimWrapper::clear_all_gap_junctions, "清空所有间隙连接")
        .def("get_next_available_sid", &SimWrapper::get_next_available_sid, "获取下一个可用的源ID")
        .def("register_netstim_batch",
             [](SimWrapper& self,
                const std::vector<std::tuple<int, int, int>>& handles,
                double interval_scale,
                double start_base,
                double epsilon,
                double number) {
                 SimWrapper::NetStimBatchParams params;
                 params.interval_scale = interval_scale;
                 params.start_base = start_base;
                 params.epsilon = epsilon;
                 params.number = number;
                 return self.register_netstim_batch(handles, params);
             },
             nb::arg("handle_triplets"),
             nb::arg("interval_scale") = 5.0,
             nb::arg("start_base") = 9.0,
             nb::arg("epsilon") = 0.01,
             nb::arg("number") = 100.0)
        .def("register_vecstim_batch",
             [](SimWrapper& self,
                const std::vector<int>& mech_indices,
                double spike_scale,
                double start_base,
                double epsilon,
                int spike_count) {
                 SimWrapper::VecStimBatchParams params;
                 params.spike_scale = spike_scale;
                 params.start_base = start_base;
                 params.epsilon = epsilon;
                 params.spike_count = spike_count;
                 return self.register_vecstim_batch(mech_indices, params);
             },
             nb::arg("mech_indices"),
             nb::arg("spike_scale") = 5.0,
             nb::arg("start_base") = 9.0,
             nb::arg("epsilon") = 0.01,
             nb::arg("spike_count") = 20)
        .def("set_input_batch_pixels",
             [](SimWrapper& self,
                int batch_id,
                nb::ndarray<double, nb::shape<-1>, nb::c_contig> pixels) {
                 std::span<const double> span(pixels.data(), pixels.shape(0));
                 return self.set_input_batch_pixels(batch_id, span);
             },
             nb::arg("batch_id"),
             nb::arg("pixels"));
}
