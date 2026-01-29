#ifndef SIM_WRAPPER_H
#define SIM_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "neuron.h"
#include "mechanism.h"
#include "simulate.h"
#include "utils.h"
#include "cuda_utils.h"
#include "coredat_structs.h"
#include "permute_order.h"
#include "read_coredat.h"
#include "cxxopts.hpp"
#include "coredat_to_innerdat.h"
#include "magic_enum/magic_enum.hpp"
#include "global_vars.h"
#include <set>
#include <netinet/in.h>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include "args_to_any.hpp"
#include "runtime_api/core/SimRuntimeCore.h"
#include "runtime_api/learn/LearnRuntime.h"
namespace nb = nanobind;
using namespace std;

/**
 * @brief 神经元模拟器类，提供Python可调用的接口
 */
class SimWrapper {
	private:
	    heliox::runtime_api::core::SimRuntimeCore core_;
        heliox::runtime_api::learn::LearnRuntime learn_{core_};
	    // Raw pointer view for legacy code paths in SimWrapper.cpp (especially learn/replay).
	    // Ownership lives in core_.
	    Simulate* sim = nullptr;

	public:
    SimWrapper();
    ~SimWrapper();
    
    /**
     * @brief 设置数据路径
     * @param path 数据文件路径
     * @return 0表示成功，-1表示失败
     */
    int set_data_path(const string& path);
    
    /**
     * @brief 设置计算设备（CPU或GPU）
     * @param dev 设备名称："cpu"或"gpu"
     * @return 0表示成功，-1表示失败
     */
    int set_device(const string& dev);
    
    /**
     * @brief 加载模型
     * @return 0表示成功，-1表示失败
     */
    int load_model();
    
    /**
     * @brief 初始化模拟
     * @param v_init 初始膜电位
     * @return 0表示成功，-1表示失败
     */
    int finitialize(double v_init);
    
    /**
     * @brief 运行模拟
     * @param tstop 模拟结束时间
     * @return 0表示成功，-1表示失败
     */
    int run(double tstop);

    /**
     * @brief 继续运行模拟（不重置t，从当前t开始推进runtime时间）
     * @param runtime 继续运行的时间长度
     * @return 0表示成功，-1表示失败
     */
    int continue_run(double runtime);

    /**
     * @brief 单步推进一次（推进一个dt）
     * @return 0表示成功，-1表示失败
     */
    int fadvance();

    /**
     * @brief 获取当前仿真时间t
     * @return 当前时间（ms）
     */
    double get_t() const;

    /**
     * @brief 将当前 recorder 缓冲区刷入IPC/HDF5（便于在step模式下读取monitor数据）
     * @return 0表示成功，-1表示失败
     */
    int flush_recorders();

    /**
     * @brief 设置是否输出spk文件
     * @param enable true表示开启，false表示关闭
     * @return 0表示成功
     */
    int set_spike_output_enabled(bool enable);

    /**
     * @brief 当前spk文件输出是否开启
     * @return true表示开启
     */
    bool is_spike_output_enabled() const;
    
    /**
     * @brief 设置排列类型
     * @param type 排列类型
     * @return 0表示成功
     */
    int set_permute_type(int type);
    
    /**
     * @brief 添加监视点
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 监视点handle（在load_model时分配），-1表示失败
     */
    int add_monitor(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 添加监视点（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 监视点handle（在load_model时分配），-1表示失败
     */
    int add_monitor_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    
    /**
     * @brief 设置时间步长
     * @param dt 时间步长
     * @return 0表示成功
     */
    int set_dt(double dt);
    
    /**
     * @brief 获取时间步长
     * @return 当前时间步长
     */
    double get_dt();
    
    /**
     * @brief 设置输出目录
     * @param dir 输出目录路径
     * @return 0表示成功，-1表示失败
     */
    int set_output_dir(const string& dir);
    
    /**
     * @brief 获取监视点的数据
     * @param handle 监视点handle
     * @return 包含监视点数据的向量，如果失败则返回空向量
     */
    vector<double> get_monitor_data(int handle);
    
    /**
     * @brief 获取多个监视点的数据
     * @param handles 监视点handle列表
     * @return 监视点handle到数据的映射
     */
    map<int, vector<double>> get_multiple_monitor_data(const vector<int>& handles);
    
    /**
     * @brief 设置变量值
     * @param val 要设置的值
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 0表示成功，-1表示失败
     */
    int set_variable_value(double val, const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 设置变量值（支持数组索引）
     * @param val 要设置的值
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 0表示成功，-1表示失败
     */
    int set_variable_value_with_array(double val, const string& mech, const string& var, int node_or_mech_idx, int array_index);

    /**
     * @brief 读取全局变量数据
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 变量值
     */
    double get_variable_value(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 读取全局变量数据（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 变量值
     */
    double get_variable_value_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);

    int create_optimizer(const string& optimizer_type);
    int optimizer_add_param(int optimizer_id, int weight_handle, int grad_handle, double impedance);
    int optimizer_add_param_batch(int optimizer_id,
                                  const std::vector<int>& weight_handles,
                                  const std::vector<int>& grad_handles,
                                  double impedance);
    int configure_optimizer(int optimizer_id, double momentum, double beta1, double beta2, double epsilon);
    int optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt);
    int optimizer_step_with_inv_record_steps(int optimizer_id, double learning_rate, double inv_record_steps);
    int optimizer_add_external_grads(int optimizer_id, const std::vector<int>& weight_handles, double impedance);
    int optimizer_set_external_grads_f32(int optimizer_id,
                                         nb::ndarray<float, nb::shape<-1>, nb::c_contig> grads);
    int optimizer_clear_external_grads(int optimizer_id);
    int optimizer_reset_state(int optimizer_id);
    std::tuple<long long, std::vector<double>, std::vector<double>, double, double, double> optimizer_get_adam_state(
        int optimizer_id);
    int optimizer_set_adam_state(int optimizer_id,
                                 long long step_count,
                                 const std::vector<double>& m,
                                 const std::vector<double>& v,
                                 double beta1,
                                 double beta2,
                                 double epsilon);

    /**
     * @brief 调用自定义函数
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param args 参数列表
     * @return 变量值
     */
    double call_mech_func(const string& mech_name, const string& func_name, nb::args args);

    // /**
    //  * @brief VecEvent播放
    //  * @param mech_idx 机制索引
    //  * @param data 数据数组
    //  * @return 0表示成功，-1表示失败
    //  */
    // int vecevent_play(int mech_idx, const vector<double>& data);
    
    /**
     * @brief 设置用户模块数量
     * @param num 模块数量
     */
    void set_user_mod_num(int num);

    /**
     * @brief 获取某个gid对应的spk时间戳
     * @param gid 神经元ID
     * @return 神经元spk时间戳
     */
    vector<double>& get_spk_by_gid(int gid);
    
    /**
     * @brief 通过VarDescriptor查找对应的handle
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return handle，-1表示未找到
     */
    int get_monitor_handle(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 通过VarDescriptor查找对应的handle（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return handle，-1表示未找到
     */
    int get_monitor_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    
    // VecPlay相关方法
    /**
     * @brief 添加VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @param tvec 时间向量
     * @param yvec 值向量
     * @return 0表示成功，-1表示失败
     */
    int add_vecplay(const string& mech_name, const string& var_name, int instance_id,
                    const vector<double>& tvec, const vector<double>& yvec);
    
    /**
     * @brief 更新VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @param new_tvec 新的时间向量
     * @param new_yvec 新的值向量
     * @return 0表示成功，-1表示失败
     */
    int update_vecplay(const string& mech_name, const string& var_name, int instance_id,
                       const vector<double>& new_tvec, const vector<double>& new_yvec);
    
    /**
     * @brief 删除VecPlay
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @return 0表示成功，-1表示失败
     */
    int remove_vecplay(const string& mech_name, const string& var_name, int instance_id);
    
    /**
     * @brief 检查VecPlay是否存在
     * @param mech_name 机制名称
     * @param var_name 变量名称
     * @param instance_id 实例ID
     * @return true表示存在，false表示不存在
     */
    bool has_vecplay(const string& mech_name, const string& var_name, int instance_id);
    
    /**
     * @brief 获取所有VecPlay键
     * @return 包含所有VecPlay键的向量，每个键为[mech_name, var_name, instance_id]
     */
    vector<vector<string>> get_all_vecplay_keys();
    
    /**
     * @brief 获取变量的快速访问handle
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @return 变量handle，-1表示失败
     */
    int get_variable_handle(const string& mech, const string& var, int node_or_mech_idx);

    /**
     * @brief 获取变量的快速访问handle（支持数组索引）
     * @param mech 机制名称
     * @param var 变量名称
     * @param node_or_mech_idx 节点或机制索引
     * @param array_index 数组索引，默认为0
     * @return 变量handle，-1表示失败
     */
    int get_variable_handle_with_array(const string& mech, const string& var, int node_or_mech_idx, int array_index);
    
    /**
     * @brief 通过handle获取变量值
     * @param handle 变量handle
     * @return 变量值
     */
    double get_variable_by_handle(int handle);

    /**
     * @brief 批量通过handle获取变量值
     * @param handles 变量handle列表
     * @return 按输入顺序排列的变量值
     */
    std::vector<double> get_variables_by_handles(const std::vector<int>& handles);

    /**
     * @brief 批量通过handle获取变量值（float32，写入预分配buffer）
     *
     * 设计目的：降低 Python<->C++ 往返与单元素 GPU->CPU 拷贝开销，适用于频繁采样场景（如 replay 训练）。
     *
     * @param handles 变量handle列表
     * @param out 预分配的 float32 一维数组（长度必须等于 handles.size()）
     * @return 0表示成功，-1表示失败
     */
    int get_variables_by_handles_f32_into(const std::vector<int>& handles,
                                          nb::ndarray<float, nb::shape<-1>, nb::c_contig> out);
    
    /**
     * @brief 通过handle设置变量值
     * @param handle 变量handle
     * @param value 要设置的值
     * @return 0表示成功，-1表示失败
     */
    int set_variable_by_handle(int handle, double value);

    /**
     * @brief 批量通过handle设置变量值
     * @param handles 变量handle列表
     * @param values 对应的目标值
     * @return 0表示成功，-1表示失败
     */
    int set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values);

    // Gap Junction 管理相关方法
    /**
     * @brief 获取所有间隙连接信息
     * @return 包含所有间隙连接的映射，键为sid，值为包含源和目标信息的map
     */
    map<int, map<string, nb::object>> get_all_gap_junctions();
    
    /**
     * @brief 获取特定间隙连接信息
     * @param sid 源ID
     * @return 包含间隙连接信息的map，如果不存在则返回空map
     */
    map<string, nb::object> get_gap_junction(int sid);
    
    /**
     * @brief 添加间隙连接源
     * @param sid 源ID
     * @param src_mech 源机制名称
     * @param src_var 源变量名称
     * @param src_idx 源索引
     * @return 0表示成功，-1表示失败
     */
    int add_gap_source(int sid, const string& src_mech, const string& src_var, int src_idx);
    
    /**
     * @brief 为间隙连接添加目标
     * @param sid 源ID
     * @param tgt_mech 目标机制名称
     * @param tgt_var 目标变量名称  
     * @param tgt_idx 目标索引
     * @return 0表示成功，-1表示失败
     */
    int add_gap_target(int sid, const string& tgt_mech, const string& tgt_var, int tgt_idx);
    
    /**
     * @brief 清空所有间隙连接
     * @return 0表示成功
     */
    int clear_all_gap_junctions();
    
    /**
     * @brief 获取下一个可用的源ID
     * @return 下一个可用的sid
     */
    int get_next_available_sid();

    // 输入刺激批量控制 -------------------------------------------------
    struct NetStimBatchParams {
        double interval_scale = 5.0;
        double start_base = 9.0;
        double epsilon = 0.01;
        double number = 100.0;
    };

    struct VecStimBatchParams {
        double spike_scale = 5.0;
        double start_base = 9.0;
        double epsilon = 0.01;
        int spike_count = 20;
    };

    /**
     * @brief 注册一组NetStim刺激器，返回批量更新ID
     */
    int register_netstim_batch(const vector<tuple<int, int, int>>& handle_triplets,
                               const NetStimBatchParams& params);

    /**
     * @brief 注册一组VecStim刺激器，返回批量更新ID
     */
    int register_vecstim_batch(const vector<int>& mech_indices,
                               const VecStimBatchParams& params);

    /**
     * @brief 批量更新指定输入刺激批次的像素值
     */
    int set_input_batch_pixels(int batch_id, std::span<const double> pixels);

    // -----------------------------
    // Training/learning helper fast paths
    // -----------------------------
    // Upload dense blocks (list of float32 arrays shaped [bn, bn, K_len]) to GPU for later use.
    int set_dense_blocks_f32(nb::list blocks);
    void clear_dense_blocks();

    // Run simulation and capture only output v(t) into a preallocated time-major buffer:
    //   - output_vs_tn: (total_steps+1, n_output)
    //
    // Intended for streaming replay workflows where LR signals are captured/consumed on-device.
    int simulate_output_vs_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
        const std::vector<int>& output_v_handles,
        double tstop_ms,
        double v_init);

    // Replay dv/dw on GPU and directly return dw/dx (Option 1: backend outputs dw/dx).
    //
    // Inputs are CPU numpy arrays; this function copies them to GPU and runs the replay loop.
    // Expected layouts:
    //   - it_lr_nt / ditdv_lr_nt / ditdvpre_lr_nt: (N, ksteps_total+1), C-contiguous
    //   - dLtdv_lr_ot: (ksteps_total, n_output), C-contiguous (time-major so each tick is contiguous)
    //   - poutput/pinput: indices into [0..N)
    //   - pre_of_col: length N, values in [-1..N) (use -1 for "no pre")
    //
    // Outputs:
    //   - dw_out_n: (N,)
    //   - dx_lr_it: (n_input, ksteps_total)
    int replay_compute_dw_dx_from_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> it_lr_nt,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdv_lr_nt,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdvpre_lr_nt,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
        double dt_ms,
        bool percise,
        double grad_scale,
        double eps,
        double grad_l2norm_threshold,
        int clip_strategy,
        int clip_check_every);

    // Run simulation and capture mapped/accumulated learning signals using GPU-side gather + scatter-add.
    // Output layout is time-major for contiguous writes:
    //   - output_vs_tn: (total_steps+1, n_output)
    //   - it_lr_tn/ditdv_lr_tn/ditdvpre_lr_tn: (ksteps_total+1, N)
    //
    // This API is intended to replace Python per-step loops when capturing training signals.
    int simulate_and_capture_mapped_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> it_lr_tn,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdv_lr_tn,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> ditdvpre_lr_tn,
        const std::vector<int>& output_v_handles,
        const std::vector<int>& pure_i_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
        const std::vector<int>& didv_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
        const std::vector<int>& didvpre_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
        double tstop_ms,
        int k_mul,
        bool percise,
        double v_init);

    // Single-pass training helper:
    // - Run simulation and capture mapped learning signals into *cached* GPU buffers (time-major: (ksteps_total+1, N)).
    // - Also capture output v(t) into the provided CPU buffer `output_vs_tn` (time-major: (total_steps+1, n_output)).
    //
    // This enables a "one simulation pass" workflow for objectives whose dL/dv depends on the full trajectory:
    // Python can compute dL/dv from output_vs, then call replay-from-cache without re-simulating or uploading signals.
    int simulate_and_capture_mapped_signals_cached(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> output_vs_tn,
        const std::vector<int>& output_v_handles,
        const std::vector<int>& pure_i_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
        const std::vector<int>& didv_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
        const std::vector<int>& didvpre_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
        double tstop_ms,
        int k_mul,
        bool percise,
        double v_init);

    // Replay dv/dw using cached signal buffers produced by `simulate_and_capture_mapped_signals_cached`.
    // Inputs:
    // - dLtdv_lr_ot: (ksteps_total, n_output), time-major
    // - poutput/pinput/pre_of_col: index metadata
    // Outputs:
    // - dw_out_n: (N,)
    // - dx_lr_it: (n_input, ksteps_total)
    int replay_compute_dw_dx_from_cached_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
        double dt_ms,
        bool percise,
        double grad_scale,
        double eps,
        double grad_l2norm_threshold,
        int clip_strategy,
        int clip_check_every);

    // Replay dw only (no dx) from cached signal buffers produced by
    // `simulate_and_capture_mapped_signals_cached`.
    int replay_compute_dw_from_cached_signals_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
        double dt_ms,
        bool percise,
        double grad_scale,
        double eps,
        double grad_l2norm_threshold,
        int clip_strategy,
        int clip_check_every);

    // Streaming replay: run simulation, maintain only a ring-buffer of LR signals on GPU,
    // and compute dw/dx without materializing full (N, ksteps_total+1) signal matrices on CPU.
    //
    // Notes:
    // - This fast path assumes inputs are provided via VecPlay (caller installs/updates vecplay beforehand).
    // - dLtdv is time-major: (ksteps_total, n_output), contiguous.
    int simulate_and_replay_dw_dx_streaming_into(
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dLtdv_lr_ot,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> poutput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pinput,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pre_of_col,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> dw_out_n,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> dx_lr_it,
        const std::vector<int>& pure_i_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> pure_i_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> pure_i_scale,
        const std::vector<int>& didv_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didv_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didv_scale,
        const std::vector<int>& didvpre_handles,
        nb::ndarray<int32_t, nb::shape<-1>, nb::c_contig> didvpre_dest,
        nb::ndarray<float, nb::shape<-1>, nb::c_contig> didvpre_scale,
        double tstop_ms,
        int k_mul,
        bool percise,
        double v_init,
        double dt_ms,
        double grad_scale,
        double eps,
        double grad_l2norm_threshold,
        int clip_strategy,
        int clip_check_every);

	private:

    struct OptimizerExternalGrads {
        std::vector<int> weight_handles;
        VecData<double> grads;
        double impedance = 1.0;

        explicit OptimizerExternalGrads(Mode mode) : grads(mode) {}
        OptimizerExternalGrads(OptimizerExternalGrads&&) = default;
        OptimizerExternalGrads& operator=(OptimizerExternalGrads&&) = default;
        OptimizerExternalGrads(const OptimizerExternalGrads&) = delete;
        OptimizerExternalGrads& operator=(const OptimizerExternalGrads&) = delete;
    };
    std::unordered_map<int, OptimizerExternalGrads> optimizer_external_grads_;
};

#endif // SIM_WRAPPER_H
