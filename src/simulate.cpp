#include "simulate.h"
#include "cuda_utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include <magic_enum/magic_enum.hpp>
#include "mechanism.h"
namespace fs = std::filesystem;
extern void destroy_cuda_streams();
void solve_serial(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index, int ncell, int len);
void cpu_solve_permute1(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index, int nstride,
                        int* stride, int* firstnode, int* lastnode, int* cellsize, int ncell, int len);
void cpu_solve_permute3(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, int* parent_index,
                        int* max_order_each_thread, int* min_order_each_thread, int* firstnode, int* lastnode,
                        int* stride, int* map_t2c, int norder, int ncell, int nthread);
void run_vjp_gpu(
    NeuronGroupData* group,
    const std::vector<std::vector<double>>& local_grad_row_history,
    int permute_type);
void run_vjp_gpu_forward_contiguous(
    NeuronGroupData* group,
    const double* local_grad_forward,
    int local_count,
    int step_count,
    int permute_type);
void build_vjp_d_const_gpu(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const std::vector<Mechanism*>& pas_mechs,
    double vjp_cj);

namespace {

std::uint64_t spike_vjp_dense_byte_cap() {
    constexpr std::uint64_t kDefaultCap = 8ull * 1024ull * 1024ull * 1024ull;
    const char* env = std::getenv("NEURONG_SPIKE_VJP_MAX_DENSE_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultCap;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) {
        return kDefaultCap;
    }
    return static_cast<std::uint64_t>(parsed);
}

int checked_spike_vjp_dense_capacity(int step_count, int npre) {
    if (step_count <= 0 || npre <= 0) {
        return 1;
    }
    const auto elems =
        static_cast<std::int64_t>(step_count) * static_cast<std::int64_t>(npre);
    if (elems > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "prepare_vjp: spike VJP dense tape is too large, step_count=" +
            std::to_string(step_count) + " npre=" + std::to_string(npre));
    }
    const auto bytes =
        static_cast<std::uint64_t>(elems) * 2ull * static_cast<std::uint64_t>(sizeof(double));
    const auto max_bytes = spike_vjp_dense_byte_cap();
    if (bytes > max_bytes) {
        throw std::runtime_error(
            "prepare_vjp: spike VJP dense tapes would allocate about " +
            std::to_string(bytes) + " bytes per storage side, step_count=" +
            std::to_string(step_count) + " npre=" + std::to_string(npre) +
            "; set NEURONG_SPIKE_VJP_MAX_DENSE_BYTES to override");
    }
    return static_cast<int>(std::max<std::int64_t>(1, elems));
}

}  // namespace

Simulate::Simulate(Mode _mode, BufferEnable buf_enable, int buffer_size, bool recorder_use_fp32)
    : mode(_mode), hdf5_manager(mode, buffer_size, buf_enable, recorder_use_fp32), window_observer(mode)
{
    t = 0;
    dt = 0;
    hdf5_file = nullptr;
    tstop = 0;
    permute_type = 0;
    output_folder = "output";
}

void Simulate::set_spike_vjp_surrogate(const std::string& kind, double width_mv)
{
    if (!(width_mv > 0.0) || !std::isfinite(width_mv)) {
        throw std::runtime_error("set_spike_vjp_surrogate: width_mv must be finite and positive");
    }
    const auto& ops = neurong::spike_vjp::SpikeVjpSurrogateRegistry::getInstance().require(kind);
    spike_vjp_surrogate_name_ = ops.name;
    spike_vjp_surrogate_config_.width_mv = width_mv;
}

bool Simulate::ensure_finitialized_(const char* fn_name) const
{
    if (finitialize_done_) {
        return true;
    }
    printf("%s: finitialize() must be called before simulation stepping APIs\n", fn_name);
    return false;
}


void Simulate::arm_vjp_tape_recording_(int group_index) {
    if (group_index < 0 || group_index >= static_cast<int>(neuron_group_list.size())) {
        throw std::out_of_range("prepare_vjp: invalid group index for tape recording");
    }
    auto* group = neuron_group_list[static_cast<std::size_t>(group_index)];
    if (group == nullptr) {
        throw std::runtime_error("prepare_vjp: null group for tape recording");
    }
    autodiff_vjp_tape_recorded_steps_ = 0;
    autodiff_record_vjp_tape_ = autodiff_vjp_tape_step_count_ > 0;
}

bool Simulate::should_stage_vjp_tape_(const NeuronGroupData* group) const {
    return autodiff_record_vjp_tape_ &&
           prepared_vjp_valid_ &&
           prepared_vjp_group_index_ >= 0 &&
           prepared_vjp_group_index_ < static_cast<int>(neuron_group_list.size()) &&
           neuron_group_list[static_cast<std::size_t>(prepared_vjp_group_index_)] == group &&
           autodiff_vjp_tape_recorded_steps_ < autodiff_vjp_tape_step_count_;
}

void Simulate::note_vjp_tape_staged_(const NeuronGroupData* group) {
    if (!should_stage_vjp_tape_(group)) {
        return;
    }
    ++autodiff_vjp_tape_recorded_steps_;
    if (autodiff_vjp_tape_recorded_steps_ >= autodiff_vjp_tape_step_count_) {
        autodiff_record_vjp_tape_ = false;
    }
}

void Simulate::clear_current_vjp_tapes_() {
    for (auto* group : neuron_group_list) {
        if (group == nullptr) {
            continue;
        }
        for (auto* mech : group->mechanism_list) {
            if (mech != nullptr) {
                mech->clear_current_vjp_tape();
            }
        }
    }
}

// The live implementation below is the maintained hand-written VJP lane:
// - loss is injected directly as row-domain local gradients
// - solver uses transpose/VJP orientation
// - H is applied via axial/PAS current VJP
// - readout is grad_rhs
// - carry update is carry += H(g) + local
void Simulate::prepare_vjp_(
    int group_index,
    const std::vector<int>& local_grad_rows,
    int step_count,
    const std::vector<std::string>& current_vjp_mechs) {
    if (!ensure_finitialized_("prepare_vjp")) {
        throw std::runtime_error("prepare_vjp: finitialize() not called");
    }
    if (group_index < 0 || group_index >= static_cast<int>(neuron_group_list.size())) {
        throw std::out_of_range("prepare_vjp: invalid group index");
    }
    auto* group = neuron_group_list[static_cast<std::size_t>(group_index)];
    if (group == nullptr) {
        throw std::runtime_error("prepare_vjp: null group");
    }
    if (current_vjp_mechs.empty()) {
        throw std::runtime_error("prepare_vjp: current_vjp_mechs must not be empty");
    }
    for (std::size_t i = 0; i < local_grad_rows.size(); ++i) {
        const int row = local_grad_rows[i];
        if (row < 0 || row >= group->len) {
            throw std::runtime_error(
                "prepare_vjp: local grad row out of range at index " +
                std::to_string(i));
        }
    }
    if (group->vecdata_a == nullptr ||
        group->vecdata_b == nullptr ||
        group->vecdata_parent_index == nullptr) {
        throw std::runtime_error("prepare_vjp: missing group tree matrix data");
    }
    if (group->vecdata_d == nullptr || group->vecdata_rhs == nullptr) {
        throw std::runtime_error("prepare_vjp: missing group D/RHS work buffers");
    }
    if (group->mech_cap == nullptr) {
        throw std::runtime_error("prepare_vjp: group->mech_cap is null");
    }
    autodiff_current_vjp_mech_whitelist_.clear();
    for (const auto& name : current_vjp_mechs) {
        if (!name.empty()) {
            autodiff_current_vjp_mech_whitelist_.insert(name);
        }
    }
    if (autodiff_current_vjp_mech_whitelist_.empty()) {
        throw std::runtime_error("prepare_vjp: current_vjp_mechs only contained empty names");
    }
    std::vector<Mechanism*> pas_mechs;
    pas_mechs.reserve(group->mech_current_list.size());
    for (auto* mech : group->mech_current_list) {
        if (mech != nullptr && mech->name == "pas") {
            pas_mechs.push_back(mech);
        }
    }

    const double vjp_cj = 1.0 / dt;
    if (mode == CPU) {
        validate_vjp_cpu_solver_topology_(group);
    }
    if (group->vjp_workspace == nullptr) {
        group->vjp_workspace = std::make_unique<NeuronGroupData::VoltageVjpWorkspace>();
    }
    auto& workspace = *group->vjp_workspace;
    workspace.prepared = false;
    workspace.current_vjp_mechs.clear();
    workspace.current_vjp_mechs.reserve(group->mech_current_list.size());
    for (auto* mech : group->mech_current_list) {
        if (mech == nullptr || !mech->supports_current_vjp()) {
            continue;
        }
        if (autodiff_current_vjp_mech_whitelist_.count(mech->name) == 0) {
            continue;
        }
        workspace.current_vjp_mechs.push_back(mech);
    }
    if (workspace.current_vjp_mechs.empty()) {
        throw std::runtime_error("prepare_vjp: no requested current_vjp mechanisms are available");
    }
    workspace.spike_vjp_postsyns.clear();
    workspace.spike_vjp_postsyns.reserve(workspace.current_vjp_mechs.size());
    for (auto* mech : workspace.current_vjp_mechs) {
        auto* postsyn = dynamic_cast<PostSyn_trait*>(mech);
        if (postsyn == nullptr || !postsyn->supports_spike_vjp()) {
            continue;
        }
        workspace.spike_vjp_postsyns.push_back(postsyn);
    }
    const bool dense_shape_changed =
        workspace.mode != group->mode || workspace.len != group->len;
    const int local_loss_input_count = static_cast<int>(local_grad_rows.size());
    workspace.mode = group->mode;
    workspace.len = group->len;
    workspace.ncell = group->ncell;
    workspace.step_count = step_count;
    workspace.dt_ms = dt;
    workspace.local_loss_input_count = local_loss_input_count;
    workspace.spike_vjp_surrogate_ops =
        &neurong::spike_vjp::SpikeVjpSurrogateRegistry::getInstance().require(spike_vjp_surrogate_name_);
    workspace.spike_vjp_surrogate_config = spike_vjp_surrogate_config_;
    workspace.spike_vjp_has_zero_delay_edge = false;
    if (dense_shape_changed || workspace.carry_v.size() != workspace.len) {
        workspace.carry_v = VecData<double>(group->mode, 0.0, workspace.len);
        workspace.d_const = VecData<double>(group->mode, 0.0, workspace.len);
    } else {
        std::fill_n(workspace.carry_v.get_cpu_data(), workspace.len, 0.0);
        if (group->mode == GPU) {
            workspace.carry_v.update_gpu_data_from_cpu();
        }
    }
    const int local_loss_input_capacity = std::max(local_loss_input_count, 1);
    if (dense_shape_changed || workspace.local_loss_input.size() != local_loss_input_capacity) {
        workspace.local_loss_input = VecData<double>(group->mode, 0.0, local_loss_input_capacity);
    }
    if (dense_shape_changed || workspace.local_loss_input_index.size() != local_loss_input_count) {
        workspace.local_loss_input_index = VecData<int>(group->mode, local_grad_rows);
    } else if (local_loss_input_count > 0) {
        std::copy(local_grad_rows.begin(), local_grad_rows.end(), workspace.local_loss_input_index.get_cpu_data());
        if (group->mode == GPU) {
            workspace.local_loss_input_index.update_gpu_data_from_cpu();
        }
    }
    workspace.spike_vjp_pre_count = 0;
    if (!workspace.spike_vjp_postsyns.empty()) {
        if (group->presyn == nullptr) {
            throw std::runtime_error("prepare_vjp: spike VJP requires a PreSyn source table");
        }
        workspace.spike_vjp_pre_count = static_cast<int>(group->presyn->npre);
        if (workspace.spike_vjp_pre_count <= 0) {
            throw std::runtime_error("prepare_vjp: spike VJP requires at least one real PreSyn source");
        }
        for (auto* postsyn : workspace.spike_vjp_postsyns) {
            if (postsyn == nullptr || postsyn->vecdata_delay_steps == nullptr) {
                workspace.spike_vjp_has_zero_delay_edge = true;
                break;
            }
            int* delay_steps = postsyn->vecdata_delay_steps->get_cpu_data();
            const int delay_count = postsyn->vecdata_delay_steps->size();
            for (int i = 0; i < delay_count; ++i) {
                if (delay_steps[i] <= 0) {
                    workspace.spike_vjp_has_zero_delay_edge = true;
                    break;
                }
            }
            if (workspace.spike_vjp_has_zero_delay_edge) {
                break;
            }
        }
        const int spike_v_tape_capacity =
            checked_spike_vjp_dense_capacity(workspace.step_count, workspace.spike_vjp_pre_count);
        const int spike_adj_capacity = spike_v_tape_capacity;
        const bool spike_shape_changed =
            dense_shape_changed ||
            workspace.spike_vjp_pre_v_tape.size() != spike_v_tape_capacity ||
            workspace.spike_vjp_pending_pre_spike_adj.size() != spike_adj_capacity;
        if (spike_shape_changed) {
            workspace.spike_vjp_pre_v_tape = VecData<double>(group->mode, 0.0, spike_v_tape_capacity);
            workspace.spike_vjp_pending_pre_spike_adj =
                VecData<double>(group->mode, 0.0, spike_adj_capacity);
        } else {
            std::fill_n(workspace.spike_vjp_pending_pre_spike_adj.get_cpu_data(), spike_adj_capacity, 0.0);
            if (group->mode == GPU) {
                workspace.spike_vjp_pending_pre_spike_adj.update_gpu_data_from_cpu();
            }
        }
    }
    zero_vjp_adjoint_carries_(group);
    prepare_gap_vjp_routes_(group, workspace);

    if (group->mode == GPU) {
        build_vjp_d_const_gpu(group, workspace, pas_mechs, vjp_cj);
    } else {
        double* zero_v = workspace.carry_v.get_cpu_data();
        double* d_const = workspace.d_const.get_cpu_data();
        double* rhs_scratch = group->vecdata_rhs->get_cpu_data();

        // D_const is rebuilt by accumulation. carry_v has already been reset
        // above for the VJP recurrence and is reused here as the read-only
        // voltage input. RHS is only a required output argument for current_cpu();
        // its value is not consumed while building static D, and run_vjp
        // overwrites RHS before use.
        std::fill_n(d_const, workspace.len, 0.0);
        SimMechCurrentParam d_param{
            .v = zero_v,
            .d = d_const,
            .rhs = rhs_scratch,
            .t = 0.0,
        };
        for (auto* mech : pas_mechs) {
            mech->current_cpu(d_param);
        }
        group->mech_cap->cap_jacob_cpu(vjp_cj, d_const);
        double* vec_a = group->vecdata_a->get_cpu_data();
        double* vec_b = group->vecdata_b->get_cpu_data();
        int* parent_index = group->vecdata_parent_index->get_cpu_data();
        for (int i = workspace.ncell; i < workspace.len; ++i) {
            d_const[static_cast<std::size_t>(i)] -= vec_b[i];
            d_const[static_cast<std::size_t>(parent_index[i])] -= vec_a[i];
        }
    }
    workspace.prepared = true;
}

void Simulate::validate_vjp_cpu_solver_topology_(NeuronGroupData* group) const {
    if (group == nullptr) {
        throw std::runtime_error("run_vjp: null group");
    }
    switch (permute_type) {
    case 0:
        return;
    case 1:
        if (group->vecdata_stride == nullptr ||
            group->vecdata_firstnode == nullptr ||
            group->vecdata_lastnode == nullptr ||
            group->vecdata_cellsize == nullptr) {
            throw std::runtime_error("run_vjp: missing CPU permute_type=1 topology data");
        }
        return;
    case 3:
        if (group->vecdata_max_order_each_thread == nullptr ||
            group->vecdata_min_order_each_thread == nullptr ||
            group->vecdata_firstnode == nullptr ||
            group->vecdata_lastnode == nullptr ||
            group->vecdata_stride == nullptr ||
            group->vecdata_map_t2c == nullptr) {
            throw std::runtime_error("run_vjp: missing CPU permute_type=3 topology data");
        }
        return;
    default:
        throw std::runtime_error("run_vjp: unsupported CPU permute_type " + std::to_string(permute_type));
    }
}

void Simulate::solve_vjp_matrix_cpu_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace) {
    if (group == nullptr) {
        throw std::runtime_error("run_vjp: null group");
    }
    double* vec_a = group->vecdata_a->get_cpu_data();
    double* vec_b = group->vecdata_b->get_cpu_data();
    int* parent_index = group->vecdata_parent_index->get_cpu_data();
    double* d_work = group->vecdata_d->get_cpu_data();
    double* rhs_work = group->vecdata_rhs->get_cpu_data();
    switch (permute_type) {
    case 0:
        solve_serial(
            vec_b,
            vec_a,
            d_work,
            rhs_work,
            parent_index,
            workspace.ncell,
            workspace.len);
        return;
    case 1: {
        int* stride = group->vecdata_stride->get_cpu_data();
        int* firstnode = group->vecdata_firstnode->get_cpu_data();
        int* lastnode = group->vecdata_lastnode->get_cpu_data();
        int* cellsize = group->vecdata_cellsize->get_cpu_data();
        cpu_solve_permute1(
            vec_b,
            vec_a,
            d_work,
            rhs_work,
            parent_index,
            group->nstride,
            stride,
            firstnode,
            lastnode,
            cellsize,
            workspace.ncell,
            workspace.len);
        return;
    }
    case 3: {
        int* stride = group->vecdata_stride->get_cpu_data();
        int* firstnode = group->vecdata_firstnode->get_cpu_data();
        int* lastnode = group->vecdata_lastnode->get_cpu_data();
        int* max_order_each_thread = group->vecdata_max_order_each_thread->get_cpu_data();
        int* min_order_each_thread = group->vecdata_min_order_each_thread->get_cpu_data();
        int* map_t2c = group->vecdata_map_t2c->get_cpu_data();
        cpu_solve_permute3(
            vec_b,
            vec_a,
            d_work,
            rhs_work,
            parent_index,
            max_order_each_thread,
            min_order_each_thread,
            firstnode,
            lastnode,
            stride,
            map_t2c,
            group->norder,
            workspace.ncell,
            group->threads_num);
        return;
    }
    default:
        throw std::runtime_error("run_vjp: unsupported CPU permute_type " + std::to_string(permute_type));
    }
}

double* Simulate::resolve_vjp_adjoint_ptr_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const VarDescriptor& descriptor,
    Mode ptr_mode) {
    if (group == nullptr) {
        return nullptr;
    }
    const bool is_global = descriptor.mech.empty() || descriptor.mech == "global";
    if (is_global) {
        if (descriptor.var != "v") {
            printf("resolve_vjp_adjoint_ptr: only global.v has a VJP adjoint, got global.%s\n",
                   descriptor.var.c_str());
            return nullptr;
        }
        int row = descriptor.node_or_mech_idx;
        if (row < 0 || row >= group->len) {
            printf("resolve_vjp_adjoint_ptr: global.v row out of range: %d len=%d\n", row, group->len);
            return nullptr;
        }
        if (group->permute != nullptr) {
            row = group->permute[row];
        }
        if (row < 0 || row >= group->len) {
            printf("resolve_vjp_adjoint_ptr: permuted global.v row out of range: %d len=%d\n", row, group->len);
            return nullptr;
        }
        return ((ptr_mode == GPU)
            ? workspace.carry_v.get_gpu_data()
            : workspace.carry_v.get_cpu_data()) + row;
    }

    auto& mech_factory = MechanismFactory::getInstance();
    auto* var_map = mech_factory.getVarMap(descriptor.mech);
    if (var_map == nullptr) {
        printf("resolve_vjp_adjoint_ptr: mech %s not found\n", descriptor.mech.c_str());
        return nullptr;
    }
    double* ptr = var_map->getVjpAdjointPtr(descriptor, ptr_mode);
    if (ptr == nullptr) {
        printf("resolve_vjp_adjoint_ptr: adjoint not found for %s.%s[%d][%d]\n",
               descriptor.mech.c_str(),
               descriptor.var.c_str(),
               descriptor.node_or_mech_idx,
               descriptor.array_index);
    }
    return ptr;
}

void Simulate::prepare_gap_vjp_routes_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace) {
    if (group == nullptr) {
        workspace.gap_vjp_src_adj = VecData<double*>(workspace.mode);
        workspace.gap_vjp_dst_adj = VecData<double*>(workspace.mode);
        workspace.gap_vjp_trans_count = 0;
        return;
    }
    workspace.gap_vjp_src_adj = VecData<double*>(group->mode);
    workspace.gap_vjp_dst_adj = VecData<double*>(group->mode);
    workspace.gap_vjp_trans_count = 0;

    if (!group->have_gap) {
        return;
    }
    const auto& gap_info = group->cpu_gap_trans_info;
    const int ntrans = gap_info.ntrans();
    if (ntrans == 0) {
        return;
    }
    if (!gap_info.has_descriptors()) {
        throw std::runtime_error(
            "prepare_vjp: gap VJP requires descriptor-backed gap routes; "
            "legacy imported gap routes do not expose descriptors yet");
    }

    workspace.gap_vjp_src_adj.reserve(ntrans);
    workspace.gap_vjp_dst_adj.reserve(ntrans);
    std::vector<double*> src_adj_cpu_list;
    std::vector<double*> dst_adj_cpu_list;
    src_adj_cpu_list.reserve(static_cast<std::size_t>(ntrans));
    dst_adj_cpu_list.reserve(static_cast<std::size_t>(ntrans));
    for (int i = 0; i < ntrans; ++i) {
        const auto& src_desc = gap_info.src_desc[static_cast<std::size_t>(i)];
        const auto& dst_desc = gap_info.dst_desc[static_cast<std::size_t>(i)];
        double* src_adj_cpu = resolve_vjp_adjoint_ptr_(group, workspace, src_desc, CPU);
        double* dst_adj_cpu = resolve_vjp_adjoint_ptr_(group, workspace, dst_desc, CPU);
        if (src_adj_cpu == nullptr || dst_adj_cpu == nullptr) {
            throw std::runtime_error(
                "prepare_vjp: failed to resolve CPU gap VJP adjoint pointer for route " +
                std::to_string(i));
        }

        double* src_adj = src_adj_cpu;
        double* dst_adj = dst_adj_cpu;
        if (group->mode == GPU) {
            src_adj = resolve_vjp_adjoint_ptr_(group, workspace, src_desc, GPU);
            dst_adj = resolve_vjp_adjoint_ptr_(group, workspace, dst_desc, GPU);
            if (src_adj == nullptr || dst_adj == nullptr) {
                throw std::runtime_error(
                    "prepare_vjp: failed to resolve GPU gap VJP adjoint pointer for route " +
                    std::to_string(i));
            }
        }
        src_adj_cpu_list.push_back(src_adj_cpu);
        dst_adj_cpu_list.push_back(dst_adj_cpu);
        workspace.gap_vjp_src_adj.push_back(src_adj);
        workspace.gap_vjp_dst_adj.push_back(dst_adj);
    }
    std::unordered_map<double*, int> dst_owner;
    dst_owner.reserve(static_cast<std::size_t>(ntrans));
    for (int i = 0; i < ntrans; ++i) {
        double* dst = dst_adj_cpu_list[static_cast<std::size_t>(i)];
        if (!dst_owner.emplace(dst, i).second) {
            throw std::runtime_error(
                "prepare_vjp: gap VJP does not support multiple copy routes targeting "
                "the same adjoint destination; use a distinct target variable per edge");
        }
    }
    for (int i = 0; i < ntrans; ++i) {
        double* src = src_adj_cpu_list[static_cast<std::size_t>(i)];
        auto dst_it = dst_owner.find(src);
        if (dst_it != dst_owner.end() && dst_it->second != i) {
            throw std::runtime_error(
                "prepare_vjp: gap VJP currently supports independent copy edges only; "
                "a gap target variable must not also be used as another gap source");
        }
    }
    workspace.gap_vjp_trans_count = ntrans;
    if (group->mode == GPU) {
        workspace.gap_vjp_src_adj.update_gpu_data_from_cpu();
        workspace.gap_vjp_dst_adj.update_gpu_data_from_cpu();
    }
}

void Simulate::zero_vjp_adjoint_carries_(NeuronGroupData* group) {
    if (group == nullptr) {
        return;
    }
    for (auto* mech : group->mechanism_list) {
        if (mech != nullptr && mech->has_vjp_adjoint_vars()) {
            mech->zeroVjpAdjoints();
        }
    }
}

void Simulate::gap_vjp_transfer_cpu_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace) {
    (void)group;
    const int ntrans = workspace.gap_vjp_trans_count;
    if (ntrans <= 0) {
        return;
    }
    double** src_adj = workspace.gap_vjp_src_adj.get_cpu_data();
    double** dst_adj = workspace.gap_vjp_dst_adj.get_cpu_data();
    for (int i = ntrans - 1; i >= 0; --i) {
        if (src_adj[i] == nullptr || dst_adj[i] == nullptr || src_adj[i] == dst_adj[i]) {
            continue;
        }
        *src_adj[i] += *dst_adj[i];
        *dst_adj[i] = 0.0;
    }
}

void Simulate::record_spike_vjp_presyn_tape_cpu_(NeuronGroupData* group) {
    if (group == nullptr || group->vjp_workspace == nullptr || group->presyn == nullptr) {
        return;
    }
    if (!should_stage_vjp_tape_(group)) {
        return;
    }
    auto& workspace = *group->vjp_workspace;
    if (!workspace.prepared || workspace.spike_vjp_postsyns.empty()) {
        return;
    }
    const int step = autodiff_vjp_tape_recorded_steps_;
    const int npre = workspace.spike_vjp_pre_count;
    if (step < 0 || step >= workspace.step_count || npre <= 0) {
        return;
    }
    double* pre_v_tape = workspace.spike_vjp_pre_v_tape.get_cpu_data();
    const std::size_t base = static_cast<std::size_t>(step) * static_cast<std::size_t>(npre);

    int* pre_node_indices = group->presyn->vecdata_pre_node_indices->get_cpu_data();
    double* vec_v = group->vecdata_v->get_cpu_data();
    for (int ipre = 0; ipre < npre; ++ipre) {
        const int node_index = pre_node_indices[ipre];
        pre_v_tape[base + static_cast<std::size_t>(ipre)] = vec_v[node_index];
    }
}

void Simulate::spike_vjp_deposit_cpu_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    int tape_step) {
    if (group == nullptr || group->presyn == nullptr || workspace.spike_vjp_postsyns.empty()) {
        return;
    }
    const int npre = workspace.spike_vjp_pre_count;
    if (tape_step < 0 || tape_step >= workspace.step_count || npre <= 0) {
        return;
    }

    double* pending_pre_spike_adj = workspace.spike_vjp_pending_pre_spike_adj.get_cpu_data();
    SimPostSynSpikeVJPParam param{
        .pending_pre_spike_adj = pending_pre_spike_adj,
        .pre_spike_adj_count = npre,
        .step_count = workspace.step_count,
        .t = workspace.dt_ms * static_cast<double>(tape_step),
        .dt = workspace.dt_ms,
        .step_index = tape_step,
    };
    for (auto* postsyn : workspace.spike_vjp_postsyns) {
        if (postsyn != nullptr) {
            postsyn->spike_vjp_cpu(param);
        }
    }
}

void Simulate::spike_vjp_consume_cpu_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    int tape_step) {
    if (group == nullptr || group->presyn == nullptr || workspace.spike_vjp_postsyns.empty()) {
        return;
    }
    const int npre = workspace.spike_vjp_pre_count;
    if (tape_step < 0 || tape_step >= workspace.step_count || npre <= 0) {
        return;
    }

    const std::size_t base = static_cast<std::size_t>(tape_step) * static_cast<std::size_t>(npre);
    double* pending_pre_spike_adj = workspace.spike_vjp_pending_pre_spike_adj.get_cpu_data();
    double* pre_v_tape = workspace.spike_vjp_pre_v_tape.get_cpu_data() + base;
    double* pending_step = pending_pre_spike_adj + base;
    double* carry_v = workspace.carry_v.get_cpu_data();
    int* pre_node_indices = group->presyn->vecdata_pre_node_indices->get_cpu_data();
    double* threshold = group->presyn->vecdata_threshold->get_cpu_data();
    const auto* ops = workspace.spike_vjp_surrogate_ops;
    if (ops == nullptr || ops->run_cpu == nullptr) {
        std::fill_n(pending_step, npre, 0.0);
        return;
    }
    const neurong::spike_vjp::SpikeVjpSurrogateRunParam param{
        .npre = npre,
        .pre_node_indices = pre_node_indices,
        .threshold = threshold,
        .pre_v_tape = pre_v_tape,
        .pending_pre_spike_adj = pending_step,
        .carry_v = carry_v,
    };
    ops->run_cpu(param, workspace.spike_vjp_surrogate_config);
}

void Simulate::run_vjp_step_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const double* local,
    int local_count,
    int step) {
    if (local_count != workspace.local_loss_input_count) {
        throw std::runtime_error("run_vjp: local grad width mismatch at step " + std::to_string(step));
    }
    int* local_loss_input_index = workspace.local_loss_input_index.get_cpu_data();
    double* carry_v = workspace.carry_v.get_cpu_data();

    for (int i = 0; i < workspace.local_loss_input_count; ++i) {
        const std::size_t row = static_cast<std::size_t>(local_loss_input_index[i]);
        carry_v[row] += local[static_cast<std::size_t>(i)];
    }
    const int tape_step = workspace.step_count - 1 - step;
    // Consume delayed spike adjoints before the solver so dL/dspike[t] enters
    // the same transpose step as ordinary dL/dv[t]. Doing this post-solve
    // shifts the surrogate path one step earlier.
    spike_vjp_consume_cpu_(group, workspace, tape_step);
    run_vjp_step_after_local_accumulation_(group, workspace, step);
}

void Simulate::run_vjp_step_forward_contiguous_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const double* local_grad_forward,
    int local_count,
    int step_count,
    int step) {
    if (local_count != workspace.local_loss_input_count) {
        throw std::runtime_error("run_vjp: local grad width mismatch at step " + std::to_string(step));
    }
    if (step_count != workspace.step_count) {
        throw std::runtime_error("run_vjp: local grad history size mismatch");
    }
    int* local_loss_input_index = workspace.local_loss_input_index.get_cpu_data();
    double* carry_v = workspace.carry_v.get_cpu_data();
    const int forward_step = step_count - 1 - step;

    for (int i = 0; i < workspace.local_loss_input_count; ++i) {
        const std::size_t row = static_cast<std::size_t>(local_loss_input_index[i]);
        carry_v[row] += local_grad_forward[
            static_cast<std::size_t>(i) * static_cast<std::size_t>(step_count) +
            static_cast<std::size_t>(forward_step)];
    }
    spike_vjp_consume_cpu_(group, workspace, forward_step);
    run_vjp_step_after_local_accumulation_(group, workspace, step);
}

void Simulate::run_vjp_step_after_local_accumulation_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    int step) {
    double* vec_a = group->vecdata_a->get_cpu_data();
    double* vec_b = group->vecdata_b->get_cpu_data();
    int* parent_index = group->vecdata_parent_index->get_cpu_data();
    double* carry_v = workspace.carry_v.get_cpu_data();
    double* rhs = group->vecdata_rhs->get_cpu_data();
    double* d_work = group->vecdata_d->get_cpu_data();
    double* d_const = workspace.d_const.get_cpu_data();

    std::copy(carry_v, carry_v + workspace.len, rhs);
    std::copy(d_const, d_const + workspace.len, d_work);
    solve_vjp_matrix_cpu_(group, workspace);

    for (int i = workspace.ncell; i < workspace.len; ++i) {
        const int parent = parent_index[i];
        const double grad_rhs_parent = rhs[static_cast<std::size_t>(parent)];
        const double grad_rhs_child = rhs[static_cast<std::size_t>(i)];
        carry_v[static_cast<std::size_t>(parent)] +=
            vec_a[i] * grad_rhs_parent - vec_b[i] * grad_rhs_child;
        carry_v[static_cast<std::size_t>(i)] +=
            vec_b[i] * grad_rhs_child - vec_a[i] * grad_rhs_parent;
    }

    const int tape_step = workspace.step_count - 1 - step;
    const double t_step = workspace.dt_ms * static_cast<double>(tape_step);
    for (auto* mech : workspace.current_vjp_mechs) {
        if (mech == nullptr) {
            continue;
        }
        SimMechCurrentVJPParam param{
            .v = rhs,
            .grad_v = carry_v,
            .grad_rhs = rhs,
            .t = t_step,
            .dt = workspace.dt_ms,
            .step_index = tape_step,
        };
        mech->current_vjp_cpu(param);
    }
    spike_vjp_deposit_cpu_(group, workspace, tape_step);
    if (workspace.spike_vjp_has_zero_delay_edge) {
        spike_vjp_consume_cpu_(group, workspace, tape_step);
    }
    gap_vjp_transfer_cpu_(group, workspace);
}

void Simulate::finalize_vjp_tape_for_backward_(
    NeuronGroupData::VoltageVjpWorkspace& workspace) {
    const bool has_current_vjp_tape_mech = std::any_of(
        workspace.current_vjp_mechs.begin(),
        workspace.current_vjp_mechs.end(),
        [](const Mechanism* mech) {
            return mech != nullptr && mech->supports_current_vjp_tape();
        });
    const bool needs_forward_tape = has_current_vjp_tape_mech || !workspace.spike_vjp_postsyns.empty();
    if (needs_forward_tape && autodiff_vjp_tape_recorded_steps_ != workspace.step_count) {
        throw std::runtime_error(
            "run_vjp: VJP tape length mismatch, recorded=" +
            std::to_string(autodiff_vjp_tape_recorded_steps_) +
            " expected=" + std::to_string(workspace.step_count));
    }
    if (has_current_vjp_tape_mech) {
        for (auto* mech : workspace.current_vjp_mechs) {
            if (mech != nullptr && mech->supports_current_vjp_tape()) {
                mech->finalize_current_vjp_tape_for_backward();
            }
        }
    }
}

void Simulate::run_prepared_vjp_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const std::vector<std::vector<double>>& local_grad_row_history) {
    finalize_vjp_tape_for_backward_(workspace);
    zero_vjp_adjoint_carries_(group);
    if (mode == GPU) {
        run_vjp_gpu(
            group,
            local_grad_row_history,
            permute_type);
        return;
    }

    for (int step = 0; step < workspace.step_count; ++step) {
        const auto& local = local_grad_row_history[static_cast<std::size_t>(step)];
        run_vjp_step_(
            group,
            workspace,
            local.data(),
            static_cast<int>(local.size()),
            step);
    }
}

void Simulate::run_prepared_vjp_forward_contiguous_(
    NeuronGroupData* group,
    NeuronGroupData::VoltageVjpWorkspace& workspace,
    const double* local_grad_forward,
    int local_count,
    int step_count) {
    finalize_vjp_tape_for_backward_(workspace);
    zero_vjp_adjoint_carries_(group);
    if (mode == GPU) {
        run_vjp_gpu_forward_contiguous(
            group,
            local_grad_forward,
            local_count,
            step_count,
            permute_type);
        return;
    }

    for (int step = 0; step < workspace.step_count; ++step) {
        run_vjp_step_forward_contiguous_(
            group,
            workspace,
            local_grad_forward,
            local_count,
            step_count,
            step);
    }
}

void Simulate::prepare_vjp(
    int group_index,
    const std::vector<int>& local_grad_rows,
    int step_count,
    const std::vector<std::string>& current_vjp_mechs) {
    prepared_vjp_valid_ = false;
    prepared_vjp_group_index_ = -1;
    autodiff_record_vjp_tape_ = false;
    autodiff_vjp_tape_recorded_steps_ = 0;
    autodiff_vjp_tape_step_count_ = 0;
    if (step_count < 0) {
        throw std::runtime_error("prepare_vjp: step_count must be non-negative");
    }
    autodiff_vjp_tape_step_count_ = step_count;
    clear_current_vjp_tapes_();
    prepare_vjp_(
        group_index,
        local_grad_rows,
        step_count,
        current_vjp_mechs);
    prepared_vjp_group_index_ = group_index;
    prepared_vjp_valid_ = true;
    arm_vjp_tape_recording_(group_index);
}

void Simulate::run_vjp(
    const std::vector<std::vector<double>>& local_grad_row_history) {
    if (!prepared_vjp_valid_) {
        throw std::runtime_error("run_vjp: prepare_vjp must be called first");
    }
    if (prepared_vjp_group_index_ < 0 ||
        prepared_vjp_group_index_ >= static_cast<int>(neuron_group_list.size())) {
        throw std::runtime_error("run_vjp: prepared group is invalid");
    }
    auto* group = neuron_group_list[static_cast<std::size_t>(prepared_vjp_group_index_)];
    if (group == nullptr || group->vjp_workspace == nullptr || !group->vjp_workspace->prepared) {
        throw std::runtime_error("run_vjp: VJP workspace is not prepared");
    }
    auto& workspace = *group->vjp_workspace;
    if (static_cast<int>(local_grad_row_history.size()) != workspace.step_count) {
        throw std::runtime_error("run_vjp: local grad history size mismatch");
    }
    for (int step = 0; step < workspace.step_count; ++step) {
        if (local_grad_row_history[static_cast<std::size_t>(step)].size() !=
            static_cast<std::size_t>(workspace.local_loss_input_count)) {
            throw std::runtime_error(
                "run_vjp: local grad width mismatch at step " +
                std::to_string(step));
        }
    }
    run_prepared_vjp_(
        group,
        workspace,
        local_grad_row_history);
    autodiff_record_vjp_tape_ = false;
}

void Simulate::run_vjp_forward_grads_contiguous(
    const double* local_grad_forward,
    int local_count,
    int step_count) {
    if (!prepared_vjp_valid_) {
        throw std::runtime_error("run_vjp: prepare_vjp must be called first");
    }
    if (prepared_vjp_group_index_ < 0 ||
        prepared_vjp_group_index_ >= static_cast<int>(neuron_group_list.size())) {
        throw std::runtime_error("run_vjp: prepared group is invalid");
    }
    auto* group = neuron_group_list[static_cast<std::size_t>(prepared_vjp_group_index_)];
    if (group == nullptr || group->vjp_workspace == nullptr || !group->vjp_workspace->prepared) {
        throw std::runtime_error("run_vjp: VJP workspace is not prepared");
    }
    auto& workspace = *group->vjp_workspace;
    if (step_count != workspace.step_count) {
        throw std::runtime_error("run_vjp: local grad history size mismatch");
    }
    if (local_count != workspace.local_loss_input_count) {
        throw std::runtime_error("run_vjp: local grad width mismatch");
    }
    if (local_grad_forward == nullptr && local_count > 0 && step_count > 0) {
        throw std::runtime_error("run_vjp: local grad buffer is null");
    }
    run_prepared_vjp_forward_contiguous_(
        group,
        workspace,
        local_grad_forward,
        local_count,
        step_count);
    autodiff_record_vjp_tape_ = false;
}

int Simulate::set_recorder_sampling(double start_time_ms, int stride_steps) {
    if (!std::isfinite(start_time_ms) || start_time_ms < 0.0) {
        printf("set_recorder_sampling: invalid start_time_ms=%f\n", start_time_ms);
        return -1;
    }
    if (stride_steps <= 0) {
        printf("set_recorder_sampling: stride_steps must be positive, got %d\n", stride_steps);
        return -1;
    }
    recorder_start_time_ms_ = start_time_ms;
    recorder_stride_steps_ = stride_steps;
    recorder_eligible_step_count_ = 0;
    return 0;
}

bool Simulate::should_record_now_() {
    constexpr double eps = 1e-12;
    if (t + eps < recorder_start_time_ms_) {
        return false;
    }
    const bool take_now = (recorder_eligible_step_count_ % static_cast<int64_t>(recorder_stride_steps_)) == 0;
    ++recorder_eligible_step_count_;
    return take_now;
}

void Simulate::record_if_needed_cpu_() {
    if (should_record_now_()) {
        hdf5_manager.log_data_cpu();
    }
    if (window_observer.active) {
        window_observer.log_data_cpu();
    }
}

void Simulate::record_if_needed_gpu_() {
    if (should_record_now_()) {
        hdf5_manager.log_data_gpu();
    }
    if (window_observer.active) {
        window_observer.log_data_gpu();
    }
}

int Simulate::begin_window_observer_recording(const std::vector<int>& handles, int step_count) {
    int rc = window_observer.begin(handles, step_count);
    if (rc != 0) return rc;
    if (mode == CPU) {
        window_observer.log_data_cpu();
    } else {
        window_observer.log_data_gpu();
    }
    return 0;
}

int Simulate::end_window_observer_recording() {
    window_observer.reset();
    return 0;
}

Simulate::~Simulate()
{
    int n = neuron_group_list.size();
    for (int i = 0; i < n; i++) {
        delete neuron_group_list[i];
        neuron_group_list[i] = nullptr;
    }
    neuron_group_list.clear();
    if (mode == GPU) {
        destroy_cuda_streams();
    }
}

void Simulate::finitialize(double v_init)
{
    finitialize_done_ = false;
    t = 0;
    recorder_eligible_step_count_ = 0;
    clear_current_vjp_tapes_();
    CUDA_CHECK_ERR();

    // 同步所有pending的gap transfers（GPU模式）
    // 必须在finitialize_gpu之前同步，因为finitialize_gpu会调用gap_transfer_gpu
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }
    
    if (this->mode == CPU)
    {
        finitialize_cpu(v_init);
        CUDA_CHECK_ERR();
    }
    else if (this->mode == GPU)
    {
        finitialize_gpu(v_init);
        CUDA_CHECK_ERR();
    }
    finitialize_done_ = true;
}

pair<double*, double*> Simulate::getVarPtr(VarDescriptor &descriptor, bool will_panic){
    #define PANIC_OR_RETURN_NULL if(will_panic) { \
                                        assert(false); \
                                    } else { \
                                        return {nullptr, nullptr}; \
                                    }

    double *var_ptr_cpu = nullptr;
    double *var_ptr_gpu = nullptr;

    auto &mechFactory = MechanismFactory::getInstance();
    auto varMapPtr = mechFactory.getVarMap(descriptor.mech);
    if (varMapPtr == nullptr) {
        printf("mech:%s not found\n", descriptor.mech.c_str());
        PANIC_OR_RETURN_NULL;
    }

    // 简化：直接传递VarDescriptor
    var_ptr_cpu = varMapPtr->getVarPtr(descriptor, Mode::CPU);
    if (var_ptr_cpu == nullptr) {
        printf("mech:%s var:%s node_or_mech_idx:%d array_index:%d not found\n",
                    descriptor.mech.c_str(),
                    descriptor.var.c_str(),
                    descriptor.node_or_mech_idx,
                    descriptor.array_index);
        PANIC_OR_RETURN_NULL;
    }

    if(this->mode == GPU){
        var_ptr_gpu = varMapPtr->getVarPtr(descriptor, Mode::GPU);
        if (var_ptr_gpu == nullptr) {
            printf("mech:%s var:%s node_or_mech_idx:%d array_index:%d not found (GPU)\n",
                        descriptor.mech.c_str(),
                        descriptor.var.c_str(),
                        descriptor.node_or_mech_idx,
                        descriptor.array_index);
            PANIC_OR_RETURN_NULL;
        }
    }
    #undef PANIC_OR_RETURN_NULL
    return {var_ptr_cpu, var_ptr_gpu};
}
std::map<VarDescriptor, int> Simulate::init_monitor_data_sets(std::vector<VarDescriptor> &monitors) {
    std::map<VarDescriptor, int> monitor_to_handle;

    // 只在启用HDF5时创建HDF5文件
    if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
        if(hdf5_file == nullptr) {
            if(!fs::exists(output_folder)) {
                fs::create_directories(output_folder);
            }
            // 创建 HDF5 文件
            hdf5_file = std::make_unique<HighFive::File>(output_folder + "/sim_out.h5",HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        }
    }
    
    int group_size = neuron_group_list.size();

    assert(group_size == 1); //目前只支持一个group

    // 只在启用HDF5时设置数据集属性
    HighFive::DataSetCreateProps prop;
    HighFive::Group hdf5_group;
    if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
        prop.add(HighFive::Chunking({100})); 
        
        for (int i = 0; i < group_size; i++) {
            // 构造组名称，例如 "/group_0"
            std::string group_name = "/group_" + std::to_string(i);
            // 如果组不存在，则先创建组
            if (!hdf5_file->exist(group_name)) {
                hdf5_file->createGroup(group_name);
            }
            // 现在获取组不会出错
            hdf5_group = hdf5_file->getGroup(group_name);
            break; // 目前只支持一个group，所以直接break
        }
    }

        for (auto &monitorPoint : monitors) {
            // printf("adding monitor:%s %s %d\n", monitorPoint.mech.c_str(), monitorPoint.var.c_str(), monitorPoint.node_or_mech_idx);
            // 找到被监控的变量指针
            
            // 创建新的监控点
            RecordPoint watchPoint;
            auto [var_ptr_cpu, var_ptr_gpu] = getVarPtr(monitorPoint, true);
            watchPoint.var_ptr_cpu = var_ptr_cpu;
            watchPoint.var_ptr_gpu = var_ptr_gpu;

        // 只有在启用HDF5时才创建数据集
        if(hdf5_manager.isEnable(BufferEnable::HDF5)) {
            // 创建新的数据集路径：[group_i][mech_name][var_name][idx]
            std::string group_name = "/group_0"; // 目前只支持一个group
            std::string dataset_path = group_name + "/" + monitorPoint.mech + "/" + 
                                       monitorPoint.var + "/" + 
                                       std::to_string(monitorPoint.node_or_mech_idx);

            // 检查路径中的组是否存在，如果不存在则创建
            std::string mech_group_path = group_name + "/" + monitorPoint.mech;
            if (!hdf5_file->exist(mech_group_path)) {
                hdf5_group.createGroup(monitorPoint.mech);
            }

            std::string var_group_path = mech_group_path + "/" + monitorPoint.var;
            if (!hdf5_file->exist(var_group_path)) {
                hdf5_file->getGroup(mech_group_path).createGroup(monitorPoint.var);
            }

            // 创建数据集
            watchPoint.dataset = hdf5_file->createDataSet(
                dataset_path,
                HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
                HighFive::AtomicType<double>(),
                prop
            );
        }

        // 添加到监控点列表，返回的handle就是新的monitorId
        int handle = hdf5_manager.push_back(monitorPoint, watchPoint);
        monitor_to_handle[monitorPoint] = handle;
        // printf("Monitor added: %s %s %d -> handle %d\n", 
        //        monitorPoint.mech.c_str(), monitorPoint.var.c_str(), 
        //        monitorPoint.node_or_mech_idx, handle);
    }
    hdf5_manager.initialize(mode);
    return monitor_to_handle;
}

std::map<VarDescriptor, int> Simulate::init_window_observers(std::vector<VarDescriptor>& observers) {
    std::map<VarDescriptor, int> observer_to_handle;
    for (auto& observer : observers) {
        auto [var_ptr_cpu, var_ptr_gpu] = getVarPtr(observer, true);
        observer_to_handle[observer] = window_observer.push_back(observer, var_ptr_cpu, var_ptr_gpu);
    }
    window_observer.initialize();
    return observer_to_handle;
}

void Simulate::finitialize_cpu(double v_init)
{
    int n_group = neuron_group_list.size();

    // 重置所有VecPlay状态
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        p_neuron->vec_play_continuous.reset_all_cpu();
        p_neuron->vec_play_continuous.play_cpu(t);          // TODO: 目前不确定是不是放在这里
        p_neuron->vec_play_continuous.continuous_cpu(t);
    }
    // 清空spike_buffer
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        vector<PostSyn_trait*> postsyns = p_neuron->vec_postsyn;
        for (PostSyn_trait* postsyn: postsyns) {
            while (!postsyn->spike_buffer.empty()){
                postsyn->spike_buffer.pop();
            }
        }
    }
    for (int i = 0; i < n_group; i++)
    {
        auto p_group = neuron_group_list[i];
        p_group->cj = 1.0 / dt;
        double* vec_v = p_group->vecdata_v->get_cpu_data();
        if (p_group->vecdata_v_init_override != nullptr) {
            const double* vec_v_override = p_group->vecdata_v_init_override->get_cpu_data();
            for (int j = 0; j < p_group->len; j++) {
                const double ov = vec_v_override[j];
                vec_v[j] = std::isnan(ov) ? v_init : ov;
            }
        } else {
            for (int j = 0; j < p_group->len; j++) {
                vec_v[j] = v_init;
            }
        }
        if (p_group->have_gap){
            gap_transfer_cpu(p_group);
        }
        SimMechInitialParam params;
        params.v = vec_v;
        params.dt = dt;
        for (auto p_mech : p_group->mech_current_list) {
            p_mech->initialize_cpu(params);
        }
    }
    window_observer.reset();
    hdf5_manager.finitialize();
    record_if_needed_cpu_();

    clearAllSpikes_cpu();

    spike_deliver_cpu();
    
    
    for (int i = 0; i < n_group; i++)
    {
        auto p_neuron = neuron_group_list[i];
        setup_tree_matrix_cpu(p_neuron);
        if (p_neuron->need_fast_imem) {
            // Fixed-step init semantics (CoreNEURON nrn_calc_fast_imem_init):
            // i_membrane_ = (rhs + sav_rhs) * area * 0.01
            const int len = p_neuron->len;
            double* vec_area = p_neuron->vecdata_area->get_cpu_data();
            double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();       // rhs before solve
            double* vec_sav_rhs = p_neuron->vecdata_sav_rhs->get_cpu_data();
            double* vec_imem = p_neuron->vecdata_i_membrane_->get_cpu_data();
            for (int j = 0; j < len; ++j) {
                vec_imem[j] = (vec_rhs[j] + vec_sav_rhs[j]) * vec_area[j] * 0.01;
            }
        }
    }

}


/*run simulate*/
void Simulate::run()
{
    if (!ensure_finitialized_("run")) {
        return;
    }
    int nstep = tstop / dt;
	t = 0.0;
    
    // 同步所有pending的gap transfers（GPU模式）
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }
    
    if (this->mode == CPU){
        for (int iter = 0; iter < nstep; iter++)
        {
            fadvance_cpu();
        }
        hdf5_manager.flush_cpu();
    }else{
        for (int iter = 0; iter < nstep; iter++)
        {
            fadvance_gpu();
        }
        cuda_sync_all();
        hdf5_manager.flush_gpu();
    }
}

void Simulate::continue_run(double runtime)
{
    if (!ensure_finitialized_("continue_run")) {
        return;
    }
    int nstep = runtime / dt;
    if (nstep <= 0) {
        return;
    }

    // 同步所有pending的gap transfers（GPU模式）
    if (this->mode == GPU) {
        for (auto p_group : neuron_group_list) {
            p_group->gpu_gap_trans_info.sync_to_gpu();
        }
    }

    if (this->mode == CPU) {
        for (int iter = 0; iter < nstep; iter++) {
            fadvance_cpu();
        }
        hdf5_manager.flush_cpu();
    } else {
        for (int iter = 0; iter < nstep; iter++) {
            fadvance_gpu();
        }
        cuda_sync_all();
        hdf5_manager.flush_gpu();
    }
}

void Simulate::fadvance()
{
    if (!ensure_finitialized_("fadvance")) {
        return;
    }
    if (this->mode == CPU)
        fadvance_cpu();
    else
        fadvance_gpu();
}

/*one step in simulation*/
void Simulate::fadvance_cpu()
{

    int n_neuron = neuron_group_list.size();
	spike_deliver_cpu();

    t += 0.5 * dt;
    for (std::size_t group_idx = 0; group_idx < neuron_group_list.size(); ++group_idx)
    {
        auto p_neuron = neuron_group_list[group_idx];
        p_neuron->vec_play_continuous.play_cpu(t);
        p_neuron->vec_play_continuous.continuous_cpu(t);
        setup_tree_matrix_cpu(p_neuron);
        solve_matrix_cpu(p_neuron);
        if (p_neuron->need_fast_imem) {
            const int len = p_neuron->len;
            double* vec_area = p_neuron->vecdata_area->get_cpu_data();
            double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();       // now holds delta_v
            double* vec_sav_rhs = p_neuron->vecdata_sav_rhs->get_cpu_data();
            double* vec_sav_d = p_neuron->vecdata_sav_d->get_cpu_data();
            double* vec_imem = p_neuron->vecdata_i_membrane_->get_cpu_data();
            for (int i = 0; i < len; ++i) {
                vec_imem[i] = (vec_sav_d[i] * vec_rhs[i] + vec_sav_rhs[i]) * vec_area[i] * 0.01;
            }
        }
        update_cpu(p_neuron);
        if(p_neuron->have_gap){
            gap_transfer_cpu(p_neuron);
        }
    }
    last_part_cpu();

}


void Simulate::spike_deliver_cpu()
{
    network_spike_send_cpu();
    network_spike_receive_cpu();
}

/*
 * setup the matrix of calble equation
 * compute mechanism current
 */
void Simulate::setup_tree_matrix_cpu(NeuronGroupData* p_neuron)
{
    int len = p_neuron->len;
    int ncell = p_neuron->ncell;
    double* vec_v = p_neuron->vecdata_v->get_cpu_data();
    double* vec_a = p_neuron->vecdata_a->get_cpu_data();
    double* vec_b = p_neuron->vecdata_b->get_cpu_data();
    double* vec_d = p_neuron->vecdata_d->get_cpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();
    double* vec_sav_rhs = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_rhs->get_cpu_data() : nullptr;
    double* vec_sav_d = p_neuron->need_fast_imem ? p_neuron->vecdata_sav_d->get_cpu_data() : nullptr;
    int* parent_index = p_neuron->vecdata_parent_index->get_cpu_data();

    /*rhs part*/
    std::fill(vec_rhs, vec_rhs + len, 0.0);
    std::fill(vec_d, vec_d + len, 0.0);

    SimMechCurrentParam params = {
        .v = vec_v,
        .d = vec_d,
        .rhs = vec_rhs,
        .t = t,
    };

    for (auto p_mech : p_neuron->mech_current_list) {
        p_mech->current_cpu(params);
    }

    // fast_imem: save membrane-only RHS contribution (before axial terms).
    if (p_neuron->need_fast_imem) {
        for (int i = 0; i < len; ++i) {
            vec_sav_rhs[i] = -vec_rhs[i];
        }
    }

    // LHS (membrane-only): capacitance contribution (must happen after any possible cm changes by mechanisms).
    // Note: at this point vec_d contains only the membrane Jacobian from mechanisms; cap_jacob adds the capacitive part.
    p_neuron->mech_cap->cap_jacob_cpu(p_neuron->cj, vec_d);

    // fast_imem: save membrane-only diagonal contribution (after capacitance, before axial terms).
    if (p_neuron->need_fast_imem) {
        std::copy(vec_d, vec_d + len, vec_sav_d);
    }

    for (int i = ncell; i < len; i++)                    //前ncell个是根节点，没有父节点
    {
        double dv = vec_v[parent_index[i]] - vec_v[i];
        vec_rhs[i] -= vec_b[i] * dv;
        vec_rhs[parent_index[i]] += vec_a[i] * dv;
    }

    
    /*lhs part*/
    for (int i = ncell; i < len; i++)
    {
        vec_d[i] -= vec_b[i];
        vec_d[parent_index[i]] -= vec_a[i];
    }
}

void solve_serial(double* vec_a, double* vec_b, double* vec_d,
                  double* vec_rhs, int* parent_index, int ncell, 
                  int len)
{
    double p;
    //triang
    for (int i = len - 1; i >= ncell; i--)
    {
        p = vec_a[i] / vec_d[i];
        vec_d[parent_index[i]] -= p * vec_b[i];
        vec_rhs[parent_index[i]] -= p * vec_rhs[i];
    }

    //bksub//Note:这些都是根节点，因此没有parent_index
    for (int i = 0; i < ncell; i++)
    {
        vec_rhs[i] /= vec_d[i];
    }
    for (int i = ncell; i < len; i++)
    {
        vec_rhs[i] -= vec_b[i] * vec_rhs[parent_index[i]];
        vec_rhs[i] /= vec_d[i];
    }
}

void cpu_solve_permute1(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, 
                        int* parent_index, int nstride, int* stride, int* firstnode,
                        int* lastnode, int* cellsize, int ncell, int len)
{
    /*
        这些是permute状态下，要传入的变量
        int nstride = p_neuron->nstride;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* cellsize = p_neuron->vecdata_cellsize->get_cpu_data();
        */
    int tid;
    int i, icellsize;
    int istride, ip;
    double p;
    for (tid = 0; tid < ncell; tid++)//分成若干个小cell，就可以并行化了，在GPU那边是把这个for循环给并行了
    {
        /*
            处理firstnode[tid]到lastnode[tid]之间的方程
            同时，他是跳着来处理的，例如，1，3，6这几个单独处理，所以有一个stride来记录间隔
        */
        icellsize = cellsize[tid];
        i = lastnode[tid];
        for (istride = nstride - 1; istride >= 0; istride--)
        {
            if (istride < icellsize)
            {
                ip = parent_index[i];
                p = vec_a[i] / vec_d[i];
                vec_d[ip] -= p * vec_b[i];
                vec_rhs[ip] -= p * vec_rhs[i];
                i -= stride[istride];
            }
        }

        i = firstnode[tid];
        vec_rhs[tid] /= vec_d[tid];
        for (istride = 0; istride < icellsize; istride++)
        {
            ip = parent_index[i];
            vec_rhs[i] -= vec_b[i] * vec_rhs[ip];
            vec_rhs[i] /= vec_d[i];
            i += stride[istride + 1];
        }
    }
}

void cpu_solve_permute3(double* vec_a, double* vec_b, double* vec_d, double* vec_rhs, 
                        int* parent_index, int* max_order_each_thread, int* min_order_each_thread,
                        int* firstnode, int* lastnode, int* stride, int* map_t2c, 
                        int norder, int ncell, int nthread)
{
    /*
     * CPU serial version of GPU permute3 cop_solve_kernel
     * Processes all threads sequentially to maintain same data layout compatibility
     */
    
    // Triangle phase - process in reverse order (norder down to 0)
    for (int iorder = norder; iorder >= 0; iorder--) {
        for (int tid = 0; tid < nthread; tid++) {
            const int max_order = max_order_each_thread[tid];
            const int min_order = min_order_each_thread[tid];
            
            if (iorder >= min_order && iorder <= max_order) {
                int i = lastnode[tid];
                const int offset = (tid >> 5) * (norder + 1) - 1;
                
                // Navigate to the correct node for this order
                for (int order_step = norder; order_step > iorder; order_step--) {
                    if (order_step >= min_order && order_step <= max_order && i > -1) {
                        i -= stride[offset + order_step];
                    }
                }
                
                if (i > -1) {
                    const int ip = parent_index[i];
                    const double a_val = vec_a[i];
                    const double d_val = vec_d[i];
                    const double b_val = vec_b[i];
                    const double rhs_val = vec_rhs[i];
                    
                    const double p = a_val / d_val;
                    
                    // Update parent node (no atomic operations needed in serial)
                    vec_d[ip] -= p * b_val;
                    vec_rhs[ip] -= p * rhs_val;
                }
            }
        }
    }
    
    // Backsubstitution phase - process root cells first
    for (int tid = 0; tid < nthread; tid++) {
        const int icell = map_t2c[tid];
        if (icell > -1) {
            vec_rhs[icell] /= vec_d[icell];
        }
    }
    
    // Backsubstitution phase - process remaining nodes in forward order
    for (int iorder = 1; iorder <= norder; iorder++) {
        for (int tid = 0; tid < nthread; tid++) {
            const int max_order = max_order_each_thread[tid];
            const int min_order = min_order_each_thread[tid];
            
            if (iorder >= min_order && iorder <= max_order) {
                int i = firstnode[tid];
                const int offset = (tid >> 5) * (norder + 1);
                
                // Navigate to the correct node for this order
                for (int order_step = 1; order_step < iorder; order_step++) {
                    if (order_step >= min_order && order_step <= max_order && i > -1) {
                        i += stride[offset + order_step];
                    }
                }
                
                if (i > -1) {
                    const int ip = parent_index[i];
                    const double b_val = vec_b[i];
                    const double rhs_parent = vec_rhs[ip];
                    const double d_val = vec_d[i];
                    const double rhs_val = vec_rhs[i];
                    
                    const double p = rhs_val - b_val * rhs_parent;
                    vec_rhs[i] = p / d_val;
                }
            }
        }
    }
}

/*
 * solve the cable equation
 */
void Simulate::solve_matrix_cpu(NeuronGroupData* p_neuron)
{
    int len = p_neuron->len;
    int ncell = p_neuron->ncell;
    double* vec_a = p_neuron->vecdata_a->get_cpu_data();
    double* vec_b = p_neuron->vecdata_b->get_cpu_data();
    double* vec_d = p_neuron->vecdata_d->get_cpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();
    int* parent_index = p_neuron->vecdata_parent_index->get_cpu_data();

    //solve_serial(vec_a, vec_b, vec_d, vec_rhs, parent_index, ncell, len);
    if (permute_type == 0)
    {
        solve_serial(vec_a, vec_b, vec_d, vec_rhs, parent_index, ncell, len);
    }
    else if (permute_type == 1)
    {
        int nstride = p_neuron->nstride;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* cellsize = p_neuron->vecdata_cellsize->get_cpu_data();
        cpu_solve_permute1(vec_a, vec_b, vec_d, vec_rhs, parent_index, nstride, 
                           stride, firstnode, lastnode, cellsize, ncell, len);
    }
    else if (permute_type == 3)
    {
        int norder = p_neuron->norder;
        int nthread = p_neuron->threads_num;
        int* stride = p_neuron->vecdata_stride->get_cpu_data();
        int* firstnode = p_neuron->vecdata_firstnode->get_cpu_data();
        int* lastnode = p_neuron->vecdata_lastnode->get_cpu_data();
        int* max_order_each_thread = p_neuron->vecdata_max_order_each_thread->get_cpu_data();
        int* min_order_each_thread = p_neuron->vecdata_min_order_each_thread->get_cpu_data();
        int* map_t2c = p_neuron->vecdata_map_t2c->get_cpu_data();
        cpu_solve_permute3(vec_a, vec_b, vec_d, vec_rhs, parent_index, max_order_each_thread, 
                          min_order_each_thread, firstnode, lastnode, stride, map_t2c, norder, 
                          ncell, nthread);
    }
}

void Simulate::update_cpu(NeuronGroupData* p_group)
{
    int len = p_group->len;
    double* vec_v = p_group->vecdata_v->get_cpu_data();
    double* vec_rhs = p_group->vecdata_rhs->get_cpu_data();

    for (int i = 0; i < len; i++)
    {
        vec_v[i] += vec_rhs[i];
    }
    p_group->mech_cap->cap_current_cpu(p_group->cj, vec_rhs);
}

void Simulate::last_part_cpu()
{
    t += 0.5 * dt;

    int ncell = neuron_group_list.size();
    for (auto p_neuron : neuron_group_list)
    {
        p_neuron->vec_play_continuous.continuous_cpu(t);
        nonvint_cpu(p_neuron);
    }
    record_if_needed_cpu_();
}

void Simulate::nonvint_cpu(NeuronGroupData* p_neuron)
{
    SimMechStateParam params = {p_neuron->vecdata_v->get_cpu_data(), dt, t};
    const bool record_vjp_tape = should_stage_vjp_tape_(p_neuron);

    for (auto p_mech : p_neuron->mech_write_state_ion_list) {
        p_mech->state_cpu(params);
        if (record_vjp_tape && p_mech->supports_current_vjp_tape()) {
            SimMechCurrentVJPTapeParam tape_param{.v = params.v, .dt = params.dt, .t = params.t};
            p_mech->stage_current_vjp_tape_cpu(tape_param);
        }
    }
    for (auto p_mech : p_neuron->mechanism_list) {
        if (!p_mech->write_state_ion) {
            p_mech->state_cpu(params);
            if (record_vjp_tape && p_mech->supports_current_vjp_tape()) {
                SimMechCurrentVJPTapeParam tape_param{.v = params.v, .dt = params.dt, .t = params.t};
                p_mech->stage_current_vjp_tape_cpu(tape_param);
            }
        }
    }
    note_vjp_tape_staged_(p_neuron);
    
}

/*
 * for all synapse mechanism, call pre_spike_send() to 
 * put fired spikes into spike buffers
 */
void Simulate::network_spike_send_cpu()
{
    int ngroup = neuron_group_list.size();
    for (int i = 0; i < ngroup; i++)
    {
        NeuronGroupData* p_neuron = neuron_group_list[i];
        if (p_neuron == nullptr || p_neuron->presyn == nullptr || p_neuron->vecdata_spk_flags == nullptr ||
            p_neuron->vecdata_v == nullptr) {
            continue;
        }
        auto &ps = p_neuron->presyn;
        SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
        double* vec_v = p_neuron->vecdata_v->get_cpu_data();
        ps->threshold_detect_cpu(vec_v, spk_flags, t, rec_spikes);
        record_spike_vjp_presyn_tape_cpu_(p_neuron);
    }
}

void Simulate::record_output_spikes_cpu(NeuronGroupData* p_neuron)
{
    if (p_neuron == nullptr || p_neuron->vecdata_spk_flags == nullptr || p_neuron->spk_vec == nullptr) {
        return;
    }
    SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
    size_t buffer_size = p_neuron->spk_vec->v.size();
    for (int i = 0; i < buffer_size; i++)
    {
        int gid = p_neuron->spk_vec->v[i].gid;
        if (spk_flags[i] == SpikeFlag::NORMAL_EVENT && gid >= 0)
        {
            this->rec_spikes.emplace_back(t, gid);
        }
    }
}

bool comp(const pair<double, uint32_t> &p1, const pair<double, uint32_t> &p2)
{
    if (p1.first == p2.first)
        return p1.second < p2.second;
    return p1.first < p2.first;
}

void Simulate::output_spikes()
{
    if(!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    string outfile = output_folder + "/spk.dat";
    sort(rec_spikes.begin(), rec_spikes.end(), comp);    

    FILE *fp = fopen(outfile.c_str(), "w");
    for (int i = 0; i < rec_spikes.size(); i++)
    {
        if (rec_spikes[i].second > -1)
        {
            fprintf(fp, "%.8g\t%d\n", rec_spikes[i].first, rec_spikes[i].second);
        }
    }
    fclose(fp);
}

/*
 * for all synapse mechanism, call post_spike_receive() to 
 * deal with fired spikes, if firetime + delay <= t, the 
 * NET_RECEIVE block in .mod file should be called 
 */
void Simulate::network_spike_receive_cpu()
{
    bool hasSent = false;
    do {
        hasSent = false;
        for (NeuronGroupData* p_neuron : neuron_group_list)
        {
            if (p_neuron == nullptr || p_neuron->vec_postsyn.empty()) {
                continue;
            }
            if (p_neuron->vecdata_spk_flags == nullptr || p_neuron->spk_vec == nullptr) {
                continue;
            }
            SpikeFlag* spk_flags = p_neuron->vecdata_spk_flags->get_cpu_data();
            for(auto postsyn : p_neuron->vec_postsyn)
            {
                postsyn->get_spike_from_vec_cpu(p_neuron->spk_vec, spk_flags, t);
            }
            // 因为有可能多个postsyn连到同一个pre-syn上，因此，需要把所有的postsyn遍历完，再恢复
            clearValidSpkFlags(p_neuron->vecdata_spk_flags);

            for(auto postsyn : p_neuron->vec_postsyn)
            {
                postsyn->post_spike_receive_cpu(t + dt / 2);
                hasSent |= postsyn->net_receive_cpu(t);
            }
        }
    }while(hasSent);
}



//clearAll是在finitialize的时候调用的，把所有的都清空
void Simulate::clearValidSpkFlags(VecData<SpikeFlag> *vecdata_spk_flags, bool cleanAll){
    if (vecdata_spk_flags == nullptr) {
        return;
    }
    auto cpu_vec_spk_flags = vecdata_spk_flags->get_cpu_data();
    int cpu_vec_spk_len = vecdata_spk_flags->size();
    for(int i = 0;i<cpu_vec_spk_len;i++){
        if(cpu_vec_spk_flags[i] == SpikeFlag::NORMAL_EVENT || cleanAll)
            cpu_vec_spk_flags[i] = SpikeFlag::INVALID; //清空spike标志
    }
}

void Simulate::gap_transfer_cpu(NeuronGroupData *p_group){
    auto &gap_info = p_group->cpu_gap_trans_info;
    int ntrans = gap_info.ntrans();
    if (ntrans == 0) {
        return;
    }
    
    double** src = gap_info.src.get_cpu_data();
    double** dst = gap_info.dst.get_cpu_data();
    
    for(int i = 0; i < ntrans; i++){
        *(dst[i]) = *(src[i]);
    }
}

void Simulate::clearAllSpikes_cpu(){
    for(auto p_neuron : neuron_group_list){
        if (p_neuron == nullptr) {
            continue;
        }
        auto vec_spk = p_neuron->spk_vec;
        if (vec_spk) {
            vec_spk->clean();
        }

        auto vec_spk_flags = p_neuron->vecdata_spk_flags;
        if (vec_spk_flags) {
            clearValidSpkFlags(vec_spk_flags,true);
        }
    }
}

void Simulate::clearAllSpikes_gpu(){
    clearAllSpikes_cpu();
    for(auto p_neuron : neuron_group_list){
        if (p_neuron && p_neuron->vecdata_spk_flags) {
            p_neuron->vecdata_spk_flags->update_gpu_data_from_cpu();
        }
    }
}

// void Simulate::vecevent_play(int mech_idx, int len,const double *data_arr){
//     assert(neuron_group_list.size() == 1);
//     auto p_neuron = neuron_group_list[0];
//     auto p_vecevent = p_neuron->mech_vecevent;
//     if(p_vecevent == nullptr){
//         printf("vecevent is not initialized\n");
//         assert(false);
//     }
//     VecEvent::play(p_vecevent, this->mode, mech_idx, len, data_arr);
// }

int Simulate::compute_transfer_impedance_dc_nodes(int source_node_idx_internal,
                                                  const std::vector<int>& target_node_idx_internal,
                                                  double inject_amp,
                                                  double v_init,
                                                  std::vector<double>& out_impedance) {
    out_impedance.clear();
    if (neuron_group_list.empty()) {
        printf("compute_transfer_impedance_dc_nodes: neuron_group_list is empty\n");
        return -1;
    }
    if (!(std::isfinite(inject_amp)) || std::fabs(inject_amp) <= 0.0) {
        printf("compute_transfer_impedance_dc_nodes: inject_amp must be non-zero\n");
        return -1;
    }

    auto* p_neuron = neuron_group_list[0];
    const int len = p_neuron->len;
    if (source_node_idx_internal < 0 || source_node_idx_internal >= len) {
        printf("compute_transfer_impedance_dc_nodes: invalid source node idx %d (len=%d)\n",
               source_node_idx_internal,
               len);
        return -1;
    }

    for (size_t i = 0; i < target_node_idx_internal.size(); ++i) {
        const int node = target_node_idx_internal[i];
        if (node < 0 || node >= len) {
            printf("compute_transfer_impedance_dc_nodes: invalid target node idx %d at i=%zu (len=%d)\n",
                   node,
                   i,
                   len);
            return -1;
        }
    }

    double* vec_v = p_neuron->vecdata_v->get_cpu_data();
    double* vec_rhs = p_neuron->vecdata_rhs->get_cpu_data();
    double* vec_area = p_neuron->vecdata_area->get_cpu_data();

    if (!vec_v || !vec_rhs || !vec_area) {
        printf("compute_transfer_impedance_dc_nodes: required vectors are null\n");
        return -1;
    }

    const double src_area = vec_area[source_node_idx_internal];
    if (!(std::isfinite(src_area)) || src_area <= 0.0) {
        printf("compute_transfer_impedance_dc_nodes: invalid source area=%g at node=%d\n",
               src_area,
               source_node_idx_internal);
        return -1;
    }

    // 阻抗求解应在静态线性化矩阵上进行（与NEURON Impedance(freq=0)一致）。
    const std::vector<double> v_saved(vec_v, vec_v + len);
    const double old_cj = p_neuron->cj;
    const double old_t = t;

    std::fill(vec_v, vec_v + len, v_init);
    p_neuron->cj = 0.0;
    setup_tree_matrix_cpu(p_neuron);

    // 覆盖rhs为注入电流：1 nA -> 1e2/area（单位与NEURON一致，MOhm）
    std::fill(vec_rhs, vec_rhs + len, 0.0);
    vec_rhs[source_node_idx_internal] = inject_amp * (1.0e2 / src_area);

    solve_matrix_cpu(p_neuron);

    out_impedance.resize(target_node_idx_internal.size(), 0.0);
    for (size_t i = 0; i < target_node_idx_internal.size(); ++i) {
        const int node = target_node_idx_internal[i];
        out_impedance[i] = vec_rhs[node] / inject_amp;
    }

    std::copy(v_saved.begin(), v_saved.end(), vec_v);
    p_neuron->cj = old_cj;
    t = old_t;
    return 0;
}

int Simulate::create_optimizer(OptimizerType type) {
    std::unique_ptr<OptimizerBase> optimizer;
    switch (type) {
        case OptimizerType::SGD:
            optimizer = std::make_unique<SGDOptimizer>(mode);
            break;
        case OptimizerType::Momentum:
            optimizer = std::make_unique<SGDMomentumOptimizer>(mode);
            break;
        case OptimizerType::Adam:
            optimizer = std::make_unique<AdamOptimizer>(mode);
            break;
        default:
            printf("Unsupported optimizer type\n");
            return -1;
    }
    int optimizer_id = next_optimizer_id++;
    optimizers.emplace(optimizer_id, std::move(optimizer));
    return optimizer_id;
}

int Simulate::register_optimizer_param(int optimizer_id,
                                       double* weight_cpu,
                                       double* grad_cpu,
                                       double* weight_gpu,
                                       double* grad_gpu,
                                       double impedance) {
    std::vector<double*> weight_cpu_vec;
    std::vector<double*> grad_cpu_vec;
    std::vector<double*> weight_gpu_vec;
    std::vector<double*> grad_gpu_vec;

    weight_cpu_vec.push_back(weight_cpu);
    grad_cpu_vec.push_back(grad_cpu);
    if (mode == GPU) {
        weight_gpu_vec.push_back(weight_gpu);
        grad_gpu_vec.push_back(grad_gpu);
    }

    return register_optimizer_param_batch(optimizer_id,
                                          weight_cpu_vec,
                                          grad_cpu_vec,
                                          weight_gpu_vec,
                                          grad_gpu_vec,
                                          impedance);
}

int Simulate::register_optimizer_param_batch(int optimizer_id,
                                             const std::vector<double*>& weight_cpu,
                                             const std::vector<double*>& grad_cpu,
                                             const std::vector<double*>& weight_gpu,
                                             const std::vector<double*>& grad_gpu,
                                             double impedance) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("register_optimizer_param: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    if (weight_cpu.empty()) {
        printf("register_optimizer_param_batch: empty weight list\n");
        return -1;
    }
    if (weight_cpu.size() != grad_cpu.size()) {
        printf("register_optimizer_param_batch: CPU pointer count mismatch (weight=%zu, grad=%zu)\n",
               weight_cpu.size(), grad_cpu.size());
        return -1;
    }
    if (mode == GPU) {
        if (weight_gpu.size() != weight_cpu.size() || grad_gpu.size() != weight_cpu.size()) {
            printf("register_optimizer_param_batch: GPU pointer count mismatch (batch=%zu)\n",
                   weight_cpu.size());
            return -1;
        }
    }

    OptimizerParam param;
    param.weight_cpu = weight_cpu;
    param.grad_cpu = grad_cpu;
    if (mode == GPU) {
        param.weight_gpu = weight_gpu;
        param.grad_gpu = grad_gpu;
    }
    param.impedance = impedance;
    return it->second->add_param(param);
}

int Simulate::configure_optimizer(int optimizer_id, const OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("configure_optimizer: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    it->second->configure(params);
    return 0;
}

int Simulate::optimizer_step(int optimizer_id, double learning_rate, double record_time, double dt_step) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_step: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    if (record_time <= 0.0 || dt_step <= 0.0) {
        printf("optimizer_step: invalid record_time (%f) or dt (%f)\n", record_time, dt_step);
        return -1;
    }
    double inv_record_steps = dt_step / record_time;
    it->second->step(learning_rate, inv_record_steps);
    return 0;
}

int Simulate::optimizer_reset_state(int optimizer_id) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_reset_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    it->second->reset_state();
    return 0;
}

int Simulate::optimizer_get_adam_state(int optimizer_id,
                                      long long& step_count,
                                      std::vector<double>& m,
                                      std::vector<double>& v,
                                      OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_get_adam_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    AdamOptimizer* adam = dynamic_cast<AdamOptimizer*>(it->second.get());
    if (adam == nullptr) {
        printf("optimizer_get_adam_state: optimizer %d is not Adam\n", optimizer_id);
        return -1;
    }
    params = it->second->hyper_params();
    adam->export_state(step_count, m, v);
    return 0;
}

int Simulate::optimizer_set_adam_state(int optimizer_id,
                                      long long step_count,
                                      const std::vector<double>& m,
                                      const std::vector<double>& v,
                                      const OptimizerHyperParams& params) {
    auto it = optimizers.find(optimizer_id);
    if (it == optimizers.end()) {
        printf("optimizer_set_adam_state: optimizer %d not found\n", optimizer_id);
        return -1;
    }
    AdamOptimizer* adam = dynamic_cast<AdamOptimizer*>(it->second.get());
    if (adam == nullptr) {
        printf("optimizer_set_adam_state: optimizer %d is not Adam\n", optimizer_id);
        return -1;
    }
    it->second->configure(params);
    if (adam->import_state(step_count, std::span<const double>(m.data(), m.size()), std::span<const double>(v.data(), v.size())) < 0) {
        printf("optimizer_set_adam_state: import_state failed\n");
        return -1;
    }
    return 0;
}
