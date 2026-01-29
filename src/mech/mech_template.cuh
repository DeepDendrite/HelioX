#pragma once
#include <vector>
#include <map>
#include "cuda_utils.h"
#include "mechanism.h"
#include "magic_enum/magic_enum.hpp"
#include "vecdata.h"
#include "var_struct.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include "ion_table.h"
#include "mech_var_table.h"
#include "global_vars.h"
#include "mech_template_utils.cuh"
#include "debug_var.cuh"
#include "neuron.h"
#include "coredat_structs.h"
#include "legacy_index_utils.h"
#include "dparam_semantics.h"

template <MechTraitType MechTrait>
struct VarAccessor
{
    // using VarNames = typename MechTrait::VarNames;
    int idx = -1; // mech中的节点下标

    DevVarStruct dev_var;
    DevGlobalVarStruct dev_global_var;
    DevIonVarStruct dev_ion_var;
    
#ifdef DEBUG
#define DEBUG_IDX_GE_0 assert(idx >= 0);
#else
#define DEBUG_IDX_GE_0 ;
#endif

/////////////////////////////普通变量访问函数（Range和State）/////////////////////////////////////
    __host__ __device__ __forceinline__ double &operator()(typename MechTrait::VarNames varname) const
    {
        DEBUG_IDX_GE_0;
        auto varname_as_idx = static_cast<size_t>(varname);
        return dev_var[varname_as_idx][idx];
    }

    __host__ __device__ __forceinline__ double &Arr(typename MechTrait::VarNames varname, int i) const
    {
        auto varname_as_idx = static_cast<size_t>(varname);
        return dev_var.getArr(varname_as_idx, idx, i);
    }
///////////////////////////////Global变量访问函数（GlobalVar）/////////////////////////////////////
    template <typename T = MechTrait>
    __host__ __device__ __forceinline__ double &operator()(typename T::GlobalVarNames varname) const
        requires has_GlobalVarNames_v<T>
    {
        DEBUG_IDX_GE_0;
        auto varname_as_idx = static_cast<size_t>(varname);
        return dev_global_var[varname_as_idx];  //global_var是共享的，没有idx
    }
    
    template <typename T = MechTrait>
    __host__ __device__ __forceinline__ double &Arr(typename T::GlobalVarNames varname, int i) const
    requires has_GlobalVarNames_v<T>
    {
        auto varname_as_idx = static_cast<size_t>(varname);
        return dev_global_var.getArr(varname_as_idx, i);//i是数组中的下标，不需要用mech中的idx
    }

////////////////////////////////////离子变量访问函数（IonVar）/////////////////////////////////////
    template <typename T = MechTrait>
    __host__ __device__ __forceinline__ double &operator()(typename T::IonVarNames varname) const
        requires has_IonVarNames_v<T>
    {
        DEBUG_IDX_GE_0;
        auto varname_as_idx = static_cast<size_t>(varname);
        return dev_ion_var.getIonVar(varname_as_idx, idx);
    }

////////////////////////////////////POINTER变量访问函数（PointerVar）/////////////////////////////////////
    template <typename T = MechTrait>
    __host__ __device__ __forceinline__ double* PtrPtr(typename T::PointerVarNames varname) const
        requires has_PointerVarNames_v<T>
    {
        DEBUG_IDX_GE_0;
        auto var_idx = static_cast<size_t>(varname);
        if (!dev_var.ptr_targets) {
            return nullptr;
        }
        return dev_var.ptr_targets[var_idx * dev_var.ptr_stride + idx];
    }

    template <typename T = MechTrait>
    __host__ __device__ __forceinline__ double& Ptr(typename T::PointerVarNames varname) const
        requires has_PointerVarNames_v<T>
    {
        DEBUG_IDX_GE_0;
        double* p = PtrPtr<T>(varname);
#ifdef DEBUG
        assert(p != nullptr);
#endif
        return *p;
    }
    #undef DEBUG_IDX_GE_0
};
// MechTemp中，三个重要的函数的参数
struct MechTempCurParam
{
    double volt;
    double t;
    bool updateIon;
    DebugVar<int> idx;
};

struct MechTempStateParam
{
    double volt;
    double dt;
    double t;
    DebugVar<int> idx;
};

struct MechTempInitParam
{
    double dt;
    double volt;
    DebugVar<int> idx;
};

// GPU模式下的三个内核
template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_init_kernel(int nnode, int *node_indices, SimMechInitialParam param, VarAccessor<MechTrait> gpu_vars);

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_current_kernel(double t, double *vec_rhs, double *vec_d, int nnode, double *v, int *node_indices, double *area, VarAccessor<MechTrait> gpu_vars);

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_state_kernel(int nnode, double *v, double dt, double t, int *node_indices, VarAccessor<MechTrait> gpu_vars);

// GPU用于获取变量指针的内核

template <typename MechTrait, typename EnumName>
__global__ void getVarKernel(VarAccessor<MechTrait> var_access, EnumName var_name, double **gpu_var_ptr);

template <typename Derived, MechTraitType MechTrait> // 需要传入子类名，以及一个枚举类型，用于标记变量名
class MechTemp : public Mechanism
{
protected:
    static_assert(std::is_enum_v<typename MechTrait::VarNames>, "MechTrait::VarNames must be an enum type");
    using VarNames = typename MechTrait::VarNames;

    VarStruct<MechTrait> var_struct; // 变量结构体
    int dparam_size_ = 0;
    std::vector<int> pointer2type_;
    std::vector<int> pointer_dparam_slots_;
    std::vector<int> pointer_p2t_rank_;

    // 变量相关的信息存放在这
    map<VarNames, double> init_values;

    map<VarNames, CoreIdxInfo> var_in_coredata_idx;
    IonVarInfoMap<MechTrait> ion_var_map;
    GlobalVarInfoMap<MechTrait> global_info_map;

public:
    using enum MechFlags;
    constexpr static MechFlags flags = Derived::flags;

    // 构造函数，应当初始化init_values,这样reg_node_indices的时候可以初始化vecdata_vars
    // 并且应当初始化var_in_coredata_idx，这样read_data_from_coredat的时候可以初始化vecdata_vars
    MechTemp(MechInitParams &param) : Mechanism(param)
    {
        // Safety contract:
        // If a mechanism updates STATE variables (ENABLE_STATE), then it must also participate
        // in finitialize() (ENABLE_INIT) so the state can be reset/reinitialized each run.
        static_assert(!hasFlag(MechFlags::ENABLE_STATE) || hasFlag(MechFlags::ENABLE_INIT),
                      "Mechanism has ENABLE_STATE but not ENABLE_INIT: state will not be reset on finitialize(), "
                      "leading to multi-run divergence when reusing the same simulator instance.");

        if constexpr (hasFlag(MechFlags::POINT_PROCESS))
        {
            need_area = true;
        }
        if constexpr (hasFlag(MechFlags::WRITE_EION_IN_STATE))
        {
            write_state_ion = true;
        }
    }

    constexpr static bool hasFlag(MechFlags flag)
    {
        return static_cast<int>(flags) & static_cast<int>(flag);
    }

    // 注册节点索引
    virtual void reg_node_indices(MechInitParams &param) override
    {
        auto node_count = param.node_count;
        assert(node_count > 0);
        printf_debug("Mech[%s] reg_node_indices node_count:%d\n", name.c_str(), node_count);

        var_struct.init(param, var_in_coredata_idx, init_values, global_info_map);
        var_struct.initPdata(param);
        var_struct.map_ion_var(ion_var_map, param);

        dparam_size_ = param.pdata_size;
        pointer2type_.clear();
        if (param.pointer2type) {
            pointer2type_ = *param.pointer2type;
        }

        pointer_dparam_slots_.clear();
        pointer_p2t_rank_.clear();
        if constexpr (has_PointerVarNames_v<MechTrait>)
        {
            using PointerVarNames = typename MechTrait::PointerVarNames;
            const int ptr_var_count = magic_enum::enum_count<PointerVarNames>();
            if (ptr_var_count > 0)
            {
                // Prefer explicit per-mechanism slot mapping (most precise because it can map
                // per-POINTER-variable, not just "which dparam indices are POINTERs").
                if (const std::vector<int>* slots =
                        MechanismFactory::getInstance().getPointerDparamSlots(this->name);
                    slots && static_cast<int>(slots->size()) == ptr_var_count) {
                    pointer_dparam_slots_ = *slots;
                } else {
                    // Next-best: derive POINTER slot indices from CoreNEURON-style dparam semantics
                    // (where semantics[i] == -5 means dparam slot i holds a POINTER Datum).
                    // This is robust w.r.t. layout changes (non-contiguous POINTER slots, density mechs, etc.)
                    // as long as the mech registers full dparam semantics.
                    const std::vector<int>* semantics = param.dparam_semantics;
                    if (!semantics) {
                        semantics = MechanismFactory::getInstance().getDparamSemantics(this->name);
                    }
                    if (semantics && static_cast<int>(semantics->size()) == dparam_size_) {
                        std::vector<int> slots_from_sem;
                        slots_from_sem.reserve(ptr_var_count);
                        for (int i = 0; i < dparam_size_; ++i) {
                            if (semantics->at(i) == dpsem(DparamSemantics::pointer)) {
                                slots_from_sem.push_back(i);
                            }
                        }
                        if (static_cast<int>(slots_from_sem.size()) == ptr_var_count) {
                            pointer_dparam_slots_ = std::move(slots_from_sem);
                        }
                    }

                    if (pointer_dparam_slots_.empty() && hasFlag(MechFlags::POINT_PROCESS)) {
                        // Backward-compatible fallback: common nrnivmodl POINT_PROCESS layout places
                        // user POINTER vars at dparam[2..] contiguously.
                        pointer_dparam_slots_.resize(ptr_var_count);
                        for (int i = 0; i < ptr_var_count; ++i) {
                            pointer_dparam_slots_[i] = 2 + i;
                        }
                    }
                }

                // Export order of pointer2type is by increasing dparam slot within each instance.
                // Build a mapping from POINTER var index -> pointer2type "rank" (slot-sorted index).
                if (static_cast<int>(pointer_dparam_slots_.size()) == ptr_var_count)
                {
                    std::vector<std::pair<int, int>> slot_and_var;
                    slot_and_var.reserve(ptr_var_count);
                    for (int i = 0; i < ptr_var_count; ++i)
                    {
                        slot_and_var.push_back({pointer_dparam_slots_[i], i});
                    }
                    std::sort(slot_and_var.begin(), slot_and_var.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });

                    pointer_p2t_rank_.assign(ptr_var_count, 0);
                    for (int rank = 0; rank < ptr_var_count; ++rank)
                    {
                        pointer_p2t_rank_[slot_and_var[rank].second] = rank;
                    }
                }
            }
        }
    }

    // 从 CoreNeuron 读取数据
    virtual void read_data_from_coredat(MechInitParams &param) override
    {
        auto nnode = param.node_count;
        auto data = param.data;
        auto param_size = param.data_size;

        // 从global_vars中读取数据
        if constexpr (has_GlobalVarNames_v<MechTrait>)
        {
            for (auto &[var_name, var_info] : global_info_map)
            {
                if (!coreneuron::global_var_map.contains(var_info.info))
                {
                    printf("global var %s not found\n", var_info.info.c_str());
                    assert(false);
                }
                auto var_vec_data = var_struct.global_vars[(int)var_name].get();
                auto cpu_var = var_vec_data->get_cpu_data();

                auto global_var = coreneuron::global_var_map[var_info.info];
                int n = global_var.size();
                assert(n == var_vec_data->size());
                for (int i = 0; i < n; i++)
                {
                    cpu_var[i] = global_var[i];
                    // printf("global var %s[%d]:%f\n",var_info.info.c_str(),i,cpu_var[i]);
                }
                if (param.mode == Mode::GPU)
                {
                    var_vec_data->update_gpu_data_from_cpu();
                }
            }
        }

        // 从coredat中读取数据
        vector<pair<double *, CoreIdxInfo>> data_init_list; // 提前把cpu数据指针和变量索引存储起来
        for (auto [var_name, var_idx] : var_in_coredata_idx)
        {
            VecData<double> *var = var_struct[var_name];
            double *cpu_data_ptr = var->get_cpu_data();
            // assert(var->len == nnode);
            data_init_list.push_back({cpu_data_ptr, var_idx});
        }
        // coredat的数据结构是按照node中的变量排列的，所以需要按照node的顺序来读取数据

        // 计算前缀和
        if (param.array_dims != nullptr)
        { // 新版本，支持数组
            vector<int> prefix_sum(param.array_dims->size());
            std::exclusive_scan(param.array_dims->begin(), param.array_dims->end(), prefix_sum.begin(), 0); // 由于不是每个变量都是1，即标量，所以需要计算前缀和以确定每个变量的起始位置
            for (int inode = 0; inode < nnode; inode++)
            {
                int offset = inode * param_size;
                for (auto &[data_ptr, var_idx] : data_init_list)
                {
                    if (var_idx.info >= param.array_dims->size())
                    {
                        printf("var_idx.info:%d >= param.array_dims->size():%d\n", var_idx.info, param.array_dims->size());
                        assert(false);
                    }
                    int begin_idx = offset + prefix_sum[var_idx.info];
                    for (int i = 0; i < var_idx.array_size; i++)
                    {
                        data_ptr[inode * var_idx.array_size + i] = data[begin_idx + i];
                    }
                }
            }
        }
        else
        {
            for (auto &[data_ptr, var_idx] : data_init_list)
            {
                assert(!var_idx.isArray()); // 不支持数组
            }
            for (int inode = 0; inode < nnode; inode++)
            {
                int offset = inode * param_size;
                for (auto &[data_ptr, var_idx] : data_init_list)
                {
                    data_ptr[inode] = data[offset + var_idx.info];
                }
            }
        }

        if (mode == Mode::GPU)
        {
            for (auto [var_name, var_idx] : var_in_coredata_idx)
            {
                var_struct[var_name]->update_gpu_data_from_cpu();
            }
        }

        // 注册varTable
        auto &varMap = mech_var_table[param.type];
        for (auto [var_name, var_idx] : var_in_coredata_idx)
        {
            MechVarData varData;
            varData.name = this->name + "_" + string(magic_enum::enum_name(var_name));
            varData.len = var_struct[var_name]->size();
            varData.cpu_data = var_struct[var_name]->get_cpu_data();
            if (param.mode == Mode::GPU)
            {
                varData.gpu_data = var_struct[var_name]->get_gpu_data();
            }
            varMap[var_idx.info] = varData;
        }
    }

    virtual void resolve_pointers(HelioXroupData* ndat, coreneuron::CoreData* cdat) override
    {
        if constexpr (!has_PointerVarNames_v<MechTrait>)
        {
            (void)ndat;
            (void)cdat;
            return;
        }
        else
        {
            if (!ndat || !cdat)
            {
                return;
            }
            using PointerVarNames = typename MechTrait::PointerVarNames;
            const int ptr_var_count = magic_enum::enum_count<PointerVarNames>();
            if (ptr_var_count <= 0)
            {
                return;
            }
            if (!var_struct.resources.cpu_ptr_targets || !var_struct.resources.cpu_pdata)
            {
                return;
            }
            if (dparam_size_ <= 0)
            {
                return;
            }

            const int stride = nnode;
            const int base_dparam = hasFlag(MechFlags::POINT_PROCESS) ? 2 : 0;

            // If this is not a point-process, we require an explicit slot map to locate POINTERs.
            if (!hasFlag(MechFlags::POINT_PROCESS) && pointer_dparam_slots_.empty())
            {
                return;
            }

            double** out = var_struct.resources.cpu_ptr_targets;

            // NEURON/CoreNEURON bbcore_write 导出的 pointer2type 是“按 instance-major 展开”的列表，
            // 并且只包含 POINTER slots（顺序为：for inst { for POINTER slot { push(type) } }）。
            const auto* p2t = pointer2type_.empty() ? nullptr : &pointer2type_;
            const bool p2t_per_inst =
                p2t && (p2t->size() == static_cast<size_t>(nnode) * static_cast<size_t>(ptr_var_count));
            const bool p2t_per_var = p2t && (p2t->size() == static_cast<size_t>(ptr_var_count));
            const bool p2t_per_slot = p2t && (p2t->size() == static_cast<size_t>(dparam_size_));

            auto get_target_type = [&](int inst, int pvar_rank, int slot) -> int
            {
                if (!p2t || p2t->empty())
                {
                    return static_cast<int>(coreneuron::gap_idx_type::voltage);
                }
                if (p2t_per_inst)
                {
                    return (*p2t)[static_cast<size_t>(inst) * static_cast<size_t>(ptr_var_count) +
                                 static_cast<size_t>(pvar_rank)];
                }
                if (p2t_per_var)
                {
                    return (*p2t)[static_cast<size_t>(pvar_rank)];
                }
                if (p2t_per_slot && slot >= 0 && slot < dparam_size_)
                {
                    return (*p2t)[static_cast<size_t>(slot)];
                }
                return static_cast<int>(coreneuron::gap_idx_type::voltage);
            };

            for (int inst = 0; inst < nnode; ++inst)
            {
                for (auto pvar : magic_enum::enum_values<PointerVarNames>())
                {
                    const int pvar_idx = static_cast<int>(pvar);
                    const int slot = (!pointer_dparam_slots_.empty() && pvar_idx < static_cast<int>(pointer_dparam_slots_.size()))
                                         ? pointer_dparam_slots_[pvar_idx]
                                         : (base_dparam + pvar_idx);
                    if (slot < 0 || slot >= dparam_size_)
                    {
                        continue;
                    }

                    const int raw = var_struct.resources.cpu_pdata[inst * dparam_size_ + slot];
                    const int pvar_rank = (!pointer_p2t_rank_.empty() && pvar_idx < static_cast<int>(pointer_p2t_rank_.size()))
                                              ? pointer_p2t_rank_[pvar_idx]
                                              : pvar_idx;
                    const int target_type = get_target_type(inst, pvar_rank, slot);

                    double* resolved = nullptr;
                    if (target_type == static_cast<int>(coreneuron::gap_idx_type::voltage))
                    {
                        if (raw >= 0 && raw < ndat->len)
                        {
                            resolved = (mode == GPU ? ndat->vecdata_v->get_gpu_data() : ndat->vecdata_v->get_cpu_data()) + raw;
                        }
                    }
                    else if (target_type == static_cast<int>(coreneuron::gap_idx_type::i_membrane_))
                    {
                        if (ndat->vecdata_i_membrane_ && raw >= 0 && raw < ndat->len)
                        {
                            resolved = (mode == GPU ? ndat->vecdata_i_membrane_->get_gpu_data()
                                                    : ndat->vecdata_i_membrane_->get_cpu_data()) +
                                       raw;
                        }
                    }
                    else if (target_type > 0 && target_type < cdat->mech_data->nmech_type)
                    {
                        int target_inst = -1;
                        int var_index = -1;
                        int offset = -1;

                        const auto& dims = cdat->mech_data->nrn_array_dims[target_type];
                        if (!dims.empty())
                        {
                            auto decoded = legacy2soaos_index(raw, dims);
                            target_inst = decoded[0];
                            var_index = decoded[1];
                            const int array_index = decoded[2];
                            if (var_index >= 0 && var_index < static_cast<int>(dims.size()))
                            {
                                const int array_size = dims[var_index];
                                offset = target_inst * array_size + array_index;
                            }
                        }
                        else
                        {
                            // Pre-array_dims export fallback: treat each variable as scalar.
                            const int sz = cdat->mech_data->nrn_prop_param_size[target_type];
                            if (sz > 0 && raw >= 0)
                            {
                                target_inst = raw / sz;
                                var_index = raw % sz;
                                offset = target_inst;
                            }
                        }

                        if (offset >= 0 && mech_var_table.contains(target_type))
                        {
                            auto& var_map = mech_var_table[target_type];
                            if (var_map.contains(var_index))
                            {
                                auto& var_data = var_map[var_index];
                                if (offset < var_data.len)
                                {
                                    resolved = (mode == GPU ? var_data.gpu_data : var_data.cpu_data) + offset;
                                }
                            }
                        }
                    }

                    out[pvar_idx * stride + inst] = resolved;
                }
            }

            if (mode == GPU && var_struct.resources.gpu_ptr_targets)
            {
                const int total = ptr_var_count * stride;
                mem_copy_cpu2gpu_sync(var_struct.resources.gpu_ptr_targets, out, total * sizeof(double*));
            }
        }
    }

    // CPU 初始化
    virtual void initialize_cpu(SimMechInitialParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_INIT))
        {
            VarAccessor<MechTrait> cpu_vars = getCpuVarAccessor();

            int *node_indices = vecdata_node_indices->get_cpu_data();

            MechTempInitParam init_param;
            init_param.dt = param.dt;

            for (int i = 0; i < nnode; i++)
            {
                cpu_vars.idx = i;
                init_param.volt = param.v[node_indices[i]];
                init_param.idx = i;
                Derived::init_single_node(init_param, cpu_vars);
            }
        }
    }

    // GPU 初始化
    virtual void initialize_gpu(SimMechInitialParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_INIT))
        {
            int *node_indices = vecdata_node_indices->get_gpu_data();
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;

            VarAccessor<MechTrait> gpu_vars = getGpuVarAccessor();

            cuda_init_kernel<Derived><<<block_num, nthread_per_block>>>(nnode, node_indices, param, gpu_vars);
        }
    }

    int info_count = 0;
    // CPU 电流计算
    void current_cpu(SimMechCurrentParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_CURRENT))
        {
            double t = param.t;
            double *v = param.v;
            double *vec_d = param.d;
            double *vec_rhs = param.rhs;

            double _rhs, _g, _v;
            int *node_indices = this->vecdata_node_indices->get_cpu_data();
            double *nd_area_vec;

            VarAccessor<MechTrait> cpu_vars = getCpuVarAccessor();

            if constexpr (hasFlag(MechFlags::POINT_PROCESS))
            {
                nd_area_vec = this->vecdata_area->get_cpu_data();
            }

            for (int i = 0; i < nnode; i++)
            {
                cpu_vars.idx = i;
                int node_index = node_indices[i];
                _v = v[node_index];

                MechTempCurParam cur_param;
                cur_param.volt = _v + 0.001;
                cur_param.t = t;
                cur_param.updateIon = false;
                cur_param.idx = i;
                _g = Derived::current_single_node(cur_param, cpu_vars);

                cur_param.volt = _v;
                cur_param.updateIon = true;
                _rhs = Derived::current_single_node(cur_param, cpu_vars);

                _g = (_g - _rhs) / 0.001;
                if constexpr (hasFlag(MechFlags::POINT_PROCESS))
                {
                    double nd_area = nd_area_vec[node_index];
                    _g *= 1.e2 / nd_area;
                    _rhs *= 1.e2 / nd_area;
                }
                // NOTE: Only mechanisms explicitly marked as ELECTRODE_CURRENT use electrode sign conventions.
                // Regular synapses/gap junctions are membrane currents (NONSPECIFIC_CURRENT), even if they are
                // POINT_PROCESS mechanisms.
                if constexpr (hasFlag(MechFlags::ELECTRODE_CURRENT))
                {
                    vec_rhs[node_index] += _rhs;
                    vec_d[node_index] -= _g;
                }
                else
                {
                    vec_rhs[node_index] -= _rhs;
                    vec_d[node_index] += _g;
                }
            }
        }
    }

    // GPU 电流计算
    void current_gpu(SimMechCurrentParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_CURRENT))
        {
            double t = param.t;
            double *v = param.v;
            double *vec_d = param.d;
            double *vec_rhs = param.rhs;

            int *node_indices = this->vecdata_node_indices->get_gpu_data();
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);

            VarAccessor<MechTrait> gpu_vars = getGpuVarAccessor();

            double *area = nullptr;
            if (this->need_area)
            {
                area = this->vecdata_area->get_gpu_data();
            }
            cuda_current_kernel<Derived><<<block_num, nthread_per_block, 0, stream>>>(t, vec_rhs, vec_d, nnode, v, node_indices, area, gpu_vars);
        }
    }

    // CPU 状态更新
    virtual void state_cpu(SimMechStateParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_STATE))
        {
            double *v = param.v;
            double dt = param.dt;
            int *node_indices = this->vecdata_node_indices->get_cpu_data();

            VarAccessor<MechTrait> cpu_vars = getCpuVarAccessor();

            MechTempStateParam state_param;
            state_param.dt = dt;
            state_param.t = param.t;
            for (int i = 0; i < nnode; i++)
            {
                cpu_vars.idx = i;
                state_param.volt = v[node_indices[i]];
                state_param.idx = i;
                Derived::state_single_node(state_param, cpu_vars);
            }
        }
    }

    // GPU 状态更新
    virtual void state_gpu(SimMechStateParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_STATE))
        {
            double *v = param.v;
            double dt = param.dt;
            int *node_indices = this->vecdata_node_indices->get_gpu_data();
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);

            VarAccessor<MechTrait> gpu_vars = getGpuVarAccessor();

            if constexpr (hasFlag(MechFlags::WRITE_EION_IN_STATE))
            { // 如果需要在状态更新中写入离子电流,串行执行，防止出错
                cuda_state_kernel<Derived><<<block_num, nthread_per_block, 0, 0>>>(nnode, v, dt, param.t, node_indices, gpu_vars);
            }
            else
            {
                cuda_state_kernel<Derived><<<block_num, nthread_per_block, 0, stream>>>(nnode, v, dt, param.t, node_indices, gpu_vars);
            }
        }
    }
    virtual void sync_gpu() override
    {
        cudaStreamSynchronize(*reinterpret_cast<cudaStream_t *>(cuda_stream));
    }

    template <typename EnumName>
    double *getVar(Mode mode, EnumName var_name, int idx, int array_index = -1)
    {
        if (array_index >= 0) {
            // 数组访问：使用Arr方法
            if (mode == Mode::CPU)
            {
                VarAccessor<MechTrait> var_access = getCpuVarAccessor(idx);

                // 嵌套if constexpr避免访问不存在的类型
                if constexpr (std::is_same_v<EnumName, typename MechTrait::VarNames>) {
                    return &(var_access.Arr(var_name, array_index));
                } else {
                    // 只有在不是VarNames时才检查GlobalVarNames
                    if constexpr (has_GlobalVarNames_v<MechTrait>) {
                        if constexpr (std::is_same_v<EnumName, typename MechTrait::GlobalVarNames>) {
                            return &(var_access.Arr(var_name, array_index));
                        } else {
                            // 必定是IonVarNames，不支持数组
                            return nullptr;
                        }
                    } else {
                        // 没有GlobalVarNames，必定是IonVarNames，不支持数组
                        return nullptr;
                    }
                }
            }
            else
            {
                VarAccessor<MechTrait> var_access = getGpuVarAccessor(idx);

                double **gpu_var_ptr;
                cudaMalloc(&gpu_var_ptr, sizeof(double *));

                // GPU端使用相同的嵌套逻辑
                if constexpr (std::is_same_v<EnumName, typename MechTrait::VarNames>) {
                    getVarArrayKernel<<<1, 1>>>(var_access, var_name, array_index, gpu_var_ptr);
                } else {
                    if constexpr (has_GlobalVarNames_v<MechTrait>) {
                        if constexpr (std::is_same_v<EnumName, typename MechTrait::GlobalVarNames>) {
                            getVarArrayKernel<<<1, 1>>>(var_access, var_name, array_index, gpu_var_ptr);
                        } else {
                            cudaFree(gpu_var_ptr);
                            return nullptr;
                        }
                    } else {
                        cudaFree(gpu_var_ptr);
                        return nullptr;
                    }
                }

                double *gpu_var_ptr_on_host;
                cudaMemcpy(&gpu_var_ptr_on_host, gpu_var_ptr, sizeof(double *), cudaMemcpyDeviceToHost);
                cudaFree(gpu_var_ptr);
                return gpu_var_ptr_on_host;
            }
        } else {
            // 标量访问：使用原来的operator()
            if (mode == Mode::CPU)
            {
                VarAccessor<MechTrait> var_access = getCpuVarAccessor(idx);
                return &(var_access(var_name));
            }
            else
            {
                VarAccessor<MechTrait> var_access = getGpuVarAccessor(idx);

                double **gpu_var_ptr;
                cudaMalloc(&gpu_var_ptr, sizeof(double *));
                getVarKernel<<<1, 1>>>(var_access, var_name, gpu_var_ptr);

                double *gpu_var_ptr_on_host;
                cudaMemcpy(&gpu_var_ptr_on_host, gpu_var_ptr, sizeof(double *), cudaMemcpyDeviceToHost);
                cudaFree(gpu_var_ptr);
                return gpu_var_ptr_on_host;
            }
        }
    }

    VarAccessor<MechTrait> __forceinline__ getCpuVarAccessor(int idx = -1)
    {
        VarAccessor<MechTrait> var_access;
        var_access.idx = idx;
        var_access.dev_var = var_struct.cpu_dev_var;
        var_access.dev_global_var = var_struct.cpu_dev_global_var;
        var_access.dev_ion_var = var_struct.cpu_dev_ion_var;
        return var_access;
    }
    VarAccessor<MechTrait> __forceinline__ getGpuVarAccessor(int idx = -1)
    {
        VarAccessor<MechTrait> var_access;
        var_access.idx = idx;
        var_access.dev_var = var_struct.gpu_dev_var;
        var_access.dev_global_var = var_struct.gpu_dev_global_var;
        var_access.dev_ion_var = var_struct.gpu_dev_ion_var;
        return var_access;
    }

    virtual double *getVarPtr(const VarDescriptor& descriptor, Mode mode) override
    {
        int mech_idx = descriptor.node_or_mech_idx;
        const std::string& var_name = descriptor.var;
        int array_index = descriptor.array_index;

        // NOTE: mech_idx is a 0-based instance index. Valid range is [0, nnode).
        // Using `>` here can allow mech_idx == nnode to pass and then index permute[]
        // out-of-bounds, leading to GPU illegal memory access (e.g. via VecPlay).
        if (mech_idx < 0 || mech_idx >= nnode)
        {
            printf("in mech[%s] getVarPtr invalid mech_idx:%d (nnode:%d)\n", name.c_str(), mech_idx, nnode);
            return nullptr;
        }
        if(permute != nullptr)
        {
            mech_idx = permute[mech_idx];
        }
        if (auto casted_value = magic_enum::enum_cast<VarNames>(var_name); casted_value.has_value())
        {
            // 直接传递array_index给getVar，让它处理标量和数组
            return getVar(mode, casted_value.value(), mech_idx, array_index);
        }
        if constexpr (has_GlobalVarNames_v<MechTrait>)
        {
            if (auto casted_value = magic_enum::enum_cast<typename MechTrait::GlobalVarNames>(var_name); casted_value.has_value())
            {
                // 直接传递array_index给getVar
                return getVar(mode, casted_value.value(), mech_idx, array_index);
            }
        }
        if constexpr (has_IonVarNames_v<MechTrait>)
        {
            if (auto casted_value = magic_enum::enum_cast<typename MechTrait::IonVarNames>(var_name); casted_value.has_value())
            {
                // Ion变量不支持数组访问，强制使用标量模式
                return getVar(mode, casted_value.value(), mech_idx, -1);
            }
        }

        return nullptr;
    }
};

/// 后面是GPU内核函数的实现

template <typename MechTrait, typename EnumName>
__global__ void getVarKernel(VarAccessor<MechTrait> var_access, EnumName var_name, double **gpu_var_ptr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        *gpu_var_ptr = &(var_access(var_name));
}

template <typename MechTrait, typename EnumName>
__global__ void getVarArrayKernel(VarAccessor<MechTrait> var_access, EnumName var_name, int array_index, double **gpu_var_ptr)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        // 嵌套if constexpr避免访问不存在的类型
        if constexpr (std::is_same_v<EnumName, typename MechTrait::VarNames>) {
            *gpu_var_ptr = &(var_access.Arr(var_name, array_index));
        } else {
            // 只有在不是VarNames时才检查GlobalVarNames
            if constexpr (has_GlobalVarNames_v<MechTrait>) {
                if constexpr (std::is_same_v<EnumName, typename MechTrait::GlobalVarNames>) {
                    *gpu_var_ptr = &(var_access.Arr(var_name, array_index));
                } else {
                    // IonVarNames不支持数组
                    *gpu_var_ptr = nullptr;
                }
            } else {
                // 没有GlobalVarNames，必定是IonVarNames
                *gpu_var_ptr = nullptr;
            }
        }
    }
}

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_init_kernel(int nnode, int *node_indices, SimMechInitialParam param, VarAccessor<MechTrait> gpu_vars)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode)
    {
        gpu_vars.idx = i; // 传入的VarAccessor中idx没有被初始化，需要为每一个节点初始化

        auto vec_v = param.v;
        auto dt = param.dt;

        MechTempInitParam init_param;
        init_param.dt = dt;
        init_param.idx = i;
        init_param.volt = vec_v[node_indices[i]];

        Derived::init_single_node(init_param, gpu_vars);
    }
}

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_current_kernel(double t, double *vec_rhs, double *vec_d, int nnode, double *v, int *node_indices, double *area, VarAccessor<MechTrait> gpu_vars)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode)
    {
        gpu_vars.idx = i; // 传入的VarAccessor中idx没有被初始化，需要为每一个节点初始化

        int node_index = node_indices[i];
        double _v = v[node_index];

        MechTempCurParam cur_param;
        cur_param.volt = _v + 0.001;
        cur_param.t = t;
        cur_param.updateIon = false;
        cur_param.idx = i;

        double _g = Derived::current_single_node(cur_param, gpu_vars);

        cur_param.volt = _v;
        cur_param.updateIon = true;
        double _rhs = Derived::current_single_node(cur_param, gpu_vars);

        _g = (_g - _rhs) / 0.001;
        if constexpr (hasFlag(Derived::flags, MechFlags::POINT_PROCESS))
        {
            double nd_area = area[node_index];
            _g *= 1.e2 / nd_area;
            _rhs *= 1.e2 / nd_area;
        }
        if constexpr (hasFlag(Derived::flags, MechFlags::ELECTRODE_CURRENT))
        {
            atomicAdd(&vec_rhs[node_index], _rhs);
            atomicAdd(&vec_d[node_index], -_g);
        }
        else
        {
            atomicAdd(&vec_rhs[node_index], -_rhs);
            atomicAdd(&vec_d[node_index], _g);
        }
    }
}

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_state_kernel(int nnode, double *v, double dt, double t, int *node_indices, VarAccessor<MechTrait> gpu_vars)
{
    if constexpr (hasFlag(Derived::flags, MechFlags::ENABLE_STATE))
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nnode)
        {
            gpu_vars.idx = i; // 传入的VarAccessor中idx没有被初始化，需要为每一个节点初始化

            MechTempStateParam state_param;
            state_param.dt = dt;
            state_param.t = t;
            state_param.idx = i;
            state_param.volt = v[node_indices[i]];
            Derived::state_single_node(state_param, gpu_vars);
        }
    }
}
