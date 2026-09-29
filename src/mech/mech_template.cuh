#pragma once
#include <array>
#include <concepts>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <string>
#include "cuda_utils.h"
#include "autodiff/mech_scalar_tape_store.hpp"
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

    [[no_unique_address]] VarAccessorTableView<MechTrait> table{};

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

struct MechTempCurVJPParam
{
    double volt;
    double t;
    double dt;
    int step_index;
    int idx;
    int node_index;
    // Additive voltage-adjoint accumulator shared by the VJP recurrence.
    // current_vjp_single_node implementations must add, not assign.
    double* grad_v;
    double grad_mech_current;
    // Optional learnable-grad view table.
    // NOTE: current implementation supports scalar learnable vars only (no array learnable vars yet).
    double* const* learnable_grad_data = nullptr;
    int learnable_grad_count = 0;
    // Optional ordinary-variable adjoint carry table. These slots are for
    // intermediate VJP propagation (for example reverse gap-copy routes), not
    // optimizer-visible parameter gradients.
    double* const* vjp_adjoint_data = nullptr;
    int vjp_adjoint_count = 0;
    // Optional per-step scalar tape table. Derived mechanisms can declare
    // CurrentVjpTapeVars and read them via tape_ref<TapeVar>() during backward.
    double* const* current_vjp_tape_data = nullptr;
    int current_vjp_tape_count = 0;
    int current_vjp_tape_step = 0;
    int current_vjp_tape_stride = 0;
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
__global__ void cuda_current_kernel(
    double t,
    double *vec_rhs,
    double *vec_d,
    int nnode,
    double *v,
    int *node_indices,
    double *area,
    VarAccessor<MechTrait> gpu_vars);

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_state_kernel(int nnode, double *v, double dt, double t, int *node_indices, VarAccessor<MechTrait> gpu_vars);
template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_current_vjp_kernel(
    int nnode,
    double* v,
    double* grad_v,
    double* grad_rhs,
    double t,
    double dt,
    int step_index,
    int* node_indices,
    double* area,
    double* const* learnable_grad_data,
    int learnable_grad_count,
    double* const* vjp_adjoint_data,
    int vjp_adjoint_count,
    double* const* current_vjp_tape_data,
    int current_vjp_tape_count,
    int current_vjp_tape_step,
    int current_vjp_tape_stride,
    VarAccessor<MechTrait> gpu_vars);
// GPU用于获取变量指针的内核

template <typename MechTrait, typename EnumName>
__global__ void getVarKernel(VarAccessor<MechTrait> var_access, EnumName var_name, double **gpu_var_ptr);

// NOTE: `mech_lookup_table_runtime.cuh` 依赖 `VarAccessor` 的完整定义，
// 所以必须放在本文件中段（VarAccessor 定义之后）包含。
#include "mech_lookup_table_runtime.cuh"

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

    using TableRuntime = MechTableRuntime<Derived, MechTrait, mech_trait_supports_table_v<MechTrait>>;
    [[no_unique_address]] TableRuntime table_runtime_;

    // Generic learnable parameter gradient storage (optional, enabled when Derived declares LearnableVars).
    // NOTE: scalar learnable vars only for now; array learnable vars are intentionally unsupported.
    mutable std::vector<VecData<double>> learnable_grad_data_;
    mutable std::vector<double*> learnable_grad_ptr_table_;
    mutable VecData<double*> learnable_grad_ptr_table_gpu_{CPU};
    mutable bool learnable_grad_ptr_table_dirty_ = true;
    mutable bool learnable_grad_storage_initialized_ = false;

    // Generic ordinary-variable adjoint carry storage (optional, enabled when
    // Derived declares VjpCarryVars or VjpAdjointVars). This is separate from
    // LearnableVars because these adjoints are consumed by the backward graph
    // (for example by reverse gap-copy), not retained for optimizers by default.
    mutable std::vector<VecData<double>> vjp_adjoint_data_;
    mutable std::vector<double*> vjp_adjoint_ptr_table_;
    mutable VecData<double*> vjp_adjoint_ptr_table_gpu_{CPU};
    mutable bool vjp_adjoint_ptr_table_dirty_ = true;
    mutable bool vjp_adjoint_storage_initialized_ = false;

    // Generic current-VJP tape storage (optional, enabled when Derived declares
    // CurrentVjpTapeVars). Each declared scalar variable is staged once per
    // forward fadvance and replayed by step_index during current_vjp.
    std::vector<neurong::autodiff::MechScalarTapeStore> current_vjp_tape_stores_;
    mutable std::vector<double*> current_vjp_tape_ptr_table_;
    mutable VecData<double*> current_vjp_tape_ptr_table_gpu_{CPU};

    template <typename T = Derived>
    static constexpr bool has_learnable_vars_decl_v = requires {
        T::LearnableVars;
    };

    static constexpr int learnable_var_count_()
    {
        if constexpr (has_learnable_vars_decl_v<Derived>)
        {
            return static_cast<int>(std::tuple_size_v<decltype(Derived::LearnableVars)>);
        }
        return 0;
    }
    template <typename T = Derived>
    static constexpr bool has_vjp_carry_vars_decl_v = requires {
        T::VjpCarryVars;
    };
    template <typename T = Derived>
    static constexpr bool has_vjp_adjoint_vars_decl_v = requires {
        T::VjpAdjointVars;
    };
    template <typename T = Derived>
    static constexpr bool has_current_vjp_tape_vars_decl_v = requires {
        T::CurrentVjpTapeVars;
    };
    static constexpr int vjp_adjoint_var_count_()
    {
        static_assert(
            !(has_vjp_carry_vars_decl_v<Derived> && has_vjp_adjoint_vars_decl_v<Derived>),
            "Declare only one of VjpCarryVars or VjpAdjointVars.");
        if constexpr (has_vjp_carry_vars_decl_v<Derived>)
        {
            return static_cast<int>(std::tuple_size_v<decltype(Derived::VjpCarryVars)>);
        }
        else if constexpr (has_vjp_adjoint_vars_decl_v<Derived>)
        {
            return static_cast<int>(std::tuple_size_v<decltype(Derived::VjpAdjointVars)>);
        }
        return 0;
    }
    static constexpr int current_vjp_tape_var_count_()
    {
        if constexpr (has_current_vjp_tape_vars_decl_v<Derived>)
        {
            return static_cast<int>(std::tuple_size_v<decltype(Derived::CurrentVjpTapeVars)>);
        }
        return 0;
    }
    static constexpr VarNames vjp_adjoint_var_at_(int i)
    {
        if constexpr (has_vjp_carry_vars_decl_v<Derived>)
        {
            return Derived::VjpCarryVars[static_cast<size_t>(i)];
        }
        else
        {
            return Derived::VjpAdjointVars[static_cast<size_t>(i)];
        }
    }
    static constexpr VarNames current_vjp_tape_var_at_(int i)
    {
        if constexpr (has_current_vjp_tape_vars_decl_v<Derived>)
        {
            return Derived::CurrentVjpTapeVars[static_cast<size_t>(i)];
        }
        else
        {
            (void)i;
            return static_cast<VarNames>(0);
        }
    }
    template <auto LearnableVar>
    static consteval int learnable_var_index_constexpr_()
    {
        constexpr int kCount = learnable_var_count_();
        if constexpr (kCount > 0)
        {
            for (int i = 0; i < kCount; ++i)
            {
                if (Derived::LearnableVars[static_cast<size_t>(i)] == LearnableVar)
                {
                    return i;
                }
            }
        }
        return -1;
    }
    template <auto VjpAdjointVar>
    static consteval int vjp_adjoint_var_index_constexpr_()
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount > 0)
        {
            for (int i = 0; i < kCount; ++i)
            {
                if (vjp_adjoint_var_at_(i) == VjpAdjointVar)
                {
                    return i;
                }
            }
        }
        return -1;
    }
    template <auto TapeVar>
    static consteval int current_vjp_tape_var_index_constexpr_()
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount > 0)
        {
            for (int i = 0; i < kCount; ++i)
            {
                if (Derived::CurrentVjpTapeVars[static_cast<size_t>(i)] == TapeVar)
                {
                    return i;
                }
            }
        }
        return -1;
    }
    void ensure_default_learnable_grad_storage_() const
    {
        constexpr int kCount = learnable_var_count_();
        if constexpr (kCount <= 0)
        {
            learnable_grad_data_.clear();
            learnable_grad_ptr_table_.clear();
            learnable_grad_ptr_table_gpu_.clear();
            learnable_grad_ptr_table_dirty_ = true;
            learnable_grad_storage_initialized_ = true;
            return;
        }
        if (learnable_grad_storage_initialized_)
        {
            return;
        }
        if (static_cast<int>(learnable_grad_data_.size()) != kCount)
        {
            learnable_grad_data_.clear();
            learnable_grad_data_.reserve(static_cast<size_t>(kCount));
            for (int k = 0; k < kCount; ++k)
            {
                learnable_grad_data_.emplace_back(this->mode);
            }
            learnable_grad_ptr_table_dirty_ = true;
        }
        for (int k = 0; k < kCount; ++k)
        {
            auto& grad = learnable_grad_data_[static_cast<size_t>(k)];
            if (grad.size() != nnode)
            {
                grad.resize(nnode);
                std::fill_n(grad.get_cpu_data(), static_cast<std::ptrdiff_t>(nnode), 0.0);
                if (this->mode == GPU)
                {
                    grad.update_gpu_data_from_cpu();
                }
                learnable_grad_ptr_table_dirty_ = true;
            }
        }
        learnable_grad_storage_initialized_ = true;
    }
    void ensure_default_vjp_adjoint_storage_() const
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount <= 0)
        {
            vjp_adjoint_data_.clear();
            vjp_adjoint_ptr_table_.clear();
            vjp_adjoint_ptr_table_gpu_.clear();
            vjp_adjoint_ptr_table_dirty_ = true;
            vjp_adjoint_storage_initialized_ = true;
            return;
        }
        if (vjp_adjoint_storage_initialized_)
        {
            return;
        }
        if (static_cast<int>(vjp_adjoint_data_.size()) != kCount)
        {
            vjp_adjoint_data_.clear();
            vjp_adjoint_data_.reserve(static_cast<size_t>(kCount));
            for (int k = 0; k < kCount; ++k)
            {
                vjp_adjoint_data_.emplace_back(this->mode);
            }
            vjp_adjoint_ptr_table_dirty_ = true;
        }
        for (int k = 0; k < kCount; ++k)
        {
            auto& adj = vjp_adjoint_data_[static_cast<size_t>(k)];
            if (adj.size() != nnode)
            {
                adj.resize(nnode);
                std::fill_n(adj.get_cpu_data(), static_cast<std::ptrdiff_t>(nnode), 0.0);
                if (this->mode == GPU)
                {
                    adj.update_gpu_data_from_cpu();
                }
                vjp_adjoint_ptr_table_dirty_ = true;
            }
        }
        vjp_adjoint_storage_initialized_ = true;
    }
    void refresh_default_learnable_grad_pointer_table_() const
    {
        constexpr int kCount = learnable_var_count_();
        if constexpr (kCount <= 0)
        {
            return;
        }
        ensure_default_learnable_grad_storage_();
        if (!learnable_grad_ptr_table_dirty_)
        {
            return;
        }
        learnable_grad_ptr_table_.resize(static_cast<size_t>(kCount));
        learnable_grad_ptr_table_gpu_.resize(kCount);
        double** ptr_gpu_host = learnable_grad_ptr_table_gpu_.get_cpu_data();
        for (int k = 0; k < kCount; ++k)
        {
            learnable_grad_ptr_table_[static_cast<size_t>(k)] =
                learnable_grad_data_[static_cast<size_t>(k)].get_cpu_data();
            ptr_gpu_host[k] = (this->mode == GPU)
                ? learnable_grad_data_[static_cast<size_t>(k)].get_gpu_data()
                : learnable_grad_data_[static_cast<size_t>(k)].get_cpu_data();
        }
        if (this->mode == GPU)
        {
            learnable_grad_ptr_table_gpu_.update_gpu_data_from_cpu();
        }
        learnable_grad_ptr_table_dirty_ = false;
    }
    void refresh_default_vjp_adjoint_pointer_table_() const
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount <= 0)
        {
            return;
        }
        ensure_default_vjp_adjoint_storage_();
        if (!vjp_adjoint_ptr_table_dirty_)
        {
            return;
        }
        vjp_adjoint_ptr_table_.resize(static_cast<size_t>(kCount));
        vjp_adjoint_ptr_table_gpu_.resize(kCount);
        double** ptr_gpu_host = vjp_adjoint_ptr_table_gpu_.get_cpu_data();
        for (int k = 0; k < kCount; ++k)
        {
            vjp_adjoint_ptr_table_[static_cast<size_t>(k)] =
                vjp_adjoint_data_[static_cast<size_t>(k)].get_cpu_data();
            ptr_gpu_host[k] = (this->mode == GPU)
                ? vjp_adjoint_data_[static_cast<size_t>(k)].get_gpu_data()
                : vjp_adjoint_data_[static_cast<size_t>(k)].get_cpu_data();
        }
        if (this->mode == GPU)
        {
            vjp_adjoint_ptr_table_gpu_.update_gpu_data_from_cpu();
        }
        vjp_adjoint_ptr_table_dirty_ = false;
    }
    void ensure_current_vjp_tape_stores_()
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount <= 0)
        {
            current_vjp_tape_stores_.clear();
            current_vjp_tape_ptr_table_.clear();
            current_vjp_tape_ptr_table_gpu_.clear();
            return;
        }
        else
        {
            if (static_cast<int>(current_vjp_tape_stores_.size()) != kCount)
            {
                current_vjp_tape_stores_.clear();
                current_vjp_tape_stores_.reserve(static_cast<size_t>(kCount));
                for (int k = 0; k < kCount; ++k)
                {
                    current_vjp_tape_stores_.emplace_back(this->mode);
                }
            }
            for (int k = 0; k < kCount; ++k)
            {
                const VarNames tape_var = current_vjp_tape_var_at_(k);
                auto* tape_vec = var_struct[tape_var];
                current_vjp_tape_stores_[static_cast<size_t>(k)].set_nnode(this->nnode);
                current_vjp_tape_stores_[static_cast<size_t>(k)].set_spill_chunk_steps(64);
                current_vjp_tape_stores_[static_cast<size_t>(k)].set_source(
                    tape_vec->get_cpu_data(),
                    this->mode == GPU ? tape_vec->get_gpu_data() : tape_vec->get_cpu_data());
            }
        }
    }

    void bind_current_vjp_tape_cpu_(MechTempCurVJPParam& vjp_param, int step_index)
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount <= 0)
        {
            (void)vjp_param;
            (void)step_index;
        }
        else
        {
            if (static_cast<int>(current_vjp_tape_stores_.size()) != kCount)
            {
                throw std::runtime_error("current_vjp_cpu: current VJP tape stores are not initialized");
            }
            current_vjp_tape_ptr_table_.resize(static_cast<size_t>(kCount));
            for (int k = 0; k < kCount; ++k)
            {
                auto& store = current_vjp_tape_stores_[static_cast<size_t>(k)];
                const double* tape = store.tape_cpu_data_for_backward();
                if (tape == nullptr)
                {
                    throw std::runtime_error("current_vjp_cpu: empty current VJP tape");
                }
                if (step_index < 0 || step_index >= store.tape_steps())
                {
                    throw std::out_of_range("current_vjp_cpu: current VJP tape step out of range");
                }
                current_vjp_tape_ptr_table_[static_cast<size_t>(k)] = const_cast<double*>(tape);
            }
            vjp_param.current_vjp_tape_data = current_vjp_tape_ptr_table_.data();
            vjp_param.current_vjp_tape_count = kCount;
            vjp_param.current_vjp_tape_step = step_index;
            vjp_param.current_vjp_tape_stride = this->nnode;
        }
    }

    double* const* bind_current_vjp_tape_gpu_(int step_index, int& local_step, int& stride)
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount <= 0)
        {
            (void)step_index;
            local_step = 0;
            stride = 0;
            return nullptr;
        }
        else
        {
            if (static_cast<int>(current_vjp_tape_stores_.size()) != kCount)
            {
                throw std::runtime_error("current_vjp_gpu: current VJP tape stores are not initialized");
            }
            current_vjp_tape_ptr_table_gpu_.resize(kCount);
            double** ptr_gpu_host = current_vjp_tape_ptr_table_gpu_.get_cpu_data();
            local_step = -1;
            for (int k = 0; k < kCount; ++k)
            {
                auto replay = current_vjp_tape_stores_[static_cast<size_t>(k)].acquire_gpu_replay_view(step_index);
                if (local_step < 0)
                {
                    local_step = replay.local_step_index;
                }
                else if (local_step != replay.local_step_index)
                {
                    throw std::runtime_error("current_vjp_gpu: inconsistent current VJP tape replay chunk");
                }
                ptr_gpu_host[k] = replay.tape_gpu;
            }
            current_vjp_tape_ptr_table_gpu_.update_gpu_data_from_cpu();
            stride = this->nnode;
            return current_vjp_tape_ptr_table_gpu_.get_gpu_data();
        }
    }
public:
    template <auto LearnableVar>
    DUAL_EXEC double& grad_ref(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars)
    {
        constexpr int slot = learnable_var_index_constexpr_<LearnableVar>();
        static_assert(slot >= 0, "grad_ref<LearnableVar>: LearnableVar is not in LearnableVars");
        return param.learnable_grad_data[slot][vars.idx];
    }

    template <auto VjpAdjointVar>
    DUAL_EXEC double& adjoint_ref(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars)
    {
        constexpr int slot = vjp_adjoint_var_index_constexpr_<VjpAdjointVar>();
        static_assert(slot >= 0, "adjoint_ref<VjpAdjointVar>: var is not in VjpCarryVars/VjpAdjointVars");
        return param.vjp_adjoint_data[slot][vars.idx];
    }

    template <auto TapeVar>
    DUAL_EXEC double tape_ref(MechTempCurVJPParam& param, VarAccessor<MechTrait>& vars)
    {
        constexpr int slot = current_vjp_tape_var_index_constexpr_<TapeVar>();
        static_assert(slot >= 0, "tape_ref<TapeVar>: TapeVar is not in CurrentVjpTapeVars");
        return param.current_vjp_tape_data[slot][
            param.current_vjp_tape_step * param.current_vjp_tape_stride + vars.idx];
    }

    bool has_learnable_params() const override
    {
        constexpr int kCount = learnable_var_count_();
        return kCount > 0;
    }
    double* getLearnableGradPtr(const std::string& param_name, Mode mode) override
    {
        constexpr int kCount = learnable_var_count_();
        if constexpr (kCount <= 0)
        {
            (void)param_name;
            (void)mode;
            return nullptr;
        }
        else
        {
            // NOTE: scalar learnable vars only for now; array learnable vars are not supported.
            ensure_default_learnable_grad_storage_();
            refresh_default_learnable_grad_pointer_table_();
            for (int k = 0; k < kCount; ++k)
            {
                if (param_name == std::string(magic_enum::enum_name(Derived::LearnableVars[static_cast<size_t>(k)])))
                {
                    return (mode == GPU)
                        ? learnable_grad_data_[static_cast<size_t>(k)].get_gpu_data()
                        : learnable_grad_data_[static_cast<size_t>(k)].get_cpu_data();
                }
            }
            return nullptr;
        }
    }
    std::vector<std::string> listLearnableGradNames() const override
    {
        constexpr int kCount = learnable_var_count_();
        if constexpr (kCount <= 0)
        {
            return {};
        }
        else
        {
            std::vector<std::string> out;
            out.reserve(static_cast<size_t>(kCount));
            for (int k = 0; k < kCount; ++k)
            {
                out.emplace_back(std::string(magic_enum::enum_name(Derived::LearnableVars[static_cast<size_t>(k)])));
            }
            return out;
        }
    }
    bool has_vjp_adjoint_vars() const override
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        return kCount > 0;
    }
    double* getVjpAdjointPtr(const VarDescriptor& descriptor, Mode mode) override
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount <= 0)
        {
            (void)descriptor;
            (void)mode;
            return nullptr;
        }
        else
        {
            if (descriptor.array_index > 0)
            {
                return nullptr;
            }
            int mech_idx = descriptor.node_or_mech_idx;
            if (mech_idx < 0 || mech_idx >= nnode)
            {
                return nullptr;
            }
            if (permute != nullptr)
            {
                mech_idx = permute[mech_idx];
            }
            if (mech_idx < 0 || mech_idx >= nnode)
            {
                return nullptr;
            }
            auto maybe_var = magic_enum::enum_cast<VarNames>(descriptor.var);
            if (!maybe_var.has_value())
            {
                return nullptr;
            }
            ensure_default_vjp_adjoint_storage_();
            refresh_default_vjp_adjoint_pointer_table_();
            for (int k = 0; k < kCount; ++k)
            {
                if (maybe_var.value() == vjp_adjoint_var_at_(k))
                {
                    return ((mode == GPU)
                        ? vjp_adjoint_data_[static_cast<size_t>(k)].get_gpu_data()
                        : vjp_adjoint_data_[static_cast<size_t>(k)].get_cpu_data()) + mech_idx;
                }
            }
            return nullptr;
        }
    }
    std::vector<std::string> listVjpAdjointNames() const override
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount <= 0)
        {
            return {};
        }
        else
        {
            std::vector<std::string> out;
            out.reserve(static_cast<size_t>(kCount));
            for (int k = 0; k < kCount; ++k)
            {
                out.emplace_back(std::string(magic_enum::enum_name(vjp_adjoint_var_at_(k))));
            }
            return out;
        }
    }
    void zeroVjpAdjoints() override
    {
        constexpr int kCount = vjp_adjoint_var_count_();
        if constexpr (kCount <= 0)
        {
            return;
        }
        ensure_default_vjp_adjoint_storage_();
        for (int k = 0; k < kCount; ++k)
        {
            auto& adj = vjp_adjoint_data_[static_cast<size_t>(k)];
            std::fill_n(adj.get_cpu_data(), static_cast<std::ptrdiff_t>(adj.size()), 0.0);
            if (this->mode == GPU)
            {
                adj.update_gpu_data_from_cpu();
            }
        }
    }

public:
    using enum MechFlags;
    constexpr static MechFlags flags = Derived::flags;

    // 构造函数，应当初始化init_values,这样reg_node_indices的时候可以初始化vecdata_vars
    // 并且应当初始化var_in_coredata_idx，这样read_data_from_coredat的时候可以初始化vecdata_vars
    MechTemp(MechInitParams &param)
        : Mechanism(param),
          learnable_grad_ptr_table_gpu_(param.mode),
          vjp_adjoint_ptr_table_gpu_(param.mode),
          current_vjp_tape_ptr_table_gpu_(param.mode)
    {
        // Safety contract:
        // If a mechanism updates STATE variables (ENABLE_STATE), then it must also participate
        // in finitialize() (ENABLE_INIT) so the state can be reset/reinitialized each run.
        static_assert(!hasFlag(MechFlags::ENABLE_STATE) || hasFlag(MechFlags::ENABLE_INIT),
                      "Mechanism has ENABLE_STATE but not ENABLE_INIT: state will not be reset on finitialize(), "
                      "leading to multi-run divergence when reusing the same simulator instance.");
        if constexpr (mech_trait_supports_table_v<MechTrait>)
        {
            static_assert(requires { typename Derived::TableSpec; },
                          "MechTrait::MechSupportsTable is true, but Derived::TableSpec is missing.");
        }
        else
        {
            static_assert(!requires { typename Derived::TableSpec; },
                          "Derived::TableSpec exists, but MechTrait::MechSupportsTable is false.");
        }

        if constexpr (hasFlag(MechFlags::POINT_PROCESS))
        {
            need_area = true;
        }
        if constexpr (hasFlag(MechFlags::WRITE_EION_IN_STATE))
        {
            write_state_ion = true;
        }
        if constexpr (hasFlag(MechFlags::ENABLE_CURRENT_VJP))
        {
            static_assert(requires(Derived& d, MechTempCurVJPParam &p, VarAccessor<MechTrait> vars) {
                              d.current_vjp_single_node(p, vars);
                          },
                          "MechFlags::ENABLE_CURRENT_VJP is set, but current_vjp_single_node is missing.");
        }
    }

    constexpr static bool hasFlag(MechFlags flag)
    {
        return static_cast<int>(flags) & static_cast<int>(flag);
    }

    bool supports_current_vjp() const override
    {
        return hasFlag(MechFlags::ENABLE_CURRENT_VJP);
    }

    bool supports_current_vjp_tape() const override
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        return kCount > 0;
    }

    void clear_current_vjp_tape() override
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount > 0)
        {
            for (auto& store : current_vjp_tape_stores_)
            {
                store.clear();
            }
        }
    }

    void stage_current_vjp_tape_cpu(SimMechCurrentVJPTapeParam& param) override
    {
        (void)param;
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount > 0)
        {
            for (auto& store : current_vjp_tape_stores_)
            {
                store.stage_step();
            }
        }
    }

    void stage_current_vjp_tape_gpu(SimMechCurrentVJPTapeParam& param) override
    {
        (void)param;
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount > 0)
        {
            for (auto& store : current_vjp_tape_stores_)
            {
                store.stage_step(this->cuda_stream);
            }
        }
    }

    void finalize_current_vjp_tape_for_backward() override
    {
        constexpr int kCount = current_vjp_tape_var_count_();
        if constexpr (kCount > 0)
        {
            for (auto& store : current_vjp_tape_stores_)
            {
                store.finalize_tape_for_backward();
            }
        }
    }






    // 注册节点索引
    virtual void reg_node_indices(MechInitParams &param) override
    {
        auto node_count = param.node_count;
        assert(node_count > 0);
        printf_debug("Mech[%s] reg_node_indices node_count:%d\n", name.c_str(), node_count);

        var_struct.init(param, var_in_coredata_idx, init_values, global_info_map);
        var_struct.initPdata(param);
        if constexpr (has_IonVarNames_v<MechTrait>)
        {
            if (param.ion_name_overrides && !param.ion_name_overrides->empty())
            {
                const auto &overrides = *param.ion_name_overrides;
                auto wildcard_it = overrides.find("*");
                const bool has_wildcard = (wildcard_it != overrides.end());
                const std::string wildcard_value = has_wildcard ? wildcard_it->second : std::string{};

                std::unordered_set<std::string> unique_bases;
                unique_bases.reserve(ion_var_map.size());
                for (const auto &[var_name, eion_info] : ion_var_map)
                {
                    const auto &ion_name = std::get<0>(eion_info);
                    std::string base = ion_name;
                    if (base.size() > 4 && base.rfind("_ion") == base.size() - 4)
                    {
                        base = base.substr(0, base.size() - 4);
                    }
                    unique_bases.insert(base);
                }

                if (has_wildcard && unique_bases.size() != 1)
                {
                    throw std::runtime_error(
                        "ion override requires explicit mapping for mechanisms with multiple ions");
                }

                for (auto &[var_name, eion_info] : ion_var_map)
                {
                    auto &ion_name = std::get<0>(eion_info);
                    std::string base = ion_name;
                    if (base.size() > 4 && base.rfind("_ion") == base.size() - 4)
                    {
                        base = base.substr(0, base.size() - 4);
                    }

                    auto it = overrides.find(base);
                    if (it != overrides.end())
                    {
                        ion_name = it->second;
                    }
                    else if (has_wildcard)
                    {
                        ion_name = wildcard_value;
                    }
                }
            }
        }
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
        // One-time setup for learnable parameter grad buffers/pointer table.
        // These structures are static for a mechanism instance (no dynamic var add/remove),
        // so they should not be rebuilt in per-step VJP hot paths.
        ensure_default_learnable_grad_storage_();
        refresh_default_learnable_grad_pointer_table_();
        ensure_default_vjp_adjoint_storage_();
        refresh_default_vjp_adjoint_pointer_table_();
        ensure_current_vjp_tape_stores_();
    }




    // 从 CoreNeuron 读取数据
    virtual void read_data_from_coredat(MechInitParams &param) override
    {
        auto nnode = param.node_count;
        auto data = param.data;
        auto param_size = param.data_size;
        const bool has_coredat_payload = (data != nullptr && param_size > 0);

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

        // Optional: read per-node RANGE/STATE values from CoreNEURON-style row data.
        // In the in-memory builder, we may skip this entirely (data==nullptr / data_size==0)
        // and keep `init_values` defaults; range assignments can then be applied directly
        // via `mech_var_table` without fabricating bbcore rows.
        if (has_coredat_payload)
        {
            // 从coredat中读取数据
            vector<pair<double *, CoreIdxInfo>> data_init_list;
            for (auto [var_name, var_idx] : var_in_coredata_idx)
            {
                VecData<double> *var = var_struct[var_name];
                double *cpu_data_ptr = var->get_cpu_data();
                data_init_list.push_back({cpu_data_ptr, var_idx});
            }

            // 计算前缀和
            if (param.array_dims != nullptr)
            {
                vector<int> prefix_sum(param.array_dims->size());
                std::exclusive_scan(param.array_dims->begin(), param.array_dims->end(), prefix_sum.begin(), 0);
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
                            const double src = data[begin_idx + i];
                            if (!std::isnan(src))
                            {
                                data_ptr[inode * var_idx.array_size + i] = src;
                            }
                        }
                    }
                }
            }
            else
            {
                for (auto &[data_ptr, var_idx] : data_init_list)
                {
                    assert(!var_idx.isArray());
                }
                for (int inode = 0; inode < nnode; inode++)
                {
                    int offset = inode * param_size;
                    for (auto &[data_ptr, var_idx] : data_init_list)
                    {
                        const double src = data[offset + var_idx.info];
                        if (!std::isnan(src))
                        {
                            data_ptr[inode] = src;
                        }
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
        }

        // 注册varTable
        auto &varMap = mech_var_table[param.type];
        for (auto [var_name, var_idx] : var_in_coredata_idx)
        {
            MechVarData varData;
            varData.name = this->name + "_" + string(magic_enum::enum_name(var_name));
            varData.len = var_struct[var_name]->size();
            varData.cpu_data = var_struct[var_name]->get_cpu_data();
            varData.vecdata = var_struct[var_name];
            if (param.mode == Mode::GPU)
            {
                varData.gpu_data = var_struct[var_name]->get_gpu_data();
            }
            varMap[var_idx.info] = varData;
        }
    }

    virtual void resolve_pointers(const MechPointerResolveContext& context) override
    {
        auto* ndat = static_cast<NeuronGroupData*>(context.neuron_group_data);
        auto* cdat = static_cast<coreneuron::CoreData*>(context.core_data);
        if constexpr (!has_PointerVarNames_v<MechTrait>)
        {
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
            if constexpr (mech_trait_supports_table_v<MechTrait>)
            {
                table_runtime_.initialize_once(*this);
                table_runtime_.rebuild_if_needed(*this, param.dt);
            }
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
            if constexpr (mech_trait_supports_table_v<MechTrait>)
            {
                table_runtime_.initialize_once(*this);
                table_runtime_.rebuild_if_needed(*this, param.dt);
            }
            int *node_indices = vecdata_node_indices->get_gpu_data();
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);

            VarAccessor<MechTrait> gpu_vars = getGpuVarAccessor();

            cuda_init_kernel<Derived><<<block_num, nthread_per_block, 0, stream>>>(nnode, node_indices, param, gpu_vars);
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
            cuda_current_kernel<Derived><<<block_num, nthread_per_block, 0, stream>>>(
                t,
                vec_rhs,
                vec_d,
                nnode,
                v,
                node_indices,
                area,
                gpu_vars);
        }
    }

    void current_vjp_cpu(SimMechCurrentVJPParam& param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_CURRENT))
        {
            if constexpr (requires(Derived& d, MechTempCurVJPParam &vjp_param, VarAccessor<MechTrait> vars) {
                              d.current_vjp_single_node(vjp_param, vars);
                          })
            {
                int* node_indices = this->vecdata_node_indices->get_cpu_data();
                double* nd_area_vec = nullptr;
                if constexpr (hasFlag(MechFlags::POINT_PROCESS))
                {
                    nd_area_vec = this->vecdata_area->get_cpu_data();
                }

                VarAccessor<MechTrait> cpu_vars = getCpuVarAccessor();
                ensure_default_learnable_grad_storage_();
                refresh_default_learnable_grad_pointer_table_();
                ensure_default_vjp_adjoint_storage_();
                refresh_default_vjp_adjoint_pointer_table_();
                MechTempCurVJPParam tape_param{};
                bind_current_vjp_tape_cpu_(tape_param, param.step_index);
                for (int i = 0; i < nnode; ++i)
                {
                    cpu_vars.idx = i;
                    const int node_index = node_indices[i];

                    const bool is_point_process = hasFlag(MechFlags::POINT_PROCESS);
                    const bool is_electrode_current = hasFlag(MechFlags::ELECTRODE_CURRENT);
                    const double nd_area = is_point_process ? nd_area_vec[node_index] : 1.0;

                    MechTempCurVJPParam vjp_param{};
                    vjp_param.volt = param.v[node_index];
                    vjp_param.t = param.t;
                    vjp_param.dt = param.dt;
                    vjp_param.step_index = param.step_index;
                    vjp_param.idx = i;
                    vjp_param.node_index = node_index;
                    vjp_param.grad_v = param.grad_v;
                    vjp_param.grad_mech_current = mech_grad_current_from_rhs(
                        param.grad_rhs[node_index],
                        is_point_process,
                        is_electrode_current,
                        nd_area);
                    vjp_param.learnable_grad_data = learnable_grad_ptr_table_.data();
                    vjp_param.learnable_grad_count = static_cast<int>(learnable_grad_ptr_table_.size());
                    vjp_param.vjp_adjoint_data = vjp_adjoint_ptr_table_.data();
                    vjp_param.vjp_adjoint_count = static_cast<int>(vjp_adjoint_ptr_table_.size());
                    vjp_param.current_vjp_tape_data = tape_param.current_vjp_tape_data;
                    vjp_param.current_vjp_tape_count = tape_param.current_vjp_tape_count;
                    vjp_param.current_vjp_tape_step = tape_param.current_vjp_tape_step;
                    vjp_param.current_vjp_tape_stride = tape_param.current_vjp_tape_stride;

                    static_cast<Derived*>(this)->current_vjp_single_node(vjp_param, cpu_vars);
                }
            }
            else
            {
                (void)param;
            }
        }
        else
        {
            (void)param;
        }
    }

    void current_vjp_gpu(SimMechCurrentVJPParam& param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_CURRENT))
        {
            if constexpr (requires(MechTempCurVJPParam &vjp_param, VarAccessor<MechTrait> vars) {
                              Derived::current_vjp_single_node(vjp_param, vars);
                          })
            {
                int* node_indices = this->vecdata_node_indices->get_gpu_data();
                double* area = nullptr;
                if constexpr (hasFlag(MechFlags::POINT_PROCESS))
                {
                    area = this->vecdata_area->get_gpu_data();
                }
                int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
                cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);
                VarAccessor<MechTrait> gpu_vars = getGpuVarAccessor();
                ensure_default_learnable_grad_storage_();
                refresh_default_learnable_grad_pointer_table_();
                ensure_default_vjp_adjoint_storage_();
                refresh_default_vjp_adjoint_pointer_table_();
                int current_vjp_tape_step = 0;
                int current_vjp_tape_stride = 0;
                double* const* current_vjp_tape_data =
                    bind_current_vjp_tape_gpu_(param.step_index, current_vjp_tape_step, current_vjp_tape_stride);
                cuda_current_vjp_kernel<Derived><<<block_num, nthread_per_block, 0, stream>>>(
                    nnode,
                    param.v,
                    param.grad_v,
                    param.grad_rhs,
                    param.t,
                    param.dt,
                    param.step_index,
                    node_indices,
                    area,
                    learnable_grad_ptr_table_gpu_.get_gpu_data(),
                    static_cast<int>(learnable_grad_ptr_table_.size()),
                    vjp_adjoint_ptr_table_gpu_.get_gpu_data(),
                    static_cast<int>(vjp_adjoint_ptr_table_.size()),
                    current_vjp_tape_data,
                    current_vjp_tape_var_count_(),
                    current_vjp_tape_step,
                    current_vjp_tape_stride,
                    gpu_vars);
            }
            else
            {
                (void)param;
            }
        }
        else
        {
            (void)param;
        }
    }

    // CPU 状态更新
    virtual void state_cpu(SimMechStateParam &param) override
    {
        if constexpr (hasFlag(MechFlags::ENABLE_STATE))
        {
            if constexpr (mech_trait_supports_table_v<MechTrait>)
            {
                table_runtime_.rebuild_if_needed(*this, param.dt);
            }
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
            if constexpr (mech_trait_supports_table_v<MechTrait>)
            {
                table_runtime_.rebuild_if_needed(*this, param.dt);
            }
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
        if constexpr (mech_trait_supports_table_v<MechTrait>)
        {
            table_runtime_.bind_cpu(*this, var_access);
        }
        return var_access;
    }
    VarAccessor<MechTrait> __forceinline__ getGpuVarAccessor(int idx = -1)
    {
        VarAccessor<MechTrait> var_access;
        var_access.idx = idx;
        var_access.dev_var = var_struct.gpu_dev_var;
        var_access.dev_global_var = var_struct.gpu_dev_global_var;
        var_access.dev_ion_var = var_struct.gpu_dev_ion_var;
        if constexpr (mech_trait_supports_table_v<MechTrait>)
        {
            table_runtime_.bind_gpu(*this, var_access);
        }
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
__global__ void cuda_current_kernel(
    double t,
    double *vec_rhs,
    double *vec_d,
    int nnode,
    double *v,
    int *node_indices,
    double *area,
    VarAccessor<MechTrait> gpu_vars)
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

template <typename Derived, MechTraitType MechTrait>
__global__ void cuda_current_vjp_kernel(
    int nnode,
    double* v,
    double* grad_v,
    double* grad_rhs,
    double t,
    double dt,
    int step_index,
    int* node_indices,
    double* area,
    double* const* learnable_grad_data,
    int learnable_grad_count,
    double* const* vjp_adjoint_data,
    int vjp_adjoint_count,
    double* const* current_vjp_tape_data,
    int current_vjp_tape_count,
    int current_vjp_tape_step,
    int current_vjp_tape_stride,
    VarAccessor<MechTrait> gpu_vars)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode)
    {
        gpu_vars.idx = i;
        const int node_index = node_indices[i];
        const bool is_point_process = hasFlag(Derived::flags, MechFlags::POINT_PROCESS);
        const bool is_electrode_current = hasFlag(Derived::flags, MechFlags::ELECTRODE_CURRENT);
        const double nd_area = is_point_process ? area[node_index] : 1.0;

        MechTempCurVJPParam vjp_param{};
        vjp_param.volt = v[node_index];
        vjp_param.t = t;
        vjp_param.dt = dt;
        vjp_param.step_index = step_index;
        vjp_param.idx = i;
        vjp_param.node_index = node_index;
        vjp_param.grad_v = grad_v;
        vjp_param.grad_mech_current = mech_grad_current_from_rhs(
            grad_rhs[node_index],
            is_point_process,
            is_electrode_current,
            nd_area);
        vjp_param.learnable_grad_data = learnable_grad_data;
        vjp_param.learnable_grad_count = learnable_grad_count;
        vjp_param.vjp_adjoint_data = vjp_adjoint_data;
        vjp_param.vjp_adjoint_count = vjp_adjoint_count;
        vjp_param.current_vjp_tape_data = current_vjp_tape_data;
        vjp_param.current_vjp_tape_count = current_vjp_tape_count;
        vjp_param.current_vjp_tape_step = current_vjp_tape_step;
        vjp_param.current_vjp_tape_stride = current_vjp_tape_stride;
        Derived::current_vjp_single_node(vjp_param, gpu_vars);
    }
}
