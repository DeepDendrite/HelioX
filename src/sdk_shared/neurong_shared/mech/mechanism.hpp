#pragma once

#include <memory>
#include <string>

#include "neurong_shared/mech/types.hpp"
#include "neurong_shared/mech/vecdata.hpp"

class VarMapAble {
public:
    virtual ~VarMapAble() = default;
    virtual double* getVarPtr(const VarDescriptor& descriptor, Mode mode) {
        (void)descriptor;
        (void)mode;
        return nullptr;
    }
    virtual double* getVjpAdjointPtr(const VarDescriptor& descriptor, Mode mode) {
        (void)descriptor;
        (void)mode;
        return nullptr;
    }
};

struct SimMechInitialParam {
    double* v;
    double dt;
};

struct SimMechCurrentParam {
    double* v;
    double* d;
    double* rhs;
    double t;
};

struct SimMechCurrentVJPParam {
    double* v;
    // Additive voltage-adjoint accumulator. Mechanisms must add their
    // feedback contribution here (for example via mechAtomAdd) and must not
    // overwrite existing values.
    double* grad_v;
    double* grad_rhs;
    double t;
    double dt;
    int step_index;
};

struct SimMechStateParam {
    double* v;
    double dt;
    double t;
};

struct SimMechCurrentVJPTapeParam {
    double* v;
    double dt;
    double t;
};

struct SimPostSynSpikeVJPParam {
    // Dense source-side spike adjoint tape indexed as [forward_step][pre_index].
    // Indices are source spike-vector indices; the first pre_spike_adj_count
    // entries correspond to real PreSyn sources in the current experimental path.
    double* pending_pre_spike_adj;
    int pre_spike_adj_count;
    int step_count;
    double t;
    double dt;
    int step_index;
};



struct MechPointerResolveContext {
    void* neuron_group_data = nullptr;
    void* core_data = nullptr;
};

class Mechanism : public VarMapAble {
public:
    int type;
    std::string name;
    bool need_area;
    bool write_state_ion;
    Mode mode;
    int* permute = nullptr;

    VecData<double>* vecdata_area = nullptr;

    explicit Mechanism(MechInitParams& param);
    virtual ~Mechanism();

    virtual void cleanUp();
    virtual void reg_node_indices(MechInitParams& param) = 0;
    virtual void read_data_from_coredat(MechInitParams& param) = 0;
    virtual void initialize_cpu(SimMechInitialParam& param) = 0;
    virtual void initialize_gpu(SimMechInitialParam& param) = 0;
    virtual void current_cpu(SimMechCurrentParam& param) = 0;
    virtual void current_gpu(SimMechCurrentParam& param) = 0;
    virtual void current_vjp_cpu(SimMechCurrentVJPParam& param) {
        (void)param;
    }
    virtual void current_vjp_gpu(SimMechCurrentVJPParam& param) {
        (void)param;
    }
    virtual bool supports_current_vjp() const { return false; }
    virtual bool supports_current_vjp_tape() const { return false; }
    virtual void clear_current_vjp_tape() {}
    virtual void stage_current_vjp_tape_cpu(SimMechCurrentVJPTapeParam& param) {
        (void)param;
    }
    virtual void stage_current_vjp_tape_gpu(SimMechCurrentVJPTapeParam& param) {
        (void)param;
    }
    virtual void finalize_current_vjp_tape_for_backward() {}
    virtual bool has_learnable_params() const { return false; }
    virtual double* getLearnableGradPtr(const std::string& param_name, Mode mode) {
        (void)param_name;
        (void)mode;
        return nullptr;
    }
    virtual std::vector<std::string> listLearnableGradNames() const { return {}; }
    virtual bool has_vjp_adjoint_vars() const { return false; }
    virtual std::vector<std::string> listVjpAdjointNames() const { return {}; }
    virtual void zeroVjpAdjoints() {}
    virtual void sync_gpu() = 0;
    virtual void state_cpu(SimMechStateParam& param) = 0;
    virtual void state_gpu(SimMechStateParam& param) = 0;
    virtual void resolve_pointers(const MechPointerResolveContext& context) {
        (void)context;
    }

    int node_count() const { return nnode; }
    int* node_indices_cpu_data() {
        return vecdata_node_indices ? vecdata_node_indices->get_cpu_data() : nullptr;
    }

    int nnode;
    std::unique_ptr<VecData<int>> vecdata_node_indices;
    std::unique_ptr<VecData<double>> vecdata_g_mech;
    std::unique_ptr<VecData<double>> vecdata_i_mech;

    double celsius;
    void* cuda_stream = nullptr;
    void* cuda_event = nullptr;
};
