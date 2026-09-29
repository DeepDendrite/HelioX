#include "neurong_sdk/plugin_auto_init.hpp"
#include "neurong_sdk/plugin_mechanism_api.hpp"

namespace {

class SdkDemoNoopMech final : public Mechanism {
public:
    explicit SdkDemoNoopMech(MechInitParams& param) : Mechanism(param) {
        need_area = false;
        write_state_ion = false;
    }

    void reg_node_indices(MechInitParams& param) override {
        (void)param;
    }

    void read_data_from_coredat(MechInitParams& param) override {
        (void)param;
    }

    void initialize_cpu(SimMechInitialParam& param) override {
        (void)param;
    }

    void initialize_gpu(SimMechInitialParam& param) override {
        (void)param;
    }

    void current_cpu(SimMechCurrentParam& param) override {
        (void)param;
    }

    void current_gpu(SimMechCurrentParam& param) override {
        (void)param;
    }

    void sync_gpu() override {}

    void state_cpu(SimMechStateParam& param) override {
        (void)param;
    }

    void state_gpu(SimMechStateParam& param) override {
        (void)param;
    }
};

}  // namespace

REGISTER_MECHANISM("sdk_demo_noop", SdkDemoNoopMech);

NEURONG_DEFINE_MECH_PACKAGE_PLUGIN("minimal_mech_plugin", "0.1.0");
