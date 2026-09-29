#pragma once
#include "mech_template.cuh"
#include "units.h"
#include "ion_table.h"
#include "mech_var_table.h"
#include <cassert>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <vector>

// VarDescriptor通过mech_template.cuh -> mechanism.h -> utils.h 间接包含

namespace eiontemp
{
    // 通用辅助函数
    DUAL_EXEC constexpr double ktf(double celsius)
    {
        return 1000. * units::gasconstant * (celsius + 273.15) / units::faraday;
    }

    DUAL_EXEC constexpr double nrn_nernst(double ci, double co, double z, double celsius)
    {
        if (z == 0)
        {
            return 0.;
        }
        if (ci <= 0.)
        {
            return 1e6;
        }
        else if (co <= 0.)
        {
            return -1e6;
        }
        else
        {
            return ktf(celsius) / z * std::log(co / ci);
        }
    }

    __global__ void Eion_Init_Kernel(int nnode,
                                    double celsius,
                                    double charge,
                                    double default_conci,
                                    double default_conco,
                                    DevVarStruct gpu_vars)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nnode)
        {
            using enum EionVarNames;
            int iontype = gpu_vars.pdata[i];
            if (iontype & 04)
            {
                gpu_vars[(size_t)conci][i] = default_conci;
                gpu_vars[(size_t)conco][i] = default_conco;
            }
            if (iontype & 040)
            {
                gpu_vars[(size_t)erev][i] = nrn_nernst(gpu_vars[(size_t)conci][i],
                                                      gpu_vars[(size_t)conco][i],
                                                      charge,
                                                      celsius);
            }
        }
    }

    __global__ void Eion_Cur_Kernel(int nnode, double celsius, double charge, DevVarStruct gpu_vars)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nnode)
        {
            using enum EionVarNames;
            int iontype = gpu_vars.pdata[i];
            gpu_vars[(size_t)cur][i] = 0.;
            gpu_vars[(size_t)dcurdv][i] = 0.;
            if (iontype & 0100)
            {
                gpu_vars[(size_t)erev][i] = nrn_nernst(gpu_vars[(size_t)conci][i],
                                                      gpu_vars[(size_t)conco][i],
                                                      charge,
                                                      celsius);
            }
        }
    }

    __global__ void getEionVarKernel(DevVarStruct gpu_dev_var, EionVarNames var_name, int idx, double **gpu_var_ptr)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i == 0)
        {
            *gpu_var_ptr = &(gpu_dev_var[(size_t)var_name][idx]);
        }
    }

    inline std::optional<IonMeta> default_ion_meta(std::string_view ion_name) {
        if (ion_name == "na_ion") {
            return ion_meta_from_valence(std::string(ion_name), 1.0);
        }
        if (ion_name == "k_ion") {
            return ion_meta_from_valence(std::string(ion_name), 1.0);
        }
        if (ion_name == "ca_ion") {
            return ion_meta_from_valence(std::string(ion_name), 2.0);
        }
        return std::nullopt;
    }

    inline IonMeta resolve_ion_meta(const std::string &ion_name)
    {
        if (!has_ion_meta(ion_name))
        {
            if (auto meta = default_ion_meta(ion_name)) {
                return *meta;
            }
            throw std::runtime_error("missing ion metadata for ion " + ion_name);
        }
        return get_ion_meta(ion_name, /*must_exist=*/true);
    }

    class GenericIon : public Mechanism
    {
    protected:
        VarStruct<EionTrait> var_struct;
        map<EionVarNames, int> var_in_coredata_idx;
        unordered_map<int, int> node_idx_to_mech_idx;
        IonMeta meta_{};

    public:
        GenericIon(MechInitParams &param) : Mechanism(param)
        {
            using enum EionVarNames;
            var_in_coredata_idx = {
                {erev, 0},
                {conci, 1},
                {conco, 2},
                {cur, 3},
                {dcurdv, 4}};

            meta_ = resolve_ion_meta(param.name);
        }

        void reg_node_indices(MechInitParams &param) override
        {
            var_struct.init(param);
            var_struct.initPdata(param);

            EionData &eionData = get_ion_vars(name);
            unordered_map<int, int> &idxReverseMap = eionData.idx_reverse_table;
            for (int i = 0; i < nnode; i++)
            {
                int nodeIdx = param.nodeindices[i];
                idxReverseMap.insert({nodeIdx, i});
            }
        }

        void read_data_from_coredat(MechInitParams &param) override
        {
            auto nnode = param.node_count;
            auto data = param.data;
            auto param_size = param.data_size;
            vector<pair<double *, int>> data_init_list;
            for (auto [var_name, var_idx] : var_in_coredata_idx)
            {
                VecData<double> *var = var_struct[var_name];
                double *cpu_data_ptr = var->get_cpu_data();
                assert(var->size() == nnode);
                data_init_list.push_back({cpu_data_ptr, var_idx});
            }

            for (int inode = 0; inode < nnode; inode++)
            {
                int offset = inode * param_size;
                for (auto &[data_ptr, var_idx] : data_init_list)
                {
                    data_ptr[inode] = static_cast<double>(data[offset + var_idx]);
                }
            }

            if (mode == Mode::GPU)
            {
                for (auto [var_name, var_idx] : var_in_coredata_idx)
                {
                    var_struct[var_name]->update_gpu_data_from_cpu();
                }
            }

            reg_ion(param.mode,
                    name,
                    var_struct.cpu_dev_var.vars_ptr,
                    var_struct.gpu_dev_var.vars_ptr,
                    nnode);

            // POINTERs targeting ion variables (for example, setpointer(..._ref_ena))
            // are exported as POINTERs to the ion mechanism type.  The generic
            // POINTER resolver uses mech_var_table for mechanism-level targets,
            // so ion mechanisms must publish their variable storage here too.
            auto &varMap = mech_var_table[param.type];
            for (auto [var_name, var_idx] : var_in_coredata_idx)
            {
                MechVarData varData;
                varData.name = name + "_" + std::string(magic_enum::enum_name(var_name));
                varData.len = var_struct[var_name]->size();
                varData.cpu_data = var_struct[var_name]->get_cpu_data();
                varData.vecdata = var_struct[var_name];
                if (param.mode == Mode::GPU)
                {
                    varData.gpu_data = var_struct[var_name]->get_gpu_data();
                }
                varMap[var_idx] = varData;
            }
        }

        static __host__ __device__ void current_single_node(int i,
                                                            double celsius,
                                                            DevVarStruct vars,
                                                            double charge)
        {
            using enum EionVarNames;
            int iontype = vars.pdata[i];
            vars[(size_t)cur][i] = 0.;
            vars[(size_t)dcurdv][i] = 0.;
            if (iontype & 0100)
            {
                vars[(size_t)erev][i] = nrn_nernst(vars[(size_t)conci][i],
                                                  vars[(size_t)conco][i],
                                                  charge,
                                                  celsius);
            }
        }

        void current_cpu(SimMechCurrentParam &param) override
        {
            for (int i = 0; i < nnode; i++)
            {
                current_single_node(i, celsius, var_struct.cpu_dev_var, meta_.charge);
            }
        }

        void current_gpu(SimMechCurrentParam &param) override
        {
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);
            Eion_Cur_Kernel<<<block_num, nthread_per_block, 0, stream>>>(nnode,
                                                                         celsius,
                                                                         meta_.charge,
                                                                         var_struct.gpu_dev_var);
        }

        static __host__ __device__ void initialize_single_node(int i,
                                                               double celsius,
                                                               DevVarStruct vars,
                                                               double charge,
                                                               double default_conci,
                                                               double default_conco)
        {
            using enum EionVarNames;
            int iontype = vars.pdata[i];
            if (iontype & 04)
            {
                vars[(size_t)conci][i] = default_conci;
                vars[(size_t)conco][i] = default_conco;
            }
            if (iontype & 040)
            {
                vars[(size_t)erev][i] = nrn_nernst(vars[(size_t)conci][i],
                                                  vars[(size_t)conco][i],
                                                  charge,
                                                  celsius);
            }
        }

        void initialize_cpu(SimMechInitialParam &param) override
        {
            for (int i = 0; i < nnode; i++)
            {
                initialize_single_node(i,
                                       celsius,
                                       var_struct.cpu_dev_var,
                                       meta_.charge,
                                       meta_.default_conci,
                                       meta_.default_conco);
            }
        }

        void initialize_gpu(SimMechInitialParam &param) override
        {
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);
            Eion_Init_Kernel<<<block_num, nthread_per_block, 0, stream>>>(nnode,
                                                                          celsius,
                                                                          meta_.charge,
                                                                          meta_.default_conci,
                                                                          meta_.default_conco,
                                                                          var_struct.gpu_dev_var);
        }

        void sync_gpu() override
        {
            cudaStreamSynchronize(*reinterpret_cast<cudaStream_t *>(cuda_stream));
        }

        void state_cpu(SimMechStateParam &param) override {}
        void state_gpu(SimMechStateParam &param) override {}

        double *getGPUVarAddr(EionVarNames var_name, int idx)
        {
            auto gpu_dev_var = var_struct.gpu_dev_var;

            double **gpu_var_ptr;
            cudaMalloc(&gpu_var_ptr, sizeof(double *));
            getEionVarKernel<<<1, 1>>>(gpu_dev_var, var_name, idx, gpu_var_ptr);

            double *gpu_var_ptr_on_host;
            cudaMemcpy(&gpu_var_ptr_on_host, gpu_var_ptr, sizeof(double *), cudaMemcpyDeviceToHost);
            cudaFree(gpu_var_ptr);
            return gpu_var_ptr_on_host;
        }

        double *getVarPtr(const VarDescriptor &descriptor, Mode mode) override
        {
            const std::string &var_name_str = descriptor.var;
            int node_idx = descriptor.node_or_mech_idx;

            int *node_indices = vecdata_node_indices->get_cpu_data();
            if (node_idx_to_mech_idx.empty())
            {
                node_idx_to_mech_idx.reserve(nnode);
                for (int i = 0; i < nnode; i++)
                {
                    node_idx_to_mech_idx[node_indices[i]] = i;
                }
            }

            int mech_idx = node_idx_to_mech_idx[node_idx];
            if (permute)
            {
                mech_idx = permute[mech_idx];
            }

            if (mech_idx < 0 || mech_idx >= nnode)
            {
                throw std::out_of_range("Eion Index out of range");
                return nullptr;
            }

            if (auto casted_value = magic_enum::enum_cast<EionVarNames>(var_name_str);
                casted_value.has_value())
            {
                EionVarNames var_name = casted_value.value();
                size_t var_idx = static_cast<size_t>(var_name);
                if (mode == Mode::CPU)
                {
                    return (var_struct.cpu_dev_var[var_idx]) + mech_idx;
                }
                else if (mode == Mode::GPU)
                {
                    return getGPUVarAddr(var_name, mech_idx);
                }
            }
            else
            {
                throw std::invalid_argument("Invalid variable name");
                return nullptr;
            }
            return nullptr;
        }
    };
} // namespace eiontemp
