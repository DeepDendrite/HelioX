#pragma once
#include "mech_template.cuh"
#include "units.h"
#include "ion_table.h"
#include "debug_counter.cuh"
#include "global_vars.h"
#include "mech_var_table.h"
// VarDescriptor通过mech_template.cuh -> mechanism.h -> utils.h 间接包含

namespace eiontemp
{
    // 修改CUDA内核函数以使用基类模板
    template <typename EionType>
    __global__ void Eion_Init_Kernel(int nnode, double celsius ,DevVarStruct gpu_vars)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nnode)
        {
            EionType::initialize_single_node(i, celsius, gpu_vars);
        }
    }

    template <typename EionType>
    __global__ void Eion_Cur_Kernel(int nnode, double celsius, DevVarStruct gpu_vars)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nnode)
        {
            EionType::current_single_node(i, celsius, gpu_vars);
        }
    }

    // 通用辅助函数，移到类外部
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

    __global__ void getEionVarKernel(DevVarStruct gpu_dev_var, EionVarNames var_name, int idx, double **gpu_var_ptr)
    {

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i == 0){
            *gpu_var_ptr = &(gpu_dev_var[(size_t)var_name][idx]);
        }
    }

    // 基类模板
    template <typename IonTraits>
    class EionBase : public Mechanism
    {
    protected:
        VarStruct<EionTrait> var_struct;
        map<EionVarNames, int> var_in_coredata_idx;
        // hdf5记录相关
        unordered_map<int, int> node_idx_to_mech_idx;
    public:
        EionBase(MechInitParams &param) : Mechanism(param)
        {
            using enum EionVarNames;
            var_in_coredata_idx = {
                {erev, 0},
                {conci, 1},
                {conco, 2},
                {cur, 3},
                {dcurdv, 4}};
        }

        void reg_node_indices(MechInitParams &param)
        {
            var_struct.init(param);
            var_struct.initPdata(param);

            EionData &eionData = get_ion_vars(IonTraits::ion_name);
            unordered_map<int, int> &idxReverseMap = eionData.idx_reverse_table;
            for (int i = 0; i < nnode; i++)
            {
                int nodeIdx = param.nodeindices[i];
                idxReverseMap.insert({nodeIdx, i});
            }
        }

        virtual void read_data_from_coredat(MechInitParams &param) override
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
                    data_ptr[inode] = data[offset + var_idx];
                }
            }

            if (mode == Mode::GPU)
            {
                for (auto [var_name, var_idx] : var_in_coredata_idx)
                {
                    var_struct[var_name]->update_gpu_data_from_cpu();
                }
            }
            
            using enum EionVarNames;

            reg_ion(param.mode, IonTraits::ion_name, var_struct.cpu_dev_var.vars_ptr, var_struct.gpu_dev_var.vars_ptr, nnode);

            // Register ion variables in mech_var_table so POINTERs targeting ion variables
            // (e.g. pointer to `ena`) can be resolved via legacy indices.
            auto &varMap = mech_var_table[param.type];
            for (auto [var_name, var_idx] : var_in_coredata_idx)
            {
                MechVarData varData;
                varData.name = string(IonTraits::ion_name) + "_" + string(magic_enum::enum_name(var_name));
                varData.len = var_struct[var_name]->size();
                varData.cpu_data = var_struct[var_name]->get_cpu_data();
                if (param.mode == Mode::GPU)
                {
                    varData.gpu_data = var_struct[var_name]->get_gpu_data();
                }
                varMap[var_idx] = varData;
            }
        }

        static __host__ __device__ void current_single_node(int i, double celsius, DevVarStruct vars)
        {
            using enum EionVarNames;
            int iontype = vars.pdata[i];
            vars[(size_t)cur][i] = 0.;
            vars[(size_t)dcurdv][i] = 0.;
            if (iontype & 0100)
            {
                constexpr double charge = IonTraits::charge;
                vars[(size_t)erev][i] = nrn_nernst(vars[(size_t)conci][i], vars[(size_t)conco][i], charge, celsius);
            }
        }

        virtual void current_cpu(SimMechCurrentParam &param) override
        {
            for (int i = 0; i < nnode; i++)
            {
                current_single_node(i, celsius, var_struct.cpu_dev_var);
            }
        }

        virtual void current_gpu(SimMechCurrentParam &param) override
        {
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);
            Eion_Cur_Kernel<EionBase<IonTraits>><<<block_num, nthread_per_block, 0, stream>>>(nnode, celsius, var_struct.gpu_dev_var);
        }

        static __host__ __device__ void initialize_single_node(int i, double celsius, DevVarStruct vars)
        {
            using enum EionVarNames;
            int iontype = vars.pdata[i];
            if (iontype & 04)
            {
                vars[(size_t)conci][i] = IonTraits::default_conci;
                vars[(size_t)conco][i] = IonTraits::default_conco;
            }
            if (iontype & 040)
            {
                constexpr double charge = IonTraits::charge;
                vars[(size_t)erev][i] = nrn_nernst(vars[(size_t)conci][i], vars[(size_t)conco][i], charge, celsius);
            }
        }

        virtual void initialize_cpu(SimMechInitialParam &param)
        {
            for (int i = 0; i < nnode; i++)
            {
                initialize_single_node(i, celsius, var_struct.cpu_dev_var);
            }
        }

        virtual void initialize_gpu(SimMechInitialParam &param)
        {
            int block_num = (nnode + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = *reinterpret_cast<cudaStream_t *>(cuda_stream);
            Eion_Init_Kernel<EionBase<IonTraits>><<<block_num, nthread_per_block, 0, stream>>>(nnode, celsius, var_struct.gpu_dev_var);
        }

        virtual void sync_gpu() override
        {
            cudaStreamSynchronize(*reinterpret_cast<cudaStream_t *>(cuda_stream));
        }

        // do nothing:
        virtual void state_cpu(SimMechStateParam &param) {}
        virtual void state_gpu(SimMechStateParam &param) {}

        static constexpr const char *ion_name = IonTraits::ion_name;

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
        //这个比较特别，虽然是mech，但是由于查询离子浓度的时候，和Node绑定，因此传的是Node的idx
        double *getVarPtr(const VarDescriptor& descriptor, Mode mode)
        {
            const std::string& var_name_str = descriptor.var;
            int node_idx = descriptor.node_or_mech_idx;
            // Ion variables don't support arrays, descriptor.array_index is ignored

            int *node_indices = vecdata_node_indices->get_cpu_data();
            if(node_idx_to_mech_idx.empty())
            {
                node_idx_to_mech_idx.reserve(nnode);
                for (int i = 0; i < nnode; i++)
                {
                    node_idx_to_mech_idx[node_indices[i]] = i;
                }
            }

            int mech_idx = node_idx_to_mech_idx[node_idx];
            if(permute){
                mech_idx = permute[mech_idx];
            }

            if(mech_idx < 0 || mech_idx >= nnode)
            {
                throw std::out_of_range("Eion Index out of range");
                return nullptr;
            }

            if(auto casted_value = magic_enum::enum_cast<EionVarNames>(var_name_str); casted_value.has_value())
            {
                EionVarNames var_name = casted_value.value();
                size_t var_idx = static_cast<size_t>(var_name);
                if(mode == Mode::CPU)
                {
                    // Note: array_index is ignored for eion variables since they are typically scalar
                    return (var_struct.cpu_dev_var[var_idx]) + mech_idx;
                }
                else if(mode == Mode::GPU)
                {
                    // Note: array_index is ignored for GPU mode as well for eion variables
                    return getGPUVarAddr(var_name, mech_idx);
                }
            }
            else
            {
                throw std::invalid_argument("Invalid variable name");
                return nullptr;
            }
            return nullptr;//不应该到这的
        }
    };

    // 离子特性类，定义不同离子的特征
    struct NaIonTraits
    {
        static constexpr const char *ion_name = "na_ion";
        static constexpr double default_conci = 10.0;
        static constexpr double default_conco = 140.0;
        static constexpr double charge = 1.0;
    };

    struct KIonTraits
    {
        static constexpr const char *ion_name = "k_ion";
        static constexpr double default_conci = 140.0;
        static constexpr double default_conco = 4.0;
        static constexpr double charge = 1.0;
    };

    struct CaIonTraits
    {
        static constexpr const char *ion_name = "ca_ion";
        static constexpr double default_conci = 5.e-5;
        static constexpr double default_conco = 2.0;
        static constexpr double charge = 2.0;
    };
    // 为每种离子类型创建具体的派生类
    class NaIon : public EionBase<NaIonTraits>
    {
    public:
        NaIon(MechInitParams &param) : EionBase<NaIonTraits>(param) {}
    };

    class KIon : public EionBase<KIonTraits>
    {
    public:
        KIon(MechInitParams &param) : EionBase<KIonTraits>(param) {}
    };

    class CaIon : public EionBase<CaIonTraits>
    {
    public:
        CaIon(MechInitParams &param) : EionBase<CaIonTraits>(param) {}
    };

    #define REGISTER_ION_MECH(ion_class) REGISTER_MECHANISM(ion_class::ion_name, ion_class)

    REGISTER_ION_MECH(NaIon);
    REGISTER_ION_MECH(KIon);
    REGISTER_ION_MECH(CaIon);
    
    #undef REGISTER_ION_MECH
} // namespace eiontemp
