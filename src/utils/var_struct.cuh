#pragma once
#include <algorithm>
#include <map>
#include <vector>
#include <unordered_map>
#include "magic_enum/magic_enum.hpp"
#include "vecdata.h"
#include "mechanism.h"
#include "ion_table.h"
#include "mech_template_utils.cuh"
#include "utils.h"

using namespace std;

struct DevVarStructBase
{
    double **vars_ptr = nullptr;
};

struct DevVarStructWithArray : public DevVarStructBase
{
    int *arr_size = nullptr; // 数组大小
};

template <typename T>
concept DevVarType = std::is_base_of_v<DevVarStructWithArray, T>;

struct DevGlobalVarStruct : public DevVarStructWithArray{
    __device__ __host__ __forceinline__ auto &operator[](size_t var_name_as_idx) const
    {
        DEBUG_ASSERT(arr_size[var_name_as_idx] == 1);
        return *(vars_ptr[var_name_as_idx]);
    }
    __device__ __host__ __forceinline__ double &getArr(size_t var_name_as_idx, int i) const
    {
        DEBUG_ASSERT(i < arr_size[var_name_as_idx]);
        return vars_ptr[var_name_as_idx][i];
    }
};
struct DevIonVarStruct : public DevVarStructBase
{

    int **ion_idx_map = nullptr;
    __device__ __host__ __forceinline__ auto &getIonVar(size_t var_name, int i) const
    {
        auto _ion_idx_map = ion_idx_map[var_name];
        int ion_idx = _ion_idx_map[i];         // ion中的索引
        double *data_vec = vars_ptr[var_name]; // ion变量的数据
        return data_vec[ion_idx];
    }
};

// 简化为单一的DevVarStruct，不再需要CPU和GPU派生类
struct DevVarStruct : public DevVarStructWithArray
{
    int *pdata = 0;
    // Flattened POINTER targets:
    // ptr_targets[pvar * ptr_stride + idx] is a pointer to the target variable
    // for pointer variable pvar of the current mechanism instance idx.
    double** ptr_targets = nullptr;
    int ptr_stride = 0;

    __device__ __host__ __forceinline__ auto &operator[](size_t var_name_as_idx) const
    {
        DEBUG_ASSERT(arr_size[var_name_as_idx] == 1);
        return vars_ptr[var_name_as_idx];
    }

    __device__ __host__ __forceinline__ double &getArr(size_t var_name_as_idx, int mech_idx, int i) const
    {
        DEBUG_ASSERT(arr_size != nullptr);
        double *arr = vars_ptr[var_name_as_idx] + mech_idx * arr_size[var_name_as_idx];
        DEBUG_ASSERT(arr != nullptr);
        DEBUG_ASSERT(i < arr_size[var_name_as_idx]);
        return arr[i];
    }
};

template <MechTraitType MechTrait>
struct VarStruct
{
    using VarNames = typename MechTrait::VarNames;
    vector<unique_ptr<VecData<double>>> vecdata_vars; // 普通的变量，每个node都有一个
    DevVarStruct cpu_dev_var;
    DevVarStruct gpu_dev_var;

    vector<unique_ptr<VecData<double>>> global_vars; // 全局变量，可能是标量，也可能是数组，但是所有的node都共享一份
    DevGlobalVarStruct cpu_dev_global_var;
    DevGlobalVarStruct gpu_dev_global_var;

    DevIonVarStruct gpu_dev_ion_var; // 离子变量，是映射过来的，所以没有vec_data
    DevIonVarStruct cpu_dev_ion_var;

    // 统一管理所有内存资源 - 使用vector替代set
    struct MemoryResources
    {
        // CPU资源
        double **cpu_vars_ptr = nullptr;     // CPU上的vars_ptr
        int *cpu_pdata = nullptr;            // CPU上的pdata
        double **cpu_ptr_targets = nullptr;  // CPU上的POINTER目标指针数组
        int **cpu_ion_idx_map = nullptr;     // CPU上的ion_idx_map主数组
        vector<int *> cpu_ion_idx_arrays;    // 存储CPU上所有离散的ion_idx数组
        double **cpu_ion_vars_ptr = nullptr; // CPU上的ion_vars_ptr

        // GPU资源
        double **gpu_vars_ptr = nullptr;  // GPU上的vars_ptr
        int *gpu_pdata = nullptr;         // GPU上的pdata
        double **gpu_ptr_targets = nullptr;  // GPU上的POINTER目标指针数组
        int **gpu_ion_idx_map = nullptr;  // GPU上的ion_idx_map主数组
        vector<int *> gpu_ion_idx_arrays; // 存储GPU上所有离散的ion_idx数组

        double **gpu_ion_vars_ptr = nullptr;

        // 跟踪Mode
        Mode mode = Mode::CPU;
    };

    MemoryResources resources;

    // 析构函数，负责释放所有资源
    ~VarStruct()
    {
        // 释放CPU资源
        if (resources.cpu_vars_ptr != nullptr)
        {
            delete[] resources.cpu_vars_ptr;
            resources.cpu_vars_ptr = nullptr;
        }

        if (resources.cpu_pdata != nullptr)
        {
            delete[] resources.cpu_pdata;
            resources.cpu_pdata = nullptr;
        }

        if (resources.cpu_ptr_targets != nullptr)
        {
            delete[] resources.cpu_ptr_targets;
            resources.cpu_ptr_targets = nullptr;
        }

        if (resources.cpu_ion_idx_map != nullptr)
        {
            delete[] resources.cpu_ion_idx_map;
            resources.cpu_ion_idx_map = nullptr;
        }

        if (resources.cpu_ion_vars_ptr != nullptr)
        {
            delete[] resources.cpu_ion_vars_ptr;
            resources.cpu_ion_vars_ptr = nullptr;
        }

        for (auto ptr : resources.cpu_ion_idx_arrays)
        {
            delete[] ptr;
        }
        resources.cpu_ion_idx_arrays.clear();

        // 释放GPU资源 (只有在GPU模式下才需要)
        if (resources.mode == Mode::GPU)
        {
            if (resources.gpu_vars_ptr != nullptr)
            {
                gpu_mem_free((void **)&resources.gpu_vars_ptr);
            }

            if (resources.gpu_pdata != nullptr)
            {
                gpu_mem_free((void **)&resources.gpu_pdata);
            }

            if (resources.gpu_ptr_targets != nullptr)
            {
                gpu_mem_free((void **)&resources.gpu_ptr_targets);
            }

            if (resources.gpu_ion_idx_map != nullptr)
            {
                gpu_mem_free((void **)&resources.gpu_ion_idx_map);
            }

            if (resources.gpu_ion_vars_ptr != nullptr)
            {
                gpu_mem_free((void **)&resources.gpu_ion_vars_ptr);
            }

            for (auto ptr : resources.gpu_ion_idx_arrays)
            {
                cudaFree(ptr);
            }
            resources.gpu_ion_idx_arrays.clear();
        }
    }

    auto operator[](VarNames var_name)
    {
        assert(static_cast<int>(var_name) < vecdata_vars.size());
        return vecdata_vars[static_cast<int>(var_name)].get();
    }

    void init(MechInitParams &param)
    {
        init(param, {}, {}, {});
        // eion调用的时候，不会有后续的参数，因此留空。但是在定义的时候留默认空参数可能会导致调用不完全，因此额外实现一个少参数的版本
    }
    // 初始化变量,如果有初始值，则会用于初始化
    void init(MechInitParams &param, map<VarNames, CoreIdxInfo> var_in_coredata_idx, std::map<VarNames, double> init_values, GlobalVarInfoMap<MechTrait> global_var_map)
    {
        // 记录模式用于析构
        resources.mode = param.mode;
        auto var_count = magic_enum::enum_count<VarNames>();
        vecdata_vars.resize(var_count);
        int *var_len = new int[var_count]; // 记录每个变量的长度
        for (auto &var : magic_enum::enum_values<VarNames>())
        {
            double init_value = 0.0;
            if (init_values.find(var) != init_values.end())
            {
                init_value = init_values[var];
            }

            // 每个实例至少有1个元素；若机制声明了数组变量，则按声明的array_size分配。
            // 不能依赖param.array_dims：in-memory前端通常不会提供它，但数组stride仍必须正确。
            int array_len = 1;
            if (auto it = var_in_coredata_idx.find(var); it != var_in_coredata_idx.end())
            {
                array_len = std::max(1, it->second.array_size);
            }

            int actual_len = param.node_count * array_len;
            var_len[static_cast<int>(var)] = array_len;
            vecdata_vars[static_cast<int>(var)] = make_unique<VecData<double>>(param.mode, init_value, actual_len);
        }
        initDevVar<VarNames>(CPU, vecdata_vars, &cpu_dev_var, var_len);
        if (param.mode == Mode::GPU)
        {
            initDevVar<VarNames>(GPU, vecdata_vars, &gpu_dev_var, var_len);
        }

        // global var的处理：1.必须存在于global_var_map中，2.只分配相应长度的内存，不需要初始化
        if constexpr (has_GlobalVarNames_v<MechTrait>)
        {
            using GlobalVarNames = MechTrait::GlobalVarNames;
            auto global_var_count = magic_enum::enum_count<GlobalVarNames>();
            global_vars.reserve(global_var_count);
            int *var_len = new int[global_var_count]; // 记录每个变量的长度
            for (auto &var : magic_enum::enum_values<GlobalVarNames>())
            {
                assert(global_var_map.contains(var));
                auto &global_var_info = global_var_map.at(var);
                int actual_len = global_var_info.array_size;
                var_len[static_cast<int>(var)] = actual_len;
                global_vars.push_back(make_unique<VecData<double>>(param.mode, actual_len));
                // printf("global var:%s, size:%d addr=%p\n",magic_enum::enum_name(var).data(),actual_len,global_vars.back()->get_cpu_data());
            }

            initDevVar<GlobalVarNames>(CPU, global_vars, &cpu_dev_global_var, var_len);
            if (param.mode == Mode::GPU)
            {
                initDevVar<GlobalVarNames>(GPU, global_vars, &gpu_dev_global_var, var_len);
            }
        }

        // POINTER变量：为每个POINTER变量分配一个 (ptr_var_count * node_count) 的指针表。
        // 具体指针值在后续的 resolve 阶段填充（需要访问目标变量的地址）。
        if constexpr (has_PointerVarNames_v<MechTrait>)
        {
            using PointerVarNames = typename MechTrait::PointerVarNames;
            const int ptr_var_count = magic_enum::enum_count<PointerVarNames>();
            if (ptr_var_count > 0)
            {
                const int total = ptr_var_count * param.node_count;
                resources.cpu_ptr_targets = new double *[total];
                std::fill_n(resources.cpu_ptr_targets, total, nullptr);
                cpu_dev_var.ptr_targets = resources.cpu_ptr_targets;
                cpu_dev_var.ptr_stride = param.node_count;

                if (param.mode == Mode::GPU)
                {
                    gpu_mem_allocate((void **)&resources.gpu_ptr_targets, total * sizeof(double *));
                    mem_copy_cpu2gpu_sync(resources.gpu_ptr_targets, resources.cpu_ptr_targets, total * sizeof(double *));
                    gpu_dev_var.ptr_targets = resources.gpu_ptr_targets;
                    gpu_dev_var.ptr_stride = param.node_count;
                }
            }
        }
    }

    void initPdata(MechInitParams &param)
    {
        if (param.pdata_size <= 0 || param.pdata == nullptr)
        {
            printf_debug("in Mech %s,has no pdata\n", param.name.c_str());
            return;
        }
        assert(param.pdata_size > 0 && param.pdata != nullptr);
        int pdata_tot_size = param.pdata_size * param.node_count;

        // 分配并复制CPU pdata
        resources.cpu_pdata = new int[pdata_tot_size];
        memcpy(resources.cpu_pdata, param.pdata, pdata_tot_size * sizeof(int));
        cpu_dev_var.pdata = resources.cpu_pdata;

        // 如果需要GPU，则分配和复制GPU pdata
        if (param.mode == Mode::GPU)
        {
            gpu_mem_allocate((void **)&resources.gpu_pdata, pdata_tot_size * sizeof(int));
            mem_copy_cpu2gpu_sync(resources.gpu_pdata, resources.cpu_pdata, pdata_tot_size * sizeof(int));
            gpu_dev_var.pdata = resources.gpu_pdata;
        }
    }

    template <EnumType EnumVarNames, DevVarType DevVar>
    void initDevVar(Mode mode, vector<unique_ptr<VecData<double>>> &vec_data_vec, DevVar *dev_vars, int *var_lens)
    {
        int len = magic_enum::enum_count<EnumVarNames>();
        double **vars_ptr = nullptr;

        if (mode == Mode::CPU)
        {
            // 分配并跟踪CPU vars_ptr
            resources.cpu_vars_ptr = new double *[len];
            vars_ptr = resources.cpu_vars_ptr;
        }
        else
        {
            // 先在CPU上准备数据
            vars_ptr = new double *[len];
        }

        // 数组长度记录数据
        if (mode == Mode::CPU)
        {
            dev_vars->arr_size = var_lens;
        }
        else
        {
            gpu_mem_allocate((void **)&dev_vars->arr_size, len * sizeof(int));
            mem_copy_cpu2gpu_sync(dev_vars->arr_size, var_lens, len * sizeof(int));
        }

        for (auto &var : magic_enum::enum_values<EnumVarNames>())
        {
            int var_idx = static_cast<int>(var);
            if (vec_data_vec[var_idx] == nullptr)
            {
                // printf("skip ion var:%s\n",magic_enum::enum_name(var).data());
                continue;
                // 说明这个变量是ion变量，保留nullptr，方便后续挂载
            }
            if (mode == Mode::CPU)
            {
                vars_ptr[var_idx] = vec_data_vec[var_idx]->get_cpu_data();
            }
            else
            {
                vars_ptr[var_idx] = vec_data_vec[var_idx]->get_gpu_data();
            }
        }

        if (mode == Mode::GPU)
        {
            // 分配并跟踪GPU vars_ptr
            gpu_mem_allocate((void **)&resources.gpu_vars_ptr, len * sizeof(double *));
            mem_copy_cpu2gpu_sync(resources.gpu_vars_ptr, vars_ptr, len * sizeof(double *));
            delete[] vars_ptr; // 删除临时CPU数组
            vars_ptr = resources.gpu_vars_ptr;
        }

        dev_vars->vars_ptr = vars_ptr;
    }

    void map_ion_var(IonVarInfoMap<MechTrait> ion_var_map, MechInitParams &param)
    {
        if constexpr (has_IonVarNames_v<MechTrait>) // 仅当存在ion变量的时候才会执行
        {
            // 从ion table中查找对应的ion变量，然后挂载到对应的变量上
            // CPU和GPU均挂载

            // 第一部分：将ion table中的变量挂载到对应的变量上（CPU和GPU）
            // 获取枚举类VarNames的元素数量，用于后续内存分配
            using IonVarNames = typename MechTrait::IonVarNames;
            int IonVarNames_count = magic_enum::enum_count<IonVarNames>();

            // cpu的内存分配
            cpu_dev_ion_var.vars_ptr = new double *[IonVarNames_count];
            resources.cpu_ion_vars_ptr = cpu_dev_ion_var.vars_ptr; // 追踪资源，避免泄露
            // 如果是GPU模式，需要创建一个临时数组存储GPU上变量指针的主机端副本
            double **gpu_vars_ptr_on_host; // 临时数组，后面会替换到cpu_dev_ion_var内部的vars_ptr里
            if (param.mode == Mode::GPU)
            {
                cudaMalloc((void **)&(gpu_dev_ion_var.vars_ptr), IonVarNames_count * sizeof(double *)); // 为gpu的变量分配内存
                resources.gpu_ion_vars_ptr = gpu_dev_ion_var.vars_ptr;                                  // 追踪资源，避免泄露
                gpu_vars_ptr_on_host = new double *[IonVarNames_count];                                 // 同时为临时数组分配内存
            }

            // 遍历映射表，将dev_ion_var的变量指针替换为ion table中获取到的各离子变量指针
            for (const auto &[var_name, eion_info] : ion_var_map)
            {
                const auto &[ion_name_str, ion_var_name] = eion_info;
                EionData &ion_vars = get_ion_vars(ion_name_str, true); // 从eiontable中获取对应名称的离子机制的各变量

                int ion_var_idx = static_cast<int>(ion_var_name); // eion中的下标
                int var_idx = static_cast<int>(var_name);         // mech中的下标

                cpu_dev_ion_var.vars_ptr[var_idx] = ion_vars.cpu_data[ion_var_idx]; // 取出变量，放到ion dev_vars中
                if (param.mode == Mode::GPU)
                {
                    gpu_vars_ptr_on_host[var_idx] = ion_vars.gpu_data_oh_host[ion_var_idx]; // 取出变量，放到临时数组中，后续再替换到gpu_dev_ion_var中
                }
            }
            if (param.mode == Mode::GPU)
            {
                // 将临时数组中的指针复制到GPU上的vars_ptr，然后释放临时数组
                cudaMemcpy(gpu_dev_ion_var.vars_ptr, gpu_vars_ptr_on_host, IonVarNames_count * sizeof(double *), cudaMemcpyHostToDevice);
                delete[] gpu_vars_ptr_on_host;
            }
            // 第一部分结束，此时，dev_ion_var的vars_ptr中已经挂载了所有的离子变量

            // 第二部分：由于第一部分挂载的变量排序是按照eion中的顺序来排的，因此需要构建索引，才能正常地从mech中访问
            //  创建CPU和GPU端的离子索引映射表
            unordered_map<string, int *> ion_idx_map_cpu; // CPU端的ion_idx_map - 键为离子名称，值为索引数组指针
            unordered_map<string, int *> ion_idx_map_gpu; // GPU端的ion_idx_map，所分配的内存为GPU端内存

            // 2.1 为索引数组分配内存（但是没有填充数据，同时，并未挂载到mech内部）

            // 遍历ion_var_map，为每种不同的离子分配索引数组（去重处理）
            // 每个离子类型只需要一个索引数组，大小为节点数量(param.node_count)
            for (const auto &[var_name, eion_info] : ion_var_map)
            {
                const auto &[ion_name_str, ion_var_name] = eion_info;
                // 检查该离子是否已经在映射表中
                if (ion_idx_map_cpu.find(ion_name_str) == ion_idx_map_cpu.end())
                {
                    // 离子名称首次出现，为CPU分配索引数组,长度与mech的node count相同
                    int *cpu_idx_array = new int[param.node_count];
                    ion_idx_map_cpu[ion_name_str] = cpu_idx_array;
                    // 将指针添加到资源跟踪列表，以便后续清理
                    resources.cpu_ion_idx_arrays.push_back(cpu_idx_array);

                    // GPU模式下，同样为GPU分配对应的索引数组
                    if (param.mode == Mode::GPU)
                    {
                        int *gpu_idx_array;
                        // 在GPU上分配内存
                        cudaMalloc((void **)&gpu_idx_array, param.node_count * sizeof(int));
                        ion_idx_map_gpu[ion_name_str] = gpu_idx_array;
                        // 将指针添加到资源跟踪列表
                        resources.gpu_ion_idx_arrays.push_back(gpu_idx_array);
                    }
                }
            }

            // 2.2 填充索引数组
            for (auto &[ion_name, idx_list] : ion_idx_map_cpu)
            {
                // 获取该离子的反向索引表（全局节点索引到离子内部索引的映射）
                auto &idx_reverse_table = get_ion_vars(ion_name, true).idx_reverse_table;
                // 遍历当前机制(mech)涉及的所有节点
                for (int i = 0; i < param.node_count; i++)
                {
                    // 获取全局节点索引
                    int node_idx = param.nodeindices[i];
                    // 断言确保该节点在离子的节点集合中存在
                    assert(idx_reverse_table.find(node_idx) != idx_reverse_table.end()); // 确保ion中有这个节点
                    // 获取节点在离子数组中的实际索引
                    int ion_idx = idx_reverse_table[node_idx];
                    // 设置映射：mech局部索引i -> 离子内部索引ion_idx
                    idx_list[i] = ion_idx;
                }
            }

            // GPU模式下，将CPU端的索引映射复制到GPU端（目前仍然在临时的unordered_map上操作）
            if (param.mode == Mode::GPU)
            {
                for (auto &[ion_name, cpu_idx_list] : ion_idx_map_cpu)
                {
                    // 获取对应的GPU索引数组
                    auto &gpu_idx_list = ion_idx_map_gpu[ion_name];
                    // 复制索引数据到GPU
                    cudaMemcpy(gpu_idx_list, cpu_idx_list, param.node_count * sizeof(int), cudaMemcpyHostToDevice);
                }
            }

            // 2.3 将CPU端的索引映射表挂载到DevIonVarStruct中
            cpu_dev_ion_var.ion_idx_map = new int *[IonVarNames_count];
            resources.cpu_ion_idx_map = cpu_dev_ion_var.ion_idx_map; // 追踪资源，避免泄露

            // GPU模式的临时指针数组
            int **ion_idx_map_gpu_on_host = nullptr;
            if (param.mode == Mode::GPU)
            {
                cudaMalloc((void **)&(gpu_dev_ion_var.ion_idx_map), IonVarNames_count * sizeof(int *));
                resources.gpu_ion_idx_map = gpu_dev_ion_var.ion_idx_map; // 追踪资源，避免泄露
                ion_idx_map_gpu_on_host = new int *[IonVarNames_count];
            }

            // 遍历ion_var_map，设置每个变量到对应离子索引数组的映射
            for (const auto &[var_name, eion_info] : ion_var_map)
            {
                const auto &[ion_name_str, ion_var_name] = eion_info;
                // 设置CPU端映射：将var_name对应的元素指向所属的eion索引数组
                int var_idx = static_cast<int>(var_name);
                cpu_dev_ion_var.ion_idx_map[var_idx] = ion_idx_map_cpu[ion_name_str];

                // GPU模式下先填充临时数组
                if (param.mode == Mode::GPU)
                {
                    ion_idx_map_gpu_on_host[var_idx] = ion_idx_map_gpu[ion_name_str];
                }
            }

            // GPU模式下，将临时数组复制到GPU端
            if (param.mode == Mode::GPU)
            {
                // 复制临时数组到GPU端
                cudaMemcpy(gpu_dev_ion_var.ion_idx_map, ion_idx_map_gpu_on_host, IonVarNames_count * sizeof(int *), cudaMemcpyHostToDevice);
                delete[] ion_idx_map_gpu_on_host; // 释放临时数组
            }
        }
    }
};
