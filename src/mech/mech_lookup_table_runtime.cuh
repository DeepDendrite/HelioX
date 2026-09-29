#pragma once

template <typename Derived, MechTraitType MechTrait, bool Enabled>
class MechTableRuntime;

namespace mech_table_detail {

template <typename TableSpec, MechTraitType MechTrait, int DepCount>
__global__ void cuda_prepare_table_kernel(double dt,
                                          VarAccessor<MechTrait> gpu_vars,
                                          double* table_data,
                                          double* saved_deps,
                                          int* deps_valid,
                                          int* table_ready,
                                          int* table_enabled,
                                          const int* dep_global_ids,
                                          int dep_global_count,
                                          const int* dep_range_ids,
                                          int dep_range_count,
                                          int point_count,
                                          int output_count,
                                          double* table_tmin,
                                          double* table_mfac)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    gpu_vars.idx = 0;

    if (!TableSpec::use_table(gpu_vars))
    {
        table_enabled[0] = 0;
        return;
    }

    constexpr int dep_storage_count = (DepCount > 0) ? DepCount : 1;
    double deps[dep_storage_count];
    int dep_i = 0;
    if constexpr (TableSpec::dep_dt)
    {
        deps[dep_i++] = dt;
    }
    for (int i = 0; i < dep_global_count; ++i)
    {
        auto dep_global = static_cast<typename MechTrait::GlobalVarNames>(dep_global_ids[i]);
        deps[dep_i++] = gpu_vars(dep_global);
    }
    for (int i = 0; i < dep_range_count; ++i)
    {
        auto dep_range = static_cast<typename MechTrait::VarNames>(dep_range_ids[i]);
        deps[dep_i++] = gpu_vars(dep_range);
    }

    bool need_rebuild = (table_ready[0] == 0) || (deps_valid[0] == 0);
    if (!need_rebuild)
    {
        for (int i = 0; i < DepCount; ++i)
        {
            if (saved_deps[i] != deps[i])
            {
                need_rebuild = true;
                break;
            }
        }
    }

    if (need_rebuild)
    {
        const double tmin = TableSpec::table_min(gpu_vars);
        const double tmax = TableSpec::table_max(gpu_vars);
        const double dx = (tmax - tmin) / static_cast<double>(point_count - 1);
        const double mfac = 1.0 / dx;
        table_tmin[0] = tmin;
        table_mfac[0] = mfac;

        for (int i = 0; i < point_count; ++i)
        {
            const double x = tmin + static_cast<double>(i) * dx;
            TableSpec::compute_table_outputs_at_point(x, dt, gpu_vars);
            for (int out = 0; out < output_count; ++out)
            {
                table_data[out * point_count + i] = gpu_vars(TableSpec::output_var(out));
            }
        }

        for (int i = 0; i < DepCount; ++i)
        {
            saved_deps[i] = deps[i];
        }
        deps_valid[0] = 1;
        table_ready[0] = 1;
    }

    table_enabled[0] = 1;
}

} // namespace mech_table_detail

template <typename TableSpec, MechTraitType MechTrait>
DUAL_EXEC bool table_try_lookup(double x, VarAccessor<MechTrait>& vars)
{
    static_assert(mech_trait_supports_table_v<MechTrait>,
                  "TABLE lookup requires MechTrait::MechSupportsTable = true");
    bool table_enabled = vars.table.enabled;
    double table_tmin = vars.table.tmin;
    double table_mfac = vars.table.mfac;
#ifdef __CUDA_ARCH__
    if (vars.table.enabled_ptr != nullptr)
    {
        table_enabled = vars.table.enabled_ptr[0] != 0;
    }
    if (vars.table.tmin_ptr != nullptr)
    {
        table_tmin = vars.table.tmin_ptr[0];
    }
    if (vars.table.mfac_ptr != nullptr)
    {
        table_mfac = vars.table.mfac_ptr[0];
    }
#endif

    if (!table_enabled || vars.table.data == nullptr || vars.table.point_count <= 1)
    {
        return false;
    }

    constexpr int output_count = TableSpec::kOutputCount;
    static_assert(output_count > 0, "TableSpec::output_vars must not be empty");

    const int last = vars.table.point_count - 1;
    const double xi = table_mfac * (x - table_tmin);

    if (isnan(xi))
    {
        for (int out = 0; out < output_count; ++out)
        {
            vars(TableSpec::output_var(out)) = xi;
        }
        return true;
    }

    if (xi <= 0.0 || xi >= static_cast<double>(last))
    {
        const int clamped = (xi <= 0.0) ? 0 : last;
        for (int out = 0; out < output_count; ++out)
        {
            const double* table = vars.table.data + out * vars.table.point_count;
            vars(TableSpec::output_var(out)) = table[clamped];
        }
        return true;
    }

    const int i0 = static_cast<int>(xi);
    const double theta = xi - static_cast<double>(i0);
    for (int out = 0; out < output_count; ++out)
    {
        const double* table = vars.table.data + out * vars.table.point_count;
        vars(TableSpec::output_var(out)) = table[i0] + theta * (table[i0 + 1] - table[i0]);
    }
    return true;
}

template <typename Derived, MechTraitType MechTrait>
class MechTableRuntime<Derived, MechTrait, false>
{
};

template <typename Derived, MechTraitType MechTrait>
class MechTableRuntime<Derived, MechTrait, true>
{
private:
    std::unique_ptr<VecData<double>> table_data_;
    std::vector<double> table_saved_deps_;
    bool table_deps_valid_ = false;
    bool table_ready_ = false;
    bool table_enabled_ = false;
    bool table_cpu_inited_ = false;
    bool table_gpu_inited_ = false;
    int table_point_count_ = 0;
    int table_output_count_ = 0;
    double table_tmin_ = 0.0;
    double table_mfac_ = 0.0;

    std::unique_ptr<VecData<double>> table_saved_deps_dev_;
    std::unique_ptr<VecData<int>> table_deps_valid_dev_;
    std::unique_ptr<VecData<int>> table_ready_dev_;
    std::unique_ptr<VecData<int>> table_enabled_dev_;
    std::unique_ptr<VecData<double>> table_tmin_dev_;
    std::unique_ptr<VecData<double>> table_mfac_dev_;
    std::unique_ptr<VecData<int>> table_dep_global_ids_dev_;
    std::unique_ptr<VecData<int>> table_dep_range_ids_dev_;
    int table_dep_global_count_ = 0;
    int table_dep_range_count_ = 0;

    template <typename TableSpec, typename Owner>
    void initialize_once_cpu(Owner& owner)
    {
        constexpr int dep_global_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_global_vars)>>;
        constexpr int dep_range_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_range_vars)>>;
        constexpr int dep_total_count = (TableSpec::dep_dt ? 1 : 0) + dep_global_count + dep_range_count;
        constexpr int table_bins = TableSpec::kTableBins;
        constexpr int point_count = table_bins + 1;
        constexpr int output_count = TableSpec::kOutputCount;
        static_assert(table_bins > 0, "TableSpec::kTableBins must be > 0");
        static_assert(output_count > 0, "TableSpec::kOutputCount must be > 0");

        if (table_cpu_inited_)
        {
            return;
        }
        DEBUG_ASSERT(owner.nnode > 0);
        table_saved_deps_.assign(static_cast<size_t>(dep_total_count), 0.0);
        table_deps_valid_ = false;
        table_ready_ = false;
        table_enabled_ = false;
        table_point_count_ = point_count;
        table_output_count_ = output_count;

        const int total_size = point_count * output_count;
        table_data_ = std::make_unique<VecData<double>>(owner.mode, total_size);
        table_cpu_inited_ = true;
    }

    template <typename TableSpec, typename Owner>
    void rebuild_if_needed_cpu(Owner& owner, double dt)
    {
        constexpr int dep_global_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_global_vars)>>;
        constexpr int dep_range_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_range_vars)>>;
        constexpr int dep_total_count = (TableSpec::dep_dt ? 1 : 0) + dep_global_count + dep_range_count;
        constexpr int table_bins = TableSpec::kTableBins;

        table_enabled_ = false;

        // TABLE/DEPEND 语义对齐 NEURON/CoreNEURON：
        // 对 dep_range_vars 使用 mech 实例 0 的值作为代表来判定是否重建表。
        // 如果 RANGE 在不同 node 上不一致，这里会保留与 NEURON 相同的
        // “代表值”行为（这是 bug-to-bug 兼容，未来可能考虑增加一个严格模式，禁止range类型的变量进depend）。
        auto dep_vars = owner.getCpuVarAccessor(0);
        if (!TableSpec::use_table(dep_vars))
        {
            return;
        }

        std::array<double, static_cast<size_t>(dep_total_count)> deps{};
        int dep_i = 0;
        if constexpr (TableSpec::dep_dt)
        {
            deps[dep_i++] = dt;
        }
        for (auto dep_global : TableSpec::dep_global_vars)
        {
            deps[dep_i++] = dep_vars(dep_global);
        }
        for (auto dep_range : TableSpec::dep_range_vars)
        {
            deps[dep_i++] = dep_vars(dep_range);
        }

        bool need_rebuild = !table_ready_ || !table_deps_valid_;
        if (!need_rebuild)
        {
            for (size_t i = 0; i < deps.size(); ++i)
            {
                if (table_saved_deps_[i] != deps[i])
                {
                    need_rebuild = true;
                    break;
                }
            }
        }

        if (need_rebuild)
        {
            const int point_count = table_point_count_;
            const int output_count = table_output_count_;

            auto table_vars = owner.getCpuVarAccessor(0);
            table_tmin_ = TableSpec::table_min(table_vars);
            const double table_tmax = TableSpec::table_max(table_vars);
            const double dx = (table_tmax - table_tmin_) / static_cast<double>(table_bins);
            table_mfac_ = 1.0 / dx;

            double* table_cpu = table_data_->get_cpu_data();
            for (int i = 0; i < point_count; ++i)
            {
                const double x = table_tmin_ + static_cast<double>(i) * dx;
                TableSpec::compute_table_outputs_at_point(x, dt, table_vars);
                for (int out = 0; out < output_count; ++out)
                {
                    table_cpu[out * point_count + i] = table_vars(TableSpec::output_var(out));
                }
            }

            table_point_count_ = point_count;
            table_output_count_ = output_count;
            table_ready_ = true;
            table_saved_deps_.assign(deps.begin(), deps.end());
            table_deps_valid_ = true;
        }

        table_enabled_ = table_ready_;
    }

    template <typename TableSpec, typename Owner>
    void initialize_once_gpu(Owner& owner)
    {
        constexpr int table_bins = TableSpec::kTableBins;
        constexpr int point_count = table_bins + 1;
        constexpr int output_count = TableSpec::kOutputCount;
        constexpr int dep_global_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_global_vars)>>;
        constexpr int dep_range_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_range_vars)>>;
        constexpr int dep_total_count = (TableSpec::dep_dt ? 1 : 0) + dep_global_count + dep_range_count;
        static_assert(table_bins > 0, "TableSpec::kTableBins must be > 0");
        static_assert(output_count > 0, "TableSpec::kOutputCount must be > 0");

        if (table_gpu_inited_)
        {
            return;
        }
        DEBUG_ASSERT(owner.nnode > 0);

        const int total_size = point_count * output_count;
        table_data_ = std::make_unique<VecData<double>>(owner.mode, total_size);

        constexpr int dep_storage_count = (dep_total_count > 0) ? dep_total_count : 1;
        table_saved_deps_dev_ = std::make_unique<VecData<double>>(owner.mode, dep_storage_count);

        constexpr int dep_global_storage_count = (dep_global_count > 0) ? dep_global_count : 1;
        table_dep_global_ids_dev_ = std::make_unique<VecData<int>>(owner.mode, dep_global_storage_count);
        int* dep_global_ids = table_dep_global_ids_dev_->get_cpu_data();
        if constexpr (dep_global_count > 0)
        {
            int dep_i = 0;
            for (auto dep_global : TableSpec::dep_global_vars)
            {
                dep_global_ids[dep_i++] = static_cast<int>(dep_global);
            }
        }
        else
        {
            dep_global_ids[0] = 0;
        }
        table_dep_global_ids_dev_->update_gpu_data_from_cpu(owner.cuda_stream);

        constexpr int dep_range_storage_count = (dep_range_count > 0) ? dep_range_count : 1;
        table_dep_range_ids_dev_ = std::make_unique<VecData<int>>(owner.mode, dep_range_storage_count);
        int* dep_range_ids = table_dep_range_ids_dev_->get_cpu_data();
        if constexpr (dep_range_count > 0)
        {
            int dep_i = 0;
            for (auto dep_range : TableSpec::dep_range_vars)
            {
                dep_range_ids[dep_i++] = static_cast<int>(dep_range);
            }
        }
        else
        {
            dep_range_ids[0] = 0;
        }
        table_dep_range_ids_dev_->update_gpu_data_from_cpu(owner.cuda_stream);

        table_deps_valid_dev_ = std::make_unique<VecData<int>>(owner.mode, 0, 1);
        table_ready_dev_ = std::make_unique<VecData<int>>(owner.mode, 0, 1);
        table_enabled_dev_ = std::make_unique<VecData<int>>(owner.mode, 0, 1);
        table_tmin_dev_ = std::make_unique<VecData<double>>(owner.mode, 0.0, 1);
        table_mfac_dev_ = std::make_unique<VecData<double>>(owner.mode, 0.0, 1);

        table_point_count_ = point_count;
        table_output_count_ = output_count;
        table_dep_global_count_ = dep_global_count;
        table_dep_range_count_ = dep_range_count;
        table_gpu_inited_ = true;
    }

    template <typename TableSpec, typename Owner>
    void rebuild_if_needed_gpu(Owner& owner, double dt)
    {
        constexpr int dep_global_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_global_vars)>>;
        constexpr int dep_range_count =
            std::tuple_size_v<std::remove_cvref_t<decltype(TableSpec::dep_range_vars)>>;
        constexpr int dep_total_count = (TableSpec::dep_dt ? 1 : 0) + dep_global_count + dep_range_count;
        DEBUG_ASSERT(table_gpu_inited_);

        VarAccessor<MechTrait> gpu_vars = owner.getGpuVarAccessor(0);
        cudaStream_t stream = *reinterpret_cast<cudaStream_t*>(owner.cuda_stream);

        // NOTE:
        // NEURON/CoreNEURON performs TABLE DEPEND checking on CPU. We intentionally keep the
        // full check+rebuild path on device stream to avoid per-step D2H/H2D synchronization.
        // In edge cases where DEPEND values mutate only on device, this can rebuild earlier
        // than CoreNEURON host-check behavior.
        mech_table_detail::cuda_prepare_table_kernel<TableSpec, MechTrait, dep_total_count>
            <<<1, 1, 0, stream>>>(dt,
                                  gpu_vars,
                                  table_data_->get_gpu_data(),
                                  table_saved_deps_dev_->get_gpu_data(),
                                  table_deps_valid_dev_->get_gpu_data(),
                                  table_ready_dev_->get_gpu_data(),
                                  table_enabled_dev_->get_gpu_data(),
                                  table_dep_global_ids_dev_->get_gpu_data(),
                                  table_dep_global_count_,
                                  table_dep_range_ids_dev_->get_gpu_data(),
                                  table_dep_range_count_,
                                  table_point_count_,
                                  table_output_count_,
                                  table_tmin_dev_->get_gpu_data(),
                                  table_mfac_dev_->get_gpu_data());
    }

public:
    template <typename Owner>
    void initialize_once(Owner& owner)
    {
        using TableSpec = typename Derived::TableSpec;

        if (owner.mode == Mode::GPU)
        {
            initialize_once_gpu<TableSpec>(owner);
            return;
        }

        initialize_once_cpu<TableSpec>(owner);
    }

    template <typename Owner>
    void rebuild_if_needed(Owner& owner, double dt)
    {
        using TableSpec = typename Derived::TableSpec;

        if (owner.mode == Mode::GPU)
        {
            rebuild_if_needed_gpu<TableSpec>(owner, dt);
            return;
        }

        rebuild_if_needed_cpu<TableSpec>(owner, dt);
    }

    template <typename Owner>
    void bind_cpu(const Owner& owner, VarAccessor<MechTrait>& var_access) const
    {
        (void)owner;
        if (table_enabled_ && table_ready_ && table_data_)
        {
            var_access.table.enabled = true;
            var_access.table.point_count = table_point_count_;
            var_access.table.output_count = table_output_count_;
            var_access.table.tmin = table_tmin_;
            var_access.table.mfac = table_mfac_;
            var_access.table.data = table_data_->get_cpu_data();
        }
    }

    template <typename Owner>
    void bind_gpu(const Owner& owner, VarAccessor<MechTrait>& var_access) const
    {
        if (owner.mode == Mode::GPU && table_data_ && table_enabled_dev_ && table_tmin_dev_ && table_mfac_dev_)
        {
            var_access.table.enabled = true;
            var_access.table.point_count = table_point_count_;
            var_access.table.output_count = table_output_count_;
            var_access.table.tmin = table_tmin_;
            var_access.table.mfac = table_mfac_;
            var_access.table.data = table_data_->get_gpu_data();
            var_access.table.enabled_ptr = table_enabled_dev_->get_gpu_data();
            var_access.table.tmin_ptr = table_tmin_dev_->get_gpu_data();
            var_access.table.mfac_ptr = table_mfac_dev_->get_gpu_data();
        }
    }
};
