#include "runtime_api/core/SimRuntimeCore.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <magic_enum/magic_enum.hpp>

#include "cuda_utils.h"
#include "coredat_to_innerdat.h"
#include "global_vars.h"
#include "ion_table.h"
#include "morph/section_distance.hpp"
#include "permute_order.h"
#include "spike/postsyn.h"
#include "spike/vecevent.h"

namespace neurong::runtime_api::core {

extern "C" void launch_batch_copy_doubles(double** d_gpu_ptrs, const double* d_cpu_values, int count);
extern "C" void launch_batch_gather_floats(double** d_gpu_ptrs, float* d_out, int count);

namespace {

std::string to_lower_ascii(std::string source) {
    std::transform(source.begin(), source.end(), source.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return source;
}

std::optional<std::filesystem::path> resolve_default_mechlib_candidate() {
    const auto cwd_candidate = std::filesystem::current_path() / "build-mech" / "libneurong_mechs.so";
    if (std::filesystem::exists(cwd_candidate)) {
        return cwd_candidate;
    }
    return std::nullopt;
}

std::filesystem::path canonicalize_existing_path(const std::filesystem::path& path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        return std::filesystem::absolute(path);
    }
    return canonical;
}

std::atomic<bool>& process_mech_registry_frozen() {
    static std::atomic<bool> frozen{false};
    return frozen;
}

bool is_process_mech_registry_frozen() {
    return process_mech_registry_frozen().load(std::memory_order_acquire);
}

void freeze_process_mech_registry() {
    process_mech_registry_frozen().store(true, std::memory_order_release);
}

const char* source_to_string(SimRuntimeCore::ModelSource source) {
    switch (source) {
    case SimRuntimeCore::ModelSource::Neuron:
        return "neuron";
    case SimRuntimeCore::ModelSource::Neurong:
        return "neurong";
    case SimRuntimeCore::ModelSource::Unset:
    default:
        return "unset";
    }
}

}  // namespace

bool SimRuntimeCore::build_mech_node_index_cache_(const std::string& mech) {
    auto* var_map = MechanismFactory::getInstance().getVarMap(mech);
    auto* mech_obj = dynamic_cast<Mechanism*>(var_map);
    if (mech_obj == nullptr) {
        printf("build_mech_node_index_cache_: mech '%s' not found or not a Mechanism\n", mech.c_str());
        return false;
    }

    const int ninst = mech_obj->node_count();
    int* node_indices = mech_obj->node_indices_cpu_data();
    if (ninst <= 0 || node_indices == nullptr) {
        printf("build_mech_node_index_cache_: mech '%s' has no node indices\n", mech.c_str());
        return false;
    }

    std::unordered_map<int, std::vector<int>> node_to_inst;
    node_to_inst.reserve(static_cast<std::size_t>(ninst) * 2);
    for (int inst = 0; inst < ninst; ++inst) {
        const int permuted_node = node_indices[inst];
        node_to_inst[permuted_node].push_back(inst);
    }

    mech_permuted_node_to_inst_[mech] = std::move(node_to_inst);
    return true;
}

int SimRuntimeCore::resolve_mech_index_by_node_internal_(const std::string& mech,
                                                         int node_idx_original,
                                                         int slot) {
    if (sim_ == nullptr) {
        printf("resolve_mech_index_by_node: Simulate not initialized\n");
        return -1;
    }
    if (sim_->neuron_group_list.empty() || sim_->neuron_group_list[0] == nullptr) {
        printf("resolve_mech_index_by_node: neuron_group_list is empty\n");
        return -1;
    }
    auto* group = sim_->neuron_group_list[0];
    if (node_idx_original < 0 || node_idx_original >= group->len) {
        printf("resolve_mech_index_by_node: invalid node index %d (len=%d)\n", node_idx_original, group->len);
        return -1;
    }
    if (slot < 0) {
        printf("resolve_mech_index_by_node: invalid slot %d\n", slot);
        return -1;
    }

    int permuted_node = node_idx_original;
    if (group->permute != nullptr) {
        permuted_node = group->permute[node_idx_original];
    }

    auto cache_it = mech_permuted_node_to_inst_.find(mech);
    if (cache_it == mech_permuted_node_to_inst_.end()) {
        if (!build_mech_node_index_cache_(mech)) {
            return -1;
        }
        cache_it = mech_permuted_node_to_inst_.find(mech);
        if (cache_it == mech_permuted_node_to_inst_.end()) {
            return -1;
        }
    }

    const auto& node_to_inst = cache_it->second;
    auto inst_it = node_to_inst.find(permuted_node);
    if (inst_it == node_to_inst.end()) {
        printf("resolve_mech_index_by_node: mech '%s' has no instance at node(original=%d permuted=%d)\n",
               mech.c_str(),
               node_idx_original,
               permuted_node);
        return -1;
    }
    const auto& instances = inst_it->second;
    const auto occ = static_cast<std::size_t>(slot);
    if (occ >= instances.size()) {
        printf("resolve_mech_index_by_node: mech '%s' slot out of range at node(original=%d permuted=%d), "
               "slot=%d available=%zu\n",
               mech.c_str(),
               node_idx_original,
               permuted_node,
               slot,
               instances.size());
        return -1;
    }
    return instances[occ];
}

int SimRuntimeCore::resolve_segment_node_index_by_loc_internal_(int gid,
                                                                const std::string& section_name,
                                                                double loc) {
    if (!std::isfinite(loc)) {
        printf("resolve_segment_node_index_by_loc: loc must be finite (got=%g)\n", loc);
        return -1;
    }

    int seg_idx = -1;
    int node_idx_original = -1;
    if (!resolve_segment_info_by_loc_internal_(
            gid, section_name, loc, &seg_idx, nullptr, &node_idx_original)) {
        return -1;
    }
    if (node_idx_original < 0) {
        printf("resolve_segment_node_index_by_loc: failed to map segment->node "
               "(gid=%d section=%s loc=%g seg_idx=%d)\n",
               gid,
               section_name.c_str(),
               loc,
               seg_idx);
        return -1;
    }
    return node_idx_original;
}

bool SimRuntimeCore::resolve_segment_info_by_loc_internal_(int gid,
                                                           const std::string& section_name,
                                                           double loc,
                                                           int* out_seg_idx,
                                                           int* out_nseg,
                                                           int* out_node_idx_original) const {
    if (out_seg_idx) {
        *out_seg_idx = -1;
    }
    if (out_nseg) {
        *out_nseg = -1;
    }
    if (out_node_idx_original) {
        *out_node_idx_original = -1;
    }
    if (!std::isfinite(loc)) {
        printf("resolve_segment_info_by_loc: loc must be finite (got=%g)\n", loc);
        return false;
    }
    if (!neurong_morph_layout_.has_value()) {
        printf("resolve_segment_info_by_loc: morphology layout not available (load_celltemplate_morphology first)\n");
        return false;
    }

    const auto& morph = *neurong_morph_layout_;
    if (gid < 0 || gid >= morph.num_cells_total) {
        printf("resolve_segment_info_by_loc: invalid gid=%d (num_cells_total=%d)\n", gid, morph.num_cells_total);
        return false;
    }
    const auto cell = static_cast<std::size_t>(gid);
    if (cell >= morph.cell_template_id.size() || cell >= morph.cell_nonroot_base.size()) {
        printf("resolve_segment_info_by_loc: internal cell layout mismatch for gid=%d\n", gid);
        return false;
    }
    const std::size_t tpl_id = morph.cell_template_id[cell];
    if (tpl_id >= morph.templates.size()) {
        printf("resolve_segment_info_by_loc: invalid template id=%zu for gid=%d\n", tpl_id, gid);
        return false;
    }

    const auto& tpl = morph.templates[tpl_id];
    const auto sec_it = tpl.section_name_to_id.find(section_name);
    if (sec_it == tpl.section_name_to_id.end()) {
        printf("resolve_segment_info_by_loc: section '%s' not found in template '%s'\n",
               section_name.c_str(),
               tpl.name.c_str());
        return false;
    }

    const auto sec_idx = static_cast<std::size_t>(sec_it->second);
    if (sec_idx >= tpl.section_nseg.size()) {
        printf("resolve_segment_info_by_loc: section metadata mismatch for section '%s'\n",
               section_name.c_str());
        return false;
    }

    const int nseg = tpl.section_nseg[sec_idx];
    if (nseg <= 0) {
        printf("resolve_segment_info_by_loc: invalid nseg=%d for section '%s'\n",
               nseg,
               section_name.c_str());
        return false;
    }

    std::int32_t seg_idx = -1;
    if (!neurong_morph::resolve_segment_index_for_nseg(static_cast<std::int32_t>(nseg), loc, &seg_idx)) {
        printf("resolve_segment_info_by_loc: failed for section='%s' loc=%g nseg=%d\n",
               section_name.c_str(),
               loc,
               nseg);
        return false;
    }
    if (out_seg_idx) {
        *out_seg_idx = static_cast<int>(seg_idx);
    }
    if (out_nseg) {
        *out_nseg = nseg;
    }
    if (out_node_idx_original) {
        const int tpl_node = tpl.section_node_base_id[sec_idx] + static_cast<int>(seg_idx);
        if (tpl_node == 0) {
            *out_node_idx_original = gid;
        } else {
            const std::size_t nonroot_base = morph.cell_nonroot_base[cell];
            const std::size_t original = nonroot_base + static_cast<std::size_t>(tpl_node - 1);
            *out_node_idx_original = static_cast<int>(original);
        }
    }
    return true;
}

int SimRuntimeCore::resolve_mech_index_by_loc_internal_(const std::string& mech,
                                                        int gid,
                                                        const std::string& section_name,
                                                        double loc,
                                                        int slot,
                                                        double* out_resolved_loc,
                                                        int* out_resolved_segment_index) {
    if (out_resolved_loc != nullptr) {
        *out_resolved_loc = std::numeric_limits<double>::quiet_NaN();
    }
    if (out_resolved_segment_index != nullptr) {
        *out_resolved_segment_index = -1;
    }
    if (slot < 0) {
        printf("resolve_mech_index_by_loc: invalid slot %d\n", slot);
        return -1;
    }
    if (!std::isfinite(loc)) {
        printf("resolve_mech_index_by_loc: loc must be finite (got=%g)\n", loc);
        return -1;
    }
    if (loc < 0.0 || loc > 1.0) {
        printf("resolve_mech_index_by_loc: loc out of range (0 <= loc <= 1), got=%g\n", loc);
        return -1;
    }

    const double loc_x = loc;
    const std::string mech_name = to_lower_ascii(mech);
    const bool is_global = mech_name.empty() || mech_name == "global";
    if (is_global) {
        // For global.* without variable-type context, keep legacy behavior:
        // map by-loc to the segment node that loc falls into.
        // Variable-type-specific NEURON alignment (e.g. v/i_membrane_ via
        // node_exact-style boundary semantics) is handled by higher-level
        // callers that know `var`.
        if (out_resolved_loc != nullptr) {
            *out_resolved_loc = loc_x;
        }
        return resolve_segment_node_index_by_loc_internal_(gid, section_name, loc_x);
    }

    // Do not canonicalize user loc to segment-center loc here.
    // Keep handle metadata aligned with caller-provided by-loc addressing.
    if (out_resolved_loc != nullptr) {
        *out_resolved_loc = loc_x;
    }

    int seg_idx = -1;
    int seg_node_idx_original = -1;
    if (!resolve_segment_info_by_loc_internal_(
            gid, section_name, loc_x, &seg_idx, nullptr, &seg_node_idx_original)) {
        return -1;
    }
    if (out_resolved_segment_index != nullptr) {
        *out_resolved_segment_index = seg_idx;
    }

    if (seg_node_idx_original < 0) {
        printf("resolve_mech_index_by_loc: failed to map segment->node "
               "(mech=%s gid=%d section=%s loc=%g seg_idx=%d)\n",
               mech.c_str(),
               gid,
               section_name.c_str(),
               loc_x,
               seg_idx);
        return -1;
    }

    // Non-global mechanism instances are segment-based; resolve by the segment
    // node that loc falls into (including boundary loc=0/1 folded to first/last
    // segment), consistent with mechanism instance materialization.
    const int node_idx_original = seg_node_idx_original;
    const int mech_idx = resolve_mech_index_by_node_internal_(mech, node_idx_original, slot);
    if (mech_idx < 0) {
        printf("resolve_mech_index_by_loc: failed to resolve mech index "
               "(mech=%s gid=%d section=%s loc=%g node=%d slot=%d)\n",
               mech.c_str(),
               gid,
               section_name.c_str(),
               loc_x,
               node_idx_original,
               slot);
        return -1;
    }
    return mech_idx;
}

SimRuntimeCore::SimRuntimeCore() = default;
SimRuntimeCore::~SimRuntimeCore() {
    sim_.reset();
    for (auto& [id, batch] : input_batches_) {
        release_input_batch_resources_(batch);
    }
    input_batches_.clear();

    if (flush_values_device_ != nullptr) {
        gpu_mem_free((void**)&flush_values_device_);
        flush_values_device_ = nullptr;
    }
    if (flush_gpu_ptrs_device_ != nullptr) {
        gpu_mem_free((void**)&flush_gpu_ptrs_device_);
        flush_gpu_ptrs_device_ = nullptr;
    }
    flush_capacity_ = 0;

    if (gather_gpu_ptrs_device_ != nullptr) {
        gpu_mem_free((void**)&gather_gpu_ptrs_device_);
        gather_gpu_ptrs_device_ = nullptr;
    }
    if (gather_values_device_ != nullptr) {
        gpu_mem_free((void**)&gather_values_device_);
        gather_values_device_ = nullptr;
    }
    gather_capacity_ = 0;
}

int SimRuntimeCore::SimInitParam::get_permute_type() const {
    if (permute_type == -1) {
        return mode == GPU ? 3 : 0;
    }
    return permute_type;
}

double SimRuntimeCore::SimInitParam::get_dt() const {
    if (dt < 0) {
        double dt_from_file = coreneuron::global_var_map.at("dt")[0];
        printf("dt not set, using dt from global.dat: %f\n", dt_from_file);
        return dt_from_file;
    }
    return dt;
}

int SimRuntimeCore::set_data_path(const std::string& path) {
    if (sim_ != nullptr) {
        printf("Cannot set data path after simulator is initialized\n");
        return -1;
    }
    sim_param_.data_path = path;
    return 0;
}

int SimRuntimeCore::set_device(const std::string& dev) {
    if (dev == "cpu") {
        sim_param_.mode = CPU;
    } else if (dev == "gpu") {
        sim_param_.mode = GPU;
    } else {
        printf("Unknown device: %s\n", dev.c_str());
        return -1;
    }
    return 0;
}

int SimRuntimeCore::set_source(const std::string& source) {
    const auto normalized = to_lower_ascii(source);
    ModelSource parsed = ModelSource::Unset;
    if (normalized == "neuron") {
        parsed = ModelSource::Neuron;
    } else if (normalized == "neurong") {
        parsed = ModelSource::Neurong;
    } else {
        printf("Unknown source: %s (expected 'neuron' or 'neurong')\n", source.c_str());
        return -1;
    }

    if (sim_ != nullptr && parsed != model_source_) {
        printf("Cannot switch source after simulator is initialized (current=%s, requested=%s)\n",
               source_to_string(model_source_), source_to_string(parsed));
        return -1;
    }

    model_source_ = parsed;
    return 0;
}

std::string SimRuntimeCore::get_source() const {
    return std::string(source_to_string(model_source_));
}

int SimRuntimeCore::set_output_dir(const std::string& dir) {
    if (sim_ != nullptr) {
        printf("Output dir must be set before loading model\n");
        return -1;
    }
    sim_param_.output_dir = dir;
    return 0;
}

int SimRuntimeCore::load_mech_library(const std::string& path) {
    if (is_process_mech_registry_frozen()) {
        printf("Cannot load mech plugin after model processing has begun in this process; "
               "load plugins before the first load_model()/load_celltemplate_morphology().\n");
        return -1;
    }
    if (sim_ != nullptr) {
        printf("Cannot load mech plugin after simulator is initialized; load plugins before load_model()/load_celltemplate_morphology().\n");
        return -1;
    }
    if (path.empty()) {
        printf("load_mech_library: empty path\n");
        return -1;
    }
    return mech_plugin_manager_.load_library(path);
}

int SimRuntimeCore::register_ion_meta(const std::string& ion_name,
                                      double charge,
                                      double default_conci,
                                      double default_conco) {
    if (sim_ != nullptr) {
        printf("register_ion_meta must be called before load_model()/load_celltemplate_morphology().\n");
        return -1;
    }
    try {
        ::register_ion_meta(ion_name, IonMeta{charge, default_conci, default_conco});
    } catch (const std::exception& ex) {
        printf("register_ion_meta failed for '%s': %s\n", ion_name.c_str(), ex.what());
        return -1;
    }
    return 0;
}

std::vector<std::string> SimRuntimeCore::get_loaded_mech_libraries() const {
    std::vector<std::string> out;
    out.reserve(mech_plugin_manager_.loaded_plugins().size());
    for (const auto& plugin : mech_plugin_manager_.loaded_plugins()) {
        out.push_back(plugin.path);
    }
    return out;
}

int SimRuntimeCore::set_permute_type(int type) {
    sim_param_.permute_type = type;
    return 0;
}

int SimRuntimeCore::set_dt(double dt) {
    if (sim_ == nullptr) {
        sim_param_.dt = dt;
    } else {
        sim_->dt = dt;
    }
    return 0;
}

double SimRuntimeCore::get_dt() const {
    if (sim_ == nullptr) {
        return sim_param_.get_dt();
    }
    return sim_->dt;
}

void SimRuntimeCore::set_user_mod_num(int num) {
    sim_param_.user_mod_num = num;
}

int SimRuntimeCore::set_recorder_buffer_capacity(int capacity) {
    if (capacity <= 0) {
        printf("Invalid recorder buffer capacity: %d\n", capacity);
        return -1;
    }
    if (sim_ != nullptr) {
        printf("Cannot set recorder buffer capacity after simulator is initialized\n");
        return -1;
    }
    sim_param_.recorder_buffer_capacity = capacity;
    return 0;
}

int SimRuntimeCore::set_recorder_storage_dtype(const std::string& dtype) {
    if (sim_ != nullptr) {
        printf("Cannot change recorder storage dtype after simulator is initialized\n");
        return -1;
    }
    std::string lower = dtype;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower == "fp32" || lower == "float32" || lower == "f32") {
        sim_param_.recorder_use_fp32 = true;
    } else if (lower == "fp64" || lower == "float64" || lower == "f64") {
        sim_param_.recorder_use_fp32 = false;
    } else {
        printf("Invalid recorder storage dtype: %s (expected fp32/fp64)\n", dtype.c_str());
        return -1;
    }
    printf("Recorder storage dtype: %s\n", sim_param_.recorder_use_fp32 ? "fp32" : "fp64");
    return 0;
}

int SimRuntimeCore::set_recorder_start_time(double start_time_ms) {
    if (!std::isfinite(start_time_ms) || start_time_ms < 0.0) {
        printf("Invalid recorder start time: %f\n", start_time_ms);
        return -1;
    }
    sim_param_.recorder_start_time_ms = start_time_ms;
    if (sim_ != nullptr) {
        return sim_->set_recorder_sampling(sim_param_.recorder_start_time_ms, sim_param_.recorder_stride_steps);
    }
    return 0;
}

int SimRuntimeCore::set_recorder_stride(int stride_steps) {
    if (stride_steps <= 0) {
        printf("Invalid recorder stride: %d\n", stride_steps);
        return -1;
    }
    sim_param_.recorder_stride_steps = stride_steps;
    if (sim_ != nullptr) {
        return sim_->set_recorder_sampling(sim_param_.recorder_start_time_ms, sim_param_.recorder_stride_steps);
    }
    return 0;
}

int SimRuntimeCore::maybe_autoload_default_mech_library_() {
    if (default_mech_autoload_attempted_) {
        return 0;
    }
    default_mech_autoload_attempted_ = true;

    if (sim_ != nullptr) {
        return 0;
    }

    const auto candidate = resolve_default_mechlib_candidate();
    if (!candidate.has_value()) {
        return 0;
    }
    if (!std::filesystem::exists(*candidate)) {
        return 0;
    }

    const auto canonical_candidate = canonicalize_existing_path(*candidate);
    for (const auto& plugin : mech_plugin_manager_.loaded_plugins()) {
        if (canonicalize_existing_path(plugin.path) == canonical_candidate) {
            return 0;
        }
    }

    return load_mech_library(canonical_candidate.string());
}

int SimRuntimeCore::load_model() {
    using enum BufferEnable;
    if (sim_ != nullptr) {
        printf("Simulate already initialized\n");
        return -1;
    }
    if (model_source_ == ModelSource::Neurong) {
        printf("Source is 'neurong' (in-memory). Use load_celltemplate_morphology() instead of load_model().\n");
        return -1;
    }
    if (model_source_ == ModelSource::Unset) {
        model_source_ = ModelSource::Neuron;
    }
    if (sim_param_.data_path.empty()) {
        printf("Data path not set\n");
        return -1;
    }
    if (maybe_autoload_default_mech_library_() != 0) {
        printf("Default mech autoload failed before load_model().\n");
        return -1;
    }

    std::string filesdat = sim_param_.data_path + "/files.dat";
    permute_type = sim_param_.get_permute_type();
    int ngroup = read_coredat(
        coredata_arr_, sim_param_.data_path.c_str(), filesdat.c_str(), 0, false, sim_param_.user_mod_num);

    BufferEnable buffer_enable = IPC;  // default: IPC only
    if (sim_param_.enable_hdf5) {
        buffer_enable = buffer_enable | HDF5;
    }

    sim_ = std::make_unique<Simulate>(
        sim_param_.mode, buffer_enable, sim_param_.recorder_buffer_capacity, sim_param_.recorder_use_fp32);
    if (!sim_param_.output_dir.empty()) {
        sim_->output_folder = sim_param_.output_dir;
    }
    if (sim_->set_recorder_sampling(sim_param_.recorder_start_time_ms, sim_param_.recorder_stride_steps) != 0) {
        printf("Failed to configure recorder sampling (start=%f, stride=%d)\n",
               sim_param_.recorder_start_time_ms,
               sim_param_.recorder_stride_steps);
        sim_.reset();
        return -1;
    }

    sim_->tstop = -1;
    sim_->dt = sim_param_.get_dt();
    sim_->permute_type = sim_param_.get_permute_type();
    data_format_trans(sim_->neuron_group_list, coredata_arr_, ngroup, sim_param_.mode, sim_->dt);
    neurong_morph_layout_.reset();
    neurong_template_gid_ranges_.clear();
    neurong_biophys_applied_ = false;
    neurong_next_type_ = 0;
    mech_permuted_node_to_inst_.clear();


    std::vector<VarDescriptor> pre_monitors(sim_param_.pre_registered_monitors.begin(),
                                           sim_param_.pre_registered_monitors.end());
    monitor_to_handle_ = sim_->init_monitor_data_sets(pre_monitors);
    std::vector<VarDescriptor> pre_window_observers(sim_param_.pre_registered_window_observers.begin(),
                                                    sim_param_.pre_registered_window_observers.end());
    window_observer_to_handle_ = sim_->init_window_observers(pre_window_observers);
    freeze_process_mech_registry();
    return 0;
}

std::vector<std::tuple<std::string, int, int>> SimRuntimeCore::load_celltemplate_morphology(
    const std::vector<neurong_celltemplate::CellTemplateMorphSpec>& templates,
    std::optional<std::uint64_t> random_seed) {
    using enum BufferEnable;

    if (sim_ != nullptr) {
        if (model_source_ == ModelSource::Neuron) {
            printf("Source is 'neuron' (CoreNEURON export). Use load_model() instead of load_celltemplate_morphology().\n");
            return {};
        }
        printf(
            "Simulate already initialized; replacing existing in-memory model. "
            "Existing handles/input batches will be invalidated.\n");
        for (auto& [id, batch] : input_batches_) {
            (void)id;
            release_input_batch_resources_(batch);
        }
        input_batches_.clear();
        next_input_batch_id_ = 0;
        var_pointer_cache_.clear();
        cpu_ptr_to_handle_.clear();
        dirty_handles_.clear();
        flush_cpu_values_.clear();
        flush_gpu_ptrs_.clear();
        monitor_to_handle_.clear();
        window_observer_to_handle_.clear();
        neurong_morph_layout_.reset();
        neurong_template_gid_ranges_.clear();
        neurong_biophys_applied_ = false;
        neurong_next_type_ = 0;
        mech_permuted_node_to_inst_.clear();
        sim_.reset();
    }
    if (model_source_ == ModelSource::Neuron) {
        printf("Source is 'neuron' (CoreNEURON export). Use load_model() instead of load_celltemplate_morphology().\n");
        return {};
    }
    if (model_source_ == ModelSource::Unset) {
        model_source_ = ModelSource::Neurong;
    }
    if (maybe_autoload_default_mech_library_() != 0) {
        printf("Default mech autoload failed before load_celltemplate_morphology().\n");
        return {};
    }

    neurong_celltemplate::LoadOptions opt{};
    opt.dt = sim_param_.get_dt();
    opt.permute_type = sim_param_.get_permute_type();
    opt.output_dir = sim_param_.output_dir;
    opt.random_seed = random_seed;
    opt.buffer_enable = IPC;
    if (sim_param_.enable_hdf5) {
        opt.buffer_enable = opt.buffer_enable | HDF5;
    }

    auto result = neurong_celltemplate::load_celltemplate_morphology(opt, templates);
    sim_ = std::move(result.sim);
    if (sim_ == nullptr) {
        printf("load_celltemplate_morphology: builder returned null simulator\n");
        return {};
    }
    if (sim_->set_recorder_sampling(sim_param_.recorder_start_time_ms, sim_param_.recorder_stride_steps) != 0) {
        printf("Failed to configure recorder sampling (start=%f, stride=%d)\n",
               sim_param_.recorder_start_time_ms,
               sim_param_.recorder_stride_steps);
        sim_.reset();
        return {};
    }
    monitor_to_handle_.clear();
    neurong_morph_layout_ = std::move(result.morph_layout);
    neurong_template_gid_ranges_.clear();
    neurong_template_gid_ranges_.reserve(result.gid_ranges.size());
    for (const auto& range : result.gid_ranges) {
        neurong_template_gid_ranges_.emplace_back(range.name, range.gid_begin, range.gid_end_exclusive);
    }
    neurong_biophys_applied_ = false;
    neurong_next_type_ = 0;
    mech_permuted_node_to_inst_.clear();
    freeze_process_mech_registry();

    return neurong_template_gid_ranges_;
}

std::vector<std::tuple<int, int, int>> SimRuntimeCore::apply_celltemplate_biophysics(
    const std::vector<neurong_celltemplate::CellTemplateBiophysSpec>& templates,
    double celsius,
    const std::vector<int>& record_gids) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized. Load morphology first.\n");
        return {};
    }
    if (model_source_ != ModelSource::Neurong) {
        printf("apply_celltemplate_biophysics() is only available for source='neurong'.\n");
        return {};
    }
    if (!neurong_morph_layout_.has_value()) {
        printf("Missing in-memory morphology layout. Call load_celltemplate_morphology() first.\n");
        return {};
    }
    if (neurong_biophys_applied_) {
        printf("Biophysics already applied for this in-memory model.\n");
        return {};
    }

    std::vector<VarDescriptor> extra_monitors(sim_param_.pre_registered_monitors.begin(),
                                              sim_param_.pre_registered_monitors.end());
    try {
        auto biophys = neurong_celltemplate::apply_celltemplate_biophysics(
            *sim_,
            *neurong_morph_layout_,
            templates,
            celsius,
            record_gids,
            extra_monitors,
            &neurong_next_type_);
        monitor_to_handle_ = std::move(biophys.monitor_to_handle);
        std::vector<VarDescriptor> pre_window_observers(sim_param_.pre_registered_window_observers.begin(),
                                                        sim_param_.pre_registered_window_observers.end());
        window_observer_to_handle_ = sim_->init_window_observers(pre_window_observers);
        neurong_biophys_applied_ = true;
        mech_permuted_node_to_inst_.clear();
        return biophys.record_handles;
    } catch (const std::exception& e) {
        printf("apply_celltemplate_biophysics failed: %s\n", e.what());
        return {};
    }
}

std::vector<std::tuple<std::string, int, int>> SimRuntimeCore::get_celltemplate_gid_ranges() const {
    return neurong_template_gid_ranges_;
}

const neurong_biophysical::CellTemplateMorphLayout* SimRuntimeCore::in_memory_morph_layout() const {
    if (!neurong_morph_layout_.has_value()) {
        return nullptr;
    }
    return &(*neurong_morph_layout_);
}

bool SimRuntimeCore::is_in_memory_biophysics_applied() const {
    return neurong_biophys_applied_;
}

int SimRuntimeCore::resolve_node_index(int gid, const std::string& section_name, int seg_index) const {
    if (!neurong_morph_layout_.has_value()) {
        printf("resolve_node_index: morphology layout not available (load_celltemplate_morphology first)\n");
        return -1;
    }
    const auto& morph = *neurong_morph_layout_;
    if (gid < 0 || gid >= morph.num_cells_total) {
        printf("resolve_node_index: invalid gid=%d (num_cells_total=%d)\n", gid, morph.num_cells_total);
        return -1;
    }

    const auto cell = static_cast<std::size_t>(gid);
    if (cell >= morph.cell_template_id.size() || cell >= morph.cell_nonroot_base.size()) {
        printf("resolve_node_index: internal cell layout mismatch for gid=%d\n", gid);
        return -1;
    }

    const std::size_t tpl_id = morph.cell_template_id[cell];
    if (tpl_id >= morph.templates.size()) {
        printf("resolve_node_index: invalid template id=%zu for gid=%d\n", tpl_id, gid);
        return -1;
    }
    const auto& tpl = morph.templates[tpl_id];
    auto sec_it = tpl.section_name_to_id.find(section_name);
    if (sec_it == tpl.section_name_to_id.end()) {
        printf("resolve_node_index: section '%s' not found in template '%s'\n",
               section_name.c_str(),
               tpl.name.c_str());
        return -1;
    }

    const auto sec_idx = static_cast<std::size_t>(sec_it->second);
    if (sec_idx >= tpl.section_node_base_id.size() || sec_idx >= tpl.section_nseg.size()) {
        printf("resolve_node_index: section metadata mismatch for section '%s'\n", section_name.c_str());
        return -1;
    }

    const int nseg = tpl.section_nseg[sec_idx];
    if (seg_index < 0 || seg_index >= nseg) {
        printf("resolve_node_index: invalid seg index=%d for section '%s' (nseg=%d)\n",
               seg_index,
               section_name.c_str(),
               nseg);
        return -1;
    }

    const int tpl_node = tpl.section_node_base_id[sec_idx] + seg_index;
    if (tpl_node == 0) {
        return gid;
    }

    const std::size_t nonroot_base = morph.cell_nonroot_base[cell];
    const std::size_t original = nonroot_base + static_cast<std::size_t>(tpl_node - 1);
    return static_cast<int>(original);
}

int SimRuntimeCore::resolve_segment_index_by_loc(int gid,
                                                 const std::string& section_name,
                                                 double loc) const {
    int seg_idx = -1;
    if (!resolve_segment_info_by_loc_internal_(gid, section_name, loc, &seg_idx, nullptr)) {
        return -1;
    }
    return seg_idx;
}

double SimRuntimeCore::resolve_segment_center_loc_by_loc(int gid,
                                                         const std::string& section_name,
                                                         double loc) const {
    int seg_idx = -1;
    int nseg = -1;
    if (!resolve_segment_info_by_loc_internal_(gid, section_name, loc, &seg_idx, &nseg)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double denom = 2.0 * static_cast<double>(nseg);
    return (2.0 * static_cast<double>(seg_idx) + 1.0) / denom;
}

double SimRuntimeCore::resolve_effective_loc_by_loc(const std::string& mech,
                                                    int gid,
                                                    const std::string& section_name,
                                                    double loc) const {
    if (!std::isfinite(loc)) {
        printf("resolve_effective_loc_by_loc: loc must be finite (got=%g)\n", loc);
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (loc < 0.0 || loc > 1.0) {
        printf("resolve_effective_loc_by_loc: loc out of range (0 <= loc <= 1), got=%g\n", loc);
        return std::numeric_limits<double>::quiet_NaN();
    }
    (void)mech;
    (void)gid;
    (void)section_name;
    return loc;
}

int SimRuntimeCore::resolve_node_index_by_loc(int gid,
                                              const std::string& section_name,
                                              double loc) const {
    if (!neurong_morph_layout_.has_value()) {
        printf("resolve_node_index_by_loc: morphology layout not available (load_celltemplate_morphology first)\n");
        return -1;
    }
    const auto& morph = *neurong_morph_layout_;
    if (gid < 0 || gid >= morph.num_cells_total) {
        printf("resolve_node_index_by_loc: invalid gid=%d (num_cells_total=%d)\n", gid, morph.num_cells_total);
        return -1;
    }

    const auto cell = static_cast<std::size_t>(gid);
    if (cell >= morph.cell_template_id.size()) {
        printf("resolve_node_index_by_loc: internal cell layout mismatch for gid=%d\n", gid);
        return -1;
    }
    const std::size_t tpl_id = morph.cell_template_id[cell];
    if (tpl_id >= morph.templates.size()) {
        printf("resolve_node_index_by_loc: invalid template id=%zu for gid=%d\n", tpl_id, gid);
        return -1;
    }
    const auto& tpl = morph.templates[tpl_id];
    std::int32_t tpl_node = -1;
    if (!neurong_morph::resolve_template_node_by_loc(tpl.section_name_to_id,
                                                     tpl.section_node_base_id,
                                                     tpl.section_nseg,
                                                     tpl.section_parent_node_id,
                                                     section_name,
                                                     loc,
                                                     &tpl_node)) {
        printf("resolve_node_index_by_loc: failed for template='%s' section='%s' loc=%g\n",
               tpl.name.c_str(),
               section_name.c_str(),
               loc);
        return -1;
    }
    if (tpl_node == 0) {
        return gid;
    }

    const std::size_t nonroot_base = morph.cell_nonroot_base[cell];
    const std::size_t original = nonroot_base + static_cast<std::size_t>(tpl_node - 1);
    return static_cast<int>(original);
}

int SimRuntimeCore::resolve_segment_node_index_by_loc(int gid,
                                                      const std::string& section_name,
                                                      double loc) {
    return resolve_segment_node_index_by_loc_internal_(gid, section_name, loc);
}

int SimRuntimeCore::resolve_mech_index_by_loc(const std::string& mech,
                                              int gid,
                                              const std::string& section_name,
                                              double loc,
                                              int slot) {
    return resolve_mech_index_by_loc_internal_(mech, gid, section_name, loc, slot);
}

int SimRuntimeCore::set_spike_output_enabled(bool enable) {
    spike_output_enabled_ = enable;
    printf("Spike file output %s\n", enable ? "enabled" : "disabled");
    return 0;
}

bool SimRuntimeCore::is_spike_output_enabled() const {
    return spike_output_enabled_;
}

int SimRuntimeCore::add_monitor(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return add_monitor_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::add_monitor_with_array(const std::string& mech,
                                           const std::string& var,
                                           int node_or_mech_idx,
                                           int array_index) {
    VarDescriptor monitor;
    monitor.mech = mech;
    monitor.var = var;
    monitor.node_or_mech_idx = node_or_mech_idx;
    monitor.array_index = array_index;

    if (sim_ != nullptr) {
        if (model_source_ == ModelSource::Neurong && neurong_morph_layout_.has_value() &&
            !neurong_biophys_applied_) {
            if (sim_param_.pre_registered_monitors.find(monitor) != sim_param_.pre_registered_monitors.end()) {
                return -1;
            }
            sim_param_.pre_registered_monitors.insert(monitor);
            return -1;
        }
        auto [var_ptr_cpu, var_ptr_gpu] = sim_->getVarPtr(monitor, false);
        if (var_ptr_cpu == nullptr) {
            printf("Monitor variable not found: %s %s %d\n", monitor.mech.c_str(), monitor.var.c_str(),
                   monitor.node_or_mech_idx);
            return -1;
        }

        RecordPoint recordPoint;
        recordPoint.var_ptr_cpu = var_ptr_cpu;
        recordPoint.var_ptr_gpu = var_ptr_gpu;

        int handle = sim_->hdf5_manager.push_back(monitor, recordPoint);
        monitor_to_handle_[monitor] = handle;
        return handle;
    }

    // Model not loaded yet: preregister, dedup via set.
    if (sim_param_.pre_registered_monitors.find(monitor) != sim_param_.pre_registered_monitors.end()) {
        return -1;
    }
    sim_param_.pre_registered_monitors.insert(monitor);
    return -1;
}

int SimRuntimeCore::get_monitor_handle(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return get_monitor_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::get_monitor_handle_with_array(const std::string& mech,
                                                  const std::string& var,
                                                  int node_or_mech_idx,
                                                  int array_index) {
    VarDescriptor monitor;
    monitor.mech = mech;
    monitor.var = var;
    monitor.node_or_mech_idx = node_or_mech_idx;
    monitor.array_index = array_index;

    if (sim_ == nullptr) {
        printf("Model not loaded, cannot get monitor handle\n");
        return -1;
    }

    auto it = monitor_to_handle_.find(monitor);
    if (it != monitor_to_handle_.end()) {
        return it->second;
    }

    return -1;
}

int SimRuntimeCore::flush_recorders() {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    if (sim_->mode == CPU) {
        sim_->hdf5_manager.flush_cpu();
        return 0;
    }
    sim_->hdf5_manager.flush_gpu();
    return 0;
}

int SimRuntimeCore::get_monitor_sample_count(int handle) {
    if (sim_ == nullptr || handle == -1) {
        return -1;
    }
    auto search_result = sim_->hdf5_manager.get_single_irq_buffer(handle);
    if (!search_result.has_value()) {
        return -1;
    }
    return static_cast<int>(search_result.value().get().size());
}

std::vector<double> SimRuntimeCore::get_monitor_data(int handle) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return {};
    }
    if (handle == -1) {
        printf("Handle is -1, cannot get data for pending monitor\n");
        return {};
    }
    auto search_result = sim_->hdf5_manager.get_single_irq_buffer(handle);
    if (search_result.has_value()) {
        auto& var_vector = search_result.value().get();
        return var_vector;
    }
    return {};
}

std::map<int, std::vector<double>> SimRuntimeCore::get_multiple_monitor_data(const std::vector<int>& handles) {
    std::map<int, std::vector<double>> result;
    for (int h : handles) {
        result[h] = get_monitor_data(h);
    }
    return result;
}

int SimRuntimeCore::get_multiple_monitor_data_f32(const std::vector<int>& handles,
                                                  float* out_cpu,
                                                  int n_handle,
                                                  int n_step) {
    return get_multiple_monitor_data_f32_from(handles, 0, out_cpu, n_handle, n_step);
}

int SimRuntimeCore::get_multiple_monitor_data_f32_from(const std::vector<int>& handles,
                                                       int start_step,
                                                       float* out_cpu,
                                                       int n_handle,
                                                       int n_step) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (out_cpu == nullptr) {
        printf("get_multiple_monitor_data_f32: out_cpu is null\n");
        return -1;
    }
    if (n_handle < 0 || n_step < 0) {
        printf("get_multiple_monitor_data_f32: invalid shape (%d, %d)\n", n_handle, n_step);
        return -1;
    }
    if (start_step < 0) {
        printf("get_multiple_monitor_data_f32: invalid start_step %d\n", start_step);
        return -1;
    }
    if (static_cast<int>(handles.size()) != n_handle) {
        printf("get_multiple_monitor_data_f32: handles size mismatch (%zu vs %d)\n", handles.size(), n_handle);
        return -1;
    }
    if (n_handle == 0 || n_step == 0) {
        return 0;
    }

    for (int i = 0; i < n_handle; ++i) {
        const int handle = handles[i];
        if (handle == -1) {
            printf("get_multiple_monitor_data_f32: invalid handle -1 at index %d\n", i);
            return -1;
        }
        auto search_result = sim_->hdf5_manager.get_single_irq_buffer(handle);
        if (!search_result.has_value()) {
            printf("get_multiple_monitor_data_f32: monitor handle %d has no data\n", handle);
            return -1;
        }
        auto& src = search_result.value().get();

        float* dst = out_cpu + static_cast<size_t>(i) * static_cast<size_t>(n_step);
        std::fill(dst, dst + n_step, 0.0f);
        const int src_size = static_cast<int>(src.size());
        const int read_start = std::min(start_step, src_size);
        const int available = src_size - read_start;
        const int copy_n = std::min(available, n_step);
        auto src_begin = src.begin() + read_start;
        std::transform(src_begin, src_begin + copy_n, dst, [](double v) {
            return static_cast<float>(v);
        });
    }
    return 0;
}

int SimRuntimeCore::add_window_observer(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return add_window_observer_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::add_window_observer_with_array(const std::string& mech,
                                                   const std::string& var,
                                                   int node_or_mech_idx,
                                                   int array_index) {
    VarDescriptor observer;
    observer.mech = mech;
    observer.var = var;
    observer.node_or_mech_idx = node_or_mech_idx;
    observer.array_index = array_index;

    if (sim_ == nullptr || (model_source_ == ModelSource::Neurong && !neurong_biophys_applied_)) {
        if (sim_param_.pre_registered_window_observers.find(observer) !=
            sim_param_.pre_registered_window_observers.end()) {
            return -1;
        }
        sim_param_.pre_registered_window_observers.insert(observer);
        return -1;
    }

    std::vector<VarDescriptor> observers{observer};
    auto handles = sim_->init_window_observers(observers);
    auto it = handles.find(observer);
    if (it == handles.end()) return -1;
    window_observer_to_handle_[observer] = it->second;
    return it->second;
}

int SimRuntimeCore::get_window_observer_handle(const std::string& mech,
                                               const std::string& var,
                                               int node_or_mech_idx) {
    return get_window_observer_handle_with_array(mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::get_window_observer_handle_with_array(const std::string& mech,
                                                          const std::string& var,
                                                          int node_or_mech_idx,
                                                          int array_index) {
    VarDescriptor observer;
    observer.mech = mech;
    observer.var = var;
    observer.node_or_mech_idx = node_or_mech_idx;
    observer.array_index = array_index;

    auto it = window_observer_to_handle_.find(observer);
    if (it != window_observer_to_handle_.end()) {
        return it->second;
    }
    return -1;
}

int SimRuntimeCore::begin_window_observer_recording(const std::vector<int>& handles, int step_count) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim_->begin_window_observer_recording(handles, step_count);
}

int SimRuntimeCore::end_window_observer_recording() {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim_->end_window_observer_recording();
}

int SimRuntimeCore::get_window_observer_data_f32(const std::vector<int>& handles,
                                                 float* out_cpu,
                                                 int n_handle,
                                                 int n_step) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    return sim_->window_observer.read_f32(handles, out_cpu, n_handle, n_step);
}

std::vector<std::vector<std::string>> SimRuntimeCore::get_vecplay_keys() {
    std::vector<std::vector<std::string>> result;
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot get vecplay keys\n");
        return result;
    }
    try {
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        auto keys = vecplay_continuous.getAllKeys();
        for (const auto& key : keys) {
            std::vector<std::string> key_str = {key.mech_name, key.var_name, std::to_string(key.instance_id)};
            result.push_back(key_str);
        }
        return result;
    } catch (const std::exception& e) {
        printf("Failed to get vecplay keys: %s\n", e.what());
        return result;
    }
}

static bool validate_vecplay_tvec(const std::vector<double>& tvec) {
    for (size_t i = 0; i < tvec.size(); i++) {
        if (tvec[i] < 0) {
            return false;
        }
        if (i > 0 && tvec[i] < tvec[i - 1]) {
            return false;
        }
    }
    return true;
}

int SimRuntimeCore::add_vecplay(const std::string& mech_name,
                                const std::string& var_name,
                                int instance_id,
                                const std::vector<double>& tvec,
                                const std::vector<double>& yvec) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot add vecplay\n");
        return -1;
    }
    if (tvec.size() != yvec.size()) {
        printf("tvec and yvec must have the same size\n");
        return -1;
    }
    if (tvec.empty()) {
        printf("tvec and yvec cannot be empty\n");
        return -1;
    }
    if (!validate_vecplay_tvec(tvec)) {
        printf("Invalid time vector for vecplay\n");
        return -1;
    }
    try {
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.addVecPlay(mech_name, var_name, instance_id, tvec, yvec);
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to add vecplay: %s\n", e.what());
        return -1;
    }
}

int SimRuntimeCore::update_vecplay(const std::string& mech_name,
                                   const std::string& var_name,
                                   int instance_id,
                                   const std::vector<double>& new_tvec,
                                   const std::vector<double>& new_yvec) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot update vecplay\n");
        return -1;
    }
    if (new_tvec.size() != new_yvec.size()) {
        printf("new_tvec and new_yvec must have the same size\n");
        return -1;
    }
    if (new_tvec.empty()) {
        printf("new_tvec and new_yvec cannot be empty\n");
        return -1;
    }
    if (!validate_vecplay_tvec(new_tvec)) {
        printf("Invalid time vector for vecplay\n");
        return -1;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.updateVecPlay(key, new_tvec, new_yvec);
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to update vecplay: %s\n", e.what());
        return -1;
    }
}

int SimRuntimeCore::remove_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot remove vecplay\n");
        return -1;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        vecplay_continuous.removeVecPlay(key);
        return 0;
    } catch (const std::exception& e) {
        printf("Failed to remove vecplay: %s\n", e.what());
        return -1;
    }
}

bool SimRuntimeCore::has_vecplay(const std::string& mech_name, const std::string& var_name, int instance_id) {
    if (sim_ == nullptr) {
        printf("Model not loaded, cannot check vecplay\n");
        return false;
    }
    try {
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        auto& vecplay_continuous = sim_->neuron_group_list[0]->vec_play_continuous;
        return vecplay_continuous.hasVecPlay(key);
    } catch (const std::exception& e) {
        printf("Failed to check vecplay: %s\n", e.what());
        return false;
    }
}

int SimRuntimeCore::get_variable_handle_by_descriptor_internal_(const std::string& mech,
                                                                const std::string& var,
                                                                int index,
                                                                int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }

    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = index;
    desc.array_index = array_index;

    auto [cpu_var_ptr, gpu_var_ptr] = sim_->getVarPtr(desc, true);
    if (cpu_var_ptr == nullptr && gpu_var_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), index);
        return -1;
    }
    auto it = cpu_ptr_to_handle_.find(cpu_var_ptr);
    if (it != cpu_ptr_to_handle_.end()) {
        return it->second;
    }

    VarPointer vp;
    vp.cpu_ptr = cpu_var_ptr;
    vp.gpu_ptr = gpu_var_ptr;
    vp.descriptor = desc;
    vp.cached_cpu_value = 0.0;
    vp.is_dirty = false;

    int handle = static_cast<int>(var_pointer_cache_.size());
    var_pointer_cache_.push_back(vp);
    cpu_ptr_to_handle_[cpu_var_ptr] = handle;
    return handle;
}

int SimRuntimeCore::get_variable_handle_with_array(const std::string& mech,
                                                   const std::string& var,
                                                   int index,
                                                   int array_index) {
    return get_variable_handle_by_descriptor_internal_(
        mech,
        var,
        index,
        array_index);
}

int SimRuntimeCore::get_variable_handle_with_array_by_loc(const std::string& mech,
                                                          const std::string& var,
                                                          int gid,
                                                          const std::string& section_name,
                                                          double loc,
                                                          int array_index,
                                                          int slot) {
    double resolved_loc = std::numeric_limits<double>::quiet_NaN();
    int resolved_segment_index = -1;
    int resolved_index = -1;
    const std::string mech_name = to_lower_ascii(mech);
    const bool is_global = mech_name.empty() || mech_name == "global";
    if (is_global) {
        if (!std::isfinite(loc)) {
            printf("get_variable_handle_with_array_by_loc: loc must be finite (got=%g)\n", loc);
            return -1;
        }
        if (loc < 0.0 || loc > 1.0) {
            printf("get_variable_handle_with_array_by_loc: loc out of range (0 <= loc <= 1), got=%g\n", loc);
            return -1;
        }
        resolved_loc = loc;
        const std::string var_name = to_lower_ascii(var);
        if (var_name == "v" || var_name == "i_membrane_") {
            resolved_index = resolve_node_index_by_loc(gid, section_name, loc);
        } else {
            resolved_index = resolve_segment_node_index_by_loc_internal_(gid, section_name, loc);
        }
        // Keep debug metadata available for callers/inspection.
        (void)resolve_segment_info_by_loc_internal_(
            gid, section_name, loc, &resolved_segment_index, nullptr);
    } else {
        resolved_index = resolve_mech_index_by_loc_internal_(
            mech,
            gid,
            section_name,
            loc,
            slot,
            &resolved_loc,
            &resolved_segment_index);
    }
    if (resolved_index < 0) {
        return -1;
    }
    const int handle = get_variable_handle_by_descriptor_internal_(
        mech,
        var,
        resolved_index,
        array_index);
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        return handle;
    }
    auto& vp = var_pointer_cache_[handle];
    if (std::isfinite(resolved_loc)) {
        vp.has_resolved_loc = true;
        vp.resolved_loc = resolved_loc;
    } else {
        vp.has_resolved_loc = false;
        vp.resolved_loc = std::numeric_limits<double>::quiet_NaN();
    }
    vp.resolved_segment_index = resolved_segment_index;
    return handle;
}

double SimRuntimeCore::get_handle_resolved_loc(int handle) const {
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("get_handle_resolved_loc: invalid handle=%d\n", handle);
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto& vp = var_pointer_cache_[handle];
    if (!vp.has_resolved_loc) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return vp.resolved_loc;
}

int SimRuntimeCore::get_handle_resolved_segment_index(int handle) const {
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("get_handle_resolved_segment_index: invalid handle=%d\n", handle);
        return -1;
    }
    return var_pointer_cache_[handle].resolved_segment_index;
}

bool SimRuntimeCore::get_cached_pointers(int handle, double*& cpu_ptr, double*& gpu_ptr) const {
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        return false;
    }
    const VarPointer& vp = var_pointer_cache_[handle];
    cpu_ptr = vp.cpu_ptr;
    gpu_ptr = vp.gpu_ptr;
    return true;
}

int SimRuntimeCore::resolve_node_index_internal_from_original(int node_idx_original) const {
    if (sim_ == nullptr) {
        printf("resolve_node_index_internal_from_original: Simulate not initialized\n");
        return -1;
    }
    if (sim_->neuron_group_list.empty() || sim_->neuron_group_list[0] == nullptr) {
        printf("resolve_node_index_internal_from_original: neuron_group_list is empty\n");
        return -1;
    }
    const auto* group = sim_->neuron_group_list[0];
    if (node_idx_original < 0 || node_idx_original >= group->len) {
        printf("resolve_node_index_internal_from_original: invalid node=%d (len=%d)\n",
               node_idx_original,
               group->len);
        return -1;
    }
    if (group->permute != nullptr) {
        return group->permute[node_idx_original];
    }
    return node_idx_original;
}

int SimRuntimeCore::resolve_node_index_permuted_from_handle(int handle) const {
    if (sim_ == nullptr) {
        printf("resolve_node_index_permuted_from_handle: Simulate not initialized\n");
        return -1;
    }
    if (sim_->neuron_group_list.empty() || sim_->neuron_group_list[0] == nullptr) {
        printf("resolve_node_index_permuted_from_handle: neuron_group_list is empty\n");
        return -1;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        static bool warned_invalid_handle = false;
        if (!warned_invalid_handle) {
            warned_invalid_handle = true;
            printf(
                "resolve_node_index_permuted_from_handle: invalid handle=%d "
                "(suppressing further invalid-handle logs)\n",
                handle);
        }
        return -1;
    }

    const auto* group = sim_->neuron_group_list[0];
    const auto& desc = var_pointer_cache_[handle].descriptor;

    const bool is_global = (desc.mech.empty() || desc.mech == "global");
    if (is_global) {
        const int node_original = desc.node_or_mech_idx;
        const int node_internal = resolve_node_index_internal_from_original(node_original);
        if (node_internal < 0) {
            printf("resolve_node_index_permuted_from_handle: invalid global node=%d (len=%d)\n",
                   node_original,
                   group->len);
            return -1;
        }
        return node_internal;
    }

    const int mech_idx_raw = desc.node_or_mech_idx;
    int offset = 0;
    for (auto* mech_obj : group->mechanism_list) {
        if (mech_obj == nullptr || mech_obj->name != desc.mech) {
            continue;
        }
        const int count = mech_obj->node_count();
        int* node_indices = mech_obj->node_indices_cpu_data();
        int* mech_permute = mech_obj->permute;
        if (node_indices == nullptr) {
            printf("resolve_node_index_permuted_from_handle: runtime mech '%s' has null node index buffer\n",
                   desc.mech.c_str());
            return -1;
        }
        if (mech_idx_raw >= offset && mech_idx_raw < offset + count) {
            const int local_raw = mech_idx_raw - offset;
            const int local_internal = mech_permute ? mech_permute[local_raw] : local_raw;
            if (local_internal < 0 || local_internal >= count) {
                printf("resolve_node_index_permuted_from_handle: invalid permuted mech idx (mech=%s raw=%d permuted=%d count=%d)\n",
                       desc.mech.c_str(),
                       local_raw,
                       local_internal,
                       count);
                return -1;
            }
            return node_indices[local_internal];
        }
        offset += count;
    }

    // Some callers historically pass mechanism-local rows rather than the
    // group-offset domain used by runtime scans above. Preserve that fallback,
    // but source the node map from the runtime mechanism instance rather than
    // the prototype/var_map object.
    for (auto* mech_obj : group->mechanism_list) {
        if (mech_obj == nullptr || mech_obj->name != desc.mech) {
            continue;
        }
        const int count = mech_obj->node_count();
        if (mech_idx_raw < 0 || mech_idx_raw >= count) {
            continue;
        }
        int* node_indices = mech_obj->node_indices_cpu_data();
        int* mech_permute = mech_obj->permute;
        if (node_indices == nullptr) {
            printf("resolve_node_index_permuted_from_handle: runtime mech '%s' has null node index buffer\n",
                   desc.mech.c_str());
            return -1;
        }
        const int local_internal = mech_permute ? mech_permute[mech_idx_raw] : mech_idx_raw;
        if (local_internal < 0 || local_internal >= count) {
            printf("resolve_node_index_permuted_from_handle: invalid local permuted mech idx (mech=%s raw=%d permuted=%d count=%d)\n",
                   desc.mech.c_str(),
                   mech_idx_raw,
                   local_internal,
                   count);
            return -1;
        }
        return node_indices[local_internal];
    }

    printf(
        "resolve_node_index_permuted_from_handle: mech idx out of range (mech=%s idx=%d)\n",
        desc.mech.c_str(),
        mech_idx_raw);
    return -1;
}

int SimRuntimeCore::resolve_mech_index_permuted_from_handle(int handle) const {
    if (sim_ == nullptr) {
        printf("resolve_mech_index_permuted_from_handle: Simulate not initialized\n");
        return -1;
    }
    if (sim_->neuron_group_list.empty() || sim_->neuron_group_list[0] == nullptr) {
        printf("resolve_mech_index_permuted_from_handle: neuron_group_list is empty\n");
        return -1;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("resolve_mech_index_permuted_from_handle: invalid handle=%d\n", handle);
        return -1;
    }

    const auto* group = sim_->neuron_group_list[0];
    const auto& desc = var_pointer_cache_[handle].descriptor;
    const bool is_global = (desc.mech.empty() || desc.mech == "global");
    if (is_global) {
        printf("resolve_mech_index_permuted_from_handle: handle=%d is global, not mech\n", handle);
        return -1;
    }

    const int mech_idx_raw = desc.node_or_mech_idx;
    int offset = 0;
    for (auto* mech_obj : group->mechanism_list) {
        if (mech_obj == nullptr || mech_obj->name != desc.mech) {
            continue;
        }
        const int count = mech_obj->node_count();
        if (mech_idx_raw >= offset && mech_idx_raw < offset + count) {
            const int local_raw = mech_idx_raw - offset;
            const int local_internal = mech_obj->permute ? mech_obj->permute[local_raw] : local_raw;
            if (local_internal < 0 || local_internal >= count) {
                printf("resolve_mech_index_permuted_from_handle: invalid permuted group-offset mech idx (mech=%s raw=%d permuted=%d count=%d)\n",
                       desc.mech.c_str(),
                       local_raw,
                       local_internal,
                       count);
                return -1;
            }
            return offset + local_internal;
        }
        offset += count;
    }

    offset = 0;
    for (auto* mech_obj : group->mechanism_list) {
        if (mech_obj == nullptr || mech_obj->name != desc.mech) {
            continue;
        }
        const int count = mech_obj->node_count();
        const int local_raw = mech_idx_raw;
        if (local_raw < 0 || local_raw >= count) {
            offset += count;
            continue;
        }
        const int local_internal = mech_obj->permute ? mech_obj->permute[local_raw] : local_raw;
        if (local_internal < 0 || local_internal >= count) {
            printf("resolve_mech_index_permuted_from_handle: invalid permuted local mech idx (mech=%s raw=%d permuted=%d count=%d)\n",
                   desc.mech.c_str(),
                   local_raw,
                   local_internal,
                   count);
            return -1;
        }
        return local_internal;
    }

    printf(
        "resolve_mech_index_permuted_from_handle: mech idx out of range (mech=%s idx=%d)\n",
        desc.mech.c_str(),
        mech_idx_raw);
    return -1;
}

double SimRuntimeCore::get_variable_by_handle(int handle) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return 0.0;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("Invalid handle: %d\n", handle);
        return 0.0;
    }
    VarPointer& vp = var_pointer_cache_[handle];
    if (sim_->mode == CPU) {
        if (vp.cpu_ptr == nullptr) {
            printf("CPU pointer is null for handle %d\n", handle);
            return 0.0;
        }
        return *vp.cpu_ptr;
    }
    if (vp.is_dirty) {
        return vp.cached_cpu_value;
    }
    if (vp.gpu_ptr == nullptr) {
        printf("GPU pointer is null for handle %d\n", handle);
        return 0.0;
    }
    double val = 0.0;
    mem_copy_gpu2cpu(&val, vp.gpu_ptr, sizeof(double));
    return val;
}

int SimRuntimeCore::set_variable_by_handle(int handle, double value) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("Invalid handle: %d\n", handle);
        return -1;
    }
    VarPointer& vp = var_pointer_cache_[handle];
    if (sim_->mode == CPU) {
        if (vp.cpu_ptr == nullptr) {
            printf("CPU pointer is null for handle %d\n", handle);
            return -1;
        }
        *vp.cpu_ptr = value;
        return 0;
    }
    if (vp.cpu_ptr == nullptr) {
        printf("CPU pointer is null for handle %d\n", handle);
        return -1;
    }
    *vp.cpu_ptr = value;
    vp.cached_cpu_value = value;
    if (!vp.is_dirty) {
        vp.is_dirty = true;
        dirty_handles_.push_back(handle);
    }
    return 0;
}

int SimRuntimeCore::set_variables_by_handles(const std::vector<int>& handles, const std::vector<double>& values) {
    if (handles.size() != values.size()) {
        printf("set_variables_by_handles: handle/value size mismatch (%zu vs %zu)\n", handles.size(), values.size());
        return -1;
    }
    for (size_t i = 0; i < handles.size(); ++i) {
        if (set_variable_by_handle(handles[i], values[i]) < 0) {
            return -1;
        }
    }
    return 0;
}

void SimRuntimeCore::flush_dirty_variables() {
    if (sim_ == nullptr || sim_->mode != GPU || dirty_handles_.empty()) {
        return;
    }

    int count = static_cast<int>(dirty_handles_.size());
    if (count > flush_capacity_) {
        if (flush_values_device_ != nullptr) {
            gpu_mem_free((void**)&flush_values_device_);
            flush_values_device_ = nullptr;
        }
        if (flush_gpu_ptrs_device_ != nullptr) {
            gpu_mem_free((void**)&flush_gpu_ptrs_device_);
            flush_gpu_ptrs_device_ = nullptr;
        }
        gpu_mem_allocate((void**)&flush_values_device_, count * static_cast<int>(sizeof(double)));
        gpu_mem_allocate((void**)&flush_gpu_ptrs_device_, count * static_cast<int>(sizeof(double*)));
        flush_capacity_ = count;
    }

    flush_cpu_values_.resize(count);
    flush_gpu_ptrs_.resize(count);

    for (int i = 0; i < count; ++i) {
        const int handle = dirty_handles_[i];
        VarPointer& vp = var_pointer_cache_[handle];
        flush_cpu_values_[i] = vp.cached_cpu_value;
        flush_gpu_ptrs_[i] = vp.gpu_ptr;
        vp.is_dirty = false;
    }

    mem_copy_cpu2gpu_sync(flush_values_device_, flush_cpu_values_.data(), count * static_cast<int>(sizeof(double)));
    mem_copy_cpu2gpu_sync(flush_gpu_ptrs_device_, flush_gpu_ptrs_.data(), count * static_cast<int>(sizeof(double*)));
    launch_batch_copy_doubles(flush_gpu_ptrs_device_, flush_values_device_, count);
    dirty_handles_.clear();
}

int SimRuntimeCore::get_variables_by_handles_f32(const std::vector<int>& handles, float* out_cpu, int count) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (count <= 0) {
        return 0;
    }
    if (sim_->mode != GPU) {
        // CPU fallback: fill by scalar reads.
        for (int i = 0; i < count; ++i) {
            out_cpu[i] = static_cast<float>(get_variable_by_handle(handles[i]));
        }
        return 0;
    }

    flush_dirty_variables();

    if (count > gather_capacity_) {
        if (gather_gpu_ptrs_device_ != nullptr) {
            gpu_mem_free((void**)&gather_gpu_ptrs_device_);
            gather_gpu_ptrs_device_ = nullptr;
        }
        if (gather_values_device_ != nullptr) {
            gpu_mem_free((void**)&gather_values_device_);
            gather_values_device_ = nullptr;
        }
        gpu_mem_allocate((void**)&gather_gpu_ptrs_device_, count * sizeof(double*));
        gpu_mem_allocate((void**)&gather_values_device_, count * sizeof(float));
        gather_capacity_ = count;
    }

    std::vector<double*> gpu_ptrs;
    gpu_ptrs.resize(count);
    for (int i = 0; i < count; ++i) {
        const int handle = handles[i];
        if (handle < 0 || handle >= static_cast<int>(var_pointer_cache_.size())) {
            gpu_ptrs[i] = nullptr;
            continue;
        }
        VarPointer& vp = var_pointer_cache_[handle];
        gpu_ptrs[i] = vp.gpu_ptr;
    }

    mem_copy_cpu2gpu_sync(gather_gpu_ptrs_device_, gpu_ptrs.data(), count * sizeof(double*));
    launch_batch_gather_floats(gather_gpu_ptrs_device_, gather_values_device_, count);
    mem_copy_gpu2cpu(out_cpu, gather_values_device_, count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        if (gpu_ptrs[i] == nullptr) {
            out_cpu[i] = 0.0f;
        }
    }
    return 0;
}

std::tuple<double, int> SimRuntimeCore::debug_get_postsyn_delay_info(
    const std::string& mech,
    int instance_index) const {
    if (sim_ == nullptr || sim_->neuron_group_list.empty()) {
        printf("debug_get_postsyn_delay_info: model is not loaded\n");
        return {std::numeric_limits<double>::quiet_NaN(), -1};
    }
    const std::string target = to_lower_ascii(mech);
    for (auto* group : sim_->neuron_group_list) {
        if (group == nullptr) {
            continue;
        }
        for (auto* postsyn : group->vec_postsyn) {
            auto* mech_obj = dynamic_cast<Mechanism*>(postsyn);
            if (postsyn == nullptr || mech_obj == nullptr || to_lower_ascii(mech_obj->name) != target) {
                continue;
            }
            if (instance_index < 0 || instance_index >= mech_obj->node_count()) {
                printf("debug_get_postsyn_delay_info: invalid instance %d for %s count=%d\n",
                       instance_index,
                       mech.c_str(),
                       mech_obj->node_count());
                return {std::numeric_limits<double>::quiet_NaN(), -1};
            }
            if (postsyn->vecdata_delay == nullptr || postsyn->vecdata_delay_steps == nullptr) {
                printf("debug_get_postsyn_delay_info: %s has no delay storage\n", mech.c_str());
                return {std::numeric_limits<double>::quiet_NaN(), -1};
            }
            return {
                postsyn->vecdata_delay->get_cpu_data()[instance_index],
                postsyn->vecdata_delay_steps->get_cpu_data()[instance_index],
            };
        }
    }
    printf("debug_get_postsyn_delay_info: postsyn mechanism not found: %s\n", mech.c_str());
    return {std::numeric_limits<double>::quiet_NaN(), -1};
}

int SimRuntimeCore::get_learnable_grad_handle_from_variable_handle(int variable_handle) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (sim_->neuron_group_list.empty() || sim_->neuron_group_list[0] == nullptr) {
        printf("get_learnable_grad_handle_from_variable_handle: neuron_group_list is empty\n");
        return -1;
    }
    if (variable_handle < 0 || variable_handle >= static_cast<int>(var_pointer_cache_.size())) {
        printf("get_learnable_grad_handle_from_variable_handle: invalid variable handle %d\n", variable_handle);
        return -1;
    }

    const auto& src = var_pointer_cache_[variable_handle];
    const auto& desc = src.descriptor;
    const std::string mech_name = to_lower_ascii(desc.mech);
    if (mech_name.empty() || mech_name == "global") {
        printf("get_learnable_grad_handle_from_variable_handle: global variables do not have learnable grads\n");
        return -1;
    }
    if (desc.var.rfind("grad_", 0) == 0) {
        printf("get_learnable_grad_handle_from_variable_handle: handle already names a grad variable: %s\n",
               desc.var.c_str());
        return -1;
    }
    if (desc.array_index > 0) {
        printf("get_learnable_grad_handle_from_variable_handle: array learnable vars are not supported: %s.%s[%d]\n",
               desc.mech.c_str(),
               desc.var.c_str(),
               desc.array_index);
        return -1;
    }

    auto resolve_grad_ptrs = [&]() -> std::pair<double*, double*> {
        auto* group = sim_->neuron_group_list[0];
        const int mech_idx_raw = desc.node_or_mech_idx;
        int offset = 0;
        for (auto* mech_obj : group->mechanism_list) {
            if (mech_obj == nullptr || mech_obj->name != desc.mech) {
                continue;
            }
            const int count = mech_obj->node_count();
            if (mech_idx_raw >= offset && mech_idx_raw < offset + count) {
                const int local_raw = mech_idx_raw - offset;
                const int local_internal = mech_obj->permute ? mech_obj->permute[local_raw] : local_raw;
                if (local_internal < 0 || local_internal >= count) {
                    printf("get_learnable_grad_handle_from_variable_handle: invalid permuted mech idx "
                           "(mech=%s raw=%d permuted=%d count=%d)\n",
                           desc.mech.c_str(),
                           local_raw,
                           local_internal,
                           count);
                    return {nullptr, nullptr};
                }
                double* grad_cpu = mech_obj->getLearnableGradPtr(desc.var, CPU);
                double* grad_gpu = (sim_->mode == GPU) ? mech_obj->getLearnableGradPtr(desc.var, GPU) : nullptr;
                if (grad_cpu == nullptr && grad_gpu == nullptr) {
                    printf("get_learnable_grad_handle_from_variable_handle: learnable grad not found: %s.%s\n",
                           desc.mech.c_str(),
                           desc.var.c_str());
                    return {nullptr, nullptr};
                }
                return {
                    grad_cpu ? grad_cpu + local_internal : nullptr,
                    grad_gpu ? grad_gpu + local_internal : nullptr,
                };
            }
            offset += count;
        }

        for (auto* mech_obj : group->mechanism_list) {
            if (mech_obj == nullptr || mech_obj->name != desc.mech) {
                continue;
            }
            const int count = mech_obj->node_count();
            if (mech_idx_raw < 0 || mech_idx_raw >= count) {
                continue;
            }
            const int local_internal = mech_obj->permute ? mech_obj->permute[mech_idx_raw] : mech_idx_raw;
            if (local_internal < 0 || local_internal >= count) {
                printf("get_learnable_grad_handle_from_variable_handle: invalid local permuted mech idx "
                       "(mech=%s raw=%d permuted=%d count=%d)\n",
                       desc.mech.c_str(),
                       mech_idx_raw,
                       local_internal,
                       count);
                return {nullptr, nullptr};
            }
            double* grad_cpu = mech_obj->getLearnableGradPtr(desc.var, CPU);
            double* grad_gpu = (sim_->mode == GPU) ? mech_obj->getLearnableGradPtr(desc.var, GPU) : nullptr;
            if (grad_cpu == nullptr && grad_gpu == nullptr) {
                printf("get_learnable_grad_handle_from_variable_handle: learnable grad not found: %s.%s\n",
                       desc.mech.c_str(),
                       desc.var.c_str());
                return {nullptr, nullptr};
            }
            return {
                grad_cpu ? grad_cpu + local_internal : nullptr,
                grad_gpu ? grad_gpu + local_internal : nullptr,
            };
        }

        printf("get_learnable_grad_handle_from_variable_handle: mech idx out of range (mech=%s idx=%d)\n",
               desc.mech.c_str(),
               mech_idx_raw);
        return {nullptr, nullptr};
    };

    auto [grad_cpu_ptr, grad_gpu_ptr] = resolve_grad_ptrs();
    if (grad_cpu_ptr == nullptr && grad_gpu_ptr == nullptr) {
        return -1;
    }
    if (grad_cpu_ptr != nullptr) {
        auto it = cpu_ptr_to_handle_.find(grad_cpu_ptr);
        if (it != cpu_ptr_to_handle_.end()) {
            return it->second;
        }
    }

    VarPointer vp;
    vp.cpu_ptr = grad_cpu_ptr;
    vp.gpu_ptr = grad_gpu_ptr;
    vp.descriptor = desc;
    vp.descriptor.var = "grad_" + desc.var;
    vp.cached_cpu_value = 0.0;
    vp.is_dirty = false;

    const int grad_handle = static_cast<int>(var_pointer_cache_.size());
    var_pointer_cache_.push_back(vp);
    if (grad_cpu_ptr != nullptr) {
        cpu_ptr_to_handle_[grad_cpu_ptr] = grad_handle;
    }
    return grad_handle;
}

std::vector<int> SimRuntimeCore::get_learnable_grad_handles_from_variable_handles(
    const std::vector<int>& variable_handles) {
    std::vector<int> out;
    out.reserve(variable_handles.size());
    for (int handle : variable_handles) {
        out.push_back(get_learnable_grad_handle_from_variable_handle(handle));
    }
    return out;
}

int SimRuntimeCore::set_variable_value(double val, const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return set_variable_value_with_array(val, mech, var, node_or_mech_idx, 0);
}

int SimRuntimeCore::set_variable_value_with_array(
    double val, const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = node_or_mech_idx;
    desc.array_index = array_index;
    auto [cpu_ptr, gpu_ptr] = sim_->getVarPtr(desc, false);
    if (cpu_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), node_or_mech_idx);
        return -1;
    }
    *cpu_ptr = val;
    if (sim_->mode == GPU && gpu_ptr != nullptr) {
        mem_copy_cpu2gpu_sync(gpu_ptr, cpu_ptr, sizeof(double));
    }
    return 0;
}

double SimRuntimeCore::get_variable_value(const std::string& mech, const std::string& var, int node_or_mech_idx) {
    return get_variable_value_with_array(mech, var, node_or_mech_idx, 0);
}

double SimRuntimeCore::get_variable_value_with_array(
    const std::string& mech, const std::string& var, int node_or_mech_idx, int array_index) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return 0.0;
    }
    VarDescriptor desc;
    desc.mech = mech;
    desc.var = var;
    desc.node_or_mech_idx = node_or_mech_idx;
    desc.array_index = array_index;
    auto [cpu_ptr, gpu_ptr] = sim_->getVarPtr(desc, false);
    if (cpu_ptr == nullptr) {
        printf("Variable not found: %s %s %d\n", mech.c_str(), var.c_str(), node_or_mech_idx);
        return 0.0;
    }
    const bool is_grad_var = (var.rfind("grad_", 0) == 0);
    if (sim_->mode == GPU && gpu_ptr != nullptr && !is_grad_var) {
        double val = 0.0;
        mem_copy_gpu2cpu(&val, gpu_ptr, sizeof(double));
        return val;
    }
    return *cpu_ptr;
}

SimRuntimeCore::InputStimBatch* SimRuntimeCore::find_input_batch_(int batch_id) {
    auto it = input_batches_.find(batch_id);
    if (it == input_batches_.end()) {
        printf("Input batch %d not found\n", batch_id);
        return nullptr;
    }
    return &it->second;
}

void SimRuntimeCore::release_input_batch_resources_(InputStimBatch& batch) {
    if (batch.net_device.interval_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.interval_ptrs);
        batch.net_device.interval_ptrs = nullptr;
    }
    if (batch.net_device.start_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.start_ptrs);
        batch.net_device.start_ptrs = nullptr;
    }
    if (batch.net_device.number_ptrs != nullptr) {
        gpu_mem_free((void**)&batch.net_device.number_ptrs);
        batch.net_device.number_ptrs = nullptr;
    }
    if (batch.net_device.interval_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.interval_values);
        batch.net_device.interval_values = nullptr;
    }
    if (batch.net_device.start_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.start_values);
        batch.net_device.start_values = nullptr;
    }
    if (batch.net_device.number_values != nullptr) {
        gpu_mem_free((void**)&batch.net_device.number_values);
        batch.net_device.number_values = nullptr;
    }
}

int SimRuntimeCore::register_netstim_batch(const std::vector<std::tuple<int, int, int>>& handle_triplets,
                                          double interval_scale,
                                          double start_base,
                                          double epsilon,
                                          double number) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (handle_triplets.empty()) {
        printf("register_netstim_batch: handle list is empty\n");
        return -1;
    }

    InputStimBatch batch;
    batch.type = InputStimBatch::Type::NetStim;
    batch.net_params.interval_scale = interval_scale;
    batch.net_params.start_base = start_base;
    batch.net_params.epsilon = epsilon;
    batch.net_params.number = number;
    batch.mode = sim_->mode;
    batch.expected_size = handle_triplets.size();
    batch.net_entries.reserve(handle_triplets.size());

    if (sim_->mode == Mode::GPU) {
        batch.net_interval_values.resize(handle_triplets.size(), 0.0);
        batch.net_start_values.resize(handle_triplets.size(), 0.0);
        batch.net_number_values.resize(handle_triplets.size(), number);
        batch.net_interval_gpu_ptrs.reserve(handle_triplets.size());
        batch.net_start_gpu_ptrs.reserve(handle_triplets.size());
        batch.net_number_gpu_ptrs.reserve(handle_triplets.size());
    }

    for (size_t idx = 0; idx < handle_triplets.size(); ++idx) {
        auto [interval_h, start_h, number_h] = handle_triplets[idx];
        NetStimEntry entry;
        entry.interval_handle = interval_h;
        entry.start_handle = start_h;
        entry.number_handle = number_h;

        double* interval_cpu = nullptr;
        double* interval_gpu = nullptr;
        double* start_cpu = nullptr;
        double* start_gpu = nullptr;
        double* number_cpu = nullptr;
        double* number_gpu = nullptr;
        if (!get_cached_pointers(interval_h, interval_cpu, interval_gpu) ||
            !get_cached_pointers(start_h, start_cpu, start_gpu) ||
            !get_cached_pointers(number_h, number_cpu, number_gpu)) {
            printf("register_netstim_batch: invalid handle at index %zu\n", idx);
            release_input_batch_resources_(batch);
            return -1;
        }
        if (interval_cpu == nullptr || start_cpu == nullptr || number_cpu == nullptr) {
            printf("register_netstim_batch: missing CPU pointer at index %zu\n", idx);
            release_input_batch_resources_(batch);
            return -1;
        }

        entry.interval_cpu = interval_cpu;
        entry.interval_gpu = interval_gpu;
        entry.start_cpu = start_cpu;
        entry.start_gpu = start_gpu;
        entry.number_cpu = number_cpu;
        entry.number_gpu = number_gpu;

        batch.net_entries.push_back(entry);

        if (sim_->mode == Mode::GPU) {
            if (entry.interval_gpu == nullptr || entry.start_gpu == nullptr || entry.number_gpu == nullptr) {
                printf("register_netstim_batch: missing GPU pointer at index %zu\n", idx);
                release_input_batch_resources_(batch);
                return -1;
            }
            batch.net_interval_gpu_ptrs.push_back(entry.interval_gpu);
            batch.net_start_gpu_ptrs.push_back(entry.start_gpu);
            batch.net_number_gpu_ptrs.push_back(entry.number_gpu);
        }
    }

    if (sim_->mode == Mode::GPU) {
        int count = static_cast<int>(batch.net_entries.size());
        gpu_mem_allocate((void**)&batch.net_device.interval_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.start_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.number_ptrs, count * static_cast<int>(sizeof(double*)));
        gpu_mem_allocate((void**)&batch.net_device.interval_values, count * static_cast<int>(sizeof(double)));
        gpu_mem_allocate((void**)&batch.net_device.start_values, count * static_cast<int>(sizeof(double)));
        gpu_mem_allocate((void**)&batch.net_device.number_values, count * static_cast<int>(sizeof(double)));

        mem_copy_cpu2gpu_sync(batch.net_device.interval_ptrs, batch.net_interval_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(batch.net_device.start_ptrs, batch.net_start_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
        mem_copy_cpu2gpu_sync(batch.net_device.number_ptrs, batch.net_number_gpu_ptrs.data(),
                              count * static_cast<int>(sizeof(double*)));
    }

    int batch_id = next_input_batch_id_++;
    input_batches_.emplace(batch_id, std::move(batch));
    return batch_id;
}

int SimRuntimeCore::register_vecstim_batch(const std::vector<int>& mech_indices,
                                          double spike_scale,
                                          double start_base,
                                          double epsilon,
                                          int spike_count) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    if (mech_indices.empty()) {
        printf("register_vecstim_batch: mech list is empty\n");
        return -1;
    }

    InputStimBatch batch;
    batch.type = InputStimBatch::Type::VecStim;
    batch.vec_params.spike_scale = spike_scale;
    batch.vec_params.start_base = start_base;
    batch.vec_params.epsilon = epsilon;
    batch.vec_params.spike_count = spike_count;
    batch.mode = sim_->mode;
    batch.expected_size = mech_indices.size();
    batch.vec_spike_buffer.resize(spike_count);
    batch.vec_entries.reserve(mech_indices.size());
    VecEvent* vecstim = VecEvent::getInstance();
    if (vecstim == nullptr) {
        printf("register_vecstim_batch: VecStim instance not ready\n");
        return -1;
    }

    for (size_t idx = 0; idx < mech_indices.size(); ++idx) {
        const int mech_idx = mech_indices[idx];
        if (mech_idx < 0 || static_cast<size_t>(mech_idx) >= vecstim->vec_ptr.size()) {
            printf("register_vecstim_batch: invalid mech index at %zu\n", idx);
            return -1;
        }
        VecStimEntry entry;
        entry.mech_index = mech_idx;
        batch.vec_entries.push_back(entry);
    }

    int batch_id = next_input_batch_id_++;
    input_batches_.emplace(batch_id, std::move(batch));
    return batch_id;
}

int SimRuntimeCore::register_vecstim_batch_by_locs(const std::string& mech,
                                                  const std::vector<int>& gids,
                                                  const std::vector<std::string>& section_names,
                                                  const std::vector<double>& locs,
                                                  double spike_scale,
                                                  double start_base,
                                                  double epsilon,
                                                  int spike_count) {
    if (gids.empty()) {
        printf("register_vecstim_batch_by_locs: location list is empty\n");
        return -1;
    }
    if (gids.size() != section_names.size() || gids.size() != locs.size()) {
        printf("register_vecstim_batch_by_locs: size mismatch gids=%zu sections=%zu locs=%zu\n",
               gids.size(),
               section_names.size(),
               locs.size());
        return -1;
    }
    std::vector<int> mech_indices;
    mech_indices.reserve(gids.size());
    for (std::size_t i = 0; i < gids.size(); ++i) {
        const int mech_idx = resolve_mech_index_by_loc_internal_(
            mech,
            gids[i],
            section_names[i],
            locs[i],
            0);
        if (mech_idx < 0) {
            printf("register_vecstim_batch_by_locs: failed to resolve mech index "
                   "(i=%zu mech=%s gid=%d section=%s loc=%g)\n",
                   i,
                   mech.c_str(),
                   gids[i],
                   section_names[i].c_str(),
                   locs[i]);
            return -1;
        }
        mech_indices.push_back(mech_idx);
    }
    return register_vecstim_batch(mech_indices, spike_scale, start_base, epsilon, spike_count);
}

int SimRuntimeCore::set_input_batch_pixels(int batch_id, std::span<const double> pixels) {
    if (sim_ == nullptr) {
        printf("Simulate not initialized\n");
        return -1;
    }
    InputStimBatch* batch = find_input_batch_(batch_id);
    if (batch == nullptr) {
        return -1;
    }
    if (pixels.size() != batch->expected_size) {
        printf("set_input_batch_pixels: size mismatch batch=%zu, got=%zu\n",
               batch->expected_size, pixels.size());
        return -1;
    }

    if (batch->type == InputStimBatch::Type::NetStim) {
        const auto count = batch->net_entries.size();
        if (count == 0) {
            return 0;
        }
        for (size_t i = 0; i < count; ++i) {
            const double pixel = pixels[i];
            const double denom = std::max(pixel + batch->net_params.epsilon, batch->net_params.epsilon);
            const double interval = batch->net_params.interval_scale / denom;
            const double start = batch->net_params.start_base + interval;
            const double number = batch->net_params.number;

            NetStimEntry& entry = batch->net_entries[i];
            if (entry.interval_cpu == nullptr || entry.start_cpu == nullptr || entry.number_cpu == nullptr) {
                printf("set_input_batch_pixels: missing CPU pointer at index %zu\n", i);
                return -1;
            }

            *entry.interval_cpu = interval;
            *entry.start_cpu = start;
            *entry.number_cpu = number;

            if (batch->mode == Mode::GPU) {
                batch->net_interval_values[i] = interval;
                batch->net_start_values[i] = start;
                batch->net_number_values[i] = number;
            }
        }

        if (batch->mode == Mode::GPU) {
            const int count_i = static_cast<int>(count);
            mem_copy_cpu2gpu_sync(batch->net_device.interval_values, batch->net_interval_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            mem_copy_cpu2gpu_sync(batch->net_device.start_values, batch->net_start_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            mem_copy_cpu2gpu_sync(batch->net_device.number_values, batch->net_number_values.data(),
                                  count_i * static_cast<int>(sizeof(double)));
            launch_batch_copy_doubles(batch->net_device.interval_ptrs, batch->net_device.interval_values, count_i);
            launch_batch_copy_doubles(batch->net_device.start_ptrs, batch->net_device.start_values, count_i);
            launch_batch_copy_doubles(batch->net_device.number_ptrs, batch->net_device.number_values, count_i);
        }
        return 0;
    }

    if (VecEvent::getInstance() == nullptr) {
        printf("VecStim instance not ready\n");
        return -1;
    }

    const int spike_count = batch->vec_params.spike_count;
    batch->vec_spike_buffer.resize(spike_count);
    VecEvent::setCurrentTime(sim_->t);

    for (size_t i = 0; i < pixels.size(); ++i) {
        const double pixel = pixels[i];
        const double denom = std::max(pixel + batch->vec_params.epsilon, batch->vec_params.epsilon);
        const double step = batch->vec_params.spike_scale / denom;

        for (int k = 0; k < spike_count; ++k) {
            batch->vec_spike_buffer[k] = batch->vec_params.start_base + step * static_cast<double>(k + 1);
        }

        VecEvent::update_sequence(batch->mode,
                                  batch->vec_entries[i].mech_index,
                                  batch->vec_spike_buffer.data(),
                                  spike_count);
    }
    return 0;
}

}  // namespace neurong::runtime_api::core
