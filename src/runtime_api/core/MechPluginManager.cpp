#include "runtime_api/core/MechPluginManager.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace neurong::runtime_api::core {

namespace {

std::string null_to_empty(const char* s) {
    return s ? std::string(s) : std::string();
}

std::string display_value(const char* s) {
    return s ? std::string(s) : std::string("(null)");
}

std::string sdk_rebuild_hint() {
    return "\n"
           "  Rebuild this mech plugin with the same NeuronG SDK/runtime prefix.\n"
           "  External plugin package:\n"
           "    rm -rf build-mech && neurong-mech-build mech\n"
           "  Built-in/source-tree mech package:\n"
           "    cmake --build <neuron-g-build> --target <mech-package-target> neurong_sdk_support\n"
           "    cmake --install <neuron-g-build> --prefix <same-sdk-prefix>";
}

std::string metadata_mismatch_message(const char* field, std::string plugin_value, std::string runtime_value) {
    std::string message = "plugin ";
    message += field;
    message += " mismatch\n  plugin ";
    message += field;
    message += ": ";
    message += plugin_value;
    message += "\n  runtime ";
    message += field;
    message += ": ";
    message += runtime_value;
    return message;
}

std::string metadata_mismatch_message(const char* field, const char* plugin_value, const char* runtime_value) {
    return metadata_mismatch_message(field, display_value(plugin_value), display_value(runtime_value));
}

struct GlobalLoadedHandle {
    void* handle = nullptr;
};

struct GlobalPluginState {
    std::mutex mutex;
    std::vector<GlobalLoadedHandle> loaded_handles;
    std::vector<MechPluginManager::LoadedPluginInfo> loaded_plugins;
    std::unordered_map<std::string, std::size_t> path_to_index;
};

GlobalPluginState& global_plugin_state() {
    static GlobalPluginState state;
    return state;
}

std::string canonicalize_plugin_path(const std::string& path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(std::filesystem::path(path), ec);
    if (ec) {
        return path;
    }
    return canonical.string();
}

}  // namespace

int MechPluginManager::register_mechanism_bridge_(void* user_data,
                                                  const neurong::sdk::MechanismRegistration* registration) {
    auto* ctx = static_cast<TransactionContext*>(user_data);
    if (ctx == nullptr || registration == nullptr) {
        return -1;
    }
    if (registration->struct_size < sizeof(neurong::sdk::MechanismRegistration)) {
        ctx->error_message = "plugin registration struct_size too small";
        return -1;
    }
    if (registration->name == nullptr || registration->name[0] == '\0') {
        ctx->error_message = "plugin mechanism registration missing name";
        return -1;
    }
    if (registration->create_fn == nullptr) {
        ctx->error_message = "plugin mechanism registration missing create_fn";
        return -1;
    }

    PendingRegistration pending{};
    pending.registration.name = registration->name;
    pending.registration.creator = registration->create_fn;
    pending.registration.source_file = make_source_file_(ctx->plugin_path, registration->source_file);
    pending.registration.source_symbol = make_source_symbol_(ctx->plugin_path, registration->source_symbol);
    if (registration->has_pnt_receive_size) {
        pending.registration.has_pnt_receive_size = true;
        pending.registration.pnt_receive_size = registration->pnt_receive_size;
    }
    if (registration->pointer_dparam_slots != nullptr && registration->pointer_dparam_slots_count > 0) {
        pending.registration.pointer_dparam_slots.assign(
            registration->pointer_dparam_slots,
            registration->pointer_dparam_slots + registration->pointer_dparam_slots_count);
    }
    if (registration->dparam_semantics != nullptr && registration->dparam_semantics_count > 0) {
        pending.registration.dparam_semantics.assign(
            registration->dparam_semantics,
            registration->dparam_semantics + registration->dparam_semantics_count);
    }

    ctx->pending.push_back(std::move(pending));
    return 0;
}

int MechPluginManager::log_message_bridge_(void* user_data,
                                           neurong::sdk::LogLevel level,
                                           const char* message) {
    auto* ctx = static_cast<TransactionContext*>(user_data);
    const char* text = message ? message : "";
    const char* level_text = "INFO";
    switch (level) {
    case neurong::sdk::LogLevel::Debug:
        level_text = "DEBUG";
        break;
    case neurong::sdk::LogLevel::Info:
        level_text = "INFO";
        break;
    case neurong::sdk::LogLevel::Warning:
        level_text = "WARN";
        break;
    case neurong::sdk::LogLevel::Error:
        level_text = "ERROR";
        break;
    }
    std::cerr << "[mech-plugin " << level_text << "] "
              << (ctx ? ctx->plugin_path : "<unknown>") << ": " << text << "\n";
    return 0;
}

bool MechPluginManager::validate_plugin_info_(const neurong::sdk::PluginInfo& info,
                                              const neurong::sdk::PluginApi& api,
                                              std::string* error) {
    if (info.struct_size < sizeof(neurong::sdk::PluginInfo)) {
        if (error) {
            *error = "plugin info struct_size too small";
        }
        return false;
    }
    if (info.plugin_name == nullptr || info.plugin_name[0] == '\0') {
        if (error) {
            *error = "plugin info missing plugin_name";
        }
        return false;
    }
    if (info.sdk_abi_major != NEURONG_SDK_ABI_MAJOR) {
        if (error) {
            *error = metadata_mismatch_message(
                "sdk_abi_major",
                std::to_string(info.sdk_abi_major),
                std::to_string(NEURONG_SDK_ABI_MAJOR));
        }
        return false;
    }
    if (info.sdk_abi_minor > NEURONG_SDK_ABI_MINOR) {
        if (error) {
            *error = "plugin sdk_abi_minor is newer than runtime\n  plugin sdk_abi_minor: "
                     + std::to_string(info.sdk_abi_minor)
                     + "\n  runtime sdk_abi_minor: "
                     + std::to_string(NEURONG_SDK_ABI_MINOR);
        }
        return false;
    }
    if (info.sdk_build_id == nullptr || std::strcmp(info.sdk_build_id, api.runtime_build_id) != 0) {
        if (error) {
            *error = metadata_mismatch_message("sdk_build_id", info.sdk_build_id, api.runtime_build_id)
                     + sdk_rebuild_hint();
        }
        return false;
    }
    if (info.compiler_id == nullptr || std::strcmp(info.compiler_id, api.compiler_id) != 0) {
        if (error) {
            *error = metadata_mismatch_message("compiler_id", info.compiler_id, api.compiler_id)
                     + sdk_rebuild_hint();
        }
        return false;
    }
    // Host compiler VERSION is intentionally not a hard gate: libstdc++/libc++ keep a
    // stable C++ ABI across compiler versions within the same family, and CUDA builds
    // legitimately use a different host g++ for .cu (nvcc -ccbin) than for plain .cpp.
    // The compiler family (compiler_id) above stays a hard check; here we only warn.
    if (info.compiler_version == nullptr || std::strcmp(info.compiler_version, api.compiler_version) != 0) {
        std::cerr << "load_mech_library: warning: plugin host compiler_version '"
                  << (info.compiler_version ? info.compiler_version : "(null)")
                  << "' differs from runtime '" << api.compiler_version
                  << "' (accepted; same compiler family ABI is stable)\n";
    }
    if (info.cxx_standard != api.cxx_standard) {
        if (error) {
            *error = "plugin cxx_standard mismatch\n  plugin cxx_standard: "
                     + std::to_string(info.cxx_standard)
                     + "\n  runtime cxx_standard: "
                     + std::to_string(api.cxx_standard)
                     + sdk_rebuild_hint();
        }
        return false;
    }
    if (info.pointer_size != api.pointer_size) {
        if (error) {
            *error = "plugin pointer_size mismatch\n  plugin pointer_size: "
                     + std::to_string(info.pointer_size)
                     + "\n  runtime pointer_size: "
                     + std::to_string(api.pointer_size)
                     + sdk_rebuild_hint();
        }
        return false;
    }
    const std::string cuda_id = info.cuda_compiler_id ? info.cuda_compiler_id : "";
    const std::string cuda_ver = info.cuda_compiler_version ? info.cuda_compiler_version : "";
    if (cuda_id.empty()) {
        if (error) {
            *error = "plugin cuda_compiler_id missing" + sdk_rebuild_hint();
        }
        return false;
    }
    if (cuda_id != "host-only") {
        if (std::strcmp(cuda_id.c_str(), api.runtime_cuda_compiler_id) != 0) {
            if (error) {
                *error = metadata_mismatch_message("cuda_compiler_id", cuda_id.c_str(), api.runtime_cuda_compiler_id)
                         + sdk_rebuild_hint();
            }
            return false;
        }
        if (std::strcmp(cuda_ver.c_str(), api.runtime_cuda_compiler_version) != 0) {
            auto parse_major = [](const std::string& ver) -> int {
                if (ver.empty()) {
                    return -1;
                }
                const auto pos = ver.find('.');
                const std::string major = (pos == std::string::npos) ? ver : ver.substr(0, pos);
                try {
                    return std::stoi(major);
                } catch (...) {
                    return -1;
                }
            };
            const int plugin_major = parse_major(cuda_ver);
            const int runtime_major = parse_major(api.runtime_cuda_compiler_version ? api.runtime_cuda_compiler_version : "");
            if (plugin_major <= 0 || runtime_major <= 0 || plugin_major != runtime_major) {
                if (error) {
                    *error = metadata_mismatch_message(
                                 "cuda_compiler_version",
                                 cuda_ver.c_str(),
                                 api.runtime_cuda_compiler_version)
                             + sdk_rebuild_hint();
                }
                return false;
            }
            std::cerr << "load_mech_library: warning: plugin cuda_compiler_version '" << cuda_ver
                      << "' differs from runtime '" << api.runtime_cuda_compiler_version
                      << "' (accepted by major-version compatibility)\n";
        }
    }
    return true;
}

std::string MechPluginManager::make_source_file_(const std::string& plugin_path, const char* source_file) {
    std::string out = "plugin:";
    out += plugin_path;
    if (source_file != nullptr && source_file[0] != '\0') {
        out += "::";
        out += source_file;
    }
    return out;
}

std::string MechPluginManager::make_source_symbol_(const std::string& plugin_name, const char* source_symbol) {
    std::string out = plugin_name;
    if (source_symbol != nullptr && source_symbol[0] != '\0') {
        out += "::";
        out += source_symbol;
    }
    return out;
}

int MechPluginManager::load_library(const std::string& path) {
    const std::string canonical_path = canonicalize_plugin_path(path);
    auto& state = global_plugin_state();
    {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (state.path_to_index.contains(canonical_path)) {
            const auto idx = state.path_to_index.at(canonical_path);
            const auto& plugin = state.loaded_plugins.at(idx);
            std::cerr << "load_mech_library: plugin '" << plugin.plugin_name
                      << "' already loaded from '" << canonical_path << "', reusing existing registration\n";
            return 0;
        }
    }

    void* handle = dlopen(canonical_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "load_mech_library: dlopen failed for '" << canonical_path << "': " << dlerror() << "\n";
        return -1;
    }

    dlerror();
    auto* init_fn = reinterpret_cast<neurong::sdk::PluginInitFn>(dlsym(handle, NEURONG_MECH_PLUGIN_INIT_SYMBOL));
    const char* sym_err = dlerror();
    if (sym_err != nullptr || init_fn == nullptr) {
        std::cerr << "load_mech_library: missing symbol '" << NEURONG_MECH_PLUGIN_INIT_SYMBOL
                  << "' in '" << canonical_path << "': " << (sym_err ? sym_err : "<null>") << "\n";
        dlclose(handle);
        return -1;
    }

    TransactionContext ctx{};
    ctx.plugin_path = canonical_path;

    neurong::sdk::PluginApi api{};
    api.struct_size = sizeof(api);
    api.sdk_abi_major = NEURONG_SDK_ABI_MAJOR;
    api.sdk_abi_minor = NEURONG_SDK_ABI_MINOR;
    api.runtime_version = NEURONG_SDK_PROJECT_VERSION;
    api.runtime_build_id = NEURONG_SDK_BUILD_ID;
    api.compiler_id = neurong::sdk::current_host_compiler_id();
    api.compiler_version = neurong::sdk::current_host_compiler_version();
    api.cxx_standard = neurong::sdk::current_cxx_standard();
    api.runtime_cuda_compiler_id = NEURONG_RUNTIME_CUDA_COMPILER_ID;
    api.runtime_cuda_compiler_version = NEURONG_RUNTIME_CUDA_COMPILER_VERSION;
    api.pointer_size = neurong::sdk::current_pointer_size();
    api.registrar.struct_size = sizeof(api.registrar);
    api.registrar.user_data = &ctx;
    api.registrar.register_mechanism = &MechPluginManager::register_mechanism_bridge_;
    api.registrar.log_message = &MechPluginManager::log_message_bridge_;

    neurong::sdk::PluginInfo info{};
    info.struct_size = sizeof(info);
    const int init_rc = init_fn(&api, &info);
    if (init_rc != 0) {
        std::cerr << "load_mech_library: plugin init failed for '" << canonical_path << "' with rc=" << init_rc;
        if (!ctx.error_message.empty()) {
            std::cerr << " error=" << ctx.error_message;
        }
        std::cerr << "\n";
        dlclose(handle);
        return -1;
    }
    if (!ctx.error_message.empty()) {
        std::cerr << "load_mech_library: plugin registration callback failed for '" << canonical_path
                  << "': " << ctx.error_message << "\n";
        dlclose(handle);
        return -1;
    }

    std::string validation_error;
    if (!validate_plugin_info_(info, api, &validation_error)) {
        std::cerr << "load_mech_library: plugin metadata rejected for '" << canonical_path
                  << "': " << validation_error << "\n";
        dlclose(handle);
        return -1;
    }

    std::vector<MechanismFactory::PendingRegistration> pending;
    pending.reserve(ctx.pending.size());
    std::vector<std::string> mech_names;
    mech_names.reserve(ctx.pending.size());
    for (const auto& entry : ctx.pending) {
        pending.push_back(entry.registration);
        mech_names.push_back(entry.registration.name);
    }

    auto already_registered_from_same_plugin = [&]() -> bool {
        auto& factory = MechanismFactory::getInstance();
        for (const auto& reg : pending) {
            const auto* info = factory.getRegistrationInfo(reg.name);
            if (info == nullptr) {
                return false;
            }
            if (info->source_file != reg.source_file || info->source_symbol != reg.source_symbol) {
                return false;
            }
        }
        return !pending.empty();
    };

    if (already_registered_from_same_plugin()) {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (!state.path_to_index.contains(canonical_path)) {
            state.loaded_handles.push_back(GlobalLoadedHandle{handle});
            state.loaded_plugins.push_back(LoadedPluginInfo{
                .path = canonical_path,
                .plugin_name = null_to_empty(info.plugin_name),
                .plugin_version = null_to_empty(info.plugin_version),
                .sdk_build_id = null_to_empty(info.sdk_build_id),
                .mechanism_names = mech_names,
            });
            state.path_to_index.emplace(canonical_path, state.loaded_plugins.size() - 1);
        } else {
            dlclose(handle);
        }
        std::cerr << "load_mech_library: plugin '" << null_to_empty(info.plugin_name)
                  << "' already present in global registry from '" << canonical_path
                  << "', reusing existing registration\n";
        return 0;
    }

    std::string batch_error;
    if (MechanismFactory::getInstance().registerMechanismBatch(pending, &batch_error) != 0) {
        std::cerr << "load_mech_library: failed to commit plugin registrations for '" << canonical_path
                  << "': " << batch_error << "\n";
        dlclose(handle);
        return -1;
    }

    {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (state.path_to_index.contains(canonical_path)) {
            std::cerr << "load_mech_library: plugin became loaded concurrently for '" << canonical_path
                      << "', keeping process-global handle resident\n";
            return 0;
        }
        state.loaded_handles.push_back(GlobalLoadedHandle{handle});
        state.loaded_plugins.push_back(LoadedPluginInfo{
        .path = canonical_path,
        .plugin_name = null_to_empty(info.plugin_name),
        .plugin_version = null_to_empty(info.plugin_version),
        .sdk_build_id = null_to_empty(info.sdk_build_id),
        .mechanism_names = std::move(mech_names),
        });
        state.path_to_index.emplace(canonical_path, state.loaded_plugins.size() - 1);
        std::cerr << "load_mech_library: loaded plugin '" << state.loaded_plugins.back().plugin_name
                  << "' from '" << canonical_path << "' with " << state.loaded_plugins.back().mechanism_names.size()
                  << " mechanism(s); plugins remain resident for process lifetime\n";
    }
    return 0;
}

const std::vector<MechPluginManager::LoadedPluginInfo>& MechPluginManager::loaded_plugins() const {
    return global_plugin_state().loaded_plugins;
}

}  // namespace neurong::runtime_api::core
