#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "neurong_sdk/plugin_api.hpp"

namespace neurong::sdk {

struct DefaultPluginMetadata {
    const char* plugin_name = nullptr;
    const char* plugin_version = nullptr;
    std::uint32_t plugin_flags = 0;
};

NEURONG_SDK_LOCAL inline int log_plugin_init_message(const PluginApi* api, LogLevel level, const std::string& message) {
    if (api != nullptr && api->registrar.log_message != nullptr) {
        return api->registrar.log_message(api->registrar.user_data, level, message.c_str());
    }
    std::fprintf(stderr, "[neurong-sdk] %s\n", message.c_str());
    return 0;
}

NEURONG_SDK_LOCAL inline int fill_default_plugin_info(PluginInfo* out_info,
                                                      const DefaultPluginMetadata& metadata,
                                                      std::uint32_t mechanism_count) {
    if (out_info == nullptr) {
        return -1;
    }
    out_info->struct_size = sizeof(*out_info);
    out_info->plugin_name = metadata.plugin_name;
    out_info->plugin_version = metadata.plugin_version;
    out_info->sdk_abi_major = NEURONG_SDK_ABI_MAJOR;
    out_info->sdk_abi_minor = NEURONG_SDK_ABI_MINOR;
    out_info->sdk_build_id = NEURONG_SDK_BUILD_ID;
    out_info->compiler_id = current_host_compiler_id();
    out_info->compiler_version = current_host_compiler_version();
    out_info->cxx_standard = current_cxx_standard();
    out_info->cuda_compiler_id = current_cuda_compiler_id();
    out_info->cuda_compiler_version = current_cuda_compiler_version();
    out_info->pointer_size = current_pointer_size();
    out_info->plugin_flags = metadata.plugin_flags;
    out_info->mechanism_count = mechanism_count;
    return 0;
}

NEURONG_SDK_LOCAL inline int flush_plugin_local_registry(const PluginApi* api,
                                                         PluginInfo* out_info,
                                                         const DefaultPluginMetadata& metadata) {
    if (api == nullptr || out_info == nullptr) {
        return -1;
    }
    if (api->registrar.register_mechanism == nullptr) {
        return -1;
    }

    if (!plugin_local_registry_ok()) {
        const char* error = plugin_local_registry_error_message();
        log_plugin_init_message(api, LogLevel::Error, error == nullptr ? "plugin registration error" : error);
        return -1;
    }

    auto& registry = plugin_local_registry();
    for (const auto& mech_name : registry.registration_order) {
        const auto it = registry.records.find(mech_name);
        if (it == registry.records.end()) {
            continue;
        }
        const auto& record = it->second;
        if (record.create_fn == nullptr || record.kind == LocalRegistrationKind::None) {
            log_plugin_init_message(
                api,
                LogLevel::Error,
                "plugin registration error for '" + record.name + "': missing creator registration");
            return -1;
        }

        MechanismRegistration registration{};
        registration.struct_size = sizeof(registration);
        registration.name = record.name.c_str();
        registration.create_fn = record.create_fn;
        registration.source_file = record.source_file.empty() ? nullptr : record.source_file.c_str();
        registration.source_symbol = record.source_symbol.empty() ? nullptr : record.source_symbol.c_str();
        registration.has_pnt_receive_size = record.has_pnt_receive_size ? 1 : 0;
        registration.pnt_receive_size = record.pnt_receive_size;
        if (record.has_pointer_dparam_slots) {
            registration.pointer_dparam_slots = record.pointer_dparam_slots.data();
            registration.pointer_dparam_slots_count = record.pointer_dparam_slots.size();
        }
        if (record.has_dparam_semantics) {
            registration.dparam_semantics = record.dparam_semantics.data();
            registration.dparam_semantics_count = record.dparam_semantics.size();
        }

        if (api->registrar.register_mechanism(api->registrar.user_data, &registration) != 0) {
            log_plugin_init_message(
                api,
                LogLevel::Error,
                "plugin registration flush failed for mechanism '" + record.name + "'");
            return -1;
        }
    }

    return fill_default_plugin_info(
        out_info,
        metadata,
        static_cast<std::uint32_t>(registry.registration_order.size()));
}

}  // namespace neurong::sdk

#define NEURONG_DEFINE_MECH_PACKAGE_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION) \
    NEURONG_DECLARE_MECH_PLUGIN_INIT() { \
        return ::neurong::sdk::flush_plugin_local_registry( \
            api, \
            out_info, \
            ::neurong::sdk::DefaultPluginMetadata{PLUGIN_NAME, PLUGIN_VERSION, 0}); \
    }
