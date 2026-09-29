#pragma once

#include <string>
#include <vector>

#include <cstdint>

#include "mechanism.h"
#include "neurong_sdk/plugin_api.hpp"

namespace neurong::runtime_api::core {

class MechPluginManager final {
public:
    struct LoadedPluginInfo {
        std::string path;
        std::string plugin_name;
        std::string plugin_version;
        std::string sdk_build_id;
        std::vector<std::string> mechanism_names;
    };

    MechPluginManager() = default;
    ~MechPluginManager() = default;

    MechPluginManager(const MechPluginManager&) = delete;
    MechPluginManager& operator=(const MechPluginManager&) = delete;

    int load_library(const std::string& path);
    const std::vector<LoadedPluginInfo>& loaded_plugins() const;

private:
    struct PendingRegistration {
        MechanismFactory::PendingRegistration registration;
    };

    struct TransactionContext {
        std::string plugin_path;
        std::vector<PendingRegistration> pending;
        std::string error_message;
    };

    static int register_mechanism_bridge_(void* user_data, const neurong::sdk::MechanismRegistration* registration);
    static int log_message_bridge_(void* user_data, neurong::sdk::LogLevel level, const char* message);

    static std::string make_source_file_(const std::string& plugin_path, const char* source_file);
    static std::string make_source_symbol_(const std::string& plugin_name, const char* source_symbol);

    static bool validate_plugin_info_(const neurong::sdk::PluginInfo& info,
                                      const neurong::sdk::PluginApi& api,
                                      std::string* error);
};

}  // namespace neurong::runtime_api::core
