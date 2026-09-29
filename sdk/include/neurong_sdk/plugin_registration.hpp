#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Mechanism;
struct MechInitParams;

#if !defined(NEURONG_SDK_LOCAL)
  #if defined(__GNUC__) || defined(__clang__)
    #define NEURONG_SDK_LOCAL __attribute__((visibility("hidden")))
  #else
    #define NEURONG_SDK_LOCAL
  #endif
#endif

namespace neurong::sdk {

using MechanismCreateFn = Mechanism* (*)(MechInitParams&);

struct MechanismRegistration {
    std::uint32_t struct_size = sizeof(MechanismRegistration);
    const char* name = nullptr;
    MechanismCreateFn create_fn = nullptr;
    const char* source_file = nullptr;
    const char* source_symbol = nullptr;
    int has_pnt_receive_size = 0;
    int pnt_receive_size = -1;
    const int* pointer_dparam_slots = nullptr;
    std::size_t pointer_dparam_slots_count = 0;
    const int* dparam_semantics = nullptr;
    std::size_t dparam_semantics_count = 0;
};

using RegisterMechanismFn = int (*)(void* user_data, const MechanismRegistration* registration);

enum class LocalRegistrationKind : std::uint32_t {
    None = 0,
    Mechanism = 1,
    Postsyn = 2,
};

struct LocalMechanismRegistration {
    std::string name;
    MechanismCreateFn create_fn = nullptr;
    std::string source_file;
    std::string source_symbol;
    LocalRegistrationKind kind = LocalRegistrationKind::None;
    bool has_pnt_receive_size = false;
    int pnt_receive_size = -1;
    bool has_pointer_dparam_slots = false;
    std::vector<int> pointer_dparam_slots;
    bool has_dparam_semantics = false;
    std::vector<int> dparam_semantics;
};

struct PluginLocalRegistry {
    std::unordered_map<std::string, LocalMechanismRegistration> records;
    std::vector<std::string> registration_order;
    std::string error_message;
};

NEURONG_SDK_LOCAL inline PluginLocalRegistry& plugin_local_registry() {
    static PluginLocalRegistry registry;
    return registry;
}

NEURONG_SDK_LOCAL inline bool plugin_local_registry_ok() {
    return plugin_local_registry().error_message.empty();
}

NEURONG_SDK_LOCAL inline const char* plugin_local_registry_error_message() {
    const auto& error = plugin_local_registry().error_message;
    return error.empty() ? nullptr : error.c_str();
}

NEURONG_SDK_LOCAL inline void plugin_local_registry_fail(std::string message) {
    auto& registry = plugin_local_registry();
    if (registry.error_message.empty()) {
        registry.error_message = std::move(message);
    }
}

NEURONG_SDK_LOCAL inline LocalMechanismRegistration& plugin_local_registry_record_for(const char* mech_name) {
    auto& registry = plugin_local_registry();
    auto [it, inserted] = registry.records.try_emplace(mech_name == nullptr ? "" : mech_name);
    if (inserted) {
        it->second.name = it->first;
        registry.registration_order.push_back(it->first);
    }
    return it->second;
}

NEURONG_SDK_LOCAL inline bool plugin_local_register_creator(const char* mech_name,
                                                            MechanismCreateFn create_fn,
                                                            const char* source_file,
                                                            const char* source_symbol,
                                                            LocalRegistrationKind kind,
                                                            int pnt_receive_size = -1) {
    if (mech_name == nullptr || mech_name[0] == '\0') {
        plugin_local_registry_fail("plugin registration error: mechanism name must not be empty");
        return false;
    }
    if (create_fn == nullptr) {
        plugin_local_registry_fail("plugin registration error for '" + std::string(mech_name) + "': creator must not be null");
        return false;
    }

    auto& record = plugin_local_registry_record_for(mech_name);
    if (record.kind != LocalRegistrationKind::None) {
        const char* prior = record.kind == LocalRegistrationKind::Postsyn ? "REGISTER_POSTSYN" : "REGISTER_MECHANISM";
        const char* current = kind == LocalRegistrationKind::Postsyn ? "REGISTER_POSTSYN" : "REGISTER_MECHANISM";
        plugin_local_registry_fail(
            "plugin registration error for '" + record.name +
            "': multiple creator registrations are not allowed; already registered via " +
            prior + ", saw " + current);
        return false;
    }

    record.create_fn = create_fn;
    record.source_file = source_file == nullptr ? "" : source_file;
    record.source_symbol = source_symbol == nullptr ? "" : source_symbol;
    record.kind = kind;
    if (kind == LocalRegistrationKind::Postsyn) {
        record.has_pnt_receive_size = true;
        record.pnt_receive_size = pnt_receive_size;
    }
    return true;
}

NEURONG_SDK_LOCAL inline bool plugin_local_register_mechanism(const char* mech_name,
                                                              MechanismCreateFn create_fn,
                                                              const char* source_file,
                                                              const char* source_symbol) {
    return plugin_local_register_creator(
        mech_name, create_fn, source_file, source_symbol, LocalRegistrationKind::Mechanism);
}

NEURONG_SDK_LOCAL inline bool plugin_local_register_postsyn(const char* mech_name,
                                                            MechanismCreateFn create_fn,
                                                            const char* source_file,
                                                            const char* source_symbol,
                                                            int pnt_receive_size) {
    return plugin_local_register_creator(
        mech_name, create_fn, source_file, source_symbol, LocalRegistrationKind::Postsyn, pnt_receive_size);
}

NEURONG_SDK_LOCAL inline bool plugin_local_register_pointer_dparam_slots(const char* mech_name, std::vector<int> slots) {
    if (mech_name == nullptr || mech_name[0] == '\0') {
        plugin_local_registry_fail("plugin registration error: mechanism name must not be empty");
        return false;
    }
    auto& record = plugin_local_registry_record_for(mech_name);
    if (!record.has_pointer_dparam_slots) {
        record.pointer_dparam_slots = std::move(slots);
        record.has_pointer_dparam_slots = true;
        return true;
    }
    if (record.pointer_dparam_slots != slots) {
        plugin_local_registry_fail(
            "plugin registration error for '" + record.name +
            "': conflicting REGISTER_POINTER_DPARAM_SLOTS definitions");
        return false;
    }
    return true;
}

NEURONG_SDK_LOCAL inline bool plugin_local_register_dparam_semantics(const char* mech_name, std::vector<int> semantics) {
    if (mech_name == nullptr || mech_name[0] == '\0') {
        plugin_local_registry_fail("plugin registration error: mechanism name must not be empty");
        return false;
    }
    auto& record = plugin_local_registry_record_for(mech_name);
    if (!record.has_dparam_semantics) {
        record.dparam_semantics = std::move(semantics);
        record.has_dparam_semantics = true;
        return true;
    }
    if (record.dparam_semantics != semantics) {
        plugin_local_registry_fail(
            "plugin registration error for '" + record.name +
            "': conflicting REGISTER_DPARAM_SEMANTICS definitions");
        return false;
    }
    return true;
}

}  // namespace neurong::sdk
