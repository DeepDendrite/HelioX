#pragma once

#include <cstddef>
#include <cstdint>

#include "neurong_sdk/compile_env.hpp"
#include "neurong_sdk/export.hpp"
#include "neurong_sdk/plugin_registration.hpp"
#include "neurong_sdk/version.hpp"

namespace neurong::sdk {

enum class LogLevel : std::uint32_t {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
};

struct PluginInfo {
    std::uint32_t struct_size = sizeof(PluginInfo);
    const char* plugin_name = nullptr;
    const char* plugin_version = nullptr;
    std::uint32_t sdk_abi_major = NEURONG_SDK_ABI_MAJOR;
    std::uint32_t sdk_abi_minor = NEURONG_SDK_ABI_MINOR;
    const char* sdk_build_id = NEURONG_SDK_BUILD_ID;
    const char* compiler_id = current_host_compiler_id();
    const char* compiler_version = current_host_compiler_version();
    std::uint32_t cxx_standard = current_cxx_standard();
    const char* cuda_compiler_id = current_cuda_compiler_id();
    const char* cuda_compiler_version = current_cuda_compiler_version();
    std::uint32_t pointer_size = current_pointer_size();
    std::uint32_t plugin_flags = 0;
    std::uint32_t mechanism_count = 0;
};

using LogMessageFn = int (*)(void* user_data, LogLevel level, const char* message);

struct RegistrarApi {
    std::uint32_t struct_size = sizeof(RegistrarApi);
    void* user_data = nullptr;
    RegisterMechanismFn register_mechanism = nullptr;
    LogMessageFn log_message = nullptr;
};

struct PluginApi {
    std::uint32_t struct_size = sizeof(PluginApi);
    std::uint32_t sdk_abi_major = NEURONG_SDK_ABI_MAJOR;
    std::uint32_t sdk_abi_minor = NEURONG_SDK_ABI_MINOR;
    const char* runtime_version = NEURONG_SDK_PROJECT_VERSION;
    const char* runtime_build_id = NEURONG_SDK_BUILD_ID;
    const char* compiler_id = current_host_compiler_id();
    const char* compiler_version = current_host_compiler_version();
    std::uint32_t cxx_standard = current_cxx_standard();
    const char* runtime_cuda_compiler_id = NEURONG_RUNTIME_CUDA_COMPILER_ID;
    const char* runtime_cuda_compiler_version = NEURONG_RUNTIME_CUDA_COMPILER_VERSION;
    std::uint32_t pointer_size = current_pointer_size();
    RegistrarApi registrar{};
};

using PluginInitFn = int (*)(const PluginApi* api, PluginInfo* out_info);

}  // namespace neurong::sdk

#define NEURONG_MECH_PLUGIN_INIT_SYMBOL "neurong_mech_plugin_init"
#define NEURONG_DECLARE_MECH_PLUGIN_INIT() \
    NEURONG_SDK_EXTERN_C NEURONG_SDK_EXPORT int neurong_mech_plugin_init( \
        const ::neurong::sdk::PluginApi* api, ::neurong::sdk::PluginInfo* out_info)
