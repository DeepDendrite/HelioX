#pragma once

#include <cstdint>

namespace neurong::sdk {

#define NEURONG_SDK_STR_INNER(x) #x
#define NEURONG_SDK_STR(x) NEURONG_SDK_STR_INNER(x)

inline constexpr const char* current_host_compiler_id() {
#if defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GNU";
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "Unknown";
#endif
}

inline constexpr const char* current_host_compiler_version() {
#if defined(__clang__)
    return NEURONG_SDK_STR(__clang_major__) "." NEURONG_SDK_STR(__clang_minor__) "." NEURONG_SDK_STR(__clang_patchlevel__);
#elif defined(__GNUC__)
    return NEURONG_SDK_STR(__GNUC__) "." NEURONG_SDK_STR(__GNUC_MINOR__) "." NEURONG_SDK_STR(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_FULL_VER)
    return NEURONG_SDK_STR(_MSC_FULL_VER);
#elif defined(_MSC_VER)
    return NEURONG_SDK_STR(_MSC_VER);
#else
    return "unknown";
#endif
}

inline constexpr std::uint32_t current_cxx_standard() {
#if __cplusplus >= 202302L
    return 23;
#elif __cplusplus >= 202002L
    return 20;
#elif __cplusplus >= 201703L
    return 17;
#elif __cplusplus >= 201402L
    return 14;
#elif __cplusplus >= 201103L
    return 11;
#else
    return 0;
#endif
}

inline constexpr std::uint32_t current_pointer_size() {
    return static_cast<std::uint32_t>(sizeof(void*));
}

inline constexpr bool current_compiled_with_cuda_frontend() {
#if defined(__CUDACC__)
    return true;
#else
    return false;
#endif
}

inline constexpr const char* current_cuda_compiler_id() {
#if defined(__NVCC__)
    return "NVIDIA";
#elif defined(__clang__) && defined(__CUDA__)
    return "ClangCUDA";
#else
    return "host-only";
#endif
}

inline constexpr const char* current_cuda_compiler_version() {
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
    return NEURONG_SDK_STR(__CUDACC_VER_MAJOR__) "." NEURONG_SDK_STR(__CUDACC_VER_MINOR__) "." NEURONG_SDK_STR(__CUDACC_VER_BUILD__);
#else
    return "n/a";
#endif
}

#undef NEURONG_SDK_STR
#undef NEURONG_SDK_STR_INNER

}  // namespace neurong::sdk
