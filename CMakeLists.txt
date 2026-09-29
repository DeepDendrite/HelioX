cmake_minimum_required(VERSION 3.24)

set(NEURONG_CUDA_ARCHITECTURES "" CACHE STRING "Override CUDA architectures, e.g. native, 70, 120, 70;120")
if(NEURONG_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${NEURONG_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

project(NEURONG 
        LANGUAGES C CXX CUDA
        DESCRIPTION "神经网络模拟项目"
        VERSION 1.0.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(NEURONG_SDK_ABI_MAJOR 0)
set(NEURONG_SDK_ABI_MINOR 1)
execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NEURONG_GIT_SHORT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT NEURONG_GIT_SHORT_HASH)
    set(NEURONG_GIT_SHORT_HASH "nogit")
endif()
execute_process(
    COMMAND git diff --quiet --ignore-submodules HEAD --
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE NEURONG_GIT_DIRTY_RESULT
    ERROR_QUIET
)
set(NEURONG_SDK_BUILD_ID_BASE "${PROJECT_VERSION}-${NEURONG_GIT_SHORT_HASH}")
if(NEURONG_GIT_DIRTY_RESULT AND NOT NEURONG_GIT_DIRTY_RESULT EQUAL 0)
    string(APPEND NEURONG_SDK_BUILD_ID_BASE "-dirty")
endif()

# 设置全局属性和编译标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 抑制 nvcc 关于废弃 GPU 架构的告警（例如 sm_75 / lto_75）
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
add_link_options("$<$<LINK_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

# 添加位置无关代码标志，对所有目标
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 保持更严格的浮点行为，尽量贴近NEURON数值路径
add_compile_options("$<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:-fno-fast-math>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:-fexcess-precision=standard>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fexcess-precision=standard>")

# CUDA 数学精度开关：
# - OFF (default): 性能优先，使用 CUDA 默认行为（允许 FMA / 可能使用更快的除法/开方实现）。
# - ON:  更严格的数值路径（更贴近某些 CPU/NEURON 数值行为），但可能显著变慢。
option(ENABLE_STRICT_CUDA_MATH "Use strict CUDA math flags (--fmad=false --prec-div=true --prec-sqrt=true)" OFF)
if(ENABLE_STRICT_CUDA_MATH)
    message(STATUS "Strict CUDA math flags: ENABLED (--fmad=false --prec-div=true --prec-sqrt=true)")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>")
else()
    message(STATUS "Strict CUDA math flags: disabled (performance default)")
endif()

# 控制是否启用 LTO（Link Time Optimization）
option(ENABLE_LTO "Enable Link Time Optimization" OFF)
if(ENABLE_LTO)
    message(STATUS "LTO is ENABLED")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO is not supported: ${error}")
    endif()
endif()

# 使用target_compile_options替代直接设置全局编译标志
# add_compile_options("$<$<CONFIG:Release>:-O2>")
# add_compile_options("$<$<CONFIG:Debug>:-O0>")
# add_compile_options(-g)

# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -g --generate-line-info -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g --generate-line-info -Wno-deprecated-gpu-targets")

# Debug 模式处理
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

# DEBUG_PRINTF 开关
option(ENABLE_DEBUG_PRINTF "Enable debug print functionality" OFF)
if(ENABLE_DEBUG_PRINTF)
    add_compile_definitions(DEBUG_PRINTF)
endif()

option(NEURON_USE_MANAGED_MEMORY "Use CUDA Unified Managed Memory for VecData buffers" OFF)
if(NEURON_USE_MANAGED_MEMORY)
    message(STATUS "Unified managed memory support: ENABLED")
endif()

add_library(neurong_sdk_build_contract INTERFACE)
target_compile_definitions(neurong_sdk_build_contract INTERFACE
    $<$<BOOL:${NEURON_USE_MANAGED_MEMORY}>:NEURON_USE_MANAGED_MEMORY>
)

# 引入依赖库
set(HIGHFIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/HighFive)
list(APPEND CMAKE_PREFIX_PATH ${HIGHFIVE_DIR})

find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(NEURONG_SDK_BUILD_ID "${NEURONG_SDK_BUILD_ID_BASE}")
string(APPEND NEURONG_SDK_BUILD_ID
    "|cxx=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    "|cudatk=${CUDAToolkit_VERSION}"
    "|cxxstd=${CMAKE_CXX_STANDARD}"
    "|custd=${CMAKE_CUDA_STANDARD}"
    "|ptr=${CMAKE_SIZEOF_VOID_P}"
)
message(STATUS "NEURONG_SDK_BUILD_ID: ${NEURONG_SDK_BUILD_ID}")

# 添加nanobind支持
set(Python_FIND_VIRTUALENV FIRST)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

# 添加nanobind支持 - 使用Python方式查找
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
if(NOT nanobind_ROOT)
    message(FATAL_ERROR "Failed to locate nanobind via `${Python_EXECUTABLE} -m nanobind --cmake_dir`.")
endif()
message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")
list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
set(nanobind_DIR "${nanobind_ROOT}" CACHE PATH "nanobind cmake dir from selected Python interpreter" FORCE)
find_package(nanobind CONFIG REQUIRED)

# 定义包含目录列表
		set(INCLUDE_DIRS
		    ${CMAKE_SOURCE_DIR}/sdk/include
            ${CMAKE_SOURCE_DIR}/src/sdk_shared
		    ${CMAKE_BINARY_DIR}/generated
		    ${CMAKE_SOURCE_DIR}/src
		    ${CMAKE_SOURCE_DIR}/src/io
		    ${CMAKE_SOURCE_DIR}/src/celltemplate
		    ${CMAKE_SOURCE_DIR}/src/morph
		    ${CMAKE_SOURCE_DIR}/src/biophysical
		    ${CMAKE_SOURCE_DIR}/src/mech
		    ${CMAKE_SOURCE_DIR}/src/mech/build-in-mech
		    ${CMAKE_SOURCE_DIR}/src/mech/user-mech
            ${CMAKE_SOURCE_DIR}/mech-packages/l5pc
            ${CMAKE_SOURCE_DIR}/mech-packages/hl23
            ${CMAKE_SOURCE_DIR}/mech-packages/eworm
            ${CMAKE_SOURCE_DIR}/mech-packages/eworm-lr
	    ${CMAKE_SOURCE_DIR}/src/utils
	    ${CMAKE_SOURCE_DIR}/src/spike
	    ${CMAKE_SOURCE_DIR}/src/spike/postsyn_mech
	    ${CMAKE_SOURCE_DIR}/src/spike/user_postsyn_mech
	    ${CMAKE_SOURCE_DIR}/src/third_party/CXXOPTS
    ${CMAKE_SOURCE_DIR}/src/third_party/magic_enum
    ${CMAKE_SOURCE_DIR}/src/third_party/Random123
    ${CMAKE_SOURCE_DIR}/sdk/include
    ${CMAKE_BINARY_DIR}/generated
    ${Python_INCLUDE_DIRS}  # 显式添加Python头文件目录
)

configure_file(
    ${CMAKE_SOURCE_DIR}/sdk/include/neurong_sdk/version.hpp.in
    ${CMAKE_BINARY_DIR}/generated/neurong_sdk/version.hpp
    @ONLY
)

set(NEURONG_INSTALL_PYTHON_DIR "python")
set(NEURONG_INSTALL_PYEXT_DIR "${CMAKE_INSTALL_LIBDIR}/python")
set(NEURONG_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/NeuronGSDK")
set(NEURONG_INSTALL_SHARE_DIR "${CMAKE_INSTALL_DATADIR}/neurong")

# 收集核心源码文件（排除 main.cpp 和 Python API 目录）
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
)
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*main\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*src/python_api/.*\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*src/mech/mechanism\\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*src/mech/mech_var_table\\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*src/io/global_vars\\.cpp$")
# runtime_api is part of the core build, not the Python binding layer.

# 单独获取 main.cpp 文件
set(MAIN_CPP_SOURCE "")
file(GLOB MAIN_CPP_GLOB CONFIGURE_DEPENDS "src/main.cpp")
if(MAIN_CPP_GLOB)
    set(MAIN_CPP_SOURCE ${MAIN_CPP_GLOB})
endif()

# 收集 Python API（nanobind module + 相关CUDA核）源文件（包括 .cpp, .cu 等）
file(GLOB_RECURSE PYTHON_API_CPP_SOURCES CONFIGURE_DEPENDS "src/python_api/*.cpp")
file(GLOB_RECURSE PYTHON_API_CU_SOURCES CONFIGURE_DEPENDS "src/python_api/*.cu")
file(GLOB_RECURSE PYTHON_API_H_SOURCES CONFIGURE_DEPENDS "src/python_api/*.h" "src/python_api/*.hpp")
file(GLOB_RECURSE PYTHON_API_CUH_SOURCES CONFIGURE_DEPENDS "src/python_api/*.cuh")

# 合并所有 Python API 源文件
set(PYTHON_API_SOURCES 
    ${PYTHON_API_CPP_SOURCES}
    ${PYTHON_API_CU_SOURCES}
)

# Python API 头文件列表（用于包含路径）
set(PYTHON_API_HEADERS
    ${PYTHON_API_H_SOURCES}
    ${PYTHON_API_CUH_SOURCES}
)

# 收集 CUDA 源码（排除 Python API 目录下的文件）
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/python_api/.*\.cu$")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/utils/cuda_utils\\.cu$")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/mech/ion_table\\.cu$")

	# 自动收集独立机制目录下的所有.cu文件（独立编译）
	file(GLOB STANDALONE_MECH_SOURCES CONFIGURE_DEPENDS
	    "${CMAKE_SOURCE_DIR}/mech-packages/l5pc/*.cu"
	    "${CMAKE_SOURCE_DIR}/src/mech/user-mech/*.cu"
	    "${CMAKE_SOURCE_DIR}/mech-packages/hl23/*.cu"
	    "${CMAKE_SOURCE_DIR}/mech-packages/eworm/*.cu"
	    "${CMAKE_SOURCE_DIR}/mech-packages/eworm-lr/*.cu"
	)

    file(GLOB SPIKE_VJP_SURROGATE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/spike/surrogates/*.cu"
    )

    list(FILTER STANDALONE_MECH_SOURCES EXCLUDE REGEX ".*/mech-packages/.*/package\\.cu$")

    # HL23 mechs are shipped as a dedicated SDK plugin, not part of the
    # default standalone static mechanism set.
    list(FILTER STANDALONE_MECH_SOURCES EXCLUDE REGEX ".*/mech-packages/hl23/.*\\.cu$")

    # L5PC mechanisms are now shipped as a dedicated package plugin.
    list(FILTER STANDALONE_MECH_SOURCES EXCLUDE REGEX ".*/mech-packages/l5pc/.*\\.cu$")

    # eworm non-lr mechanisms are now shipped as a dedicated package plugin.
    list(FILTER STANDALONE_MECH_SOURCES EXCLUDE REGEX ".*/mech-packages/eworm/.*\\.cu$")

    # eworm-lr mechanisms are now shipped as a dedicated package plugin.
    list(FILTER STANDALONE_MECH_SOURCES EXCLUDE REGEX ".*/mech-packages/eworm-lr/.*\\.cu$")

	# 从CUDA_SOURCES中排除这些独立编译的机制
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/mech/user-mech/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/spike/surrogates/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*mech-packages/l5pc/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*mech-packages/hl23/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*mech-packages/eworm/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*mech-packages/eworm-lr/.*\.cu$")

file(GLOB_RECURSE CUDA_HEADERS CONFIGURE_DEPENDS "src/*.cuh")
list(FILTER CUDA_HEADERS EXCLUDE REGEX ".*src/python_api/.*\.cuh$")

# 输出源文件统计信息，用于调试
message(STATUS "Core CPP sources: ${CPP_SOURCES}")
message(STATUS "Main CPP source: ${MAIN_CPP_SOURCE}")
message(STATUS "Python API CPP sources: ${PYTHON_API_CPP_SOURCES}")
message(STATUS "Python API CU sources: ${PYTHON_API_CU_SOURCES}")
message(STATUS "CUDA sources: ${CUDA_SOURCES}")
message(STATUS "Standalone mechanisms: ${STANDALONE_MECH_SOURCES}")
message(STATUS "Spike VJP surrogates: ${SPIKE_VJP_SURROGATE_SOURCES}")

# 创建独立机制库（使用STATIC库 + whole-archive强制完全链接）
add_library(standalone_mechs STATIC ${STANDALONE_MECH_SOURCES})
target_include_directories(standalone_mechs PUBLIC ${INCLUDE_DIRS})
target_compile_options(standalone_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(standalone_mechs PUBLIC HDF5::HDF5 HighFive)
target_link_libraries(standalone_mechs PUBLIC neurong_sdk_build_contract)
set_target_properties(standalone_mechs PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(spike_vjp_surrogates STATIC ${SPIKE_VJP_SURROGATE_SOURCES})
target_include_directories(spike_vjp_surrogates PUBLIC ${INCLUDE_DIRS})
target_compile_options(spike_vjp_surrogates PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(spike_vjp_surrogates PUBLIC CUDA::cudart)
target_link_libraries(spike_vjp_surrogates PUBLIC neurong_sdk_build_contract)
set_target_properties(spike_vjp_surrogates PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(neurong_sdk_support SHARED
    src/mech/mechanism.cpp
    src/mech/mech_var_table.cpp
    src/mech/ion_table.cu
    src/io/global_vars.cpp
    src/io/staging_channel.cu
    src/utils/cuda_utils.cu
)
add_library(neurong_shared_mech_headers INTERFACE)
target_include_directories(neurong_shared_mech_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/sdk_shared>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(neurong_shared_mech_headers INTERFACE neurong_sdk_build_contract)
set_target_properties(neurong_sdk_build_contract PROPERTIES EXPORT_NAME SDKBuildContract)
set_target_properties(neurong_shared_mech_headers PROPERTIES EXPORT_NAME SharedMech)

target_include_directories(neurong_sdk_support
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${INCLUDE_DIRS}
)
target_compile_options(neurong_sdk_support PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(neurong_sdk_support
    PRIVATE
        HDF5::HDF5
        HighFive
    PUBLIC
        CUDA::cudart
        neurong_shared_mech_headers
        neurong_sdk_build_contract
)
set_target_properties(neurong_sdk_support PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME "neurong_sdk_support")
set_target_properties(neurong_sdk_support PROPERTIES EXPORT_NAME Runtime)

add_library(neurong_sdk_headers INTERFACE)
target_include_directories(neurong_sdk_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(neurong_sdk_headers INTERFACE neurong_shared_mech_headers neurong_sdk_build_contract)
set_target_properties(neurong_sdk_headers PROPERTIES EXPORT_NAME SDKHeaders)

set(NEURONG_MECH_TEMPLATE_BUILD_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/mech
    ${CMAKE_SOURCE_DIR}/src/mech/build-in-mech
    ${CMAKE_SOURCE_DIR}/src/autodiff
    ${CMAKE_SOURCE_DIR}/src/spike
    ${CMAKE_SOURCE_DIR}/src/third_party/HighFive/include
    ${CMAKE_SOURCE_DIR}/src/third_party/magic_enum
    ${CMAKE_SOURCE_DIR}/src/third_party/Random123
)
set(NEURONG_MECH_TEMPLATE_INSTALL_INCLUDE_DIRS
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/io
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/utils
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/mech
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/mech/build-in-mech
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/autodiff
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/spike
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/third_party/HighFive/include
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/third_party/magic_enum
    ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/third_party/Random123
)

add_library(neurong_mech_template_headers INTERFACE)
foreach(include_dir IN LISTS NEURONG_MECH_TEMPLATE_BUILD_INCLUDE_DIRS)
    target_include_directories(neurong_mech_template_headers INTERFACE
        $<BUILD_INTERFACE:${include_dir}>
    )
endforeach()
foreach(include_dir IN LISTS NEURONG_MECH_TEMPLATE_INSTALL_INCLUDE_DIRS)
    target_include_directories(neurong_mech_template_headers INTERFACE
        $<INSTALL_INTERFACE:${include_dir}>
    )
endforeach()
target_link_libraries(neurong_mech_template_headers INTERFACE
    neurong_sdk_headers
    neurong_shared_mech_headers
    neurong_sdk_build_contract
    neurong_sdk_support
    HDF5::HDF5
)
set_target_properties(neurong_mech_template_headers PROPERTIES EXPORT_NAME MechTemplate)

add_library(neurong_mech_sdk_internal_compat INTERFACE)
target_include_directories(neurong_mech_sdk_internal_compat INTERFACE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/mech
    ${CMAKE_SOURCE_DIR}/src/mech/build-in-mech
    ${CMAKE_SOURCE_DIR}/mech-packages/l5pc
    ${CMAKE_SOURCE_DIR}/mech-packages/hl23
    ${CMAKE_SOURCE_DIR}/mech-packages/eworm
    ${CMAKE_SOURCE_DIR}/mech-packages/eworm-lr
    ${CMAKE_SOURCE_DIR}/src/spike
    ${CMAKE_SOURCE_DIR}/src/third_party/magic_enum
)
target_link_libraries(neurong_mech_sdk_internal_compat INTERFACE HDF5::HDF5 HighFive)
target_link_libraries(neurong_mech_sdk_internal_compat INTERFACE neurong_shared_mech_headers neurong_sdk_build_contract)

# CUDA 静态库（使用位置无关代码）
add_library(gpulib STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(gpulib PUBLIC ${INCLUDE_DIRS})
target_compile_options(gpulib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(gpulib PUBLIC HDF5::HDF5 HighFive)
target_link_libraries(gpulib PUBLIC neurong_sdk_build_contract)
set_target_properties(gpulib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(gpulib PUBLIC CUDA::cudart)

# 使用 whole-archive 链接独立机制库，强制包含所有符号
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Linux/GCC/Clang
    target_link_libraries(gpulib PUBLIC
        -Wl,--whole-archive
        standalone_mechs
        spike_vjp_surrogates
        -Wl,--no-whole-archive
    )
elseif(MSVC)
    # Windows/MSVC
    target_link_libraries(gpulib PUBLIC
        /WHOLEARCHIVE:standalone_mechs
        /WHOLEARCHIVE:spike_vjp_surrogates
    )
else()
    # 其他平台降级为普通链接
    target_link_libraries(gpulib PUBLIC standalone_mechs)
    target_link_libraries(gpulib PUBLIC spike_vjp_surrogates)
    message(WARNING "Whole-archive linking not supported on this platform, standalone mechanisms may not register")
endif()
# 主可执行程序 
add_executable(neurong_exec ${CPP_SOURCES} ${MAIN_CPP_SOURCE})
set_target_properties(neurong_exec PROPERTIES OUTPUT_NAME "neurong_exec")
set_target_properties(neurong_exec PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
target_include_directories(neurong_exec PRIVATE ${INCLUDE_DIRS})
target_link_libraries(neurong_exec PRIVATE 
    neurong_sdk_support
    gpulib 
    HDF5::HDF5 
    HighFive 
    ${CMAKE_DL_LIBS}
)

# Python绑定模块 
nanobind_add_module(neurong_py 
    ${PYTHON_API_SOURCES}
    ${CPP_SOURCES}
)
target_include_directories(neurong_py PRIVATE 
    ${INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/python_api  # Python API 目录到包含路径
)
target_link_libraries(neurong_py PRIVATE 
    neurong_sdk_support
    gpulib 
    HDF5::HDF5 
    HighFive 
    ${CMAKE_DL_LIBS}
)
set_target_properties(neurong_py PROPERTIES OUTPUT_NAME "neurong")
set_target_properties(neurong_py PROPERTIES INSTALL_RPATH "$ORIGIN/..")

option(NEURONG_BUILD_SDK_DEMO_PLUGIN "Build the in-tree mech SDK demo plugin" OFF)
if(NEURONG_BUILD_SDK_DEMO_PLUGIN)
    add_library(neurong_sdk_demo_plugin SHARED EXCLUDE_FROM_ALL
        sdk/examples/minimal_plugin/minimal_plugin.cpp
    )
    target_compile_definitions(neurong_sdk_demo_plugin PRIVATE
        NEURONG_SDK_BUILDING_PLUGIN
        NEURONG_DISABLE_STATIC_MECH_REGISTRATION
    )
    target_link_libraries(neurong_sdk_demo_plugin PRIVATE neurong_sdk_headers neurong_sdk_support)
    set_target_properties(neurong_sdk_demo_plugin PROPERTIES OUTPUT_NAME "neurong_demo_mechs")
endif()

file(GLOB HL23_PACKAGE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/mech-packages/hl23/*.cu"
)
add_library(neurong_hl23_mechs SHARED
    ${HL23_PACKAGE_SOURCES}
)
target_compile_definitions(neurong_hl23_mechs PRIVATE
    NEURONG_SDK_BUILDING_PLUGIN
    NEURONG_DISABLE_STATIC_MECH_REGISTRATION
)
target_compile_options(neurong_hl23_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(neurong_hl23_mechs PRIVATE
    neurong_sdk_headers
    neurong_mech_sdk_internal_compat
    neurong_sdk_support
    CUDA::cudart
)
set_target_properties(neurong_hl23_mechs PROPERTIES OUTPUT_NAME "neurong_hl23_mechs")
set_target_properties(neurong_hl23_mechs PROPERTIES INSTALL_RPATH "$ORIGIN")

file(GLOB L5PC_PACKAGE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/mech-packages/l5pc/*.cu"
)
add_library(neurong_l5pc_mechs SHARED
    ${L5PC_PACKAGE_SOURCES}
)
target_compile_definitions(neurong_l5pc_mechs PRIVATE
    NEURONG_SDK_BUILDING_PLUGIN
    NEURONG_DISABLE_STATIC_MECH_REGISTRATION
)
target_compile_options(neurong_l5pc_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(neurong_l5pc_mechs PRIVATE
    neurong_sdk_headers
    neurong_mech_sdk_internal_compat
    neurong_sdk_support
    CUDA::cudart
)
set_target_properties(neurong_l5pc_mechs PROPERTIES OUTPUT_NAME "neurong_l5pc_mechs")
set_target_properties(neurong_l5pc_mechs PROPERTIES INSTALL_RPATH "$ORIGIN")

file(GLOB EWORM_PACKAGE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/mech-packages/eworm/*.cu"
)
add_library(neurong_eworm_mechs SHARED
    ${EWORM_PACKAGE_SOURCES}
)
target_compile_definitions(neurong_eworm_mechs PRIVATE
    NEURONG_SDK_BUILDING_PLUGIN
    NEURONG_DISABLE_STATIC_MECH_REGISTRATION
)
target_compile_options(neurong_eworm_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(neurong_eworm_mechs PRIVATE
    neurong_sdk_headers
    neurong_mech_sdk_internal_compat
    neurong_sdk_support
    CUDA::cudart
)
set_target_properties(neurong_eworm_mechs PROPERTIES OUTPUT_NAME "neurong_eworm_mechs")
set_target_properties(neurong_eworm_mechs PROPERTIES INSTALL_RPATH "$ORIGIN")

file(GLOB EWORM_LR_PACKAGE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/mech-packages/eworm-lr/*.cu"
)
add_library(neurong_eworm_lr_mechs SHARED
    ${EWORM_LR_PACKAGE_SOURCES}
)
target_compile_definitions(neurong_eworm_lr_mechs PRIVATE
    NEURONG_SDK_BUILDING_PLUGIN
    NEURONG_DISABLE_STATIC_MECH_REGISTRATION
)
target_compile_options(neurong_eworm_lr_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(neurong_eworm_lr_mechs PRIVATE
    neurong_sdk_headers
    neurong_mech_sdk_internal_compat
    neurong_sdk_support
    CUDA::cudart
)
set_target_properties(neurong_eworm_lr_mechs PROPERTIES OUTPUT_NAME "neurong_eworm_lr_mechs")
set_target_properties(neurong_eworm_lr_mechs PROPERTIES INSTALL_RPATH "$ORIGIN")

# 显示编译选项信息
foreach(tgt neurong_sdk_support gpulib neurong_exec neurong_py)
  get_target_property(opts ${tgt} COMPILE_OPTIONS)
  message(STATUS "Compile options for ${tgt}: ${opts}")
endforeach()

get_directory_property(compile_definitions COMPILE_DEFINITIONS)
message(STATUS "Compile definitions: ${compile_definitions}")

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/NeuronGSDKConfig.cmake.in
    ${CMAKE_BINARY_DIR}/generated/NeuronGSDKConfig.cmake
    INSTALL_DESTINATION ${NEURONG_INSTALL_CMAKE_DIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_DATADIR
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/generated/NeuronGSDKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS
    neurong_exec
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS
    neurong_sdk_support
    neurong_hl23_mechs
    neurong_l5pc_mechs
    neurong_eworm_mechs
    neurong_eworm_lr_mechs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS
    neurong_py
    LIBRARY DESTINATION ${NEURONG_INSTALL_PYEXT_DIR}
)

if(TARGET neurong_sdk_demo_plugin)
    install(TARGETS neurong_sdk_demo_plugin
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(TARGETS
    neurong_sdk_build_contract
    neurong_shared_mech_headers
    neurong_sdk_headers
    neurong_mech_template_headers
    neurong_sdk_support
    EXPORT NeuronGSDKTargets
)

install(EXPORT NeuronGSDKTargets
    FILE NeuronGSDKTargets.cmake
    NAMESPACE NeuronG::
    DESTINATION ${NEURONG_INSTALL_CMAKE_DIR}
)

install(FILES
    ${CMAKE_BINARY_DIR}/generated/NeuronGSDKConfig.cmake
    ${CMAKE_BINARY_DIR}/generated/NeuronGSDKConfigVersion.cmake
    DESTINATION ${NEURONG_INSTALL_CMAKE_DIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/sdk_shared/neurong_shared
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/sdk/include/neurong_sdk
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*.in" EXCLUDE
)

install(FILES
    ${CMAKE_SOURCE_DIR}/src/neuron.h
    ${CMAKE_SOURCE_DIR}/src/vecplay.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal
)

install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/src/autodiff
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/mech
    ${CMAKE_SOURCE_DIR}/src/spike
    ${CMAKE_SOURCE_DIR}/src/utils
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/src/third_party/HighFive/include
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/third_party/HighFive
    FILES_MATCHING
        PATTERN "*.hpp"
)

install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/src/third_party/magic_enum
    ${CMAKE_SOURCE_DIR}/src/third_party/Random123
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neurong_template_internal/third_party
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(FILES
    ${CMAKE_BINARY_DIR}/generated/neurong_sdk/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neurong_sdk
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/python_lib/
    DESTINATION ${NEURONG_INSTALL_PYTHON_DIR}
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.so" EXCLUDE
    PATTERN "*.pyd" EXCLUDE
)

install(PROGRAMS ${CMAKE_SOURCE_DIR}/tools/neurong-mech-build
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_SOURCE_DIR}/share/neurong/mech/CMakeLists.plugin.in
    ${CMAKE_SOURCE_DIR}/share/neurong/mech/plugin_init.cpp.in
    ${CMAKE_SOURCE_DIR}/share/neurong/mech/toolchain-contract.cmake
    DESTINATION ${NEURONG_INSTALL_SHARE_DIR}/mech
)

install(FILES
    ${CMAKE_SOURCE_DIR}/sdk/examples/minimal_plugin/minimal_plugin.cpp
    DESTINATION ${NEURONG_INSTALL_SHARE_DIR}/mech/examples/minimal_plugin
)

install(FILES
    ${CMAKE_SOURCE_DIR}/sdk/examples/templated_gap_plugin/sdk_template_gap.cu
    DESTINATION ${NEURONG_INSTALL_SHARE_DIR}/mech/examples/templated_gap_plugin
)
