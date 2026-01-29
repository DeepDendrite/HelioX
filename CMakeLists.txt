cmake_minimum_required(VERSION 3.20)
project(HELIOX 
        LANGUAGES C CXX CUDA
        DESCRIPTION "神经网络模拟项目"
        VERSION 1.0.0)

# 设置全局属性和编译标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 推荐写法：支持主流架构（如只需本机架构可只写一个数字）
# CUDA 13 已移除部分旧架构（例如 sm_70），避免在新 NVCC 上构建失败。
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
# set(CMAKE_CUDA_ARCHITECTURES 80 86)

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

# 引入依赖库
set(HIGHFIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/HighFive)
list(APPEND CMAKE_PREFIX_PATH ${HIGHFIVE_DIR})

find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 添加nanobind支持
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

# 添加nanobind支持 - 使用Python方式查找
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")
list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
find_package(nanobind CONFIG REQUIRED)

# 定义包含目录列表
	set(INCLUDE_DIRS
	    ${CMAKE_SOURCE_DIR}/src
	    ${CMAKE_SOURCE_DIR}/src/io
	    ${CMAKE_SOURCE_DIR}/src/mech
	    ${CMAKE_SOURCE_DIR}/src/mech/build-in-mech
	    ${CMAKE_SOURCE_DIR}/src/mech/user-mech
	    ${CMAKE_SOURCE_DIR}/src/mech/worm-mech
	    ${CMAKE_SOURCE_DIR}/src/mech/worm-lr-mech
	    ${CMAKE_SOURCE_DIR}/src/utils
	    ${CMAKE_SOURCE_DIR}/src/spike
	    ${CMAKE_SOURCE_DIR}/src/spike/postsyn_mech
	    ${CMAKE_SOURCE_DIR}/src/spike/user_postsyn_mech
	    ${CMAKE_SOURCE_DIR}/src/third_party/CXXOPTS
    ${CMAKE_SOURCE_DIR}/src/third_party/magic_enum
    ${CMAKE_SOURCE_DIR}/src/third_party/Random123
    ${Python_INCLUDE_DIRS}  # 显式添加Python头文件目录
)

# 收集核心源码文件（排除 main.cpp 和 Python API 目录）
file(GLOB_RECURSE CPP_SOURCES 
    "src/*.cpp"
)
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*main\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*src/python_api/.*\.cpp$")
# runtime_api is part of the core build, not the Python binding layer.

# 单独获取 main.cpp 文件
set(MAIN_CPP_SOURCE "")
file(GLOB MAIN_CPP_GLOB "src/main.cpp")
if(MAIN_CPP_GLOB)
    set(MAIN_CPP_SOURCE ${MAIN_CPP_GLOB})
endif()

# 收集 Python API（nanobind module + 相关CUDA核）源文件（包括 .cpp, .cu 等）
file(GLOB_RECURSE PYTHON_API_CPP_SOURCES "src/python_api/*.cpp")
file(GLOB_RECURSE PYTHON_API_CU_SOURCES "src/python_api/*.cu")
file(GLOB_RECURSE PYTHON_API_H_SOURCES "src/python_api/*.h" "src/python_api/*.hpp")
file(GLOB_RECURSE PYTHON_API_CUH_SOURCES "src/python_api/*.cuh")

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
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/python_api/.*\.cu$")

	# 自动收集user-mech和worm-mech目录下的所有.cu文件（独立编译）
	file(GLOB STANDALONE_MECH_SOURCES
	    "${CMAKE_SOURCE_DIR}/src/mech/user-mech/*.cu"
	    "${CMAKE_SOURCE_DIR}/src/mech/worm-mech/*.cu"
	    "${CMAKE_SOURCE_DIR}/src/mech/worm-lr-mech/*.cu"
	)

	# 从CUDA_SOURCES中排除这些独立编译的机制
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/mech/user-mech/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/mech/worm-mech/.*\.cu$")
	list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*src/mech/worm-lr-mech/.*\.cu$")

file(GLOB_RECURSE CUDA_HEADERS "src/*.cuh")
list(FILTER CUDA_HEADERS EXCLUDE REGEX ".*src/python_api/.*\.cuh$")

# 输出源文件统计信息，用于调试
message(STATUS "Core CPP sources: ${CPP_SOURCES}")
message(STATUS "Main CPP source: ${MAIN_CPP_SOURCE}")
message(STATUS "Python API CPP sources: ${PYTHON_API_CPP_SOURCES}")
message(STATUS "Python API CU sources: ${PYTHON_API_CU_SOURCES}")
message(STATUS "CUDA sources: ${CUDA_SOURCES}")
message(STATUS "Standalone mechanisms (user-mech): ${STANDALONE_MECH_SOURCES}")

# 创建独立机制库（使用STATIC库 + whole-archive强制完全链接）
add_library(standalone_mechs STATIC ${STANDALONE_MECH_SOURCES})
target_include_directories(standalone_mechs PUBLIC ${INCLUDE_DIRS})
target_compile_options(standalone_mechs PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(standalone_mechs PUBLIC HDF5::HDF5 HighFive)
set_target_properties(standalone_mechs PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CUDA 静态库（使用位置无关代码）
add_library(gpulib STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(gpulib PUBLIC ${INCLUDE_DIRS})
target_compile_options(gpulib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20040>")
target_link_libraries(gpulib PUBLIC HDF5::HDF5 HighFive)
set_target_properties(gpulib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(gpulib PUBLIC CUDA::cudart)

# 使用 whole-archive 链接独立机制库，强制包含所有符号
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Linux/GCC/Clang
    target_link_libraries(gpulib PUBLIC
        -Wl,--whole-archive
        standalone_mechs
        -Wl,--no-whole-archive
    )
elseif(MSVC)
    # Windows/MSVC
    target_link_libraries(gpulib PUBLIC
        /WHOLEARCHIVE:standalone_mechs
    )
else()
    # 其他平台降级为普通链接
    target_link_libraries(gpulib PUBLIC standalone_mechs)
    message(WARNING "Whole-archive linking not supported on this platform, standalone mechanisms may not register")
endif()
# 主可执行程序 
add_executable(heliox_exec ${CPP_SOURCES} ${MAIN_CPP_SOURCE})
set_target_properties(heliox_exec PROPERTIES OUTPUT_NAME "heliox_exec")
target_include_directories(heliox_exec PRIVATE ${INCLUDE_DIRS})
target_link_libraries(heliox_exec PRIVATE 
    gpulib 
    HDF5::HDF5 
    HighFive 
)

# Python绑定模块 
nanobind_add_module(heliox_py 
    ${PYTHON_API_SOURCES}
    ${CPP_SOURCES}
)
target_include_directories(heliox_py PRIVATE 
    ${INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/python_api  # Python API 目录到包含路径
)
target_link_libraries(heliox_py PRIVATE 
    gpulib 
    HDF5::HDF5 
    HighFive 
)
set_target_properties(heliox_py PROPERTIES OUTPUT_NAME "heliox")

# 显示编译选项信息
foreach(tgt gpulib heliox_exec heliox_py)
  get_target_property(opts ${tgt} COMPILE_OPTIONS)
  message(STATUS "Compile options for ${tgt}: ${opts}")
endforeach()

if(NEURON_USE_MANAGED_MEMORY)
    foreach(tgt standalone_mechs gpulib heliox_exec heliox_py)
        if(TARGET ${tgt})
            target_compile_definitions(${tgt} PRIVATE NEURON_USE_MANAGED_MEMORY)
        endif()
    endforeach()
endif()

get_directory_property(compile_definitions COMPILE_DEFINITIONS)
message(STATUS "Compile definitions: ${compile_definitions}")
