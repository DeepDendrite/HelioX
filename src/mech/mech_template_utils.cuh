#pragma once
#include <string>
#include "utils.h"
#include "magic_enum/magic_enum.hpp"
#include "mechanism.h"
#include "ion_table.h"

#define DUAL_EXEC static __device__ __host__ __forceinline__

using namespace magic_enum::bitwise_operators;


//定义一下可选枚举类，以及对应的信息map
DEFINE_HAS_ENUM(GlobalVarNames);
DEFINE_ENUM_MAP_ALIAS(GlobalVarNames,CoreGlobalVarInfo,GlobalVarInfoMap);

using IonVarMapInfo = std::tuple<std::string, EionVarNames>;
DEFINE_HAS_ENUM(IonVarNames);
DEFINE_ENUM_MAP_ALIAS(IonVarNames,IonVarMapInfo, IonVarInfoMap);

// Optional POINTER variables (per-instance pointers) support.
DEFINE_HAS_ENUM(PointerVarNames);



enum class MechFlags{
    NONE = 0,
    ENABLE_INIT = 1 << 0,
    ENABLE_CURRENT = 1 << 1,
    ENABLE_STATE = 1 << 2,
    POINT_PROCESS = 1 << 3,
    ELECTRODE_CURRENT = 1 << 4,
    WRITE_EION_IN_STATE = 1 << 5
};
template <typename T>
concept EnumType = std::is_enum_v<T>;

template <>
struct magic_enum::customize::enum_range<MechFlags> {
  static constexpr bool is_flags = true;
};

__host__ __device__ bool constexpr hasFlag(MechFlags flags, MechFlags flag){
    return static_cast<int>(flags) & static_cast<int>(flag);
}

__host__ __device__ void __forceinline__  mechAtomAdd(double* address, double val){
    #if defined(__CUDA_ARCH__)
    atomicAdd(address,val);
    #else
    *address += val;
    #endif
}


template<typename T>
concept MechTraitType = requires {
    typename T::VarNames;
}&&std::is_enum_v<typename T::VarNames>;
