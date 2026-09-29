#pragma once
#include <string>
#include <type_traits>
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
    WRITE_EION_IN_STATE = 1 << 5,
    ENABLE_CURRENT_VJP = 1 << 6
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

// Unified conversion from matrix-domain grad_rhs to mechanism-current adjoint.
// NOTE:
// - point-process uses area scaling (100/area)
// - electrode current uses positive RHS sign; non-electrode current uses negative RHS sign
DUAL_EXEC double mech_grad_current_from_rhs(
    double grad_rhs,
    bool point_process,
    bool electrode_current,
    double node_area)
{
    double grad_current = grad_rhs;
    if (point_process) {
        grad_current *= 1.0e2 / node_area;
    }
    return electrode_current ? grad_current : -grad_current;
}


template<typename T>
concept MechTraitType = requires {
    typename T::VarNames;
}&&std::is_enum_v<typename T::VarNames>;

template <typename T, typename = void>
struct mech_trait_supports_table : std::false_type
{
};

template <typename T>
struct mech_trait_supports_table<T, std::void_t<decltype(T::MechSupportsTable)>>
    : std::bool_constant<static_cast<bool>(T::MechSupportsTable)>
{
};

template <typename T>
inline constexpr bool mech_trait_supports_table_v = mech_trait_supports_table<T>::value;

struct TableViewEnabled
{
    const double* data = nullptr;
    int point_count = 0;
    int output_count = 0;
    double tmin = 0.0;
    double mfac = 0.0;
    bool enabled = false;
    const int* enabled_ptr = nullptr;
    const double* tmin_ptr = nullptr;
    const double* mfac_ptr = nullptr;
};

struct TableViewDisabled
{
};

template <typename MechTrait>
using VarAccessorTableView =
    std::conditional_t<mech_trait_supports_table_v<MechTrait>, TableViewEnabled, TableViewDisabled>;
