#pragma once

#include "neurong_shared/mech/types.hpp"

#include <string>
#include <vector>
#include <memory>
#include <highfive/highfive.hpp>
#include <magic_enum/magic_enum.hpp>
#include <cassert>

#include <map>
#include <unordered_map>
#include <type_traits>
#include <variant> // for std::monostate
#include <tuple>   // for std::tie

const int nthread_per_block = 128;
bool isVersionGreater(const std::string& version, const std::string& threshold);

void unique_sort(std::vector<int>& vec);

struct hdf5_info {
    std::unique_ptr<HighFive::File> hdf5_file;
    std::vector<HighFive::DataSet> voltages;
};

bool eof(FILE* fp);



#ifdef DEBUG_PRINTF
#define printf_debug(...) printf(__VA_ARGS__)
#else
#define printf_debug(...) (void)0
#endif

#ifdef DEBUG
#define DEBUG_ASSERT(condition) assert(condition);
#else
#define DEBUG_ASSERT(condition, ...) (void)0;
#endif



// ----------------------------------------------------------------------------
// 1. 生成某个嵌套枚举是否存在的检查工具
// 用法：DEFINE_HAS_ENUM(EnumName)
// 会生成：has_EnumName<T>, has_EnumName_v<T>
// ----------------------------------------------------------------------------
#define DEFINE_HAS_ENUM(EnumName)                                               \
template <typename T, typename = void>                                          \
struct has_##EnumName : std::false_type {};                                     \
                                                                                \
template <typename T>                                                           \
struct has_##EnumName<T, std::void_t<typename T::EnumName>> : std::true_type {};\
                                                                                \
template <typename T>                                                           \
inline constexpr bool has_##EnumName##_v = has_##EnumName<T>::value;

// ----------------------------------------------------------------------------
// 2. 生成 enum 对应 map 类型别名（自定义别名名）
// 用法：DEFINE_ENUM_MAP_ALIAS(EnumName, ValueType, AliasName)
// 例：DEFINE_ENUM_MAP_ALIAS(GlobalVarNames, CoreGlobalVarInfo, GlobalVarInfoMap)
//     生成 GlobalVarInfoMap<Trait>，如果 Enum 存在是 map，否则是 monostate
// ----------------------------------------------------------------------------
#define DEFINE_ENUM_MAP_ALIAS(EnumName, ValueType, AliasName)                  \
template <typename Trait, bool = has_##EnumName##_v<Trait>>                    \
struct AliasName##_impl {                                                      \
    using type = std::map<typename Trait::EnumName, ValueType>;                \
};                                                                             \
                                                                                \
template <typename Trait>                                                      \
struct AliasName##_impl<Trait, false> {                                        \
    using type = std::monostate;                                               \
};                                                                             \
                                                                                \
template <typename Trait>                                                      \
using AliasName = typename AliasName##_impl<Trait>::type;
