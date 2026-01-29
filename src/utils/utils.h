#pragma once
#include <string>
#include <vector>
#include <memory>
#include <highfive/highfive.hpp>
#include <magic_enum/magic_enum.hpp>
#include <cassert>

#include <map>
#include <type_traits>
#include <variant> // for std::monostate
#include <tuple>   // for std::tie

enum Mode {CPU, GPU};
const int nthread_per_block = 128;

// ============================================================================
// VarDescriptor - 通用的变量描述符结构体
// 用于唯一标识系统中的任何变量（mechanism变量、全局变量、数组元素等）
// ============================================================================
struct VarDescriptor
{
    std::string mech;  // 机制名称（空字符串或"global"表示全局变量）
    std::string var;   // 变量名称
    union {
        int idx;           // 新名称：索引
        int node_or_mech_idx;  // 旧名称：保持兼容性
    };
    int array_index = -1;   // 数组索引（-1表示标量，>=0表示数组元素索引）

    // 构造函数
    VarDescriptor() : idx(0), array_index(-1) {}
    VarDescriptor(const std::string& m, const std::string& v, int i)
        : mech(m), var(v), idx(i), array_index(-1) {}
    VarDescriptor(const std::string& m, const std::string& v, int i, int arr_idx)
        : mech(m), var(v), idx(i), array_index(arr_idx) {}

    // 比较运算符（用于map/set）
    bool operator < (const VarDescriptor &other) const
    {
        return std::tie(mech, var, idx, array_index) < std::tie(other.mech, other.var, other.idx, other.array_index);
    }

    // 相等运算符
    bool operator == (const VarDescriptor &other) const
    {
        return mech == other.mech && var == other.var && idx == other.idx && array_index == other.array_index;
    }

    // 便于调试的字符串表示
    std::string to_string() const {
        return mech + "." + var + "[" + std::to_string(idx) + "][" + std::to_string(array_index) + "]";
    }
};

// 为VarDescriptor添加hash函数支持，用于unordered_map
namespace std {
    template<>
    struct hash<VarDescriptor> {
        size_t operator()(const VarDescriptor& vd) const {
            return std::hash<std::string>()(vd.mech) ^
                   (std::hash<std::string>()(vd.var) << 1) ^
                   (std::hash<int>()(vd.idx) << 2) ^
                   (std::hash<int>()(vd.array_index) << 3);
        }
    };
}
// ============================================================================

template<typename T>
struct CoreVarInfo{
    T info;
    int array_size;//普通变量是标量，因此array_size为1，如果是数组变量，array_size为数组大小
    CoreVarInfo(T info){
        this->info = info;
        this->array_size = 1;
    }
    CoreVarInfo(T info,int array_size){
        this->info = info;
        this->array_size = array_size;
    }
    bool isArray(){
        return array_size > 1;
    }
};
using CoreIdxInfo = CoreVarInfo<int>;
using CoreGlobalVarInfo = CoreVarInfo<std::string>;


struct MechInitParams
{
    Mode mode; // cpu?gpu?
    int type;  // 唯一的数字编号
    std::string name;
    int node_count;   // 这个mech有几个node
    int *nodeindices; // 对应的node的index

    int data_size; // AKA: szp,param_size,是data的元素个数
    double *data;  // 每一个mech内部向量（double类型）的数据

    int pdata_size; // AKA: szdp   node_count * pdata_size 可得到pdata的元素个数
    int *pdata;     // 一些用于索引的下表，参数等其他的数据

    int *permute;
    std::vector<int> *array_dims;//实际上叫lens会更好，但是为了和coreneuron保持一致，所以叫array_dims

    // v1.8+ bbcore: POINTER 元信息（每个 POINTER 实例的 target type）
    // - nullptr 表示没有 POINTER 或者导出未提供该信息
    // - 编码来自 NEURON/CoreNEURON：
    //   -1: voltage, -2: i_membrane_, >0: mechanism type id
    // - 顺序与 NEURON 导出保持一致：按 instance-major 展开，并且只包含 POINTER slots
    //   （即：for inst in nodecount { for each POINTER slot { push_back(type) } }）
    const std::vector<int>* pointer2type = nullptr;

    // Optional CoreNEURON-style dparam semantics metadata for this mechanism type.
    // - nullptr 表示该机制未注册该元信息（保持后向兼容）
    // - 若非空，长度应等于 pdata_size (nrn_prop_dparam_size[type])
    // - 编码与 NEURON/CoreNEURON 保持一致（见 coreneuron/mechanism/register_mech.cpp）：
    //   -1: area, -2: iontype, -3: cvodeieq, -4: netsend, -5: pointer, -6: pntproc,
    //   -7: bbcorepointer, -8: watch, -9: diam, -10: fornetcon, -11: random,
    //   >=0: ion semantics encoding (nrn_semantics_from_ion)
    const std::vector<int>* dparam_semantics = nullptr;
};


template <typename T>
void try_delete_arr(T*& ptr) {
    if (ptr) { // 如果指针有效
        delete [] ptr; // 释放指针指向的对象
        ptr = nullptr; // 将指针设为nullptr
    }
}

template <typename T>
void try_delete(T*& ptr) {
    if (ptr) { // 如果指针有效
        delete ptr; // 释放指针指向的对象
        ptr = nullptr; // 将指针设为nullptr
    }
}
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
