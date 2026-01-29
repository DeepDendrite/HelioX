#pragma once

/*
    这个头文件提供了一个用于调试的变量类 DebugVar。
    在 DEBUG 模式下，DebugVar 会正常工作，允许设置和获取值。
    否则，它会被编译为一个空壳，所有的操作都不会有任何效果，但是也不会报错。
    这样可以在调试时使用 DebugVar 来跟踪变量的值，而在发布版本中则不会有任何开销。
*/

#ifdef DEBUG
constexpr bool DebugEnabledDefault = true;
#else
constexpr bool DebugEnabledDefault = false;
#endif

// Debug 默认值设定器
template<typename T>
struct DebugDefaultVal {
    static constexpr T value = T{};
};

template<>
struct DebugDefaultVal<int> {
    static constexpr int value = -1;
};

template<>
struct DebugDefaultVal<double> {
    static constexpr double value = -1.0;
};

template<>
struct DebugDefaultVal<bool> {
    static constexpr bool value = false;
};

// DebugVar 主模板
template<
    typename T,
    T DefaultVal = DebugDefaultVal<T>::value,
    bool Enabled = DebugEnabledDefault
>
class DebugVar;

// DEBUG 模式：正常变量
template<typename T, T DefaultVal>
class DebugVar<T, DefaultVal, true> {
    T value;
public:
    __host__ __device__ DebugVar() {}
    __host__ __device__ DebugVar(T val) : value(val) {}

    __host__ __device__ T& operator()() { return value; }
    __host__ __device__ const T& operator()() const { return value; }

    __host__ __device__ DebugVar& operator=(T val) { value = val; return *this; }
    __host__ __device__ operator T() const { return value; }
};

// RELEASE 模式：空壳
template<typename T, T DefaultVal>
class DebugVar<T, DefaultVal, false> {
public:
    __host__ __device__ DebugVar() {}
    __host__ __device__ DebugVar(T) {}

    __host__ __device__ DebugVar& operator=(T) { return *this; }

    __host__ __device__ DebugVar& operator()() { return *this; }
    __host__ __device__ const DebugVar& operator()() const { return *this; }

    __host__ __device__ operator T() const { return DefaultVal; }
};
