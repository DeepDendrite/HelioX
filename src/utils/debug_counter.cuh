#pragma once
#include <iostream>
#include <string>

// 简化版的计数器模板函数
// 仅返回是否应该打印，不包含任何打印逻辑
template <int ID, int MaxPrintSteps = 10>
__host__ __device__ bool shouldPrint() {
    static int counter = 0;
    return ++counter <= MaxPrintSteps;
}

// 使用类型作为唯一标识符的版本
template <typename T, int MaxPrintSteps = 10>
__host__ __device__ bool typedShouldPrint() {
    static int counter = 0;
    return ++counter <= MaxPrintSteps;
}