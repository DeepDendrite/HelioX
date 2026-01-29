#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <any>
#include <vector>
#include <string>
#include <stdexcept>

namespace nb = nanobind;

// 你允许的 C++ 目标类型列表，可随时扩充
using AllowedTypes = std::variant<
    bool,
    long long,         // Python int
    double,            // Python float
    std::string,       // Python str
    std::vector<std::any> // 递归解析 list
>;

// ---------- helpers ----------
inline std::any to_any(const nb::handle& h);          // 前置声明

inline std::vector<std::any> list_to_any_vector(const nb::handle& h) {
    std::vector<std::any> out;
    nb::sequence seq = nb::borrow<nb::sequence>(h);
    out.reserve(nb::len(seq));
    for (size_t i = 0; i < nb::len(seq); ++i) {
        out.push_back(to_any(seq[i]));
    }
    return out;
}

// ---------- 主函数 ----------
inline std::any to_any(const nb::handle& h) {
    // None -> 空 std::any
    if (h.is_none()){
        return std::any{};
    }

    // 检查 bool (需要在 int 之前，因为 bool 也是 int 的子类)
    if (nb::isinstance<nb::bool_>(h)){
        return std::any(nb::cast<bool>(h));
    }

    // 检查 int
    if (nb::isinstance<nb::int_>(h)){
        return std::any(nb::cast<int>(h));
    }

    // 检查 float
    if (nb::isinstance<nb::float_>(h)){
        return std::any(nb::cast<double>(h));
    }

    // 检查 str
    if (nb::isinstance<nb::str>(h)){
        return std::any(nb::cast<std::string>(h));
    }

    // 检查 list/tuple (但不是 str)
    if (nb::isinstance<nb::sequence>(h) && !nb::isinstance<nb::str>(h)){
        auto v = list_to_any_vector(h);
        return std::any(v);
    }

    // 兜底：保留 nb::object，晚点再说
    return std::any(nb::borrow<nb::object>(h));
}

inline std::vector<std::any> args_to_vector(Mode mode, const nb::args& args) {
    std::vector<std::any> v;
    size_t args_len = args.size();  // nb::args 有 size() 方法
    v.reserve(args_len + 1);
    v.emplace_back(std::any(mode));
    for (size_t i = 0; i < args_len; ++i) {
        v.emplace_back(to_any(args[i]));
    }
    return v;
}
