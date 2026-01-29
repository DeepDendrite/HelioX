#pragma once

#include <iostream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <any>
#include <stdexcept>
#include <tuple>
#include <memory>

// 主模板：对普通类型
template<typename T>
struct convert_any_helper {
    static T from(const std::any& a) {
        // 如果 a 里本来就是 T 类型，直接取
        if (a.type() == typeid(T)) {
            return std::any_cast<T>(a);
        }
        // 常见类型：int, double, float, long, etc.
        // 你可以根据需要扩展
        if (a.type() == typeid(int)) {
            return static_cast<T>(std::any_cast<int>(a));
        }
        if (a.type() == typeid(double)) {
            return static_cast<T>(std::any_cast<double>(a));
        }
        if (a.type() == typeid(float)) {
            return static_cast<T>(std::any_cast<float>(a));
        }
        if (a.type() == typeid(long)) {
            return static_cast<T>(std::any_cast<long>(a));
        }
        // 其他类型可以继续加

        throw std::runtime_error("bad any cast: 20 type=" + std::string(a.type().name()));
    }
};

// 偏特化：对 vector<U>
template<typename U>
struct convert_any_helper<std::vector<U>> {
    static std::vector<U> from(const std::any& a) {
        try {
            const auto& vec_any = std::any_cast<const std::vector<std::any>&>(a);
            std::vector<U> res;
            res.reserve(vec_any.size());
            for (const auto& item : vec_any) {
                res.push_back(convert_any_helper<U>::from(item));
            }
            return res;
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("bad any cast: 38" + std::string(e.what()));
        }
    }
};

// 包装函数
template<typename T>
T convert_any(const std::any& a) {
    return convert_any_helper<T>::from(a);
}

class FunctionWrapper {
public:
    virtual ~FunctionWrapper() = default;
    virtual std::any call(const std::vector<std::any>& args) = 0;
};
template<typename R, typename... Args>
class ConcreteFunction : public FunctionWrapper {
public:
    using FuncType = std::function<R(Args...)>;

    ConcreteFunction(FuncType f) : func(std::move(f)) {}
    
    std::any call(const std::vector<std::any>& args) override {
        if (args.size() != sizeof...(Args))
            throw std::runtime_error("参数数量不匹配");

        return callImpl(args, std::index_sequence_for<Args...>{});
    }

private:
    FuncType func;

    template<std::size_t... I>
    std::any callImpl(const std::vector<std::any>& args, std::index_sequence<I...>) {
        if constexpr (std::is_void<R>::value) {
            func(convert_any<Args>(args[I])...);
            return 0;
        } else {
            return func(convert_any<Args>(args[I])...);
        }
    }
};
class FunctionRegistry {
public:
    FunctionRegistry() {}

    static FunctionRegistry& getInstance() {
        static FunctionRegistry instance;
        return instance;
    }
    
    template<typename R, typename... Args>
    void register_func(const std::string& func_name, std::function<R(Args...)> f) {
        funcs[func_name] = std::make_unique<ConcreteFunction<R, Args...>>(std::move(f));
    }
    template<typename R, typename... Args>
    void register_func(const std::string& mech_name, const std::string& func_name, std::function<R(Args...)> f) {
        funcs[mech_name + "." + func_name] = std::make_unique<ConcreteFunction<R, Args...>>(std::move(f));
    }
    std::any call(const std::string& func_name, const std::vector<std::any>& args) {
        auto it = funcs.find(func_name);
        if (it == funcs.end())
            throw std::runtime_error("函数未注册: " + func_name);
        return it->second->call(args);
    }
    std::any call(const std::string& mech_name, const std::string& func_name, const std::vector<std::any>& args) {
        auto it = funcs.find(mech_name + "." + func_name);
        if (it == funcs.end())
            throw std::runtime_error("函数未注册: " + mech_name + "." + func_name);
        return it->second->call(args);
    }

private:
    std::unordered_map<std::string, std::unique_ptr<FunctionWrapper>> funcs;
};