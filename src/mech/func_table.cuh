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
#include <algorithm>
#include <cctype>

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
        if (a.type() == typeid(bool)) {
            return static_cast<T>(std::any_cast<bool>(a));
        }
        // 其他类型可以继续加

        throw std::runtime_error("bad any cast: 20 type=" + std::string(a.type().name()));
    }
};

template<>
struct convert_any_helper<std::string> {
    static std::string from(const std::any& a) {
        if (a.type() == typeid(std::string)) {
            return std::any_cast<std::string>(a);
        }
        if (a.type() == typeid(const char*)) {
            return std::string(std::any_cast<const char*>(a));
        }
        throw std::runtime_error("bad any cast: string type=" + std::string(a.type().name()));
    }
};

template<>
struct convert_any_helper<bool> {
    static bool from(const std::any& a) {
        if (a.type() == typeid(bool)) {
            return std::any_cast<bool>(a);
        }
        if (a.type() == typeid(int)) {
            return std::any_cast<int>(a) != 0;
        }
        throw std::runtime_error("bad any cast: bool type=" + std::string(a.type().name()));
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

class StaticInterfaceRegistry {
public:
    struct Property {
        std::function<std::any()> getter;
        std::function<void(const std::any&)> setter;
    };

    static StaticInterfaceRegistry& getInstance() {
        static StaticInterfaceRegistry instance;
        return instance;
    }

    template<typename GetterR, typename SetterArg>
    void register_property(const std::string& mech_name,
                           const std::string& property_name,
                           std::function<GetterR()> getter,
                           std::function<void(SetterArg)> setter) {
        Property prop;
        prop.getter = [getter = std::move(getter)]() -> std::any {
            return getter();
        };
        prop.setter = [setter = std::move(setter)](const std::any& value) {
            setter(convert_any<SetterArg>(value));
        };
        properties_[mech_name][property_name] = std::move(prop);
    }

    template<typename GetterR>
    void register_readonly_property(const std::string& mech_name,
                                    const std::string& property_name,
                                    std::function<GetterR()> getter) {
        Property prop;
        prop.getter = [getter = std::move(getter)]() -> std::any {
            return getter();
        };
        properties_[mech_name][property_name] = std::move(prop);
    }

    template<typename R, typename... Args>
    void register_method(const std::string& mech_name,
                         const std::string& method_name,
                         std::function<R(Args...)> func) {
        methods_[mech_name][method_name] = std::make_unique<ConcreteFunction<R, Args...>>(std::move(func));
    }

    bool has_interface(const std::string& mech_name) const {
        return properties_.find(mech_name) != properties_.end() || methods_.find(mech_name) != methods_.end();
    }

    std::string resolve_interface(const std::string& query) const {
        if (has_interface(query)) {
            return query;
        }
        const std::string lowered = to_lower(query);
        for (const auto& name : list_interfaces()) {
            if (to_lower(name) == lowered) {
                return name;
            }
        }
        return "";
    }

    bool has_property(const std::string& mech_name, const std::string& property_name) const {
        auto mech_it = properties_.find(mech_name);
        if (mech_it == properties_.end()) {
            return false;
        }
        return mech_it->second.find(property_name) != mech_it->second.end();
    }

    bool has_method(const std::string& mech_name, const std::string& method_name) const {
        auto mech_it = methods_.find(mech_name);
        if (mech_it == methods_.end()) {
            return false;
        }
        return mech_it->second.find(method_name) != mech_it->second.end();
    }

    std::any get_property(const std::string& mech_name, const std::string& property_name) const {
        auto mech_it = properties_.find(mech_name);
        if (mech_it == properties_.end()) {
            throw std::runtime_error("static interface not found: " + mech_name);
        }
        auto prop_it = mech_it->second.find(property_name);
        if (prop_it == mech_it->second.end()) {
            throw std::runtime_error("static property not found: " + mech_name + "." + property_name);
        }
        if (!prop_it->second.getter) {
            throw std::runtime_error("static property is write-only: " + mech_name + "." + property_name);
        }
        return prop_it->second.getter();
    }

    void set_property(const std::string& mech_name, const std::string& property_name, const std::any& value) const {
        auto mech_it = properties_.find(mech_name);
        if (mech_it == properties_.end()) {
            throw std::runtime_error("static interface not found: " + mech_name);
        }
        auto prop_it = mech_it->second.find(property_name);
        if (prop_it == mech_it->second.end()) {
            throw std::runtime_error("static property not found: " + mech_name + "." + property_name);
        }
        if (!prop_it->second.setter) {
            throw std::runtime_error("static property is read-only: " + mech_name + "." + property_name);
        }
        prop_it->second.setter(value);
    }

    std::any call_method(const std::string& mech_name, const std::string& method_name, const std::vector<std::any>& args) const {
        auto mech_it = methods_.find(mech_name);
        if (mech_it == methods_.end()) {
            throw std::runtime_error("static interface not found: " + mech_name);
        }
        auto method_it = mech_it->second.find(method_name);
        if (method_it == mech_it->second.end()) {
            throw std::runtime_error("static method not found: " + mech_name + "." + method_name);
        }
        return method_it->second->call(args);
    }

    std::vector<std::string> list_interfaces() const {
        std::vector<std::string> names;
        names.reserve(properties_.size() + methods_.size());
        for (const auto& [name, _] : properties_) {
            names.push_back(name);
        }
        for (const auto& [name, _] : methods_) {
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                names.push_back(name);
            }
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    std::vector<std::string> list_properties(const std::string& mech_name) const {
        std::vector<std::string> names;
        auto mech_it = properties_.find(mech_name);
        if (mech_it == properties_.end()) {
            return names;
        }
        for (const auto& [name, _] : mech_it->second) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    std::vector<std::string> list_methods(const std::string& mech_name) const {
        std::vector<std::string> names;
        auto mech_it = methods_.find(mech_name);
        if (mech_it == methods_.end()) {
            return names;
        }
        for (const auto& [name, _] : mech_it->second) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());
        return names;
    }

private:
    static std::string to_lower(const std::string& input) {
        std::string out = input;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    }

    std::unordered_map<std::string, std::unordered_map<std::string, Property>> properties_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<FunctionWrapper>>> methods_;
};
