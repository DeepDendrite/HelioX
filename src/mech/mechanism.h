#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "neurong_shared/mech/mechanism.hpp"
#include "neurong_sdk/plugin_registration_macros.hpp"
#include "dparam_semantics.h"
#include "utils.h"

class MechanismFactory {
public:
    using Creator = std::function<Mechanism*(MechInitParams&)>;
    struct RegistrationInfo {
        std::string source_file;
        std::string source_symbol;
    };
    struct PendingRegistration {
        std::string name;
        Creator creator;
        std::string source_file;
        std::string source_symbol;
        bool has_pnt_receive_size = false;
        int pnt_receive_size = -1;
        std::vector<int> pointer_dparam_slots;
        std::vector<int> dparam_semantics;
    };

    static MechanismFactory& getInstance() {
        static MechanismFactory instance;
        return instance;
    }
    void registerMechanism(const std::string& name, Creator creator, const char* source_file, const char* source_symbol);
    int registerMechanismBatch(const std::vector<PendingRegistration>& registrations, std::string* error_message);
    void registerVarMap(const std::string& name, VarMapAble* varMap);
    void registerPntReceiveSize(const std::string& name, int pnt_receive_size);
    // Register the dparam (pdata) slot indices that correspond to POINTER variables for this mechanism.
    // Slots must be in the same order as the mechanism's POINTER variables (trait enum order).
    void registerPointerDparamSlots(const std::string& name, std::vector<int> slots);
    // Optional dparam semantics (CoreNEURON-style).
    // Length must match bbcore dparam size for the mechanism type. Encoding is compatible with NEURON:
    // See `DparamSemantics` in `src/utils/dparam_semantics.h`.
    void registerDparamSemantics(const std::string& name, std::vector<int> semantics);
    const RegistrationInfo* getRegistrationInfo(const std::string& name) const;
    Mechanism* createMechanism(const std::string& name, MechInitParams& initParam);
    VarMapAble* getVarMap(const std::string& name);
    int getPntReceiveSize(const std::string& name);
    const std::vector<int>* getPointerDparamSlots(const std::string& name) const;
    const std::vector<int>* getDparamSemantics(const std::string& name) const;

private:
    std::unordered_map<std::string, Creator> registry_;
    std::unordered_map<std::string, RegistrationInfo> registry_info_;
    std::vector<Mechanism*> allocatedMechanisms_;
    std::unordered_map<int, std::string> type2name_;
    std::unordered_map<std::string, int> name2type_;
    std::unordered_map<std::string, VarMapAble*> name2varMap_;
    std::unordered_map<std::string, int> name2pntReceiveSize_;
    std::unordered_map<std::string, std::vector<int>> name2pointerDparamSlots_;
    std::unordered_map<std::string, std::vector<int>> name2dparamSemantics_;
    MechanismFactory() = default;
    ~MechanismFactory() = default;
    MechanismFactory(const MechanismFactory&) = delete;
    MechanismFactory& operator=(const MechanismFactory&) = delete;
};

#if !defined(REGISTER_MECHANISM)
#ifndef NEURONG_DISABLE_STATIC_MECH_REGISTRATION
// 注册宏
#define REGISTER_MECHANISM(NAME, CLASS) \
    inline bool CLASS##_registered = []() { \
        MechanismFactory::getInstance().registerMechanism(NAME, [](MechInitParams &param) { return new CLASS(param); }, __FILE__, #CLASS); \
        return true; }();

// 新的后突触注册宏
#define REGISTER_POSTSYN(NAME, CLASS, PNT_RECEIVE_SIZE) \
    inline bool CLASS##_postsyn_registered = []() { \
        MechanismFactory::getInstance().registerMechanism(NAME, [](MechInitParams &param) { return new CLASS(param); }, __FILE__, #CLASS); \
        MechanismFactory::getInstance().registerPntReceiveSize(NAME, PNT_RECEIVE_SIZE); \
        return true; }();
#endif
#endif

// Macro helpers for unique variable names in registration.
#ifndef NEURONG_CONCAT_INNER
#define NEURONG_CONCAT_INNER(a, b) a##b
#endif
#ifndef NEURONG_CONCAT
#define NEURONG_CONCAT(a, b) NEURONG_CONCAT_INNER(a, b)
#endif

// Register POINTER dparam slot indices for a mechanism.
// Example: for a POINT_PROCESS with `POINTER x, y` that ends up in `_ppvar[2]` and `_ppvar[3]`:
//   REGISTER_POINTER_DPARAM_SLOTS("mechname", 2, 3);
#if !defined(REGISTER_POINTER_DPARAM_SLOTS)
#define REGISTER_POINTER_DPARAM_SLOTS(NAME, ...) \
    inline bool NEURONG_CONCAT(_pointer_slots_registered_, __COUNTER__) = []() { \
        MechanismFactory::getInstance().registerPointerDparamSlots(NAME, std::vector<int>{__VA_ARGS__}); \
        return true; }();
#endif

// Register optional dparam semantics for a mechanism.
// Example:
//   REGISTER_DPARAM_SEMANTICS("mechname", -1, -6, -5);
#if !defined(REGISTER_DPARAM_SEMANTICS)
#define REGISTER_DPARAM_SEMANTICS(NAME, ...) \
    inline bool NEURONG_CONCAT(_dparam_semantics_registered_, __COUNTER__) = []() { \
        MechanismFactory::getInstance().registerDparamSemantics(NAME, std::vector<int>{__VA_ARGS__}); \
        return true; }();
#endif

//
