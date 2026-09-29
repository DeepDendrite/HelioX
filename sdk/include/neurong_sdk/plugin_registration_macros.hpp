#pragma once

#include <vector>

#include "neurong_sdk/plugin_registration.hpp"

#if defined(NEURONG_DISABLE_STATIC_MECH_REGISTRATION)

#ifndef NEURONG_CONCAT_INNER
#define NEURONG_CONCAT_INNER(a, b) a##b
#endif
#ifndef NEURONG_CONCAT
#define NEURONG_CONCAT(a, b) NEURONG_CONCAT_INNER(a, b)
#endif

#ifndef REGISTER_MECHANISM
#define REGISTER_MECHANISM(NAME, CLASS) \
    static const bool NEURONG_CONCAT(CLASS##_plugin_registered_, __LINE__) = []() { \
        (void)::neurong::sdk::plugin_local_register_mechanism( \
            NAME, \
            [](MechInitParams& param) -> Mechanism* { return new CLASS(param); }, \
            __FILE__, \
            #CLASS); \
        return true; \
    }()
#endif

#ifndef REGISTER_POSTSYN
#define REGISTER_POSTSYN(NAME, CLASS, PNT_RECEIVE_SIZE) \
    static const bool NEURONG_CONCAT(CLASS##_plugin_postsyn_registered_, __LINE__) = []() { \
        (void)::neurong::sdk::plugin_local_register_postsyn( \
            NAME, \
            [](MechInitParams& param) -> Mechanism* { return new CLASS(param); }, \
            __FILE__, \
            #CLASS, \
            PNT_RECEIVE_SIZE); \
        return true; \
    }()
#endif

#ifndef REGISTER_POINTER_DPARAM_SLOTS
#define REGISTER_POINTER_DPARAM_SLOTS(NAME, ...) \
    static const bool NEURONG_CONCAT(_pointer_slots_registered_, __LINE__) = []() { \
        (void)::neurong::sdk::plugin_local_register_pointer_dparam_slots( \
            NAME, std::vector<int>{__VA_ARGS__}); \
        return true; \
    }()
#endif

#ifndef REGISTER_DPARAM_SEMANTICS
#define REGISTER_DPARAM_SEMANTICS(NAME, ...) \
    static const bool NEURONG_CONCAT(_dparam_semantics_registered_, __LINE__) = []() { \
        (void)::neurong::sdk::plugin_local_register_dparam_semantics( \
            NAME, std::vector<int>{__VA_ARGS__}); \
        return true; \
    }()
#endif

#endif
