#pragma once

#if defined(_WIN32)
  #if defined(NEURONG_SDK_BUILDING_PLUGIN)
    #define NEURONG_SDK_EXPORT __declspec(dllexport)
  #else
    #define NEURONG_SDK_EXPORT __declspec(dllimport)
  #endif
#else
  #define NEURONG_SDK_EXPORT __attribute__((visibility("default")))
#endif

#define NEURONG_SDK_EXTERN_C extern "C"
