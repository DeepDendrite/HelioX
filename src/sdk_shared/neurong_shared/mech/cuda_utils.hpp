#pragma once

#include <cstddef>

#ifdef DEBUG
#define CUDA_CHECK_ERR() cuda_check_err_func(__FILE__, __LINE__)
#else
#define CUDA_CHECK_ERR() (void)0
#endif

extern "C" {
void gpu_mem_allocate(void** arr, size_t size);
void gpu_mem_free(void** arr);
void managed_mem_allocate(void** arr, int size);
void managed_mem_free(void** arr);
void mem_copy_gpu2gpu(void* dst, void* src, int size, void* cuda_stream);
void mem_copy_cpu2gpu(void* dst, const void* src, int size, void* cuda_stream = nullptr);
void mem_copy_cpu2gpu_sync(void* dst, const void* src, int size);
void cpu_mem_allocate(void** arr, int size);
void cpu_mem_allocate_mapped(void** arr, int size);
void cpu_mem_free(void** arr);
void mem_copy_gpu2cpu(void* dst, void* src, int size);
void mem_prefetch_to_gpu(void* ptr, int size, int device_id = -1, void* stream = nullptr);
void mem_prefetch_to_cpu(void* ptr, int size, void* stream = nullptr);
void tag_event(const char* msg);
void tag_event_end();
void cuda_sync_all();
void cuda_check_err_func(const char* file, int line);
void cuda_stream_initialize(void** cuda_stream);
void cuda_stream_sync(void* cuda_stream);
void cuda_stream_destroy(void** cuda_stream);
void cuda_event_initialize(void** cuda_event);
void cuda_event_destroy(void** cuda_event);
}
