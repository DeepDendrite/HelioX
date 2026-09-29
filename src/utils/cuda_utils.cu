#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace {
static bool should_use_pinned_host_alloc() {
    // AddressSanitizer reserves huge shadow memory mappings; CUDA pinned allocations
    // (cudaHostAlloc/cudaMallocHost) can fail or behave unexpectedly. Prefer plain
    // heap allocations under ASAN to keep debugging usable.
#if defined(__SANITIZE_ADDRESS__)
    return false;
#else
    return true;  // pinned host allocation (historical default)
#endif
}

static size_t round_up(size_t n, size_t align) {
    return (n + align - 1) / align * align;
}
}  // namespace

void gpu_mem_allocate(void** arr, size_t size)
{
    if (arr == nullptr) {
        return;
    }
    if (size == 0) {
        *arr = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL ERROR: cudaMalloc failed!\n");
        fprintf(stderr, "  Size: %zu bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemset(*arr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL ERROR: cudaMemset failed after cudaMalloc!\n");
        fprintf(stderr, "  Size: %zu bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        cudaFree(*arr);
        abort();
    }
}

void managed_mem_allocate(void** arr, int size)
{
    if (arr == nullptr) {
        return;
    }
    if (size <= 0) {
        *arr = nullptr;
        return;
    }
    cudaError_t err = cudaMallocManaged(arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL ERROR: cudaMallocManaged failed!\n");
        fprintf(stderr, "  Size: %d bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemset(*arr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL ERROR: cudaMemset failed after cudaMallocManaged!\n");
        fprintf(stderr, "  Size: %d bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        cudaFree(*arr);
        abort();
    }
}

void gpu_mem_free(void** arr)
{
    if (arr == nullptr || *arr == nullptr) {
        return;
    }
    cudaFree(*arr);
    *arr = nullptr;
}

void managed_mem_free(void** arr)
{
    if (arr == nullptr || *arr == nullptr) {
        return;
    }
    cudaFree(*arr);
    *arr = nullptr;
}

void cpu_mem_allocate(void** arr, int size)
{
    if (arr == nullptr) {
        return;
    }
    if (size <= 0) {
        *arr = nullptr;
        return;
    }
    if (!should_use_pinned_host_alloc()) {
        // Use regular heap allocation (aligned) for CPU-only / sanitizer builds.
        void* p = nullptr;
        constexpr size_t kAlign = 64;
        const size_t nbytes = round_up(static_cast<size_t>(size), kAlign);
        if (posix_memalign(&p, kAlign, nbytes) != 0 || p == nullptr) {
            fprintf(stderr, "FATAL ERROR: posix_memalign failed!\n");
            fprintf(stderr, "  Size: %d bytes\n", size);
            abort();
        }
        *arr = p;
        memset(*arr, 0, static_cast<size_t>(size));
        return;
    }

    cudaError_t err = cudaHostAlloc(arr, size, cudaHostAllocDefault);
    if (err != cudaSuccess || *arr == nullptr) {
        fprintf(stderr, "FATAL ERROR: cudaHostAlloc failed!\n");
        fprintf(stderr, "  Size: %d bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        abort();
    }
    memset(*arr, 0, static_cast<size_t>(size));
}

void cpu_mem_allocate_mapped(void** arr, int size)
{
    if (arr == nullptr) {
        return;
    }
    if (size <= 0) {
        *arr = nullptr;
        return;
    }
    if (!should_use_pinned_host_alloc()) {
        // Mapped pinned buffers don't make sense without pinned alloc; fall back.
        cpu_mem_allocate(arr, size);
        return;
    }

    cudaError_t err = cudaHostAlloc(arr, size, cudaHostAllocMapped);
    if (err != cudaSuccess || *arr == nullptr) {
        fprintf(stderr, "FATAL ERROR: cudaHostAlloc (mapped) failed!\n");
        fprintf(stderr, "  Size: %d bytes\n", size);
        fprintf(stderr, "  Error: %s\n", cudaGetErrorString(err));
        abort();
    }
    memset(*arr, 0, static_cast<size_t>(size));
}

void cpu_mem_free(void** arr)
{
    if (arr == nullptr || *arr == nullptr) {
        return;
    }
    if (!should_use_pinned_host_alloc()) {
        free(*arr);
        *arr = nullptr;
        return;
    }
    cudaFreeHost(*arr);
    *arr = nullptr;
}
void cuda_check_err_func(const char* file, int line){
    cuda_sync_all();
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) { 
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl; 
        assert(0); 
    } 
}

void mem_copy_cpu2gpu(void* dst,const void* src, int size, void* cuda_stream)
{
    if(cuda_stream){
        cudaStream_t *stream = reinterpret_cast<cudaStream_t*>(cuda_stream);
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,*stream);
    }else{
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,0);
    }
}

void mem_copy_cpu2gpu_sync(void* dst,const void* src, int size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void mem_copy_gpu2cpu(void* dst, void* src, int size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void mem_copy_gpu2gpu(void* dst, void* src, int size, void* cuda_stream)
{
    if(cuda_stream){
        cudaStream_t *stream = reinterpret_cast<cudaStream_t*>(cuda_stream);
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,*stream);
    }else{
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,0);
    }
}

extern "C" void cuda_stream_initialize(void** cuda_stream) {
    if (cuda_stream == nullptr) {
        return;
    }
    *cuda_stream = nullptr;
    cudaStream_t* stream = new cudaStream_t;
    cudaError_t err = cudaStreamCreate(stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        delete stream;
    } else {
        *cuda_stream = reinterpret_cast<void*>(stream);
    }
}

/**
 * @brief 同步 CUDA 流，确保流中所有任务完成。
 * @param cuda_stream 指向 cudaStream_t 的 void* 指针。
 */
extern "C" void cuda_stream_sync(void* cuda_stream) {
    if (__glibc_unlikely(cuda_stream == nullptr)) {
        std::cerr << "cuda_stream_sync: Invalid stream pointer!" << std::endl;
        return;
    }

    // 将 void* 转换为 cudaStream_t*
    cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(cuda_stream);

    cudaError_t err = cudaStreamSynchronize(*stream);
    if (err != cudaSuccess) {
        std::cerr << "cuda_stream_sync: Failed to synchronize CUDA stream: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

/**
 * @brief 销毁 CUDA 流，释放资源。
 * @param cuda_stream 指向 cudaStream_t 的 void* 指针。
 */
extern "C" void cuda_stream_destroy(void** cuda_stream) {
    if (__glibc_unlikely(cuda_stream == nullptr || *cuda_stream == nullptr)) {
        std::cerr << "cuda_stream_destroy: Invalid stream pointer!" << std::endl;
        return;
    }

    // 将 void* 转换为 cudaStream_t*
    cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(*cuda_stream);

    cudaError_t err = cudaStreamDestroy(*stream);
    if (err != cudaSuccess) {
        std::cerr << "cuda_stream_destroy: Failed to destroy CUDA stream: "
                  << cudaGetErrorString(err) << std::endl;
    } else {
        // std::cout << "cuda_stream_destroy: CUDA stream destroyed successfully." << std::endl;
    }

    // 释放指针并清空
    delete stream;
    *cuda_stream = nullptr;
}

extern "C" void cuda_event_initialize(void** cuda_event) {
    if (cuda_event == nullptr) {
        return;
    }
    *cuda_event = nullptr;
    cudaEvent_t* event = new cudaEvent_t;
    cudaError_t err = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA event: " << cudaGetErrorString(err) << std::endl;
        delete event;
    } else {
        *cuda_event = reinterpret_cast<void*>(event);
    }
}

extern "C" void cuda_event_destroy(void** cuda_event) {
    if (__glibc_unlikely(cuda_event == nullptr || *cuda_event == nullptr)) {
        std::cerr << "cuda_event_destroy: Invalid event pointer!" << std::endl;
        return;
    }

    cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(*cuda_event);
    cudaError_t err = cudaEventDestroy(*event);
    if (err != cudaSuccess) {
        std::cerr << "cuda_event_destroy: Failed to destroy CUDA event: "
                  << cudaGetErrorString(err) << std::endl;
    }

    delete event;
    *cuda_event = nullptr;
}

extern "C" void tag_event(const char* msg){
    nvtxRangePushA(msg);
}
extern "C" void tag_event_end(){
    nvtxRangePop();
}

void cuda_sync_all(){
    cudaDeviceSynchronize();
}

void mem_prefetch_to_gpu(void* ptr, int size, int device_id, void* stream)
{
    if (device_id < 0) {
        cudaGetDevice(&device_id);
    }
    cudaStream_t cuda_stream_handle = stream
        ? *reinterpret_cast<cudaStream_t*>(stream)
#if defined(CUDART_VERSION) && CUDART_VERSION < 12020
        : static_cast<cudaStream_t>(0);
#else
        : static_cast<cudaStream_t>(nullptr);
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 12020
    cudaMemPrefetchAsync(ptr, static_cast<size_t>(size), device_id, cuda_stream_handle);
#else
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device_id;
    unsigned int flags = 0;
    cudaMemPrefetchAsync(ptr, static_cast<size_t>(size), location, flags, cuda_stream_handle);
#endif
}

void mem_prefetch_to_cpu(void* ptr, int size, void* stream)
{
    cudaStream_t cuda_stream_handle = stream
        ? *reinterpret_cast<cudaStream_t*>(stream)
#if defined(CUDART_VERSION) && CUDART_VERSION < 12020
        : static_cast<cudaStream_t>(0);
#else
        : static_cast<cudaStream_t>(nullptr);
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 12020
    cudaMemPrefetchAsync(ptr, static_cast<size_t>(size), cudaCpuDeviceId, cuda_stream_handle);
#else
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeHost;
    location.id = 0;
    unsigned int flags = 0;
    cudaMemPrefetchAsync(ptr, static_cast<size_t>(size), location, flags, cuda_stream_handle);
#endif
}
