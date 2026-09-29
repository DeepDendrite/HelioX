#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "neurong_shared/mech/cuda_utils.hpp"
#include "neurong_shared/mech/types.hpp"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

template <typename Dtype>
class VecData
{
public:
    using value_type = Dtype;
    using size_type = int;

private:
    size_type len_ = 0;
    size_type cap_ = 0;
    bool cpu_dirty_ = false;
    Mode mode_ = CPU;

    Dtype* data_cpu_ = nullptr;
    Dtype* data_gpu_ = nullptr;

    bool using_managed_memory() const noexcept
    {
#ifdef NEURON_USE_MANAGED_MEMORY
        return mode_ == GPU;
#else
        return false;
#endif
    }

    void destroy_elements() noexcept
    {
        if constexpr (!std::is_trivially_destructible_v<Dtype>)
        {
            for (size_type i = 0; i < len_; ++i)
            {
                data_cpu_[i].~Dtype();
            }
        }
    }

    void allocate_cpu_data()
    {
        if (cap_ == 0)
        {
            return;
        }
        const size_t bytes = sizeof(Dtype) * cap_;

        const bool use_managed = using_managed_memory();
        if (use_managed)
        {
            managed_mem_allocate(reinterpret_cast<void**>(&data_cpu_), static_cast<int>(bytes));
            if (!data_cpu_)
            {
                throw std::bad_alloc{};
            }
            data_gpu_ = data_cpu_;
        }
        else
        {
            cpu_mem_allocate(reinterpret_cast<void**>(&data_cpu_), static_cast<int>(bytes));
            if (!data_cpu_)
            {
                throw std::bad_alloc{};
            }
        }
    }

    void allocate_gpu_data()
    {
        if (mode_ != GPU || cap_ == 0)
        {
            return;
        }

        if (using_managed_memory())
        {
            return;
        }

        const size_t bytes = sizeof(Dtype) * cap_;
        gpu_mem_allocate(reinterpret_cast<void**>(&data_gpu_), static_cast<int>(bytes));
        if (!data_gpu_)
        {
            throw std::bad_alloc{};
        }
    }

    void free_cpu_data()
    {
        if (!data_cpu_)
        {
            return;
        }

        if (using_managed_memory())
        {
            managed_mem_free(reinterpret_cast<void**>(&data_cpu_));
            data_cpu_ = nullptr;
            data_gpu_ = nullptr;
            return;
        }

        cpu_mem_free(reinterpret_cast<void**>(&data_cpu_));
        data_cpu_ = nullptr;
    }

    void free_gpu_data()
    {
        if (!data_gpu_)
        {
            return;
        }

        if (using_managed_memory())
        {
            data_gpu_ = nullptr;
            return;
        }

        gpu_mem_free(reinterpret_cast<void**>(&data_gpu_));
        data_gpu_ = nullptr;
    }

public:
    explicit VecData(Mode m = CPU) noexcept : mode_(m) {}

    VecData(Mode m, size_type n) : len_(n), cap_(n), mode_(m)
    {
        allocate_cpu_data();
        if (mode_ == GPU)
        {
            allocate_gpu_data();
        }
    }

    VecData(Mode m, const Dtype* arr, size_type n) : VecData(m, n)
    {
        if constexpr (std::is_trivially_copyable_v<Dtype>)
        {
            std::memcpy(data_cpu_, arr, sizeof(Dtype) * n);
        }
        else
        {
            for (size_type i = 0; i < n; ++i)
            {
                ::new (data_cpu_ + i) Dtype(arr[i]);
            }
        }

        if (mode_ == GPU)
        {
            update_gpu_data_from_cpu();
        }
    }

    VecData(Mode m, const std::vector<Dtype>& v) : VecData(m, v.data(), static_cast<size_type>(v.size())) {}

    VecData(Mode m, const Dtype& value, size_type n) : VecData(m, n)
    {
        if constexpr (std::is_trivially_copyable_v<Dtype>)
        {
            std::fill_n(data_cpu_, n, value);
        }
        else
        {
            for (size_type i = 0; i < n; ++i)
            {
                ::new (data_cpu_ + i) Dtype(value);
            }
        }

        if (mode_ == GPU)
        {
            update_gpu_data_from_cpu();
        }
    }

    VecData(const VecData&) = delete;
    VecData& operator=(const VecData&) = delete;

    VecData(VecData&& other) noexcept
        : len_(other.len_), cap_(other.cap_), cpu_dirty_(other.cpu_dirty_), mode_(other.mode_),
          data_cpu_(other.data_cpu_), data_gpu_(other.data_gpu_)
    {
        other.len_ = other.cap_ = 0;
        other.data_cpu_ = other.data_gpu_ = nullptr;
        other.cpu_dirty_ = false;
    }

    VecData& operator=(VecData&& other) noexcept
    {
        if (this != &other)
        {
            destroy_elements();
            free_cpu_data();
            free_gpu_data();

            len_ = other.len_;
            cap_ = other.cap_;
            cpu_dirty_ = other.cpu_dirty_;
            mode_ = other.mode_;
            data_cpu_ = other.data_cpu_;
            data_gpu_ = other.data_gpu_;

            other.len_ = other.cap_ = 0;
            other.data_cpu_ = other.data_gpu_ = nullptr;
            other.cpu_dirty_ = false;
        }
        return *this;
    }

    ~VecData()
    {
        destroy_elements();
        free_cpu_data();
        free_gpu_data();
        len_ = 0;
    }

    HD size_type size() const noexcept { return len_; }
    HD size_type capacity() const noexcept { return cap_; }
    HD bool empty() const noexcept { return len_ == 0; }

    void reserve(size_type new_cap)
    {
        if (new_cap <= cap_)
        {
            return;
        }
        const bool use_managed = using_managed_memory();

        Dtype* new_cpu = nullptr;
        const size_t bytes = sizeof(Dtype) * new_cap;

        if (use_managed)
        {
            managed_mem_allocate(reinterpret_cast<void**>(&new_cpu), static_cast<int>(bytes));
        }
        else
        {
            cpu_mem_allocate(reinterpret_cast<void**>(&new_cpu), static_cast<int>(bytes));
        }

        if (!new_cpu)
        {
            throw std::bad_alloc{};
        }

        if constexpr (std::is_trivially_move_constructible_v<Dtype>)
        {
            std::memcpy(new_cpu, data_cpu_, sizeof(Dtype) * len_);
        }
        else
        {
            for (size_type i = 0; i < len_; ++i)
            {
                ::new (new_cpu + i) Dtype(std::move_if_noexcept(data_cpu_[i]));
            }
            destroy_elements();
        }

        Dtype* new_gpu = nullptr;
        if (mode_ == GPU)
        {
            if (use_managed)
            {
                new_gpu = new_cpu;
            }
            else
            {
                gpu_mem_allocate(reinterpret_cast<void**>(&new_gpu), static_cast<int>(bytes));
                if (!new_gpu)
                {
                    throw std::bad_alloc{};
                }
                mem_copy_cpu2gpu(new_gpu, new_cpu, static_cast<int>(sizeof(Dtype) * len_));
            }
        }

        free_cpu_data();
        free_gpu_data();

        data_cpu_ = new_cpu;
        data_gpu_ = new_gpu;
        cap_ = new_cap;
    }

    void update_gpu_data_from_cpu(void* cuda_stream = nullptr)
    {
        if (mode_ != GPU || len_ == 0)
        {
            return;
        }
        const bool use_managed = using_managed_memory();
        const size_t bytes = sizeof(Dtype) * len_;

        if (use_managed)
        {
            mem_prefetch_to_gpu(data_gpu_, static_cast<int>(bytes), -1, cuda_stream);
        }
        else
        {
            mem_copy_cpu2gpu(data_gpu_, data_cpu_, static_cast<int>(bytes), cuda_stream);
        }

        cpu_dirty_ = false;
    }

    void update_cpu_data_from_gpu()
    {
        if (mode_ != GPU || len_ == 0)
        {
            return;
        }
        const bool use_managed = using_managed_memory();
        const size_t bytes = sizeof(Dtype) * len_;

        if (use_managed)
        {
            mem_prefetch_to_cpu(data_cpu_, static_cast<int>(bytes), nullptr);
        }
        else
        {
            mem_copy_gpu2cpu(data_cpu_, data_gpu_, static_cast<int>(bytes));
        }
    }

    HD Dtype& cpu(int idx)
    {
        assert(idx >= 0 && idx < len_);
        return data_cpu_[idx];
    }

    void push_back(const Dtype& v) { emplace_back(v); }
    void push_back(Dtype&& v) { emplace_back(std::move(v)); }

    template <typename... Args>
    Dtype& emplace_back(Args&&... args)
    {
        if (len_ >= cap_)
        {
            reserve(cap_ == 0 ? 8 : cap_ * 2);
        }

        if constexpr (std::is_trivially_constructible_v<Dtype, Args&&...>)
        {
            data_cpu_[len_] = Dtype(std::forward<Args>(args)...);
        }
        else
        {
            ::new (data_cpu_ + len_) Dtype(std::forward<Args>(args)...);
        }

        cpu_dirty_ = true;
        return data_cpu_[len_++];
    }

    void pop_back(bool update_gpu = false)
    {
        assert(len_ > 0);
        --len_;

        if constexpr (!std::is_trivially_destructible_v<Dtype>)
        {
            data_cpu_[len_].~Dtype();
        }

        cpu_dirty_ = true;
        if (update_gpu && mode_ == GPU)
        {
            update_gpu_data_from_cpu();
        }
    }

    void resize(size_type new_size)
    {
        if (new_size > cap_)
        {
            reserve(new_size);
        }

        if (new_size > len_)
        {
            for (size_type i = len_; i < new_size; ++i)
            {
                if constexpr (std::is_trivially_constructible_v<Dtype>)
                {
                    data_cpu_[i] = Dtype{};
                }
                else
                {
                    ::new (data_cpu_ + i) Dtype{};
                }
            }
        }
        else if (new_size < len_)
        {
            if constexpr (!std::is_trivially_destructible_v<Dtype>)
            {
                for (size_type i = new_size; i < len_; ++i)
                {
                    data_cpu_[i].~Dtype();
                }
            }
        }

        len_ = new_size;
        cpu_dirty_ = true;
    }

    Dtype* get_cpu_data()
    {
        return this->data_cpu_;
    }

    HD Dtype* get_gpu_data()
    {
        return this->data_gpu_;
    }

    HD Dtype* get_dev_data()
    {
#ifdef __CUDA_ARCH__
        return this->data_gpu_;
#else
        return this->data_cpu_;
#endif
    }

    Dtype& cpu(int idx) const
    {
        assert(idx >= 0 && idx < len_);
        return data_cpu_[idx];
    }

    void clear()
    {
        destroy_elements();
        len_ = 0;
        cpu_dirty_ = true;
    }

    void erase(size_type index, bool update_gpu = false)
    {
        assert(index >= 0 && index < len_);

        if constexpr (!std::is_trivially_destructible_v<Dtype>)
        {
            data_cpu_[index].~Dtype();
        }

        if (index < len_ - 1)
        {
            if constexpr (std::is_trivially_move_constructible_v<Dtype> &&
                          std::is_trivially_destructible_v<Dtype>)
            {
                std::memmove(data_cpu_ + index, data_cpu_ + index + 1,
                             sizeof(Dtype) * (len_ - index - 1));
            }
            else
            {
                for (size_type i = index; i < len_ - 1; ++i)
                {
                    ::new (data_cpu_ + i) Dtype(std::move_if_noexcept(data_cpu_[i + 1]));
                    data_cpu_[i + 1].~Dtype();
                }
            }
        }

        --len_;
        cpu_dirty_ = true;

        if (update_gpu && mode_ == GPU)
        {
            update_gpu_data_from_cpu();
        }
    }

    void erase_swap(size_type index, bool update_gpu = false)
    {
        assert(index >= 0 && index < len_);

        if (index == len_ - 1)
        {
            pop_back(update_gpu);
            return;
        }

        if constexpr (!std::is_trivially_destructible_v<Dtype>)
        {
            data_cpu_[index].~Dtype();
        }

        ::new (data_cpu_ + index) Dtype(std::move_if_noexcept(data_cpu_[len_ - 1]));

        if constexpr (!std::is_trivially_destructible_v<Dtype>)
        {
            data_cpu_[len_ - 1].~Dtype();
        }

        --len_;
        cpu_dirty_ = true;

        if (update_gpu && mode_ == GPU)
        {
            update_gpu_data_from_cpu();
        }
    }
};

#undef HD
