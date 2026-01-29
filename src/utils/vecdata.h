// VecData_movable.h — 完整的可移动版本 VecData
// 主要改动：
// 1. 显式删除拷贝构造 / 赋值，防止双重释放。
// 2. 实现移动构造 / 移动赋值，将所有权安全移交。
// 3. push_back 改为完美转发版本 emplace_back，
//    能直接移动或原地构造（支持 VecData 嵌套）。
// 4. reserve / destructor / allocate_cpu_data 支持非平凡类型，
//    在必要时调用构造 / 析构函数（placement‑new + 手动析构）。
// 5. 代码整体改用 *_ 作为私有成员后缀，避免与函数混淆。

#pragma once

#include "utils.h"
#include "cuda_utils.h"
#include <type_traits>
#include <vector>
#include <cstring>

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
	size_type len_ = 0; // 当前元素数量
	size_type cap_ = 0; // 已分配容量（元素个数）
	bool cpu_dirty_ = false;
	Mode mode_ = CPU;

	Dtype *data_cpu_ = nullptr; // 锁页 CPU 缓冲区
	Dtype *data_gpu_ = nullptr; // GPU 缓冲区

	bool using_managed_memory() const noexcept
	{
#ifdef NEURON_USE_MANAGED_MEMORY
		return mode_ == GPU;
#else
		return false;
#endif
	}

	/* --------------------------------------------------
	 *  内部工具
	 * --------------------------------------------------*/

	// 析构 [0,len_) 区间内的对象（仅对非平凡可析构类型需要）
	void destroy_elements() noexcept
	{
		if constexpr (!std::is_trivially_destructible_v<Dtype>)
		{
			for (size_type i = 0; i < len_; ++i)
				data_cpu_[i].~Dtype();
		}
	}

	/* --------------------------------------------------
	 *  资源管理
	 * --------------------------------------------------*/

	void allocate_cpu_data()
	{
		if (cap_ == 0)
			return;
		const size_t bytes = sizeof(Dtype) * cap_;

		const bool use_managed = using_managed_memory();
		if (use_managed)
		{
			managed_mem_allocate(reinterpret_cast<void **>(&data_cpu_), static_cast<int>(bytes));
			if (!data_cpu_)
				throw std::bad_alloc{};
			data_gpu_ = data_cpu_;
		}
		else
		{
			cpu_mem_allocate(reinterpret_cast<void **>(&data_cpu_), static_cast<int>(bytes));
			if (!data_cpu_)
				throw std::bad_alloc{};
		}
	}

	void allocate_gpu_data()
	{
		if (mode_ != GPU || cap_ == 0)
			return;

		if (using_managed_memory())
			return;

		const size_t bytes = sizeof(Dtype) * cap_;
		gpu_mem_allocate(reinterpret_cast<void **>(&data_gpu_), static_cast<int>(bytes));
		if (!data_gpu_)
			throw std::bad_alloc{};
	}

	void free_cpu_data()
	{
		if (!data_cpu_)
			return;

		if (using_managed_memory())
		{
			managed_mem_free(reinterpret_cast<void **>(&data_cpu_));
			data_cpu_ = nullptr;
			data_gpu_ = nullptr;
			return;
		}

		cpu_mem_free(reinterpret_cast<void **>(&data_cpu_));
		data_cpu_ = nullptr;
	}

	void free_gpu_data()
	{
		if (!data_gpu_)
			return;

		if (using_managed_memory())
		{
			// managed 内存的释放已经在 free_cpu_data 中完成
			data_gpu_ = nullptr;
			return;
		}

		gpu_mem_free(reinterpret_cast<void **>(&data_gpu_));
		data_gpu_ = nullptr;
	}

public:
	/* --------------------------------------------------
	 *  构造 / 析构 / 赋值
	 * --------------------------------------------------*/

	explicit VecData(Mode m = CPU) noexcept : mode_(m) {}

	VecData(Mode m, size_type n) : len_(n), cap_(n), mode_(m)
	{
		allocate_cpu_data();
		if (mode_ == GPU)
			allocate_gpu_data();
	}

	// 从数组构造
	VecData(Mode m, const Dtype *arr, size_type n) : VecData(m, n)
	{
		if constexpr (std::is_trivially_copyable_v<Dtype>)
			std::memcpy(data_cpu_, arr, sizeof(Dtype) * n);
		else
			for (size_type i = 0; i < n; ++i)
				::new (data_cpu_ + i) Dtype(arr[i]);

		if (mode_ == GPU)
			update_gpu_data_from_cpu();
	}

	// 从 std::vector 构造
	VecData(Mode m, const std::vector<Dtype> &v) : VecData(m, v.data(), static_cast<size_type>(v.size())) {}

	// 从单值，复制多份构造
	VecData(Mode m, const Dtype &value, size_type n) : VecData(m, n)
	{
		if constexpr (std::is_trivially_copyable_v<Dtype>)
		{
			std::fill_n(data_cpu_, n, value);
		}
		else
		{
			for (size_type i = 0; i < n; ++i)
				::new (data_cpu_ + i) Dtype(value);
		}

		if (mode_ == GPU)
			update_gpu_data_from_cpu();
	}

	/* 禁用拷贝 */
	VecData(const VecData &) = delete;
	VecData &operator=(const VecData &) = delete;

	/* 允许移动 */
	VecData(VecData &&other) noexcept
		: len_(other.len_), cap_(other.cap_), cpu_dirty_(other.cpu_dirty_), mode_(other.mode_),
		  data_cpu_(other.data_cpu_), data_gpu_(other.data_gpu_)
	{
		other.len_ = other.cap_ = 0;
		other.data_cpu_ = other.data_gpu_ = nullptr;
		other.cpu_dirty_ = false;
	}

	VecData &operator=(VecData &&other) noexcept
	{
		if (this != &other)
		{
			// 释放旧资源
			destroy_elements();
			free_cpu_data();
			free_gpu_data();

			// 转移所有权
			len_ = other.len_;
			cap_ = other.cap_;
			cpu_dirty_ = other.cpu_dirty_;
			mode_ = other.mode_;
			data_cpu_ = other.data_cpu_;
			data_gpu_ = other.data_gpu_;

			// 置空被移动对象
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

	/* --------------------------------------------------
	 *  容量管理
	 * --------------------------------------------------*/

	HD size_type size() const noexcept { return len_; }
	HD size_type capacity() const noexcept { return cap_; }
	HD bool empty() const noexcept { return len_ == 0; }

	void reserve(size_type new_cap)
	{
		if (new_cap <= cap_)
			return;
		const bool use_managed = using_managed_memory();

		// 1. 新分配 CPU 缓冲
		Dtype *new_cpu = nullptr;
		const size_t bytes = sizeof(Dtype) * new_cap;

		if (use_managed)
		{
			managed_mem_allocate(reinterpret_cast<void **>(&new_cpu), static_cast<int>(bytes));
		}
		else
		{
			cpu_mem_allocate(reinterpret_cast<void **>(&new_cpu), static_cast<int>(bytes));
		}

		if (!new_cpu)
			throw std::bad_alloc{};

		// 2. 迁移 / 拷贝元素
		if constexpr (std::is_trivially_move_constructible_v<Dtype>)
		{
			std::memcpy(new_cpu, data_cpu_, sizeof(Dtype) * len_);
		}
		else
		{
			for (size_type i = 0; i < len_; ++i)
				::new (new_cpu + i) Dtype(std::move_if_noexcept(data_cpu_[i]));
			destroy_elements();
		}

		// 3. 处理 GPU 缓冲
		Dtype *new_gpu = nullptr;
		if (mode_ == GPU)
		{
			if (use_managed)
			{
				new_gpu = new_cpu;
			}
			else
			{
				gpu_mem_allocate(reinterpret_cast<void **>(&new_gpu), static_cast<int>(bytes));
				if (!new_gpu)
					throw std::bad_alloc{};
				mem_copy_cpu2gpu(new_gpu, new_cpu, static_cast<int>(sizeof(Dtype) * len_));
			}
		}

		// 4. 释放旧缓冲并更新指针 / 容量
		free_cpu_data();
		free_gpu_data();

		data_cpu_ = new_cpu;
		data_gpu_ = new_gpu;
		cap_ = new_cap;
	}

	/* --------------------------------------------------
	 *  数据同步
	 * --------------------------------------------------*/

	void update_gpu_data_from_cpu(void *cuda_stream = nullptr)
	{
		if (mode_ != GPU || len_ == 0)
			return;
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
			return;
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

	/* --------------------------------------------------
	 *  元素存取
	 * --------------------------------------------------*/

	HD Dtype &cpu(int idx)
	{
		assert(idx >= 0 && idx < len_);
		return data_cpu_[idx];
	}

	/* --------------------------------------------------
	 *  push / emplace
	 * --------------------------------------------------*/

	// 便捷包装：区分左值 / 右值
	void push_back(const Dtype &v) { emplace_back(v); }
	void push_back(Dtype &&v) { emplace_back(std::move(v)); }

	// 完美转发 emplace_back —— 支持 "VecData 嵌 VecData" 等复杂情形
	template <typename... Args>
	Dtype &emplace_back(Args &&...args)
	{
		if (len_ >= cap_)
			reserve(cap_ == 0 ? 8 : cap_ * 2);

		if constexpr (std::is_trivially_constructible_v<Dtype, Args &&...>)
		{
			// 对于平凡类型，直接赋值即可（少一次构造开销）
			data_cpu_[len_] = Dtype(std::forward<Args>(args)...);
		}
		else
		{
			// placement‑new 原地构造
			::new (data_cpu_ + len_) Dtype(std::forward<Args>(args)...);
		}

		cpu_dirty_ = true;
		return data_cpu_[len_++];
	}

	/* --------------------------------------------------
	 *  pop_back
	 * --------------------------------------------------*/

	void pop_back(bool update_gpu = false)
	{
		assert(len_ > 0);
		--len_;

		if constexpr (!std::is_trivially_destructible_v<Dtype>)
			data_cpu_[len_].~Dtype();

		cpu_dirty_ = true;
		if (update_gpu && mode_ == GPU)
			update_gpu_data_from_cpu();
	}

	/* --------------------------------------------------
	 *  resize
	 * --------------------------------------------------*/

	void resize(size_type new_size)
	{
		if (new_size > cap_)
		{
			// 需要扩容
			reserve(new_size);
		}
		
		if (new_size > len_)
		{
			// 增加元素，用默认值初始化
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
			// 减少元素，需要析构多余的元素
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

	/*
	 * return the pointer of cpu data buffer (this->data_cpu)
	 */
	Dtype *get_cpu_data()
	{
		return this->data_cpu_;
	}

	/*
	 * return the pointer of gpu data buffer (this->data_gpu)
	 */
	HD Dtype *get_gpu_data()
	{
		return this->data_gpu_;
	}

	HD Dtype *get_dev_data()
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

	/* --------------------------------------------------
	 *  clear
	 * --------------------------------------------------*/
	
	void clear()
	{
		destroy_elements();
		
		// 只重置长度，不释放内存（保留容量）
		len_ = 0;
		cpu_dirty_ = true;
	}

	/* --------------------------------------------------
	 *  erase
	 * --------------------------------------------------*/
	
	void erase(size_type index, bool update_gpu = false)
	{
		assert(index >= 0 && index < len_);
		
		// 1. 析构要删除的元素
		if constexpr (!std::is_trivially_destructible_v<Dtype>) {
			data_cpu_[index].~Dtype();
		}
		
		// 2. 移动后续元素向前填补空隙
		if (index < len_ - 1) {
			if constexpr (std::is_trivially_move_constructible_v<Dtype> && 
			              std::is_trivially_destructible_v<Dtype>) {
				// 平凡类型：直接使用memmove
				std::memmove(data_cpu_ + index, data_cpu_ + index + 1, 
				            sizeof(Dtype) * (len_ - index - 1));
			} else {
				// 非平凡类型：逐个移动构造
				for (size_type i = index; i < len_ - 1; ++i) {
					::new (data_cpu_ + i) Dtype(std::move_if_noexcept(data_cpu_[i + 1]));
					data_cpu_[i + 1].~Dtype();
				}
			}
		}
		
		// 3. 更新长度和标记
		--len_;
		cpu_dirty_ = true;
		
		// 4. 可选的GPU同步
		if (update_gpu && mode_ == GPU) {
			update_gpu_data_from_cpu();
		}
	}

	// 交换删除版本（O(1)复杂度，但不保持顺序）
	void erase_swap(size_type index, bool update_gpu = false)
	{
		assert(index >= 0 && index < len_);
		
		if (index == len_ - 1) {
			// 如果是最后一个元素，直接pop_back
			pop_back(update_gpu);
			return;
		}
		
		// 1. 析构要删除的元素（在index位置）
		if constexpr (!std::is_trivially_destructible_v<Dtype>) {
			data_cpu_[index].~Dtype();
		}
		
		// 2. 将最后一个元素移动到删除位置
		::new (data_cpu_ + index) Dtype(std::move_if_noexcept(data_cpu_[len_ - 1]));
		
		// 3. 析构最后一个位置的元素（已被移动，但对象仍在尾部）
		if constexpr (!std::is_trivially_destructible_v<Dtype>) {
			data_cpu_[len_ - 1].~Dtype();
		}
		
		// 4. 更新长度和标记
		--len_;
		cpu_dirty_ = true;
		
		// 5. 可选的GPU同步
		if (update_gpu && mode_ == GPU) {
			update_gpu_data_from_cpu();
		}
	}

};

#undef HD
