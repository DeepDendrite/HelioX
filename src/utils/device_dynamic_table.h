// device_dynamic_table.h — 支持移动语义的紧凑句柄表
// 关键修复：
// 1. add_or_update / update_by_handle 接受万能引用 (Item&& / CpuOnlyItem&&)，
//    以便真正触发移动，避免因 const 导致调用已删除的拷贝赋值。
// 2. 在新增时使用 emplace_back / std::forward 直接构造或移动到 VecData，
//    覆盖时使用 move‑assignment；若 Item 不可 move‑assignable 会静态断言失败。
// 3. remove_by_handle 内部移动最后元素时显式 std::move。
// 4. 添加 static_assert 保证 Item 可移动构造、可移动赋值。
// 5. 其余接口基本保持不变，兼容原来调用。

#pragma once
#include "vecdata.h"
#include <unordered_map>

// ===== DynamicDeviceTable ===================================================
// 带句柄的连续存储表，便于 GPU 批量处理。
// Item           : 真正存 GPU/CPU 的元素类型（需支持移动构造/赋值）。
// Key            : 用户侧键（可重复 -> 同句柄）。必须可哈希、可比较。
// CpuOnlyItem    : 仅驻留 CPU 的附加信息，默认为 std::monostate。
//=============================================================================

template <typename Item,
          typename Key,
          typename CpuOnlyItem = std::monostate>
class DynamicDeviceTable {
public:
    using Handle     = int;
    using size_type  = int;
    static_assert(std::is_move_constructible_v<Item>, "Item 必须可移动构造");
    static_assert(std::is_move_assignable_v<Item>,    "Item 必须可移动赋值");

private:
    std::unordered_map<Key, Handle>   key2handle_;
    std::unordered_map<Handle, int>   handle2index_;

    VecData<Item>                     item_vec_;
    std::vector<CpuOnlyItem>          cpu_only_vec_;

    Mode   mode_        = CPU;
    bool   dirty_       = false;
    Handle next_handle_ = 0;

public:
    explicit DynamicDeviceTable(Mode mode = CPU) : item_vec_(mode), mode_(mode) {}

    // ---------------------------------------------------------------------
    // add_or_update :
    //   * 若 key 已存在 -> 覆写对应元素 (move assign)
    //   * 否则          -> push_back / emplace_back 新元素
    // ---------------------------------------------------------------------
    Handle add_or_update(const Key &k,
                         Item &&h_item,
                         CpuOnlyItem &&cpu_only_item = CpuOnlyItem{}) {
        auto it = key2handle_.find(k);
        if (it != key2handle_.end()) {            // ---- 覆盖更新 ----
            Handle h   = it->second;
            int    idx = handle2index_.at(h);
            assert(idx >= 0 && idx < item_vec_.size() && "index out of bounds");

            item_vec_.cpu(idx) = std::move(h_item);
            if constexpr (!std::is_same_v<CpuOnlyItem, std::monostate>) {
                cpu_only_vec_[idx] = std::move(cpu_only_item);
            }
            dirty_ = true;
            return h;
        }

        // ---- 新增 ----
        Handle h = next_handle_++;
        item_vec_.emplace_back(std::forward<Item>(h_item));
        if constexpr (!std::is_same_v<CpuOnlyItem, std::monostate>) {
            cpu_only_vec_.emplace_back(std::forward<CpuOnlyItem>(cpu_only_item));
        }
        key2handle_[k] = h;
        handle2index_[h] = static_cast<int>(item_vec_.size()) - 1;
        dirty_ = true;
        return h;
    }

    // 仅 CPU update；GPU 数据需另行 sync
    void update_by_handle(Handle h,
                          Item &&h_item,
                          CpuOnlyItem &&cpu_only_item = CpuOnlyItem{}) {
        if (!handle2index_.contains(h)) throw std::runtime_error("Handle not found");
        int idx = handle2index_.at(h);
        assert(idx >= 0 && idx < item_vec_.size());

        item_vec_.cpu(idx) = std::move(h_item);
        if constexpr (!std::is_same_v<CpuOnlyItem, std::monostate>) {
            cpu_only_vec_[idx] = std::move(cpu_only_item);
        }
        dirty_ = true;
    }

    // ------------------------------------------------------------------
    // remove_by_handle : O(1) 删除 (swap‑pop)
    // ------------------------------------------------------------------
    void remove_by_handle(Handle h) {
        if (!handle2index_.contains(h)) throw std::runtime_error("Handle not found");
        int idx = handle2index_[h];
        int last = static_cast<int>(item_vec_.size()) - 1;

        // 若不是尾元素，则把尾元素 move 到空位
        if (idx < last) {
            item_vec_.cpu(idx) = std::move(item_vec_.cpu(last));
            if constexpr (!std::is_same_v<CpuOnlyItem, std::monostate>) {
                cpu_only_vec_[idx] = std::move(cpu_only_vec_[last]);
            }
            // 修正尾元素对应句柄的映射
            for (auto &pr : handle2index_) {
                if (pr.second == last) {
                    pr.second = idx;
                    break;
                }
            }
        }

        item_vec_.pop_back();
        if constexpr (!std::is_same_v<CpuOnlyItem, std::monostate>) cpu_only_vec_.pop_back();

        // 移除映射
        handle2index_.erase(h);
        for (auto it = key2handle_.begin(); it != key2handle_.end(); ++it) {
            if (it->second == h) { key2handle_.erase(it); break; }
        }
        dirty_ = true;
    }

    // ------------------------------------------------------------------
    // 简单查询接口
    // ------------------------------------------------------------------
    Item &get_by_handle(Handle h) {
        if (!handle2index_.contains(h)) throw std::runtime_error("Handle not found");
        int idx = handle2index_.at(h);
        return item_vec_.cpu(idx);
    }
    const Item &get_by_handle(Handle h) const {
        return const_cast<DynamicDeviceTable *>(this)->get_by_handle(h);
    }

    int index_of_handle(Handle h) const {
        auto it = handle2index_.find(h);
        if (it == handle2index_.end()) throw std::runtime_error("Handle not found");
        return it->second;
    }

    CpuOnlyItem &get_cpu_only_by_handle(Handle h) {
        if constexpr (std::is_same_v<CpuOnlyItem, std::monostate>) {
            throw std::runtime_error("CpuOnlyItem is monostate — no data available");
        } else {
            if (!handle2index_.contains(h)) throw std::runtime_error("Handle not found");
            return cpu_only_vec_[handle2index_.at(h)];
        }
    }

    // ------------------------------------------------------------------
    // 同步 GPU <‑> CPU
    // ------------------------------------------------------------------
    void update_gpu_from_cpu(void *cuda_stream = nullptr) {
        if (dirty_ && mode_ == GPU) {
            item_vec_.update_gpu_data_from_cpu(cuda_stream);
            dirty_ = false;
        }
    }
    void update_cpu_from_gpu() {
        if (mode_ == GPU) {
            assert(!dirty_ && "CPU 数据已被修改，不能覆盖");
            item_vec_.update_cpu_data_from_gpu();
        }
    }

    // ---- 其他辅助 ----
    [[nodiscard]] int  size()       const { return item_vec_.size(); }
    [[nodiscard]] bool dirty()      const { return dirty_; }
    [[nodiscard]] Mode mode()       const { return mode_; }
    [[nodiscard]] auto get_cpu_data() { return item_vec_.get_cpu_data(); }
    [[nodiscard]] auto get_gpu_data() { return item_vec_.get_gpu_data(); }
    void set_dirty(bool d = true) { dirty_ = d; }

    auto *get_cpu_only_data_vec() {
        if constexpr (std::is_same_v<CpuOnlyItem, std::monostate>) {
            return static_cast<std::vector<std::monostate> *>(nullptr);
        } else {
            return &cpu_only_vec_;
        }
    }
    
    // ------------------------------------------------------------------
    // 通过Key的操作接口
    // ------------------------------------------------------------------
    bool has_key(const Key& key) const {
        return key2handle_.find(key) != key2handle_.end();
    }
    
    Item* get_item_ptr(const Key& key) {
        auto it = key2handle_.find(key);
        if (it == key2handle_.end()) return nullptr;
        
        Handle h = it->second;
        auto handle_it = handle2index_.find(h);
        if (handle_it == handle2index_.end()) return nullptr;
        
        int idx = handle_it->second;
        return &item_vec_.cpu(idx);
    }
    
    const Item* get_item_ptr(const Key& key) const {
        return const_cast<DynamicDeviceTable*>(this)->get_item_ptr(key);
    }
    
    void remove(const Key& key) {
        auto it = key2handle_.find(key);
        if (it == key2handle_.end()) return;
        
        Handle h = it->second;
        remove_by_handle(h);
    }
    
    std::vector<Key> get_all_keys() const {
        std::vector<Key> keys;
        keys.reserve(key2handle_.size());
        for (const auto& pair : key2handle_) {
            keys.push_back(pair.first);
        }
        return keys;
    }
};
