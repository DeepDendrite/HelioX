#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_utils.h"
#include "staging_channel.h"
#include "vecdata.h"

namespace neurong::autodiff {

// Lightweight per-step scalar tape for one [step][local_node] channel.
// GPU mode keeps only a small staged buffer on device and spills to host;
// backward uploads replay chunks back to GPU on demand.
class MechScalarTapeStore {
public:
    struct GpuReplayView {
        int local_step_index = 0;
        double* tape_gpu = nullptr;
    };

    explicit MechScalarTapeStore(Mode mode)
        : mode_(mode), replay_tape_(GPU) {}

    void set_nnode(int nnode) {
        if (nnode_ == nnode) {
            return;
        }
        clear();
        nnode_ = nnode;
        staging_dirty_ = true;
    }

    void set_source(double* source_cpu, double* source_gpu) {
        source_cpu_ = source_cpu;
        source_gpu_ = source_gpu;
        staging_dirty_ = true;
    }

    void set_spill_chunk_steps(int steps) {
        const int clamped = std::max(1, steps);
        if (spill_chunk_steps_ == clamped) {
            return;
        }
        spill_chunk_steps_ = clamped;
        clear();
        staging_dirty_ = true;
    }

    void clear() {
        tape_steps_ = 0;
        host_tape_steps_ = 0;
        host_tape_.clear();
        if (staged_) {
            staged_->flush();
        }
        replay_tape_.clear();
        replay_loaded_start_step_ = -1;
        replay_loaded_step_count_ = 0;
    }

    int tape_steps() const { return tape_steps_; }
    int host_tape_steps() const { return host_tape_steps_; }

    void stage_step(void* cuda_stream = nullptr) {
        ensure_staging_channel_();
        if (!staged_ || nnode_ <= 0) {
            return;
        }
        if (staged_->staged_count() >= staged_->staged_capacity()) {
            throw std::runtime_error("MechScalarTapeStore staged channel capacity overflow");
        }

        if (mode_ == GPU) {
            const int base = staged_->staged_count() * nnode_;
            const int bytes =
                static_cast<int>(sizeof(double) * static_cast<std::size_t>(nnode_));
            mem_copy_gpu2gpu(
                staged_->buffer_f64.get_gpu_data() + base,
                source_gpu_,
                bytes,
                cuda_stream);
            staged_->len += 1;
        } else {
            staged_->stage_sample_single();
        }

        tape_steps_ += 1;
        if (staged_->staged_count() >= spill_chunk_steps_) {
            flush_staged_to_host_(cuda_stream);
        }
    }

    void finalize_tape_for_backward() {
        flush_staged_to_host_(nullptr);
        replay_loaded_start_step_ = -1;
        replay_loaded_step_count_ = 0;
    }

    const double* tape_cpu_data_for_backward() const {
        return host_tape_.empty() ? nullptr : host_tape_.data();
    }

    GpuReplayView acquire_gpu_replay_view(int global_step_index) {
        if (mode_ != GPU) {
            throw std::runtime_error("MechScalarTapeStore GPU replay requested in CPU mode");
        }
        if (global_step_index < 0 || global_step_index >= tape_steps_) {
            throw std::out_of_range("MechScalarTapeStore replay step index out of range");
        }
        ensure_replay_chunk_loaded_(global_step_index);
        return GpuReplayView{
            .local_step_index = global_step_index - replay_loaded_start_step_,
            .tape_gpu = replay_tape_.get_gpu_data(),
        };
    }

private:
    void ensure_staging_channel_() {
        if (!staging_dirty_ && staged_) {
            return;
        }
        staged_.reset();
        if (nnode_ <= 0 || source_cpu_ == nullptr || (mode_ == GPU && source_gpu_ == nullptr)) {
            return;
        }
        staged_ = std::make_unique<StagingChannel>(
            mode_, spill_chunk_steps_, nnode_, source_cpu_, source_gpu_, false);
        staging_dirty_ = false;
    }

    void flush_staged_to_host_(void* cuda_stream) {
        (void)cuda_stream;
        if (!staged_ || staged_->staged_count() == 0) {
            return;
        }
        if (mode_ == GPU) {
            staged_->sync_cpu_from_gpu();
        }

        const int staged_steps = staged_->staged_count();
        const int elem_count = staged_steps * nnode_;
        const std::size_t old_size = host_tape_.size();
        host_tape_.resize(old_size + static_cast<std::size_t>(elem_count));
        std::memcpy(
            host_tape_.data() + static_cast<std::ptrdiff_t>(old_size),
            staged_->buffer_f64.get_cpu_data(),
            sizeof(double) * static_cast<std::size_t>(elem_count));
        host_tape_steps_ += staged_steps;
        staged_->flush();
        replay_loaded_start_step_ = -1;
        replay_loaded_step_count_ = 0;
    }

    void ensure_replay_chunk_loaded_(int global_step_index) {
        if (replay_loaded_step_count_ > 0 &&
            global_step_index >= replay_loaded_start_step_ &&
            global_step_index < replay_loaded_start_step_ + replay_loaded_step_count_) {
            return;
        }
        if (host_tape_steps_ != tape_steps_) {
            throw std::runtime_error("MechScalarTapeStore replay requested before finalize");
        }
        const int chunk_end = global_step_index + 1;
        const int chunk_start = std::max(0, chunk_end - spill_chunk_steps_);
        const int chunk_steps = chunk_end - chunk_start;
        const int elem_count = chunk_steps * nnode_;
        const std::size_t src_base =
            static_cast<std::size_t>(chunk_start) * static_cast<std::size_t>(nnode_);
        replay_tape_.resize(elem_count);
        mem_copy_cpu2gpu_sync(
            replay_tape_.get_gpu_data(),
            host_tape_.data() + static_cast<std::ptrdiff_t>(src_base),
            static_cast<int>(sizeof(double) * static_cast<std::size_t>(elem_count)));
        replay_loaded_start_step_ = chunk_start;
        replay_loaded_step_count_ = chunk_steps;
    }

    Mode mode_ = CPU;
    int nnode_ = 0;
    int spill_chunk_steps_ = 64;
    int tape_steps_ = 0;
    int host_tape_steps_ = 0;
    double* source_cpu_ = nullptr;
    double* source_gpu_ = nullptr;
    bool staging_dirty_ = true;

    std::unique_ptr<StagingChannel> staged_;
    std::vector<double> host_tape_;
    VecData<double> replay_tape_;
    int replay_loaded_start_step_ = -1;
    int replay_loaded_step_count_ = 0;
};

}  // namespace neurong::autodiff
