#pragma once
#include <highfive/highfive.hpp>
#include <string>
#include "utils.h"  // VarDescriptor现在定义在utils.h中
#include "magic_enum/magic_enum.hpp"
#include "staging_channel.h"
#include "vecdata.h"
#include "device_dynamic_table.h"

// 变量记录点 - 存储需要记录的变量的指针信息
struct RecordPoint {
    double *var_ptr_cpu;
    double *var_ptr_gpu;
    HighFive::DataSet dataset;

    bool operator==(const RecordPoint& other) const {
        return var_ptr_cpu == other.var_ptr_cpu && var_ptr_gpu == other.var_ptr_gpu;
    }
};

namespace std {
    template<>
    struct hash<RecordPoint> {
        size_t operator()(const RecordPoint& key) const {
            return std::hash<double*>()(key.var_ptr_cpu) ^ (std::hash<double*>()(key.var_ptr_gpu) << 1);
        }
    };
}

struct OutPutBuffers{
    HighFive::DataSet dataset; // 对应的 HDF5 数据集
    std::vector<double> ipc_buffer; // 使用 std::vector 来管理缓冲区
};

enum class RecorderStorageDType {
    FP64 = 0,
    FP32 = 1,
};

enum class BufferEnable{
    NONE = 0,
    HDF5 = 1<<0,
    IPC = 1<<1
};
using namespace magic_enum::bitwise_operators;
template <>
struct magic_enum::customize::enum_range<BufferEnable> {
  static constexpr bool is_flags = true;
};

// 变量记录器 - 统一管理变量的记录、缓冲和输出
// 支持多种输出方式：HDF5文件、IPC缓冲区
struct VariableRecorder {
    Mode mode;

    DynamicDeviceTable<StagingChannel,VarDescriptor,OutPutBuffers> bufferTable;

    BufferEnable buffer_enable;
    RecorderStorageDType storage_dtype;

    int buffer_capacity;   // 每个 buffer 的容量
    int buffer_size;       // 记录当前缓冲区中已写入的数据条数


    //返回的是可选的const vector<double>&(引用)
    std::optional<std::reference_wrapper<const std::vector<double>>>
    // const std::vector<double>&
    get_single_irq_buffer(int handle) {
        auto &buffer = bufferTable.get_cpu_only_by_handle(handle).ipc_buffer;
        return buffer; // 返回对应的缓冲区
    }

    bool isEnable(BufferEnable flag) {
        return (int)buffer_enable & (int)flag;
    }
    VariableRecorder(Mode mode, size_t _buffer_capacity, BufferEnable _buffer_enable, bool use_fp32_storage = false)
        : mode(mode), bufferTable(mode), storage_dtype(use_fp32_storage ? RecorderStorageDType::FP32 : RecorderStorageDType::FP64) {
        buffer_capacity = _buffer_capacity;
        buffer_size = 0;
        this->buffer_enable = _buffer_enable;
    }
    // 全部 push 完成后，需要调用 initialize
    // 返回DDT分配的handle，作为新的recordId
    int push_back(const VarDescriptor& record_var, RecordPoint &recordPoint) {
        auto handle = bufferTable.add_or_update(
            record_var,
            StagingChannel(mode,
                           buffer_capacity,
                           1,
                           recordPoint.var_ptr_cpu,
                           recordPoint.var_ptr_gpu,
                           storage_dtype == RecorderStorageDType::FP32),
            OutPutBuffers{recordPoint.dataset, std::vector<double>()}
        );
        return handle;
    }

    //注意！这个函数和initialize不同，这个函数是每次sim在调用finitialize之前都要调用的
    //这个函数是为了在每次模拟开始之前，清空上次的buffer
    void finitialize(){
        auto *output_buffers = bufferTable.get_cpu_only_data_vec();
        for(auto &buf: *output_buffers){
            buf.ipc_buffer.clear();
        }

        // 重置缓冲区状态
        buffer_size = 0;  // 重置全局buffer_size

        // 重置每个 staged channel 的状态
        auto bufferItems = bufferTable.get_cpu_data();
        auto buffer_num = bufferTable.size();
        for (int i = 0; i < buffer_num; i++) {
            bufferItems[i].flush();
        }

        // 重要：将 len 重置同步到 GPU，避免 GPU 端仍使用旧 len 导致越界/NaN
        bufferTable.set_dirty(true);
        bufferTable.update_gpu_from_cpu();
    }

    void initialize(Mode mode);

    void log_data_cpu() {
        if(bufferTable.size() <= 0) return;

        auto buffer_num = bufferTable.size();
        auto bufferItems = bufferTable.get_cpu_data();
        for (int i = 0; i < buffer_num; i++) {
            bufferItems[i].stage_sample_single();
        }
        buffer_size++;
        if (buffer_size >= buffer_capacity) {
            assert(buffer_size == buffer_capacity);
            flush_cpu();
        }
    }
    void put_data_to_hdf5(){
        if(!isEnable(BufferEnable::HDF5)) return;
        //写入HDF5文件
        auto output_buffers = bufferTable.get_cpu_only_data_vec();
        auto record_buffers = bufferTable.get_cpu_data();
        auto buffer_count = bufferTable.size();
        std::vector<double> temp_f64;
        for (int i = 0; i < buffer_count; i++) {
            auto & dataset = (*output_buffers)[i].dataset;
            auto current_size = dataset.getSpace().getDimensions()[0];
            // 扩展数据集大小，写入当前 buffer 中收集的数据
            dataset.resize({ current_size + buffer_size });
            if (record_buffers[i].use_fp32_storage) {
                temp_f64.resize(buffer_size);
                auto* src = record_buffers[i].buffer_f32.get_cpu_data();
                for (int j = 0; j < buffer_size; ++j) {
                    temp_f64[j] = static_cast<double>(src[j]);
                }
                dataset.select({ current_size }, { (unsigned long)buffer_size }).write(temp_f64.data());
            } else {
                dataset.select({ current_size }, { (unsigned long)buffer_size }).write(record_buffers[i].buffer_f64.get_cpu_data());
            }
        }
    }

    void put_data_to_ipc_buf(){
        if(!isEnable(BufferEnable::IPC)) return;
        //写入IPC缓冲区
        auto output_buffers = bufferTable.get_cpu_only_data_vec();
        auto record_buffers = bufferTable.get_cpu_data();
        auto buffer_count = bufferTable.size();
        for (int i = 0; i < buffer_count; i++) {
            auto &ipc_buffer = (*output_buffers)[i].ipc_buffer;
            auto current_size = ipc_buffer.size();
            ipc_buffer.resize(current_size + buffer_size);
            auto* dst = ipc_buffer.data() + current_size;
            if (record_buffers[i].use_fp32_storage) {
                auto* src = record_buffers[i].buffer_f32.get_cpu_data();
                for (int j = 0; j < buffer_size; ++j) {
                    dst[j] = static_cast<double>(src[j]);
                }
            } else {
                auto* src = record_buffers[i].buffer_f64.get_cpu_data();
                std::copy(src, src + buffer_size, dst);
            }
        }
    }

    void flush_cpu() {
        if(buffer_size <= 0) return;
        put_data_to_hdf5();
        put_data_to_ipc_buf();
        auto buffer_num = bufferTable.size();
        auto bufferItems = bufferTable.get_cpu_data();
        for (int i = 0; i < buffer_num; i++) {
            assert(bufferItems[i].staged_count() == buffer_size && "staged channel length should match buffer_size");
            bufferItems[i].flush();
        }
        // printf("Flushed %zu items to HDF5 and IPC buffers.\n", buffer_size);
        buffer_size = 0;
    }

    void log_data_gpu();
    void flush_gpu();
};
