#pragma once
#include <highfive/highfive.hpp>
#include <string>
#include "utils.h"  // VarDescriptor现在定义在utils.h中
#include "magic_enum/magic_enum.hpp"
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


struct BufferItem{
    double *var_ptr_cpu; // CPU 端变量指针
    double *var_ptr_gpu; // GPU 端变量指针
    VecData<double> buffer; // 使用 VecData 来管理缓冲区
    int buffer_capacity; // 每个 buffer 的最大容量
    int len; // 当前缓冲区中已写入的数据条数
    BufferItem(Mode mode,int buffer_cap, double *var_ptr_cpu, double *var_ptr_gpu)
        : var_ptr_cpu(var_ptr_cpu), var_ptr_gpu(var_ptr_gpu), len(0),buffer(mode, buffer_cap), buffer_capacity(buffer_cap) {
        // printf("Item capacity = %zu\n", buffer_capacity);
    }
    void log_data_single();
    void flush();
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

    DynamicDeviceTable<BufferItem,VarDescriptor,OutPutBuffers> bufferTable;

    BufferEnable buffer_enable;

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
    VariableRecorder(Mode mode,size_t _buffer_capacity,BufferEnable _buffer_enable):mode(mode),bufferTable(mode) {
        buffer_capacity = _buffer_capacity;
        printf("Buffer Capacity = %zu\n", buffer_capacity);
        buffer_size = 0;
        this->buffer_enable = _buffer_enable;
        printf("IPC Enable = %d\n", isEnable(BufferEnable::IPC));
        printf("HDF5 Enable = %d\n", isEnable(BufferEnable::HDF5));
    }
    // 全部 push 完成后，需要调用 initialize
    // 返回DDT分配的handle，作为新的recordId
    int push_back(const VarDescriptor& record_var, RecordPoint &recordPoint) {
        auto handle = bufferTable.add_or_update(
            record_var,
            BufferItem(mode, buffer_capacity, recordPoint.var_ptr_cpu, recordPoint.var_ptr_gpu),
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

        // 重置每个BufferItem的状态
        auto bufferItems = bufferTable.get_cpu_data();
        auto buffer_num = bufferTable.size();
        for (int i = 0; i < buffer_num; i++) {
            bufferItems[i].len = 0;  // 重置每个item的长度
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
            bufferItems[i].log_data_single();
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
        for (int i = 0; i < buffer_count; i++) {
            auto & dataset = (*output_buffers)[i].dataset;
            auto current_size = dataset.getSpace().getDimensions()[0];
            // 扩展数据集大小，写入当前 buffer 中收集的数据
            dataset.resize({ current_size + buffer_size });
            dataset.select({ current_size }, { (unsigned long)buffer_size }).write(record_buffers[i].buffer.get_cpu_data());
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
            auto buffer_ptr = record_buffers[i].buffer.get_cpu_data();
            std::copy(buffer_ptr, buffer_ptr + buffer_size, ipc_buffer.data() + current_size);
        }
    }

    void flush_cpu() {
        if(buffer_size <= 0) return;
        put_data_to_hdf5();
        put_data_to_ipc_buf();
        auto buffer_num = bufferTable.size();
        auto bufferItems = bufferTable.get_cpu_data();
        for (int i = 0; i < buffer_num; i++) {
            assert(bufferItems[i].len == buffer_size && "BufferItem length should match buffer_size");
            bufferItems[i].flush();
        }
        // printf("Flushed %zu items to HDF5 and IPC buffers.\n", buffer_size);
        buffer_size = 0;
    }

    void log_data_gpu();
    void flush_gpu();
};
