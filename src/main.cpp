#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "neuron.h"
#include "mechanism.h"
#include "simulate.h"
#include "utils.h"
#include "coredat_structs.h"
#include "permute_order.h"
#include "read_coredat.h"
#include "cxxopts.hpp"
#include "coredat_to_innerdat.h"
#include "magic_enum/magic_enum.hpp"
#include "global_vars.h"
using namespace std;

extern int nthread_each_cell;

// 修改后的 MonitorParser::parse 函数，支持任意长度的点位输入
// 修改后的 MonitorParser::parse 函数，直接通过引用修改外部 monitors，并支持格式：mechname:var1:idx1:...:idxn:var2:idx1:...
namespace MonitorParser
{
    std::vector<std::string> split(const std::string &input, char delimiter)
    {
        std::vector<std::string> tokens;
        std::istringstream iss(input);
        std::string token;
        while (std::getline(iss, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    // 新的解析函数：直接修改传入的 monitors 向量，不返回拷贝
    void parse(const std::string &input, std::vector<VarDescriptor>& monitors, char delimiter = ':')
    {
        auto tokens = split(input, delimiter);
        if (tokens.size() < 3)
        {
            throw std::runtime_error("格式错误: 输入必须至少包含 mechname 以及至少一组 var:idx");
        }
        std::string mech = tokens[0];
        size_t i = 1;
        while (i < tokens.size())
        {
            // 下一个 token 视为 var 名称
            std::string var = tokens[i++];
            bool foundIndex = false;
            // 后续连续的整数 token 视为该 var 的索引
            while (i < tokens.size())
            {
                try {
                    int idx = std::stoi(tokens[i]);
                    VarDescriptor mp;
                    mp.mech = mech;
                    mp.var = var;
                    mp.node_or_mech_idx = idx;
                    
                    // 检查是否已存在
                    bool exists = false;
                    for (const auto& existing : monitors) {
                        if (existing == mp) {
                            exists = true;
                            break;
                        }
                    }
                    
                    if (!exists) {
                        monitors.push_back(mp);
                    }
                    foundIndex = true;
                    i++;
                }
                catch (const std::exception &)
                {
                    // 遇到非整数的 token，则认为是下一个 var 名称，退出当前循环
                    break;
                }
            }
            if (!foundIndex)
            {
                throw std::runtime_error("格式错误: var " + var + " 后面必须跟至少一个整数索引");
            }
        }
    }
}


int main(int argc, char *argv[])
{
    cxxopts::Options options("neurong", "A Simple Neuron Simulator");
    options.add_options()
        ("d,dir", "Model Dir", cxxopts::value<std::string>())
        ("m,mode", "mode, gpu/cpu", cxxopts::value<std::string>()->default_value("cpu"))
        ("p,permute_type", "permute_type,0-3", cxxopts::value<int>()->default_value("0"))
        ("t,tstop", "tstop", cxxopts::value<double>()->default_value("100"))
        ("dt", "dt", cxxopts::value<double>()->default_value("0.025"))
        ("mod_num", "mod_num, neurong old models set to 100, otherwise, 1000", cxxopts::value<int>()->default_value("1000"))
        ("nthread_each_cell", "thread budget per cell when splitting (permute_type=3, <=32)", cxxopts::value<int>())
        ("h,help", "Print usage")
        ("init_volt", "init voltage",cxxopts::value<double>()->default_value("-65"))
        ("w,watch_point", "监视点，格式为 Mech:Var:node_idx。可以重复指定该选项",cxxopts::value<std::vector<std::string>>())
        ("ipc_port", "进入IPC模式，需要输入端口号，如 --ipc_port 5555",cxxopts::value<int>())
        ("o,output_dir","输出目录",cxxopts::value<std::string>()->default_value("output"))
        ("disable_spk_output", "关闭spk写文件功能", cxxopts::value<bool>()->default_value("false"))
        ;
    auto arg_result = options.parse(argc, argv);

    if (arg_result.count("help") || arg_result.count("dir") == 0)
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // 存储所有解析后的监视点
    std::vector<VarDescriptor> monitors;
    if (arg_result.count("watch_point")) {
        auto monitorStrings = arg_result["watch_point"].as<std::vector<std::string>>();
        for (const auto& mstr : monitorStrings) {
            try {
                MonitorParser::parse(mstr, monitors);
            } catch (const std::exception& e) {
                std::cerr << "解析监视点 " << mstr << " 失败: " << e.what() << std::endl;
            }
        }
    }
    
    // 输出解析后的监视点信息
    for (const auto& mp : monitors) {
        std::cout << "监视点 - Mech: " << mp.mech
                  << ", Var: " << mp.var
                  << ", node_idx: " << mp.node_or_mech_idx << std::endl;
    }

    clock_t start_time, end_time;
    // string str_mod(argv[2]);
    Mode mode;
    if (arg_result["mode"].as<string>() == "gpu")
    {
        mode = GPU;
        permute_type = 3;
    }
    else{
        mode = CPU;
        permute_type = 0;
    }

    // permute_type = arg_result["permute_type"].as<int>();
    if(arg_result.count("permute_type") == 0){
        printf("permute_type is not set, default to %d\n", permute_type);
    }else{
        if(permute_type < 0 || permute_type > 3){
            printf("permute_type should be in [0,3], set to default %d\n",permute_type);
        }else{
            permute_type = arg_result["permute_type"].as<int>();
        }
    }

    if (arg_result.count("nthread_each_cell")) {
        const int v = arg_result["nthread_each_cell"].as<int>();
        if (v <= 0 || v > 32) {
            printf("nthread_each_cell should be in [1,32], keep default %d\n", nthread_each_cell);
        } else {
            nthread_each_cell = v;
            printf("nthread_each_cell set to %d\n", nthread_each_cell);
        }
    }

    printf_debug("str_mod = %s\n", arg_result["mode"].as<string>().c_str());


    // read data from CoreNEURON data files
    //  coreneuron::CoreData** coredata_arr;
    unique_ptr<coreneuron::CoreData *[]> coredata_arr;
    int ngroup = 0;
    string datapath = arg_result["dir"].as<string>();
    string filesdat = datapath + "/files.dat";

    ngroup = read_coredat(coredata_arr, datapath.c_str(), filesdat.c_str(), 0, false, arg_result["mod_num"].as<int>());


    using enum BufferEnable;
    BufferEnable buffer_enable = NONE;
    if (arg_result.count("watch_point")) {//有监视点，才开启HDF5
        buffer_enable = buffer_enable | HDF5;
    }

    Simulate *sim = new Simulate(mode, buffer_enable);
    sim->output_folder = arg_result["output_dir"].as<string>();
    sim->tstop = arg_result["tstop"].as<double>();
    sim->dt = coreneuron::global_var_map.at("dt")[0];
    if(arg_result.count("dt")){
        sim->dt = arg_result["dt"].as<double>();
    }

    data_format_trans(sim->neuron_group_list, coredata_arr, ngroup, mode,sim->dt);

    printf_debug("============================\n");
    for (int i = 0; i < ngroup; i++)
    {
        delete coredata_arr[i];
    }

    printf("load data finish\n");

    sim->permute_type = permute_type;

    printf("\n");
    printf("Initialize\n");

    double init_volt = arg_result["init_volt"].as<double>();

    auto monitor_to_handle = sim->init_monitor_data_sets(monitors);
    sim->finitialize(init_volt);

    printf("simulation start\n");
    start_time = clock();
    sim->run();
    end_time = clock();
    printf("simulation finished, time:%fs\n", ((float)end_time - start_time) / CLOCKS_PER_SEC);

    bool disable_spk_output = arg_result["disable_spk_output"].as<bool>();
    if (!disable_spk_output) {
        sim->output_spikes();
    } else {
        printf("Spike file output disabled by option\n");
    }

    delete sim;
    delete[] permute_info_arr;
    printf("delete simulation\n");
    return 0;
}
