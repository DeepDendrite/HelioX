#include "global_vars.h"
#include <cassert>
#include <string.h>

using namespace std;
namespace coreneuron
{
    std::unordered_map<std::string, std::vector<double>> global_var_map;
    int secondorder = 0;
}

void read_global_dat(const char *datapath)
{
    string file_name = string(datapath) + string("/globals.dat");

    FILE *f = fopen(file_name.c_str(), "r");
    if (f == nullptr)
    {
        printf("Error opening file: %s\n", file_name.c_str());
        assert(false);
        return;
    }
    char line[256];
    assert(fscanf(f, "%s\n", line) == 1); // 这一行是版本号

    while (true)//第一部分，读到“0 0”才结束
    {
        char name[256];
        double val;
        int n;
        assert(fgets(line, 256, f) != nullptr);//读一行到缓冲区
        
        if (sscanf(line, "%s %lf", name, &val) == 2)//普通的标量类型数据
        {
            if (strcmp(name, "0") == 0)
            {
                break;
            }
            std::vector<double> item(1,val);//分配一个标量
            coreneuron::global_var_map.insert({name, item});//插入到map中
            // printf("name:%s val:%lf\n", name, val);
        }
        else if (sscanf(line, "%[^[][%d]\n", name, &n) == 2)//形如"name[m]"的数组
        {
            if (strcmp(name, "0") == 0)
            {
                break;
            }
            std::vector<double> item(n);//分配一个数组

            for (int i = 0; i < n; ++i)
            {
                assert(fgets(line, 256, f) != nullptr);//读一行到缓冲区
                assert(sscanf(line, "%lf\n", &val) == 1);//读入数据，确保只有一个数据
                item[i] = val;//赋值
            }
            coreneuron::global_var_map.insert({name, item});//插入到map中
            // printf("name:%s val:", name);
        }
        else
        {
            assert(false&&"Error globals.dat format\n");//格式错误
        }
    }

    //第二部分
    while (fgets(line, 256, f)) {
        char name[256];
        int n;
        if (sscanf(line, "%s %d", name, &n) == 2) {
            if (strcmp(name, "secondorder") == 0) {
                coreneuron::secondorder = n;
            } else if (strcmp(name, "Random123_globalindex") == 0) {
                //TODO:暂时没有实现
                //nrnran123_set_globalindex((uint32_t) n);
            }
        }
    }

    fclose(f);
    printf("Read global.dat file: %s\n", file_name.c_str());
    printf("global var map size: %zu\n", coreneuron::global_var_map.size());
    //这部分先不用实现
    // overwrite global.dat config if seed is specified on Command line
    // if (cli_global_seed) {
    //     nrnran123_set_globalindex((uint32_t) cli_global_seed_value);
    // }
}