#include "ion_table.h"
#include "var_struct.cuh"
using namespace std;
static unordered_map<string,EionData> EionTable;

void reg_ion(Mode mode, const std::string& name,double **cpu_data,double **gpu_data,int nnodes){
    auto &eionData = EionTable[name];
    eionData.nnodes = nnodes;
    eionData.cpu_data = cpu_data;
    if(mode == Mode::GPU){
        int Eion_Var_Size = magic_enum::enum_count<EionVarNames>();
        eionData.gpu_data_oh_host = new double*[Eion_Var_Size];
        cudaMemcpy(eionData.gpu_data_oh_host,gpu_data,Eion_Var_Size*sizeof(double*),cudaMemcpyDeviceToHost);
    }

}
EionData& get_ion_vars(const std::string& name,bool must_exist){
    if(must_exist){
        assert(EionTable.find(name) != EionTable.end());
    }
    return EionTable[name];
}
