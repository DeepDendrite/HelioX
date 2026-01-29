#pragma once

#include "mechanism.h"

class HHMech : public Mechanism
{
    public:
        bool usetable = false;
        HHMech(MechInitParams &param);
		virtual ~HHMech();
        void reg_node_indices(MechInitParams &param);
        void sync_gpu();
        void initialize_cpu(SimMechInitialParam &param);
        void initialize_gpu(SimMechInitialParam &param);
        void current_cpu(SimMechCurrentParam &param); 
        void current_gpu(SimMechCurrentParam &param); 
        void state_cpu(SimMechStateParam &param);
        void state_gpu(SimMechStateParam &param);
        void read_data_from_coredat(MechInitParams &param);

    private:
        VecData<double> *vecdata_m, *vecdata_n, *vecdata_h; //state variables
        VecData<double> *vecdata_ina, *vecdata_ik, *vecdata_il;
        VecData<double> *vecdata_gnabar, *vecdata_gkbar, *vecdata_gl, *vecdata_gna, *vecdata_gk; //RANGE variables
        VecData<double> *vecdata_minf, *vecdata_hinf, *vecdata_ninf, *vecdata_mtau, *vecdata_htau, *vecdata_ntau;
        VecData<double> *vecdata_ena, *vecdata_ek, *vecdata_el;
        void rates_cpu(double v, int i);
        double cal_current_cpu(double v, int node_index);
        double vtrap_cpu(double x, double y);
        //table相关
        
        struct Table {
            unique_ptr<VecData<double>> vecdata_table_minf, vecdata_table_mtau, vecdata_table_hinf, vecdata_table_htau, vecdata_table_ninf, vecdata_table_ntau;
        };
        Table lut = {};
        bool needMakeTable = true;
        //FROM -100 TO 100 WITH 200
        double tmin = -100;
        double tmax = 100;
        int table_size = 200;
        double mfac_rates;
        void make_table();
        void table_rates_cpu(double v, int i);
};

REGISTER_MECHANISM("hh-org",HHMech);