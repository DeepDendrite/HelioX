#include "hh.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include "cuda_utils.h"
#include <cassert>


HHMech::HHMech(MechInitParams &param) : Mechanism(param)
{
    //gnabar = 0.25;
    //gl = 0.00016666;
    //el = -60.0;
    //gkbar = 0.036;
    //ena = 50;
    //ek = -77;
    // celsius = 6.2999999999999998224;//6.3
    
}

HHMech::~HHMech()
{
	delete vecdata_ina ;
	delete vecdata_ik	 ;
	delete vecdata_il	 ;
	delete vecdata_gna ;
	delete vecdata_gk	 ;
	delete vecdata_minf;
	delete vecdata_hinf;
	delete vecdata_ninf;
	delete vecdata_mtau;
	delete vecdata_htau;
	delete vecdata_ntau;
	delete vecdata_m	 ;
	delete vecdata_n	 ;
	delete vecdata_h	 ;
    delete vecdata_gnabar;
    delete vecdata_gl;
    delete vecdata_gkbar;
    delete vecdata_ena;
    delete vecdata_ek;
    delete vecdata_el;

	vecdata_ina = nullptr;
	vecdata_ik = nullptr;
	vecdata_il = nullptr;
	vecdata_gna = nullptr;
	vecdata_gk = nullptr;
	vecdata_minf = nullptr;
	vecdata_hinf = nullptr;
	vecdata_ninf = nullptr;
	vecdata_mtau = nullptr;
	vecdata_htau = nullptr;
	vecdata_ntau = nullptr;
	vecdata_m = nullptr;
	vecdata_n = nullptr;
	vecdata_h = nullptr;
    vecdata_gnabar = nullptr;
    vecdata_gl = nullptr;
    vecdata_gkbar = nullptr;
    vecdata_ena = nullptr;
    vecdata_ek = nullptr;
    vecdata_el = nullptr;

    
}

void HHMech::reg_node_indices(MechInitParams &param)
{
    auto node_count = param.node_count;
    vecdata_m = new VecData<double>(mode, 0.0, node_count);
    vecdata_n = new VecData<double>(mode, 0.0, node_count);
    vecdata_h = new VecData<double>(mode, 0.0, node_count);
    vecdata_ina = new VecData<double>(mode, 0.0, node_count);
    vecdata_ik = new VecData<double>(mode, 0.0, node_count);
    vecdata_il = new VecData<double>(mode, 0.0, node_count);
    vecdata_gna = new VecData<double>(mode, 0.0, node_count);
    vecdata_gk = new VecData<double>(mode, 0.0, node_count);
    vecdata_minf = new VecData<double>(mode, 0.0, node_count);
    vecdata_hinf = new VecData<double>(mode, 0.0, node_count);
    vecdata_ninf = new VecData<double>(mode, 0.0, node_count);
    vecdata_mtau = new VecData<double>(mode, 0.0, node_count);
    vecdata_htau = new VecData<double>(mode, 0.0, node_count);
    vecdata_ntau = new VecData<double>(mode, 0.0, node_count);
    vecdata_gnabar = new VecData<double>(mode, 0.25, node_count);
    vecdata_gkbar = new VecData<double>(mode, 0.036, node_count);
    vecdata_gl = new VecData<double>(mode, 0.00016666, node_count);
    vecdata_el = new VecData<double>(mode, -60.0, node_count);
    vecdata_ena = new VecData<double>(mode, 50.0, node_count);
    vecdata_ek = new VecData<double>(mode, -77.0, node_count);


    //gnabar = 0.25;
    //gl = 0.00016666;
    //el = -60.0;
    //gkbar = 0.036;
    //ena = 50;
    //ek = -77;
    
}

void HHMech::read_data_from_coredat(MechInitParams &param)
{
    auto param_size = param.data_size;
    auto data = param.data;
    double* gnabar = this->vecdata_gnabar->get_cpu_data();
    double* gkbar = this->vecdata_gkbar->get_cpu_data();
    double* gl = this->vecdata_gl->get_cpu_data();
    double* el =this->vecdata_el->get_cpu_data();
    double* gna = this->vecdata_gna->get_cpu_data();
    double* gk = this->vecdata_gk->get_cpu_data();
    double* il = this->vecdata_il->get_cpu_data();
    double* minf = this->vecdata_minf->get_cpu_data();
    double* hinf = this->vecdata_hinf->get_cpu_data();
    double* ninf = this->vecdata_ninf->get_cpu_data();
    double* mtau = this->vecdata_mtau->get_cpu_data();
    double* htau = this->vecdata_htau->get_cpu_data();
    double* ntau = this->vecdata_ntau->get_cpu_data();
    double* m = this->vecdata_m->get_cpu_data();
    double* h = this->vecdata_h->get_cpu_data();
    double* n = this->vecdata_n->get_cpu_data();
    double* ena = this->vecdata_ena->get_cpu_data(); //19
    double* ek = this->vecdata_ek->get_cpu_data(); //20

    // printf_debug("HHMech::read_data_from_coredat\n");
    // printf_debug("nnode=%d this->nnode=%d param_size=%d\n",nnode,this->nnode,param_size);

    for (int inode = 0; inode < nnode; inode++)
    {
        gnabar[inode] = data[inode * param_size + 0];
        gkbar[inode] = data[inode * param_size + 1];
        gl[inode] = data[inode * param_size + 2];
        el[inode] = data[inode * param_size + 3];
        gna[inode] = data[inode * param_size + 4];
        gk[inode] = data[inode * param_size + 5];
        il[inode] = data[inode * param_size + 6];
        minf[inode] = data[inode * param_size + 7];
        hinf[inode] = data[inode * param_size + 8];
        ninf[inode] = data[inode * param_size + 9];
        mtau[inode] = data[inode * param_size + 10];
        htau[inode] = data[inode * param_size + 11];
        ntau[inode] = data[inode * param_size + 12];
        ena[inode] = data[inode * param_size + 19];
        ek[inode] = data[inode * param_size + 20];
        // printf_debug("mtau[%d]=%lf\n",inode,mtau[inode]);
    }
    
    if (mode == GPU)
    {
        this->vecdata_gnabar->update_gpu_data_from_cpu();
        this->vecdata_gkbar->update_gpu_data_from_cpu();
        this->vecdata_gl->update_gpu_data_from_cpu();
        this->vecdata_el->update_gpu_data_from_cpu();
        this->vecdata_gna->update_gpu_data_from_cpu();
        this->vecdata_gk->update_gpu_data_from_cpu();
        this->vecdata_il->update_gpu_data_from_cpu();
        this->vecdata_minf->update_gpu_data_from_cpu();
        this->vecdata_hinf->update_gpu_data_from_cpu();
        this->vecdata_ninf->update_gpu_data_from_cpu();
        this->vecdata_mtau->update_gpu_data_from_cpu();
        this->vecdata_htau->update_gpu_data_from_cpu();
        this->vecdata_ntau->update_gpu_data_from_cpu();
        this->vecdata_ena->update_gpu_data_from_cpu();
        this->vecdata_ek->update_gpu_data_from_cpu();
    }
}

void HHMech::initialize_cpu(SimMechInitialParam &param)
{
    double* v = param.v;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* m = this->vecdata_m->get_cpu_data();
    double* h = this->vecdata_h->get_cpu_data();
    double* n = this->vecdata_n->get_cpu_data();
    double* minf = this->vecdata_minf->get_cpu_data();
    double* hinf = this->vecdata_hinf->get_cpu_data();
    double* ninf = this->vecdata_ninf->get_cpu_data();
    double* mtau = this->vecdata_mtau->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        rates_cpu(v[node_index], i);
        m[i] = minf[i];
        h[i] = hinf[i];
        n[i] = ninf[i];

        printf("init_single_node\n");
        // printf("i:%d m:%f h:%f n:%f\n",i,dev_vars[m][i],dev_vars[h][i],dev_vars[n][i]);
        printf("i:%d m:%f h:%f n:%f\n",i,m[i],h[i],n[i]);
    }
    printf_debug("HHMech::initialize_cpu\n");

}

void HHMech::current_cpu(SimMechCurrentParam &param)
{
    double *vec_v = param.v;
    double *vec_rhs = param.rhs;
    double *vec_d = param.d;
    double _rhs, _g, _v;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* ina = this->vecdata_ina->get_cpu_data();
    double* ik = this->vecdata_ik->get_cpu_data();
    for (int i = 0; i < nnode; i++)
    {
        int node_index = node_indices[i];
        _v = vec_v[node_index];
        _g = cal_current_cpu(_v + 0.001, i);
        // double _dik, _dina;//这部分注释掉的代码是原版的mech里面的，用于监控相关变量
        // _dina = ina[i];
        // _dik = ik[i];
        //???TODO:原版的mech里面有莫名其妙的算微分的操作
        _rhs = cal_current_cpu(_v, i);
        _g = (_g - _rhs) / 0.001;
        vec_rhs[node_index] -= _rhs;
        vec_d[node_index] += _g;
    }
}

double HHMech::cal_current_cpu(double v, int mech_index)
{
    double* _gna = this->vecdata_gna->get_cpu_data();
    double* _gnabar = this->vecdata_gnabar->get_cpu_data();
    double* _m = this->vecdata_m->get_cpu_data();
    double* _h = this->vecdata_h->get_cpu_data();
    double* _ina = this->vecdata_ina->get_cpu_data();
    double* _ena = this->vecdata_ena->get_cpu_data();
    double* _gk = this->vecdata_gk->get_cpu_data();
    double* _gkbar = this->vecdata_gkbar->get_cpu_data();
    double* _n = this->vecdata_n->get_cpu_data();
    double* _ik = this->vecdata_ik->get_cpu_data();
    double* _ek = this->vecdata_ek->get_cpu_data();
    double* _il = this->vecdata_il->get_cpu_data();
    double* _gl = this->vecdata_gl->get_cpu_data();
    double* _el = this->vecdata_el->get_cpu_data();
    double* _i_mech = this->vecdata_i_mech->get_cpu_data();
    
    auto &gna = _gna[mech_index];
    auto &gnabar = _gnabar[mech_index];
    auto &m = _m[mech_index];
    auto &h = _h[mech_index];
    auto &ina = _ina[mech_index];
    auto &ena = _ena[mech_index];
    auto &gk = _gk[mech_index];
    auto &gkbar = _gkbar[mech_index];
    auto &n = _n[mech_index];
    auto &ik = _ik[mech_index];
    auto &ek = _ek[mech_index];
    auto &il = _il[mech_index];
    auto &gl = _gl[mech_index];
    auto &el = _el[mech_index];
    auto &i_mech = _i_mech[mech_index];

    static int debug_count = 0;

    double _current = 0.;
    gna = gnabar * m * m * m * h;
    ina = gna * (v - ena);
    gk = gkbar * n * n * n * n;
    ik = gk * (v - ek);
    il = gl * (v - el);
    _current += ina;
    _current += ik;
    _current += il;
    return _current;
}
double inline hoc_Exp(double x) {
    if constexpr(1){
        return exp(x);
    }
    //下面是原版的hoc_Exp,但是，真的会出现|x|>700的情况吗？
    if (x < -700.) {
        return 0.;
    } else if (x > 700) {
        return exp(700.);
    }
    return exp(x);
}
/*
    这个函数的输入是电压v和节点编号i，以及内部变量celsius
    修改了mtau, minf, hinf, htau, ninf, ntau这6个变量
    与原始的mtau, minf, hinf, htau, ninf, ntau这6个变量无关
*/
void HHMech::rates_cpu(double v, int i)
{
    double* mtau = this->vecdata_mtau->get_cpu_data();
    double* minf = this->vecdata_minf->get_cpu_data();
    double* hinf = this->vecdata_hinf->get_cpu_data();
    double* htau = this->vecdata_htau->get_cpu_data();
    double* ninf = this->vecdata_ninf->get_cpu_data();
    double* ntau = this->vecdata_ntau->get_cpu_data();
    double alpha, beta, sum, q10;
    q10 = pow(3, (celsius - 6.3) / 10.0);
    alpha = 0.1 * vtrap_cpu(-(v + 40.0), 10.0);
    beta = 4.0 * hoc_Exp(-(v + 65.0) / 18.0);
    sum = alpha + beta;
    mtau[i] = 1.0 / (q10 * sum);
    minf[i] = alpha / sum;
    
    alpha = 0.07 * hoc_Exp(-(v + 65.0) / 20.0);
    beta = 1.0 / (hoc_Exp(-(v + 35.0) / 10.0) + 1.0);
    sum = alpha + beta;
    htau[i] = 1.0 / (q10 * sum);
    hinf[i] = alpha / sum;

    alpha = 0.01 * vtrap_cpu(-(v + 55.0), 10.0);
    beta = 0.125 * hoc_Exp(-(v + 65.0) / 80.0);
    sum = alpha + beta;
    ntau[i] = 1.0 / (q10 * sum);
    ninf[i] = alpha / sum;
}

double HHMech::vtrap_cpu(double x, double y)
{
    if (fabs(x / y) < 1e-6)
    {
        return y * (1.0 - x / y / 2.0);
    }
    else
    {
        return x / (hoc_Exp(x / y) - 1.0);
    }
}

void HHMech::state_cpu(SimMechStateParam &param)
{
    double *vec_v = param.v;
    double dt = param.dt;
    int* node_indices = this->vecdata_node_indices->get_cpu_data();
    double* m = this->vecdata_m->get_cpu_data();
    double* h = this->vecdata_h->get_cpu_data();
    double* n = this->vecdata_n->get_cpu_data();
    double* mtau = this->vecdata_mtau->get_cpu_data();
    double* minf = this->vecdata_minf->get_cpu_data();
    double* hinf = this->vecdata_hinf->get_cpu_data();
    double* htau = this->vecdata_htau->get_cpu_data();
    double* ninf = this->vecdata_ninf->get_cpu_data();
    double* ntau = this->vecdata_ntau->get_cpu_data();
    double v;
    int node_index;
    
    auto rates = usetable ? &HHMech::table_rates_cpu : &HHMech::rates_cpu;

    for (int i = 0; i < nnode; i++)
    {
        node_index = node_indices[i];
        v = vec_v[node_index];
        (this->*rates)(v, i);
		
		m[i] = m[i] + (1.0 - hoc_Exp(dt*((((-1.0))) / mtau[i])))*(-(((minf[i])) / mtau[i]) / ((((-1.0))) / mtau[i]) - m[i]);
		h[i] = h[i] + (1.0 - hoc_Exp(dt*((((-1.0))) / htau[i])))*(-(((hinf[i])) / htau[i]) / ((((-1.0))) / htau[i]) - h[i]);
		n[i] = n[i] + (1.0 - hoc_Exp(dt*((((-1.0))) / ntau[i])))*(-(((ninf[i])) / ntau[i]) / ((((-1.0))) / ntau[i]) - n[i]);
    }
}

void HHMech::make_table()
{
    printf_debug("HHMech::make_table\n");
    auto check_and_create_vecdata = [this](unique_ptr<VecData<double>> &vecdata)
    {
        vecdata = make_unique<VecData<double>>(mode, table_size + 1);
    };
    check_and_create_vecdata(lut.vecdata_table_minf);
    check_and_create_vecdata(lut.vecdata_table_mtau);
    check_and_create_vecdata(lut.vecdata_table_hinf);
    check_and_create_vecdata(lut.vecdata_table_htau);
    check_and_create_vecdata(lut.vecdata_table_ninf);
    check_and_create_vecdata(lut.vecdata_table_ntau);

    double* table_minf = lut.vecdata_table_minf->get_cpu_data();
    double* table_mtau = lut.vecdata_table_mtau->get_cpu_data();
    double* table_hinf = lut.vecdata_table_hinf->get_cpu_data();
    double* table_htau = lut.vecdata_table_htau->get_cpu_data();
    double* table_ninf = lut.vecdata_table_ninf->get_cpu_data();
    double* table_ntau = lut.vecdata_table_ntau->get_cpu_data();

    double* cpu_minf = this->vecdata_minf->get_cpu_data();
    double* cpu_mtau = this->vecdata_mtau->get_cpu_data();
    double* cpu_hinf = this->vecdata_hinf->get_cpu_data();
    double* cpu_htau = this->vecdata_htau->get_cpu_data();
    double* cpu_ninf = this->vecdata_ninf->get_cpu_data();
    double* cpu_ntau = this->vecdata_ntau->get_cpu_data();

    double dv = (tmax - tmin) / (double)table_size;
    mfac_rates = 1. / dv;
    int i;
    double v;
    for(i=0,v=tmin;i<table_size + 1;i++,v+=dv) {
        rates_cpu(v, 0);
        table_minf[i] = cpu_minf[0];
        table_mtau[i] = cpu_mtau[0];
        table_hinf[i] = cpu_hinf[0];
        table_htau[i] = cpu_htau[0];
        table_ninf[i] = cpu_ninf[0];
        table_ntau[i] = cpu_ntau[0];
    }
    printf_debug("HHMech::make_table end\n");
    return;
}

void HHMech::table_rates_cpu(double v, int idx)
{
    if(needMakeTable) {
        make_table();
        needMakeTable = false;
    }
    
    double* table_minf = lut.vecdata_table_minf->get_cpu_data();
    double* table_mtau = lut.vecdata_table_mtau->get_cpu_data();
    double* table_hinf = lut.vecdata_table_hinf->get_cpu_data();
    double* table_htau = lut.vecdata_table_htau->get_cpu_data();
    double* table_ninf = lut.vecdata_table_ninf->get_cpu_data();
    double* table_ntau = lut.vecdata_table_ntau->get_cpu_data();

    double* cpu_minf = this->vecdata_minf->get_cpu_data();
    double* cpu_mtau = this->vecdata_mtau->get_cpu_data();
    double* cpu_hinf = this->vecdata_hinf->get_cpu_data();
    double* cpu_htau = this->vecdata_htau->get_cpu_data();
    double* cpu_ninf = this->vecdata_ninf->get_cpu_data();
    double* cpu_ntau = this->vecdata_ntau->get_cpu_data();

    double *src_list[] = {table_minf, table_mtau, table_hinf, table_htau, table_ninf, table_ntau};
    double *dst_list[] = {cpu_minf, cpu_mtau, cpu_hinf, cpu_htau, cpu_ninf, cpu_ntau};
    constexpr int list_len = sizeof(src_list) / sizeof(src_list[0]);
    assert(list_len == sizeof(dst_list) / sizeof(dst_list[0]));
    //如果v是nan，那么结果也是nan，不用计算
    double idx_f = (v - tmin)* mfac_rates;

    if(std::isnan(idx_f)){
        //nan的话，那结果也设为nan
        for(auto dst:dst_list){
            dst[idx] = idx_f;
        }
        return;
    }
    //如果越界，那就用边界值
    if(idx_f <= 0.0){
        for(int i = 0; i < list_len; i++){
            dst_list[i][idx] = src_list[i][0];
        }
        return;
    }
    if(idx_f >= table_size){
        for(int i = 0; i < list_len; i++){
            dst_list[i][idx] = src_list[i][table_size];
        }
        return;
    }
    //线性插值
    int table_idx = (int)idx_f;
    double theta = idx_f - (double)table_idx;
    for(int i = 0; i < list_len; i++){
        dst_list[i][idx] = src_list[i][table_idx] + theta * (src_list[i][table_idx + 1] - src_list[i][table_idx]);
    }
    return;
}
