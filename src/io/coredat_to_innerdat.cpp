#include "coredat_to_innerdat.h"
#include "permute_order.h"
#include "mechanism.h"
#include "mech_var_table.h"
#include "legacy_index_utils.h"
#include <set>
#include "vecevent.h"
#include "netstim.h"
#include <algorithm>
#include <cstdlib>

extern vector<coreneuron::NetCon *> netcon_in_presyn_order;
void setup_permute_info(HelioXroupData *ndat, coreneuron::CoreData *cdat, PermuteInfo &pi, Mode mode)
{
    printf("setup_permute_info: permute_type=%d mode = %s\n", permute_type, magic_enum::enum_name(mode).data());
    if (permute_type == 1)
    {
        ndat->vecdata_firstnode = new VecData<int>(mode, pi.firstnode, cdat->n_real_output);
        ndat->vecdata_lastnode = new VecData<int>(mode, pi.lastnode, cdat->n_real_output);
        ndat->vecdata_cellsize = new VecData<int>(mode, pi.cellsize, cdat->n_real_output);
        ndat->vecdata_stride = new VecData<int>(mode, pi.stride, pi.nstride + 1);
        ndat->nstride = pi.nstride;
    }

    if (permute_type == 3)
    {
        ndat->vecdata_firstnode = new VecData<int>(mode, pi.firstnode, pi.threads_num);
        ndat->vecdata_lastnode = new VecData<int>(mode, pi.lastnode, pi.threads_num);
        ndat->vecdata_max_order_each_thread = new VecData<int>(mode, pi.max_order_each_thread, pi.threads_num);
        ndat->vecdata_min_order_each_thread = new VecData<int>(mode, pi.min_order_each_thread, pi.threads_num);
        ndat->vecdata_map_t2c = new VecData<int>(mode, pi.map_t2c, pi.threads_num);
        int nwarps = (pi.threads_num + 31) / 32;
        ndat->vecdata_stride = new VecData<int>(mode, pi.stride, nwarps * (pi.norder + 1));
        ndat->nstride = pi.nstride;
        ndat->norder = pi.norder;
        ndat->nwarp = nwarps;
        ndat->threads_num = pi.threads_num;
        ndat->nthread_each_cell = pi.nthread_each_cell;
    }
}

void initGroupData(Mode mode, HelioXroupData *ndat, coreneuron::CoreData *cdat)
{
    int ncell = cdat->n_real_cell;
    int len = cdat->end;
    printf("ncell=%d len=%d\n",ncell,len);

    ndat->ncell = ncell;
    ndat->len = cdat->end;
    // 直接拷贝过来的
    ndat->vecdata_a = new VecData<double>(mode, cdat->_actual_a, len);
    ndat->vecdata_b = new VecData<double>(mode, cdat->_actual_b, len);
    ndat->vecdata_area = new VecData<double>(mode, cdat->_actual_area, len);
    ndat->vecdata_parent_index = new VecData<int>(mode, cdat->_v_parent_index, len);

    // 初始化新的向量，但是没赋值的
    ndat->vecdata_d = new VecData<double>(mode, 0.0, len);
    ndat->vecdata_rhs = new VecData<double>(mode, 0.0, len);
    ndat->vecdata_v = new VecData<double>(mode, 0.0, len);

    // Enable fast-imem buffers only if the exported model contains POINTERs targeting i_membrane_.
    bool need_fast_imem = false;
    for (coreneuron::CoreMech* ml = cdat->mech_list; ml; ml = ml->next)
    {
        for (int t : ml->pointer2type)
        {
            if (t == static_cast<int>(coreneuron::gap_idx_type::i_membrane_))
            {
                need_fast_imem = true;
                break;
            }
        }
        if (need_fast_imem)
        {
            break;
        }
    }
    ndat->need_fast_imem = need_fast_imem;
    if (need_fast_imem)
    {
        ndat->vecdata_sav_rhs = new VecData<double>(mode, 0.0, len);
        ndat->vecdata_sav_d = new VecData<double>(mode, 0.0, len);
        ndat->vecdata_i_membrane_ = new VecData<double>(mode, 0.0, len);
    }

    ndat->spk_vec = new SpikeVector(cdat->n_presyn);
    ndat->vecdata_spk_flags = new VecData<SpikeFlag>(mode, SpikeFlag::INVALID, cdat->n_presyn);

    // Mech的标号映射的数组
    uint32_t mech_num = cdat->mech_data->nmech_type;
    ndat->type2postsyn_ix = new int[mech_num];
    ndat->type2articell_ix = new int[mech_num];
    memset(ndat->type2postsyn_ix, -1, sizeof(int) * mech_num);
    memset(ndat->type2articell_ix, -1, sizeof(int) * mech_num);
}

void initMechList(Mode mode, HelioXroupData *ndat, coreneuron::CoreData *cdat)
{
    // 将CoreNeuron中读出来的Mech给转换成HelioX内部的Mech形式
    //  ml: mechanism from CoreNEURON data, mech: mechanism of HelioX
    MechInitParams mechParam = {.mode = mode};
    bool mech_not_found = false;
    vector<string> not_found_mech_list;
    for (coreneuron::CoreMech *ml = cdat->mech_list; ml; ml = ml->next)
    {
        int mech_type = ml->type;
        string mech_name = cdat->mech_data->name_vec[mech_type];

        mechParam.type = mech_type;
        mechParam.name = mech_name;
        mechParam.node_count = ml->nodecount;
        mechParam.nodeindices = ml->nodeindices;

        mechParam.data_size = cdat->mech_data->nrn_prop_param_size[mech_type];
        mechParam.data = ml->_data;
        mechParam.pdata_size = cdat->mech_data->nrn_prop_dparam_size[mech_type];
        mechParam.pdata = ml->pdata;
        mechParam.permute = ml->permute;
        mechParam.pointer2type = ml->pointer2type.empty() ? nullptr : &ml->pointer2type;
        mechParam.dparam_semantics = MechanismFactory::getInstance().getDparamSemantics(mech_name);

        if(cdat->mech_data->nrn_array_dims[mech_type].size() > 0)
            mechParam.array_dims = &cdat->mech_data->nrn_array_dims[mech_type];//这个是一个指针，指向一个vector<int>，里面存放了每个变量的维度
        else
            mechParam.array_dims = nullptr;//旧版本没有这个，所以直接置空，这样后面的代码就知道这个是旧版本

        //创建mech在这一行
        Mechanism *mech = MechanismFactory::getInstance().createMechanism(mech_name, mechParam);

        // printf_debug("trying create mech[%s] type=%d data_size=%d\n", mech_name.c_str(), mech_type, mechParam.data_size);
        // printf_debug("node Idx=");
        // for (int i = 0; i < ml->nodecount; i++)
        // {
        //     printf_debug("%d ", ml->nodeindices[i]);
        // }
        // printf_debug("\n");
        if (!mech)
        {
            printf("Can't find mech type[%d] AKA[%s], skipping\n", mech_type, mech_name.c_str());
            mech_not_found = true;
            not_found_mech_list.push_back(mech_name);
            continue;
        }
        printf_debug("processing mech[%s] type=%d\n", mech_name.c_str(), mech_type);
        printf_debug("Mech:%s n=%d szp=%d szdp=%d\n", mech_name.c_str(), mechParam.node_count, mechParam.data_size, mechParam.pdata_size);
        if (cdat->mech_data->nrn_is_artificial[mech_type]) // 如果是人工细胞
        // if (dynamic_cast<ArtiCell*>(mech))//虽然这样cast好像也成，但是nrn_is_artificial是直接从模型里面读出来的，就用这个吧
        {
            ArtiCell *articell = dynamic_cast<ArtiCell *>(mech); // ArtiCell的部分
            articell->spk_vec_bkp = ndat->spk_vec;
            articell->spk_flags_bkp = ndat->vecdata_spk_flags;
            ndat->vec_articell.push_back(articell);
            ndat->type2articell_ix[mech_type] = ndat->vec_articell.size() - 1;

            PostSyn_trait *postsyn = dynamic_cast<PostSyn_trait *>(mech); // ArtiCell同时也是postsyn，所以，也要加进去
            ndat->vec_postsyn.push_back(postsyn);
            ndat->type2postsyn_ix[mech_type] = ndat->vec_postsyn.size() - 1;
        }
        else if (dynamic_cast<PostSyn_trait *>(mech)) // 单独的PoseSyn，那么加入列表
        {
            PostSyn_trait *postsyn = dynamic_cast<PostSyn_trait *>(mech);
            ndat->vec_postsyn.push_back(postsyn);
            ndat->type2postsyn_ix[mech_type] = ndat->vec_postsyn.size() - 1;
        }
        else if (mech_name == "capacitance") // 电容
            ndat->mech_cap = dynamic_cast<Capac *>(mech);

        // 如果机制需要面积数据，创建相应的向量
        if (mech->need_area) // exp2syn,iclamp这两个会在创建的时候设置为true
        {
            mech->vecdata_area = ndat->vecdata_area;
        }
        
        if (cdat->mech_data->is_ion[mech_type]) {
            ndat->vec_eion.push_back(mech);
        } else {
            ndat->mechanism_list.push_back(mech);
        }
        // Preserve CoreNEURON mech_list order for current calculation
        ndat->mech_current_list.push_back(mech);

        if(dynamic_cast<VecEvent *>(mech)){
            ndat->mech_vecevent = dynamic_cast<VecEvent *>(mech);
        }
        
        if(mech->write_state_ion){
            ndat->mech_write_state_ion_list.push_back(mech);
        }

        

        if (cdat->mech_data->nrn_is_artificial[mech_type])
        {
            assert(dynamic_cast<ArtiCell *>(mech));
            ArtiCell *arti_cell = dynamic_cast<ArtiCell *>(mech);
            arti_cell->bbcore_read(ml->icnt, ml->dcnt, ml->iArray, ml->dArray, mode);
            printf_debug("ArtiCell[%s] bbcore_read icnt=%d dcnt=%d dArray=%X\n", mech_name.c_str(), ml->icnt, ml->dcnt, ml->dArray);
        }
    }

    if(mech_not_found){
        printf("Some mechanisms not found, please check the mechanism name\n");
        for(auto &mech_name:not_found_mech_list){
            printf("mech_name=%s\n",mech_name.c_str());
        }
        exit(1);
    }
    // 打印结果进行验证
    for (auto mech_ptr : ndat->mechanism_list)
    {
        printf("mech_name=%s\n", mech_ptr->name.c_str());
    }

    const char* order_env = std::getenv("HELIOX_FORCE_MECH_ORDER");
    if (order_env && ndat->mech_current_list.size() > 1) {
        std::string mode_str(order_env);
        if (mode_str == "type") {
            std::stable_sort(ndat->mech_current_list.begin(),
                             ndat->mech_current_list.end(),
                             [](Mechanism* a, Mechanism* b) { return a->type < b->type; });
            printf("HELIOX_FORCE_MECH_ORDER=type applied to mech_current_list\n");
        } else if (mode_str == "name") {
            std::stable_sort(ndat->mech_current_list.begin(),
                             ndat->mech_current_list.end(),
                             [](Mechanism* a, Mechanism* b) { return a->name < b->name; });
            printf("HELIOX_FORCE_MECH_ORDER=name applied to mech_current_list\n");
        }
    }
}

static void initMechPointers(HelioXroupData* ndat, coreneuron::CoreData* cdat) {
    // Resolve POINTER targets after all mechanisms exist and mech_var_table is populated.
    for (auto* mech : ndat->mechanism_list) {
        mech->resolve_pointers(ndat, cdat);
    }
    for (auto* mech : ndat->vec_eion) {
        mech->resolve_pointers(ndat, cdat);
    }
}

void initSynapse(Mode mode, HelioXroupData *ndat, coreneuron::CoreData *cdat)
{
    // 初始化synapse
    printf_debug("postsyn_n=%ld articell_n=%ld \n", ndat->vec_postsyn.size(), ndat->vec_articell.size());

    int n_arti = ndat->vec_articell.size();

    double *weights;
    double *delay;
    uint32_t npre_real = cdat->n_real_output; // 真实的presyn数量;
    vector<int> pre_node_indices;
    vector<uint32_t> real_spk_vec_offset;
    vector<int> pre_gids;
    vector<vector<uint32_t>> arti_spk_vec_offset;
    arti_spk_vec_offset.resize(n_arti);
    vector<double> threshold;
    npre_real = cdat->n_real_output;
    // for (int i = 0; i < cdat->n_presyn; i++) // npre_real，指presyn_arr中，gid大于0的个数，如果小于0，那么是人工细胞
    // {
    //     if (cdat->presyn_arr[i].gid >= 0)
    //         npre_real++;
    // }
    real_spk_vec_offset.resize(npre_real);
    pre_node_indices.resize(npre_real);
    threshold.resize(npre_real);
    pre_gids.resize(npre_real);

    // Debug logging for presyn (can be extremely verbose for large models).
    // Enable explicitly to avoid slowing down initialization and flooding stdout.
    const char* dbg_presyn_env = std::getenv("HELIOX_DEBUG_PRESYN");
    const bool debug_presyn = dbg_presyn_env && dbg_presyn_env[0] != '0';
    int debug_presyn_limit = 10;
    if (const char* lim = std::getenv("HELIOX_DEBUG_PRESYN_LIMIT"); lim && lim[0]) {
        debug_presyn_limit = std::max(0, std::atoi(lim));
    }

    int ipre = 0;
    // spk_vec: 0 -- npre_real-1 real cell, npre_real -- n_presyn arti-cell
    for (int i = 0; i < cdat->n_presyn; i++)
    {
        coreneuron::CorePreSyn *ps = cdat->presyn_arr + i;
        int gid = ps->gid;
        bool is_arti;
        if(ps->pntsrc == nullptr){
            is_arti = false;//articell一定是点过程
        }else{
            is_arti = cdat->mech_data->nrn_is_artificial[ps->pntsrc->type];
        }
        if (!is_arti) 
        {
            //实际上，在coreNeuron中，是保证了前n_real_presyn个presyn都是实际的细胞
            //但是历史遗留问题，heliox用了不同的判断方式，因此，现在用assert来确保不会出毛病
            //否则的话，现在可能会导致真实的spk vec offset和arti cell的spk vec offset冲突，这就出问题了
            assert(ipre < npre_real);
            assert(i == ipre);
            pre_node_indices[ipre] = ps->thvar_index;
            threshold[ipre] = ps->threshold;
            real_spk_vec_offset[ipre] = ipre; // Note:此时认为，所有的非人工Cell都在前面
            pre_gids[ipre] = gid;
            if (debug_presyn) {
                // Print only a small prefix/suffix to keep logs readable for large models.
                if (debug_presyn_limit <= 0 ||
                    ipre < debug_presyn_limit ||
                    ipre >= (static_cast<int>(npre_real) - debug_presyn_limit)) {
                    printf_debug("RealPresyn[%d][%d]: pre_node_indices[%d] threshold[%lf] gid[%d]\n",
                                 i, ipre, pre_node_indices[ipre], threshold[ipre], gid);
                } else if (ipre == debug_presyn_limit) {
                    printf_debug("RealPresyn: ... suppressed (npre_real=%u, set HELIOX_DEBUG_PRESYN_LIMIT to change) ...\n",
                                 npre_real);
                }
            }

            ipre++;
        }
        else // artificial cell
        {
            assert(i >= npre_real);
            int mech_type = ps->pntsrc->type;
            int index = ps->pntsrc->i_instance;
            int articell_idx = ndat->type2articell_ix[mech_type];
            arti_spk_vec_offset[articell_idx].push_back(i);
        }
    }

    // setup presyn in HelioXroupData, fill in the
    // values of threshold, pre_node_indices
    ndat->presyn = new PreSyn(mode, npre_real, ndat->spk_vec,
                              pre_node_indices, threshold,
                              real_spk_vec_offset, pre_gids);
    // printf_debug("presyn num=%d tot_n_presyn=%d\n",npre_real,cdat->n_presyn);
    // ndat->presyn->vecdata_pre_node_indices = new VecData<uint32_t>(mode, pre_node_indices);   //这几行是PreSyn构造函数里面发生的事情
    // ndat->presyn->vecdata_threshold = new VecData<double>(mode, threshold);
    // ndat->presyn->vecdata_pre_flags = new VecData<bool>(mode, false, npre_real);
    // ndat->presyn->vecdata_spk_vec_offset = new VecData<uint32_t>(mode, real_spk_vec_offset);

    // fill in the values of spk_vec_offset in all
    // artificial cells
    for (int iart = 0; iart < n_arti; iart++)
    {
        ndat->vec_articell[iart]->vecdata_spk_vec_offset = new VecData<uint32_t>(mode, arti_spk_vec_offset[iart]);
    }
    // fill in corresponding values in all post_syns
    // e.g. spk_vec_offset, delay, weight
    for (int i = 0; i < cdat->n_presyn; i++) // 遍历presyn
    {
        coreneuron::CorePreSyn &ps = cdat->presyn_arr[i];
        for (int i_nc = 0; i_nc < ps.nc_cnt; i_nc++) // 遍历当前这个presyn中的netconnect
        {
            coreneuron::NetCon *p_nc = netcon_in_presyn_order[ps.nc_index + i_nc];
            if (p_nc->active) // connected with post synapse
            {
                int mech_type = p_nc->pp->type;
                // printf("mech_type[%d]\n",mech_type);
                int i_instance = p_nc->pp->i_instance;
                // printf("i_instance[%d]\n",i_instance);
                int ipost = ndat->type2postsyn_ix[mech_type];
                // printf("ipost[%d]\n",ipost);
                auto postsyn = ndat->vec_postsyn[ipost];
                uint32_t *spk_vec_idx = postsyn->vecdata_spk_vec_idx->get_cpu_data();
                double *delay = postsyn->vecdata_delay->get_cpu_data();
                double *weights = postsyn->vecdata_weights->get_cpu_data();
                spk_vec_idx[i_instance] = i;
                delay[i_instance] = p_nc->delay;
                weights[i_instance] = cdat->weights[p_nc->weight_index];
            }
            else
            {
                // printf_debug("nc[%d] not active\n",i_nc);
            }
        }
    }

    if (mode == GPU)
    {
        for (int i = 0, npost = ndat->vec_postsyn.size(); i < npost; i++)
        {
            auto post_syn = ndat->vec_postsyn[i];
            post_syn->vecdata_spk_vec_idx->update_gpu_data_from_cpu();
            post_syn->vecdata_delay->update_gpu_data_from_cpu();
            post_syn->vecdata_weights->update_gpu_data_from_cpu();
        }
    }
}
std::tuple<double *, double *> getMechVarPtrByLagacyIndex(int mech_type,int index, coreneuron::CoreData *cdat){
    assert(mech_type>=2);
    assert(mech_var_table.contains(mech_type));
    auto &mech_var_map = mech_var_table[mech_type];
    auto [mech_i, variable_index, array_index] = legacy2soaos_index(index, cdat->mech_data->nrn_array_dims[mech_type]);
    assert(mech_var_map.contains(variable_index));

    // 获取数组大小
    int array_size = 1; // 默认标量
    if (variable_index < cdat->mech_data->nrn_array_dims[mech_type].size()) {
        array_size = cdat->mech_data->nrn_array_dims[mech_type][variable_index];
    }

    // 验证数组索引有效性
    assert(array_index >= 0 && array_index < array_size);

    // if(cdat->permute){
    //     mech_i = cdat->permute[mech_i];
    // }
    if(cdat->map_type2mechptr[mech_type]->permute){
        //如果是permute的，那么需要根据permute来获取正确的下标
        mech_i = cdat->map_type2mechptr[mech_type]->permute[mech_i];
    }
    auto &var_data = mech_var_map[variable_index];

    // 计算实际偏移：考虑数组大小和数组索引
    int actual_offset = mech_i * array_size + array_index;
    assert(var_data.len > actual_offset);

    // printf("find mechVar[%s], index[%d] array_index[%d] mech_i[%d] array_size[%d] actual_offset[%d]\n",
    //        var_data.name.c_str(), index, array_index, mech_i, array_size, actual_offset);
    return make_tuple(var_data.cpu_data + actual_offset, var_data.gpu_data + actual_offset);
}

void initGapTrans(Mode mode, HelioXroupData *ndat, coreneuron::CoreData *cdat)
{
    if(cdat->gap_transfer == nullptr){
        ndat->have_gap = false;
        return;
    }
    auto &core_gap_info = cdat->gap_transfer;
    int ntrans = core_gap_info->ntrans;
    // assert(ntrans>0);
    if(ntrans==0){
        ndat->have_gap = false;
        return;
    }
    ndat->have_gap = true;
    auto &cpu_gap = ndat->cpu_gap_trans_info;
    auto &gpu_gap = ndat->gpu_gap_trans_info;

    // 使用VecData的reserve方法预分配空间，然后逐个push_back
    cpu_gap.src.reserve(ntrans);
    cpu_gap.dst.reserve(ntrans);
    if (mode == GPU) {
        gpu_gap.src.reserve(ntrans);
        gpu_gap.dst.reserve(ntrans);
    }

    using enum coreneuron::gap_idx_type;

    // 修改set_val lambda以接受index和添加到VecData
    auto add_transfer = [cdat,ndat,mode,&cpu_gap,&gpu_gap](int src_type, int src_index, int dst_type, int dst_index){
        double* src_cpu_ptr = nullptr;
        double* dst_cpu_ptr = nullptr;
        double* src_gpu_ptr = nullptr;
        double* dst_gpu_ptr = nullptr;
        
        // 处理源指针
        if(src_type == static_cast<int>(voltage)){
            auto &vec_v = ndat->vecdata_v;
            if(cdat->permute){
                src_index = cdat->permute[src_index];
            }
            src_cpu_ptr = vec_v->get_cpu_data() + src_index;
            if(mode == GPU){
                src_gpu_ptr = vec_v->get_gpu_data() + src_index;
            }
        } else {
            assert(src_type >= 2);
            auto [cpu_ptr, gpu_ptr] = getMechVarPtrByLagacyIndex(src_type, src_index, cdat);
            src_cpu_ptr = cpu_ptr;
            if(mode == GPU){
                src_gpu_ptr = gpu_ptr;
            }
        }
        
        // 处理目标指针
        if(dst_type == static_cast<int>(voltage)){
            auto &vec_v = ndat->vecdata_v;
            if(cdat->permute){
                dst_index = cdat->permute[dst_index];
            }
            dst_cpu_ptr = vec_v->get_cpu_data() + dst_index;
            if(mode == GPU){
                dst_gpu_ptr = vec_v->get_gpu_data() + dst_index;
            }
        } else {
            assert(dst_type >= 2);
            auto [cpu_ptr, gpu_ptr] = getMechVarPtrByLagacyIndex(dst_type, dst_index, cdat);
            dst_cpu_ptr = cpu_ptr;
            if(mode == GPU){
                dst_gpu_ptr = gpu_ptr;
            }
        }
        
        // 添加到VecData
        cpu_gap.src.push_back(src_cpu_ptr);
        cpu_gap.dst.push_back(dst_cpu_ptr);
        if(mode == GPU){
            gpu_gap.src.push_back(src_gpu_ptr);
            gpu_gap.dst.push_back(dst_gpu_ptr);
        }
    };

    for(int i=0;i<ntrans;i++){
        int tar_type = core_gap_info->tar_type[i];
        int tar_sid = core_gap_info->tar_sid[i];
        int tar_index = core_gap_info->tar_index[i];
        int src_id = core_gap_info->src_sid_map[tar_sid];
        int src_index = core_gap_info->src_index[src_id];
        int src_sid = core_gap_info->src_sid[src_id];
        int src_type = core_gap_info->src_type[src_id];
        printf_debug("GapTrans[%d]: src_sid[%d] tar_sid[%d] src_type[%d] tar_type[%d] src_index[%d] tar_index[%d]\n",i,src_sid,tar_sid,src_type,tar_type,src_index,tar_index);
        
        //变量在coredat所有变量中的idx，需要经过映射，找到对应变量的对应下标
        add_transfer(src_type, src_index, tar_type, tar_index);
    }
    
    //确认dst没有重复
    std::set<double*> dst_set_cpu, dst_set_gpu;
    for(int i=0;i<cpu_gap.ntrans();i++){
        assert(dst_set_cpu.find(cpu_gap.dst.get_cpu_data()[i]) == dst_set_cpu.end());
        dst_set_cpu.insert(cpu_gap.dst.get_cpu_data()[i]);
        if(mode == GPU){
            assert(dst_set_gpu.find(gpu_gap.dst.get_cpu_data()[i]) == dst_set_gpu.end());
            dst_set_gpu.insert(gpu_gap.dst.get_cpu_data()[i]);
        }
    }

    //GPU模式下同步数据到GPU
    if(mode == GPU){
        gpu_gap.src.update_gpu_data_from_cpu();
        gpu_gap.dst.update_gpu_data_from_cpu();
    }

}
void initVecPlay(Mode mode, HelioXroupData *ndat, coreneuron::CoreData *cdat){
    for(auto &vecplay_core:cdat->vec_play_continuous_core){
        auto mtype = vecplay_core.mtype;
        auto ix = vecplay_core.ix;
        auto [cpu_data_ptr, gpu_data_ptr] = getMechVarPtrByLagacyIndex(vecplay_core.mtype,vecplay_core.ix,cdat);
        printf("reg vecplay ptr[%p]\n",cpu_data_ptr);
        
        // 将legacy索引转换为新的key格式
        // 这里需要根据mtype和ix找到对应的mech_name和var_name
        // 由于这是从文件加载的legacy数据，我们需要使用mech_var_table来查找变量名
        assert(mech_var_table.contains(mtype));
        auto &mech_var_map = mech_var_table[mtype];
        
        // 使用legacy2soaos_index函数获取变量信息
        auto [mech_i, variable_index, array_index] = legacy2soaos_index(ix, cdat->mech_data->nrn_array_dims[mtype]);
        assert(mech_var_map.contains(variable_index));
        auto &var_data = mech_var_map[variable_index];
        
        // 获取mech_name
        string mech_name = cdat->mech_data->name_vec[mtype];
        string var_name = var_data.name;
        int instance_id = mech_i;
        
        VecPlayContinuousKey key{mech_name, var_name, instance_id};
        VecPlayContinuousItem vecplay(mode,vecplay_core, cpu_data_ptr,gpu_data_ptr);
        ndat->vec_play_continuous.push_back(key,std::move(vecplay));
        
        printf("reg vecplay for mechanism[%s] variable[%s] instance[%d]\n", 
               mech_name.c_str(), var_name.c_str(), instance_id);
    }
    if(mode == GPU){
        ndat->vec_play_continuous.try_update_gpu();
    }
}
/*
 * build up data structures used in simulation from CoreNEURON data
 * 输入是cdat,然后输出是ndat
 */
void init_neurondata_from_coredat(HelioXroupData *ndat, coreneuron::CoreData *cdat, Mode mode)
{
    initGroupData(mode, ndat, cdat);
    printf("initGroupData done\n");
    initMechList(mode, ndat, cdat);
    printf("initMechList done\n");
    initMechPointers(ndat, cdat);
    printf("initMechPointers done\n");
    initSynapse(mode, ndat, cdat);
    printf("initSynapse done\n");
    initGapTrans(mode, ndat, cdat);
    printf("initGapTrans done\n");
    initVecPlay(mode, ndat, cdat);
    
    if (cdat->permute)
    {
        ndat->permute = new int[ndat->len];
        memcpy(ndat->permute, cdat->permute, sizeof(int) * ndat->len);
        setup_permute_info(ndat, cdat, permute_info_arr[cdat->id], mode);
    }
}

void data_format_trans(vector<HelioXroupData *> &neuron_group_list, unique_ptr<coreneuron::CoreData *[]> &coredata_arr, int ngroup, Mode mode, double dt)
{
    // neuron_group_list.resize(ngroup);
    printf_debug("ngroup=%d\n", ngroup);
    assert(ngroup == 1);
    for (int i = 0; i < ngroup; i++)
    {
        HelioXroupData *neuron_data = new HelioXroupData(mode,dt);
        neuron_group_list.push_back(neuron_data);
        init_neurondata_from_coredat(neuron_group_list[i], coredata_arr[i], mode);
        MechanismFactory::getInstance().registerVarMap("global", neuron_group_list[i]);
    }
}
