#include <map>
#include <fstream>
#include <cassert>

#include "neuron.h"
#include "permute_order.h"
#include "permute.h"
#include "coredat_structs.h"
#include "mechanism.h"
#include "global_vars.h"

using namespace coreneuron;

int have_gaps = 0;
int setup_multiple = 1;
int permute_type = 0;
int nthread_each_cell = 16;
static int maxgid;
int chkpnt = 0;
extern PermuteInfo *permute_info_arr;

vector<map<int, CorePreSyn *>> neg_gid2out;//每个GroupData有一个.只记录了负的gid，正的gid在gid2out中
map<int, CorePreSyn *> gid2out;
map<int, InputPreSyn *> gid2in;
vector<NetCon *> netcon_in_presyn_order;

void read_filesdat(int &ngrp, int *&grp, int multiple, int *&imult, const char *filesdat);
void read_mech(CoreMechData *mech_data, const char *data_path);
void read_phase1(CoreData *&coredata, int fileid, int imult, const char *datapath);
void read_phase2(CoreData *&coredata, int fileid, int imult, const char *datapath, int user_mod_num);
void determine_inputpresyn(unique_ptr<CoreData *[]> &coredata_arr, int ncoredat);
void setup_cleanup();

void gap_setup(CoreData *&coredat, int fileid, const char *datapath);

int read_coredat(unique_ptr<CoreData *[]> &coredata_arr, const char *datapath, const char *filesdat, int byte_swap, bool run_setup_cleanup, int user_mod_num)
{
    int ngroup = 0;
    int *gidgroups = nullptr;
    int *imult = nullptr;
    maxgid = 0x7fffffff / setup_multiple;
    read_filesdat(ngroup, gidgroups, setup_multiple, imult, filesdat);//读取最基础的文件结构，即，每个文件叫啥名字
    assert(ngroup == 1);// 目前只支持一个Group
    neg_gid2out.resize(ngroup);

    coredata_arr = nullptr;
    coredata_arr = make_unique<CoreData *[]>(ngroup);
    permute_info_arr = new PermuteInfo[ngroup];
    for (int i = 0; i < ngroup; i++)
    {
        coredata_arr[i] = new CoreData(i);
        coredata_arr[i]->mech_data = new CoreMechData();
        read_mech(coredata_arr[i]->mech_data, datapath); // 读取每种Mech的信息,注意，是Mech类的信息，而不是具体的Mech实例的数据。例如，hh这种mech的pdata的大小之类的
    }
    for (int i = 0; i < ngroup; i++)
    {
        //phase1：读取presyn和netcon的基本信息,并分配内存。以及建立gid到presyn的映射
        read_phase1(coredata_arr[i], gidgroups[i], imult[i], datapath); // coreneuron/io/nrn_setup.cpp line:479，这是分散的，最后分到phase1.cpp
        printf_debug("phase1 done!\n");
    }
    //主要是建立netcon_in_presyn_order这个数组，即，每一个presyn对应的netcon的顺序
    //例如 [nc1, nc2, nc3][nc4, nc5]，表示第一个presyn对应的netcon是nc1、nc2、nc3，第二个presyn对应的netcon是nc4、nc5
    determine_inputpresyn(coredata_arr, ngroup); // nrn_setup:line 490
    for (int i = 0; i < ngroup; i++)
    {
        read_phase2(coredata_arr[i], gidgroups[i], imult[i], datapath, user_mod_num);
        printf_debug("phase2 done!\n");
        if (have_gaps)
        {
            gap_setup(coredata_arr[i], gidgroups[i], datapath);
        }
        printf_debug("gap_setup done!\n");
    }

    read_global_dat(datapath); // 读取全局变量
    
    if (run_setup_cleanup)
    {
        setup_cleanup();
    }

    delete[] gidgroups;
    delete[] imult;
    gidgroups = nullptr;
    imult = nullptr;
    return ngroup;
}

void alloc_mech(CoreMechData *mech_data, int n)
{
    /*?? ljf:比起1.2缺少：
    memb_func、
    pnt_receive、
    pnt_receive_init、
    watch_check

    bbcore_read//这好像被魔改成了bool，原本应该是函数来着
    bbcore_write(这两个应该完全没用，这两个是保存读取的函数)

    */
    // debug:
    printf_debug("alloc_mech: n=%d\n", n);

    mech_data->name_vec.resize(n);

    mech_data->nmech_type = n;
    mech_data->pnt_map = new int[n];
    mech_data->pnt_receive_size = new short[n];
    mech_data->nrn_is_artificial = new short[n];
    mech_data->nrn_artcell_qindex = new short[n];
    mech_data->nrn_prop_param_size = new int[n];
    mech_data->nrn_prop_dparam_size = new int[n];
    mech_data->nrn_mech_data_layout = new int[n];

    /*    1.8新增了
    array_dims
    */
    mech_data->nrn_array_dims.resize(n);

    mech_data->nrn_bbcore_read = new bool[n];
    memset(mech_data->nrn_bbcore_read, false, sizeof(bool) * n);
    // mech_data->nrn_bbcore_read[24] = true;//这个魔数24是个啥//TODO:vecevent写死的地方需要调整

    // setup nrn_has_net_event here temporary,
    // now only vecevent.mod has net_event

    for (int i = 0; i < n; i++)
    {
        mech_data->nrn_mech_data_layout[i] = 1; // TODO:这部分是在干啥
    }
    mech_data->is_ion = std::vector<bool>(n, false); // 新增：是否是离子通道
}
void ion_reg(const char *iname, double charge)
{
    printf_debug("TODO: reg ion [%s][%lf]\n", iname, charge);
    // TODO:后续需要加吗？
}
void read_mech(CoreMechData *mech_data, const char *data_path)
{
    ifstream s; // emmm,比较帅气的文件打开方式哈，和mk_mech一致
    char filename[500];
    sprintf(filename, "%s/bbcore_mech.dat", data_path); // coreneuron/io/mk_mech.cpp:50
    s.open(filename);
    string version; // 然后就到了80行的static void mk_mech(std::istream& s)
    int nmech;
    s >> version;
    s >> nmech; // coreneuron:是叫n
    alloc_mech(mech_data, nmech);
    printf_debug("version in mechfile[%s] is %s\n", filename, version.c_str());

    printf_debug("Mechanism Debug Information:\n");
    for (int i = 2; i < nmech; i++)
    {
        char mname[1000];
        int type = 0, pnttype = 0, is_art = 0, is_ion = 0, dsize = 0, pdsize = 0;
        int vsize = 0; // ljf
        s >> mname >> type >> pnttype >> is_art >> is_ion >> dsize >> pdsize;
        assert(i == type);
        printf_debug("Mech[%d] is [%s]\n", type, mname);

        if (isVersionGreater(version, "1.8"))
        {
            // printf_debug("%s:version ge 1.8",__func__);
            s >> vsize; // ljf
            std::vector<int> array_dims(vsize);
            for (size_t i = 0; i < vsize; ++i)
            {
                assert(s >> array_dims[i]);
            }
            mech_data->nrn_array_dims.at(type) = array_dims;
            //这个array_dims是一个vector<int>，里面存放了每个变量的维度，即，有的变量是一个数组，记录数组的长度
        }

        // 这部分省略了memb_func的部分，毕竟不兼容
        mech_data->name_vec[type] = mname;
        mech_data->pnt_map[type] = pnttype;
        mech_data->nrn_prop_param_size[type] = dsize;
        mech_data->nrn_prop_dparam_size[type] = pdsize;
        mech_data->nrn_is_artificial[type] = is_art;
        
        // 使用 MechanismFactory 获取 pnt_receive_size，如果未找到则默认为 0
        int receive_size = MechanismFactory::getInstance().getPntReceiveSize(mname);
        mech_data->pnt_receive_size[type] = (receive_size != -1) ? receive_size : 0;
        
        if (is_ion)
        {
            mech_data->is_ion[type] = true;
            double charge = 0.0;
            assert(s >> charge);

            // strip the _ion
            char iname[1000];
            strcpy(iname, mname);
            iname[strlen(iname) - 4] = '\0';
            // ion register, not implemented yet
            ion_reg(iname, charge);
        }
        printf_debug("Mech[%d]:%s type=%d pnttype=%d is_art=%d is_ion=%d dsize=%d pdsize=%d\n", i, mname, type, pnttype, is_art, is_ion, dsize, pdsize);
        if (string(mname) == "VecStim")
        {
            printf_debug("VecStim type=%d\n", type);
            mech_data->nrn_bbcore_read[type] = true;
            mech_data->nrn_has_net_event.push_back(type); // TODO:这个地方写死了，需要调整
        }
    }

    printf_debug("mech_data->has_net_event_cnt=%ld\n", mech_data->nrn_has_net_event.size());
    s.close();
    printf_debug("read mech done!\n");
}

int read_int(FILE *&fp)
{
    char line_buf[500];
    fgets(line_buf, 500, fp);
    int i;
    int n_scan;
    n_scan = sscanf(line_buf, "%d", &i);
    assert(n_scan == 1);
    return i;
}

void read_checkpoint_assert(FILE *&fp) // coreneuron/io/nrn_filehandler.cpp:85
{
    char line_buf[1024];
    fgets(line_buf, 1024, fp);
    int i, n_scan;
    n_scan = sscanf(line_buf, "chkpnt %d\n", &i);
    if (n_scan != 1)
    {
        printf("no chkpnt line for %d\n", chkpnt);
    }
    assert(n_scan == 1);
    if (i != chkpnt)
    {
        printf("file chkpnt %d != expected %d\n", i, chkpnt);
    }
    assert(i == chkpnt);
    chkpnt++; // 这是一个全局变量，每次读之后+1，然后每次读文件，值应该都和这变量一样，所以是，0，1，2，3之类的
}

template <typename T>
void read_array(FILE *&fp, T *arr, int len)
{
    // 每个array前面都有一个chkpnt #的标记，用来检查是否读取正确，两个chkpnt之间的数据是一个二进制
    read_checkpoint_assert(fp);     // coreneuron/io/nrn_filehandler.hpp:182
    fread(arr, sizeof(T), len, fp); // line 188
}
template <typename T>
void read_array_vector(FILE *&fp, std::vector<T> &arr, int len)
{
    read_checkpoint_assert(fp); // coreneuron/io/nrn_filehandler.hpp:182
    arr.resize(len);
    fread(arr.data(), sizeof(T), len, fp); // line 188
}
// coreneuron/io/nrn_setup.cpp
// void nrn_read_filesdat(int& ngrp, int*& grp, const char* filesdat)
void read_filesdat(int &ngrp, int *&grp, int multiple, int *&imult, const char *filesdat)
{
    FILE *fp = fopen(filesdat, "r");
    char version[256];
    fscanf(fp, "%s\n", version);
    printf_debug("version in data base is:%s\tHelioX supports:1.8\n", version);
    int iNumFiles = 0;

    //读取文件个数
    fscanf(fp, "%d\n", &iNumFiles);
    have_gaps = 0;
    if (iNumFiles == -1)
    {
        fscanf(fp, "%d\n", &iNumFiles);
        have_gaps = 1; // line 197
    }

    ngrp = 0;                              // 统计，并分配新的id用
    grp = new int[iNumFiles * multiple];   // line 211,出现了分歧，coreneuron中，是多线程处理，这个是单线程
    imult = new int[iNumFiles * multiple]; //???这是干啥的，好像是说在第几段的multiple内，比如说[1,1,1,1,1,1,2,2,2,2,2,2,2]这样的

    for (int iNum = 0; iNum < iNumFiles * multiple; iNum++)
    {
        int iFile;
        fscanf(fp, "%d\n", &iFile);
        grp[ngrp] = iFile; // 文件顺序和文件的idx对应
        imult[ngrp] = iNum / iNumFiles;
        ngrp++;
        if ((iNum + 1) % iNumFiles == 0) // 下次循环就要结束了
        {
            rewind(fp);
            fscanf(fp, "%*s\n");
            fscanf(fp, "%*d\n");
        }
    }

    fclose(fp);
}
// <firstgid>_1.dat
// n_presyn, n_netcon
// output_gids (npresyn) 对于那些没有 gid 的 acell，用 -(type+1000*index) 表示
// netcon_srcgid (nnetcon) -(type+1000*index) 指代没有 gid 的 acell
//                         -1 表示 netcon 没有来源（尚未实现）
// 注意，负的 gids 仅在线程内是唯一的，而在进程内并不是唯一的。
// 我们为每个线程创建一个线程特定的哈希表，用于存储负的 gids，
// 在读取 <firstgid>_1.dat 时创建，并在 <firstgid>_2.dat 使用完后销毁。
// 之前的实现尝试将线程号编码到负的 gid 中
// （即 -ith - nth*(type +1000*index)），但由于整数域大小不足而失败。
// 注意，对于文件传输，如果负的 srcgid 不在与目标相同的线程中，则会报错。
// 这是因为在 NEURON 进程中的线程可能不会在 CoreNEURON 中位于同一进程中。
// 如果出现这种情况，NEURON 会抛出错误。然而，对于直接内存传输，
// 允许负的 srcgid 位于与目标不同的线程中。因此，nrn2core_get_dat1
// 有一个最后的参数 netcon_negsrcgid_tid，指定 netcon_srcgid 中的负 gids
// （按顺序）对应的来源线程。
void read_phase1(CoreData *&coredata, int fileid, int imult, const char *datapath)
{
    char fnamebuf[500];
    char version[256];
    // int zz = imult * maxgid;                            // 通过对所有的 output_gid 和 netcon_srcgid 加上一个偏移量（zz），确保不同线程的 GID 是唯一的。这与注释中提到的“所有 GID 全局唯一”的要求一致。
    sprintf(fnamebuf, "%s/%d_1.dat", datapath, fileid); // coreneuron/io/nrn_setup.hpp:line 117

    printf_debug("reading phase1:%s\n", fnamebuf);
    FILE *fp = fopen(fnamebuf, "rb");

    fgets(version, 256, fp); // coreneuron/io/nrn_filehandler.cpp:35 Phase1::Phase1(FileHandler& F)

    //===============coreneuron/io/phase1.cpp:27
    coredata->n_presyn = read_int(fp); // Number of PreSyn-s in NrnThread nt
    coredata->n_netcon = read_int(fp); // Number of NetCon-s in NrnThread nt
    coredata->output_gid = new int[coredata->n_presyn];//每个presyn对应一个output_gid
    coredata->netcon_srcgid = new int[coredata->n_netcon];//每个netcon对应一个srcgid
    read_array<int>(fp, coredata->output_gid, coredata->n_presyn);
    read_array<int>(fp, coredata->netcon_srcgid, coredata->n_netcon);
    fclose(fp);

    //====================================
    // 对应的void Phase1::populate(NrnThread& nt, OMP_Mutex& mut)
    // for (int i = 0; i < coredata->n_presyn; i++)//给每一个presyn的output_gid加上一个偏移量，保证不同线程的GID是唯一的
    // {
    //     if (coredata->output_gid[i] >= 0)
    //     {
    //         assert(coredata->output_gid[i] < maxgid); 
    //         coredata->output_gid[i] += zz;            
    //     }
    // }
    ////////////////////////////////////// 对netcon的srcgid进行同样的操作

    coredata->netcon_arr = new NetCon[coredata->n_netcon];
    for (int i = 0; i < coredata->n_netcon; i++)
    {
        // if (coredata->netcon_srcgid[i] >= 0)
        // {
        //     assert(coredata->netcon_srcgid[i] < maxgid);
        //     // coredata->netcon_srcgid[i] += zz;                             //????
        //     coredata->netcon_arr[i].src_gid = coredata->netcon_srcgid[i]; // coreneuron/io/phase1.cpp line 58的魔改
        // }

        coredata->netcon_arr[i].src_gid = coredata->netcon_srcgid[i]; // coreneuron/io/phase1.cpp line 58的魔改
    }

    ////////////////////////////////////// 处理PreSyn
    coredata->presyn_arr = new CorePreSyn[coredata->n_presyn]; // 对应68，这里默认一定有N-presyn
    for (int i = 0; i < coredata->n_presyn; i++)
    {
        int gid = coredata->output_gid[i];
        if (gid == -1)//跳过是-1的gid，这个是没有gid的presyn，其中，PreSyn默认初始化的时候，gid=-1，所以不用再处理
            continue;

        //coreNeuron中的注释：
        // 注意，负的 (type, index) 编码信息存储在 neg_gid2out[tid] 哈希表中。
        // 有关 netpar_tid_... 函数的实现，请参见 netpar.cpp。
        // 在 setup 结束之前，可以删除该表和进程范围的 gid2out 表。

        /// 将 gid 与对应的输出 PreSyn 一起放入 gid2out 哈希表中
        /// 或者放入负的 PreSyn 映射中
        CorePreSyn *ps = coredata->presyn_arr + i; // 对应line73的，设置一个PreSyn* ps = nt.presyns;然后每次循环++
        if (gid >= 0)//建立gid到PreSyn的映射
        {
            // 此时，应该这些都是空的，所以应该都找不到
            if (gid2in.contains(gid)) // 对应line92
            {
                printf("gid=%d already exists as an input port\n", gid);
                exit(1);
            }
            if (gid2out.contains(gid)) // 对应line96
            {
                printf("gid=%d already exists as an output port\n", gid);
                exit(1);
            }
            ps->gid = gid;
            ps->output_index = gid;
            gid2out[gid] = ps;
            //gid2in在determine_inputpresyn中处理
        }
        else
        {
            assert(!neg_gid2out[coredata->id].contains(gid));
            //不设置PreSyn的gid,因为默认就是-1
            ps->output_index = -1;
            neg_gid2out[coredata->id][gid] = ps;
        }
    }
}

void netpar_tid_gid2ps(int tid, int gid, CorePreSyn **ps, InputPreSyn **psi)
{
    *ps = nullptr;
    *psi = nullptr;
    if (gid >= 0)
    {
        if (gid2out.contains(gid))
        {
            *ps = gid2out.at(gid); 
        }
        else if(gid2in.contains(gid))
        {
            *psi = gid2in.at(gid);
        }
    }
    else
    {
        if (neg_gid2out[tid].contains(gid))
        {
            *ps = neg_gid2out[tid].at(gid);
        }
    }
}

void determine_inputpresyn(unique_ptr<CoreData *[]> &coredata_arr, int ncoredat)
{
    // coreneuron/io/nrn_setup.cpp:determine_inputpresyn()
    // 分配进程范围的 InputPreSyn 数组
    // 所有的 output_gid 已经注册并与 PreSyn 关联。
    // 现在通过填充 `gid2in` 映射来计算所需的 InputPreSyn 数量。
    gid2in.clear();
    // 现在必须填充新表格
    // 不需要担心负数 gid 重叠，因为仅用于在此线程中查找 PreSyn。
    vector<InputPreSyn *> inputpresyn;
    // map<int, CorePreSyn *>::iterator gid2out_it;
    // map<int, InputPreSyn *>::iterator gid2in_it;

    //这一个循环，用于维护presyn的netcon计数，即，每一个presyn连接了多少个postsyn
    for (int i = 0; i < ncoredat; i++) // 原本coreneuron是遍历每个thread的nt结构体，现在的实现是遍历coredata结构体
    {
        CoreData *coredata = coredata_arr[i];
        coredata->n_intput_presyn = 0;
        printf_debug("in coredat[%d] has[%d] netcons\n", i, coredata->n_netcon);
        for (int i = 0; i < coredata->n_netcon; i++)
        {
            int gid = coredata->netcon_srcgid[i];
            // printf_debug("netcon[%d]: gid=%d\n",i,gid);
            if (gid >= 0)
            {
                /// If PreSyn or InputPreSyn is already in the map
                if (gid2out.contains(gid))
                {
                    /// Increase PreSyn count
                    gid2out[gid]->nc_cnt++;
                    continue;
                }
                //////ljf:这段好像一直都在，但是没添加，这就添加上去

                if (gid2in.contains(gid))
                {
                    /// Increase InputPreSyn count
                    gid2in[gid]->nc_cnt++;
                    continue;
                }
                ////
                /// Create InputPreSyn and increase its count
                InputPreSyn *psi = new InputPreSyn();
                psi->nc_cnt=1;
                gid2in[gid] = psi;
                inputpresyn.push_back(psi);
                coredata->n_intput_presyn++;
            }
            else
            {
                // note：这原本有一个修正负的tid的地方，但是neuron-G实现方法不一样，就省略了

                if (neg_gid2out[coredata->id].contains(gid))
                {
                    /// Increase negative PreSyn count
                    neg_gid2out[coredata->id][gid]->nc_cnt++;
                }
            }
        }
    }

    // 现在，我们可以机会主义地创建 NetCon* 指针数组
    // 通过统计每个 PreSyn 和 InputPreSyn 指向的 NetCon 数量，来节省
    // “大量小数组分配”所带来的内存开销。
    // 设想 nt.netcons 可以成为一个进程的全局数组，
    // 在这种情况下，NetCon* 指针数组可以变为一个整数索引数组。
    // 更具猜测性地，如果进程全局的 NetCon 数组能够合理排序，
    // 那么索引数组本身可以被消除，但这样会将不同线程的 NetCon 交错在一起。
    // 对于串行线程来说，这并不是问题，但重新排序会影响到 nt.pntprocs，
    // 尤其当 NetCon 数据指针也被整数索引取代时。

    // 首先，创建指针数组,并计算总的 NetCon 数量
    int n_nc = 0;
    for (int i = 0; i < ncoredat; i++)
    {
        n_nc += coredata_arr[i]->n_netcon;
    }
    netcon_in_presyn_order.resize(n_nc);

    n_nc = 0;
    // 重置 n_nc 计数器，后面复用这个计数器来填充 netcon_in_presyn_order 数组
    //同时，有的netcon没有对应的 presyn（即 netcon_srcgid[nt.id][i] = -1），因此，后面的处理会跳过这些，有可能最终的大小会被“压缩”，只保留真正需要处理的

    // 使用偏移值填充索引并重置 nc_cnt_，
    // 以便在接下来的循环中使用 nc_cnt_ 将 NetCon 分配到正确的位置

    // 处理 PreSyn
    int offset = 0;
    for (int i = 0; i < ncoredat; i++)
    {
        CoreData *coredata = coredata_arr[i];
        for (int ipre = 0; ipre < coredata->n_presyn; ipre++)
        {
            CorePreSyn &ps = coredata->presyn_arr[ipre];
            ps.nc_index = offset;
            offset += ps.nc_cnt;
            ps.nc_cnt = 0;
        }
    }
    // 处理InputPreSyn
    for (size_t i = 0; i < inputpresyn.size(); i++)
    {
        InputPreSyn *psi = inputpresyn[i];
        psi->nc_index = offset;
        offset += psi->nc_cnt;
        psi->nc_cnt = 0;
    }

    inputpresyn.clear();

    // 填充 netcon_in_presyn_order 数组并重新计算 nc_cnt_
    // 注意，如果有 netcon 没有对应的 presyn（即 netcon_srcgid[nt.id][i] = -1），并不是所有的 netcon_in_presyn 都会被填充，
    // 但这没关系，因为它们只通过 ps.nc_index_ 和 ps.nc_cnt_ 使用。

    for (int ith = 0; ith < ncoredat; ith++)
    {
        CoreData *coredata = coredata_arr[ith];
        // printf_debug("determine_inputpresyn: coredata[%d] has %d netcons\n",ith,coredata->n_netcon);
        for (int i = 0; i < coredata->n_netcon; i++)
        {
            NetCon *nc = coredata->netcon_arr + i;
            int gid = coredata->netcon_srcgid[i];//这个for循环在遍历所有的gid
            // Note:这原本有一个和nt的负tid相关的操作，由于HelioX里面实现不同，所以不需要操作

            //源GID有两种，一种是内部的PreSyn，另一种是其他线程的，叫InputPreSyn
            CorePreSyn *ps;
            InputPreSyn *psi;
            netpar_tid_gid2ps(ith, gid, &ps, &psi);// 通过 gid 获取对应的 PreSyn 或 InputPreSyn，只会有一个非空指针
            if (ps)
            {
                netcon_in_presyn_order[ps->nc_index + ps->nc_cnt] = nc;
                ps->nc_cnt++;
                n_nc++;
                // printf_debug("determine_inputpresyn: [PS] netcon_in_presyn_order[%d]=nc[%d] n_nc=%d ps->nc_cnt=%d\n",ps->nc_index + ps->nc_cnt - 1,i,n_nc,ps->nc_cnt);
            }
            else if (psi)
            {
                netcon_in_presyn_order[psi->nc_index + psi->nc_cnt] = nc;
                psi->nc_cnt++;
                n_nc++;
                // printf_debug("determine_inputpresyn: [PSI] netcon_in_presyn_order[%d]=nc[%d] n_nc=%d psi->nc_cnt=%d\n",psi->nc_index + psi->nc_cnt -1,i,n_nc,psi->nc_cnt);
            }
            // nc->print();
        }
    }

    /// Resize the vector to its actual size of the netcons put in it
    netcon_in_presyn_order.resize(n_nc);
}

void gap_setup(CoreData *&coredat, int fileid, const char *datapath)
{
    coredat->gap_transfer = new GapTransferInfo();
    char fnamebuf[500];
    sprintf(fnamebuf, "%s/%d_gap.dat", datapath, fileid);
    FILE *fp = fopen(fnamebuf, "rb");
    char version[200];
    fgets(version, 200, fp);

    chkpnt = 0;
    int sidt_size = read_int(fp);
    assert(sidt_size == int(sizeof(int)));
    std::size_t ntar = read_int(fp);
    std::size_t nsrc = read_int(fp);
    auto &si = coredat->gap_transfer;
    si->ntrans = ntar;
    si->src_sid_map.clear();
    si->src_sid.resize(nsrc);
    si->src_type.resize(nsrc);
    si->src_index.resize(nsrc);
    if (nsrc)
    {
        read_array<int>(fp, si->src_sid.data(), nsrc);
        read_array<int>(fp, si->src_type.data(), nsrc);
        read_array<int>(fp, si->src_index.data(), nsrc);
    }
    for (int i = 0; i < nsrc; i++)
    {
        si->src_sid_map[si->src_sid[i]] = i;
        printf_debug("src[%d]:sid=%d type=%d index=%d\n", i, si->src_sid[i], si->src_type[i], si->src_index[i]);
    }

    si->tar_sid.resize(ntar);
    si->tar_type.resize(ntar);
    si->tar_index.resize(ntar);
    if (ntar)
    {
        read_array<int>(fp, si->tar_sid.data(), ntar);
        read_array<int>(fp, si->tar_type.data(), ntar);
        read_array<int>(fp, si->tar_index.data(), ntar);
    }
    for (int i = 0; i < ntar; i++)
    {
        printf_debug("tar[%d]:sid=%d type=%d index=%d\n", i, si->tar_sid[i], si->tar_type[i], si->tar_index[i]);
    }
    fclose(fp);
}

// coreneuron/io/phase2.cpp:read_file()
void read_phase2(CoreData *&coredata, int fileid, int imult, const char *datapath, int user_mod_num)
{
    char fnamebuf[500];
    sprintf(fnamebuf, "%s/%d_2.dat", datapath, fileid);
    FILE *fp = fopen(fnamebuf, "rb");

    char version[200];
    fgets(version, 200, fp);
    int ndiam, nmech, *mech_types, *ml_nodecount;

    if (isVersionGreater(version, "1.8"))
    {
        // printf_debug("%s:version ge 1.8",__func__);
        coredata->n_real_cell = read_int(fp); // ljf:新版多了这一项
    }

    //???这对应得上吗？感觉哪不太对的样子
    int n_outputgid = read_int(fp);         // n_output AKA:ngid
    coredata->n_real_output = read_int(fp); // n_real_output 也就是n_cell AKA:n_real_gid
    if (!isVersionGreater(version, "1.8")){
        coredata->n_real_cell = coredata->n_real_output;
    }
    coredata->end = read_int(fp);           // n_node
    ndiam = read_int(fp);                   // n_diam
    nmech = read_int(fp);                   // n_mech

    mech_types = new int[nmech];    // mech_types
    ml_nodecount = new int[nmech]; // nodecounts
    int n_memb_func = coredata->mech_data->nmech_type;

    printf_debug("n_cell=%d n_node=%d\n", coredata->n_real_output, coredata->end);
    for (int i = 0; i < nmech; i++)
    {
        mech_types[i] = read_int(fp); // 对应原文是mech_types[i] = F.read_int();
        ml_nodecount[i] = read_int(fp);
    }
    // TODO:ljf:原版这里有一个检查兼容性，检查是否Mech都已经被支持了

    coredata->nidata = read_int(fp);
    coredata->nvdata = read_int(fp);
    coredata->n_weight = read_int(fp);

    CoreMech *ml_last = nullptr;
    coredata->map_type2mechptr = new CoreMech *[n_memb_func]; // 这是一个指针数组，将tml_index转换成ml节点的指针
    for (int i = 0; i < n_memb_func; i++)
    {
        coredata->map_type2mechptr[i] = nullptr;
    }

    for (int i = 0; i < nmech; i++) // HelioX的第一步采用的是链表存储，初始化链表
    {
        CoreMech *ml = new CoreMech();
        ml->type = mech_types[i];        // 链表上每个节点存的数据
        ml->nodecount = ml_nodecount[i]; // 把line 700-703的部分存到这了
        ml->next = nullptr;

        if (coredata->mech_list == nullptr) // 链表的创建操作
        {
            coredata->mech_list = ml;
        }
        else
        {
            ml_last->next = ml;
        }
        ml_last = ml; // 挂上链表

        coredata->map_type2mechptr[ml->type] = ml; // 这个数组用来把idx转换成当前节点的指针
    }
    delete[] mech_types;
    delete[] ml_nodecount; // 临时变量可以删了
    mech_types = nullptr;
    ml_nodecount = nullptr;

    coredata->idata = new int[coredata->nidata];
    coredata->vdata = make_unique<void *[]>(coredata->nvdata);
    // coreNeuron的n_weight在这，但是neuronG放后面了

    /////////////// ???TODO:看一下这是什么东西
    int npnt = 0; // 似乎是计算总的点过程的nodecount
    for (CoreMech *ml = coredata->mech_list; ml; ml = ml->next)
    {
        int n = ml->nodecount;
        int type = ml->type;
        if (coredata->mech_data->pnt_map[type] > 0) // 好像是在计算点过程需要的空间
            npnt += n;
    }

    coredata->pntprocs = new PointProcess[npnt]; // 为所有的点过程创建统一的一个数组
    coredata->n_pntproc = npnt;
    /////////////////////////////
    int len = coredata->end;
    coredata->_v_parent_index = new int[len];
    coredata->_actual_a = new double[len];
    coredata->_actual_b = new double[len];
    coredata->_actual_area = new double[len];
    coredata->_actual_v = new double[len];
    if (ndiam)
    {
        coredata->_actual_diam = new double[len];
    }

    ////////////////////////
    read_array<int>(fp, coredata->_v_parent_index, len);
    read_array<double>(fp, coredata->_actual_a, len);
    read_array<double>(fp, coredata->_actual_b, len);
    read_array<double>(fp, coredata->_actual_area, len);
    read_array<double>(fp, coredata->_actual_v, len);
    if (ndiam)
    {
        read_array<double>(fp, coredata->_actual_diam, len);
    }

    int synoffset = 0;
    auto pnt_offset = make_unique<int[]>(coredata->mech_data->nmech_type);
    for (CoreMech *ml = coredata->mech_list; ml; ml = ml->next)
    {
        int type = ml->type;
        int is_art = coredata->mech_data->nrn_is_artificial[type];
        int n = ml->nodecount;
        int sz = coredata->mech_data->nrn_prop_param_size[type];   // sz
        int dsz = coredata->mech_data->nrn_prop_dparam_size[type]; // dsz
        int layout = coredata->mech_data->nrn_mech_data_layout[type];

        if (!is_art) // 对应phase2.cpp的line 186
        {
            ml->nodeindices = new int[n];
            read_array<int>(fp, ml->nodeindices, n);
        }
        else
        {
            ml->nodeindices = nullptr;
            printf_debug("Mechanism[%d] is artificial\n", type);
        }

        ml->_data = new double[n * sz]; // neuron代码中的_data + offset
        read_array<double>(fp, ml->_data, n * sz);
        if (dsz) // 如果存在dparam的话，读一下，line 193
        {
            ml->pdata = new int[n * dsz];
            read_array<int>(fp, ml->pdata, n * dsz);
            auto &name = coredata->mech_data->name_vec[type];

        }
        else
        {
            ml->pdata = nullptr;
        }
        // NEURON 1.8+ 在 mechanism dparam 之后还会写入：
        // - pointer2type：POINTER dparam 对应的目标类型（用于后续解析）
        // - nmodlrandom：随机数状态（如使用 nmodlrandom）
        //
        // 这些字段在旧版本中不存在；这里按版本条件读取即可。

        if (isVersionGreater(version, "1.8"))
        {
            // printf_debug("%s:version ge 1.8",__func__);
            if (dsz > 0)
            {
                int sz = read_int(fp); // 对应line200
                if (sz)
                {
                    read_array_vector(fp, ml->pointer2type, sz);
                    // coreNeuron里面，采用的是不断给tml往后添加元素的方法，因此tml的最后一个元素就是当前元素
                    // neuronG里面，是先把ml链表搭完，再遍历，因此应该用ml代表当前的元素
                }
                sz = read_int(fp);
                if (sz)
                {
                    read_array_vector(fp, ml->nmodlrandom, sz);
                }
            }
        }

        /////////

        // 这一段是直接调到populate的1155行去了
        if (coredata->mech_data->pnt_map[type] > 0) // POINT_PROCESS mechanism including acell
        {
            int cnt = ml->nodecount;
            PointProcess *pnt = nullptr;
            pnt = coredata->pntprocs + synoffset;
            pnt_offset[type] = synoffset;
            synoffset += cnt;
            // 所有的点过程，都是在上面的pntprocs这个全局数组中存储
            // 现在是在把对应的offset存到每一个mech里面

            // 然后，每一个点过程都要初始化一下
            for (int i = 0; i < cnt; i++) // line:1161
            {
                PointProcess *pp = pnt + i;
                pp->type = type;
                pp->i_instance = i;
                coredata->vdata[ml->pdata[i * dsz + 1]] = pp;
                pp->cid = coredata->id;
            }
        }
    }


    // 这两部分，第一个部分是生成一个数据存储映射，第二个部分是调整数据存放位置
    if (permute_type > 0)
    {
        // 这部分，是用精细神经元划分方法来重新排列细胞数据，并且获取并行执行相关的信息
        coredata->permute = permute_order(coredata->id, coredata->n_real_cell, coredata->end, coredata->_v_parent_index); // phase2 populate部分的1213行
    }

    if (coredata->permute)
    {
        int *p = coredata->permute;
        permute_data(coredata->_actual_a, coredata->end, p);
        permute_data(coredata->_actual_b, coredata->end, p);
        permute_data(coredata->_actual_area, coredata->end, p);
        permute_data(coredata->_actual_v, coredata->end, p);
        if (coredata->_actual_diam)
        {
            permute_data(coredata->_actual_diam, coredata->end, p);
        }
        permute_ptr(coredata->_v_parent_index, coredata->end, p);
        node_permute(coredata->_v_parent_index, coredata->end, p);

        // specify the ml->permute and sort the nodeindices
        /*        // specify the ml->_permute and sort the nodeindices
        // Have to calculate all the permute before updating pdata in case
        // POINTER to data of other mechanisms exist.
        */
        for (CoreMech *ml = coredata->mech_list; ml; ml = ml->next)
        {
            if (ml->nodeindices) // not artificial
            {
                permute_nodeindices(ml, p); // phase2.cpp line 1241,这玩意儿居然是coreNeuron就存在的？
            }
        }

        // permute ml->data, ml->pdata is not used currently
        for (CoreMech *ml = coredata->mech_list; ml; ml = ml->next)
        {
            if (ml->nodeindices) // not artificial
            {
                permute_ml(ml, ml->type, coredata->mech_data, coredata);
            }
        }

        // permute Point_process.i_instance
        for (int i = 0; i < coredata->n_pntproc; i++)
        {
            PointProcess &pp = coredata->pntprocs[i];
            CoreMech *ml = coredata->map_type2mechptr[pp.type];
            if (ml->permute)
            {
                pp.i_instance = ml->permute[pp.i_instance];
            }
        }

    }

    /* deal with point processes which have net_send()
     * code should be added here
     */
    //????这是什么东西，net_event
    // 这好像又到了populate的1286附近了
    size_t has_net_event_cnt = coredata->mech_data->nrn_has_net_event.size();//所有有net_event(normal event)的mech
    int *pnttype2presyn = new int[n_memb_func]; // inverse of nrn_has_net_event
    for (int i = 0; i < n_memb_func; i++)       // coreneuron/io/nrn_setup.cpp:434
    {
        pnttype2presyn[i] = -1;
    }
    printf_debug("has_net_event_cnt=%d\n", has_net_event_cnt);
    for (int i = 0; i < has_net_event_cnt; i++)
    {
        printf_debug("coredata->mech_data->nrn_has_net_event[%d]=%d\n", i, coredata->mech_data->nrn_has_net_event[i]);
        pnttype2presyn[coredata->mech_data->nrn_has_net_event[i]] = i; // 这段初始化是在nrn_setup里面做的，现在移到这
    }

    coredata->pnt2presyn_ix = std::make_unique<std::unique_ptr<int[]>[]>(has_net_event_cnt);
    for (size_t i = 0; i < has_net_event_cnt; ++i)
    {
        CoreMech *mech = coredata->map_type2mechptr[coredata->mech_data->nrn_has_net_event[i]]; // 1288行
        if (mech && mech->nodecount > 0)
        {
            coredata->pnt2presyn_ix[i] = std::make_unique<int[]>(mech->nodecount);
        }
    }

    int *output_vindex, *pnttype, *pntindex;
    double *output_threshold, *delay;

    //presyn的真实细胞索引，或者是articell的索引
    output_vindex = new int[coredata->n_presyn]; // phase2.cpp:215行
    read_array<int>(fp, output_vindex, coredata->n_presyn);

    output_threshold = new double[coredata->n_presyn];
    read_array<double>(fp, output_threshold, coredata->n_real_output);
    // fread(output_threshold, sizeof(double), coredata->n_presyn, fp);

    if (coredata->permute) // 对应phase2.cpp:1303行，原来这玩意儿是coreNeuron自带的？
    {
        node_permute(output_vindex, coredata->n_presyn, coredata->permute);
    }

    for (int i = 0; i < coredata->n_presyn; i++) // 1312行
    {
        CorePreSyn *presyn = coredata->presyn_arr + i;
        int ix = output_vindex[i];
        // if (ix == -1 && coredata->ncell)// neuronG原版：可能写错了

        if (ix == -1 && i < coredata->n_real_output)
        {
            // 这里的ix=-1表示这个presyn没有对应的gid，即src是空的
            // 1.2版 if (ix == -1 && i < nt.ncell) {  // real cell without a presyn
            // 1.8是 i < n_real_output，但是n_real_output对应的就是 coredata->ncell = read_int(fp);//n_real_output
            continue;
        }
        //articell,解码出type和index
        if (ix < 0)
        {

            ix = -ix;
            // int index = ix / 100;//line 1321
            // int type = ix - index * 100;//???怎么改成100了，原本是1000

            int index = ix / user_mod_num;        // line 1321
            int type = ix - index * user_mod_num; // HelioX 自带的测试模型，他把生成时候的改成了100，而默认应该是1000

            PointProcess *pnt = coredata->pntprocs + (pnt_offset[type] + index);
            presyn->pntsrc = pnt;

            // pnt->_presyn = ps;
            printf_debug("pnt->type=%d  n_memb_func=%d\n", pnt->type, n_memb_func);

            int ip2ps = pnttype2presyn[pnt->type];

            if (ip2ps >= 0)
            {
                //记录不同type的presyn在整个presyn中的索引
                coredata->pnt2presyn_ix[ip2ps][pnt->i_instance] = i;
            }
            if (presyn->gid < 0)
            {

                presyn->gid = -1;
            }
        }
        else
        {

            assert(presyn->gid > -1);
            presyn->thvar_index = ix;
            assert(ix < coredata->end); // ljf:coreneuron里面有这段，就补上吧
            presyn->threshold = output_threshold[i];
        }
    }

    delete[] output_vindex;
    output_vindex = nullptr;
    delete[] output_threshold;
    output_threshold = nullptr;
    delete[] pnttype2presyn;
    pnttype2presyn = nullptr;

    int nnetcon = coredata->n_netcon; // n_netcon - nrn_setup_extracon
    int nweight = coredata->n_weight;
    pnttype = new int[nnetcon]; // >0 if it's point process
    pntindex = new int[nnetcon];

    read_array<int>(fp, pnttype, nnetcon); // 又回到了read_file,第219行
    read_array<int>(fp, pntindex, nnetcon);
    // fread(pnttype, sizeof(int), nnetcon, fp);
    // fread(pntindex, sizeof(int), nnetcon, fp);

    for (int i = 0; i < nnetcon; i++) // 然后马上来到了populate的1359行
    {
        int type = pnttype[i];
        if (type > 0)
        {
            int index = pnt_offset[type] + pntindex[i];
            NetCon &nc = coredata->netcon_arr[i];

            PointProcess *pnt = coredata->pntprocs + index;
            nc.pp = pnt;
            nc.active = true;
        }
    }
    ////////////////////////

    coredata->weights = new double[coredata->n_weight];
    read_array<double>(fp, coredata->weights, coredata->n_weight); // 这回到了read_file的222行

    int iw = 0;
    for (int i = 0; i < coredata->n_netcon; i++) // 这段是handle_weights()里了，line 843
    {
        //这一段，在原始的coreNeuron里面是放在handle_weights()函数里面的
        //其逻辑是，所有的weight变量都存在一个连续的内存里面，然后
        NetCon &nc = coredata->netcon_arr[i];
        // for (int j = 0; j < coredata->n_presyn; j++)
        //{
        //     if (coredata->presyn_arr[j].src_gid == nc.src_gid)
        //     {
        //         nc.thvar_index = coredata->presyn_arr[j].thvar_index;
        //         nc.threshold = coredata->presyn_arr[j].threshold;
        //         break;
        //     }
        // }
        nc.weight_index = iw;
        if (pnttype[i] != 0)
        {
            iw += coredata->mech_data->pnt_receive_size[pnttype[i]];
        }
        else
        {
            iw += 1;
        }
    }
    assert(iw == nweight);
    delete[] pnttype;
    delete[] pntindex;
    ///////////////////

    delay = new double[nnetcon]; // 223行，read_file
    read_array<double>(fp, delay, nnetcon);
    // fread(delay, sizeof(double), nnetcon, fp);
    for (int i = 0; i < nnetcon; i++) // 然后马上飞到populate的871行
    {
        NetCon &nc = coredata->netcon_arr[i];
        nc.delay = delay[i];
    }
    delete[] delay;

    // BBCOREPOINTER information
    npnt = read_int(fp); // coreNeuron:num_point_process line 224

    // deal with mod files which implement bbcore_read and bbcore_write
    for (CoreMech *ml = coredata->mech_list; ml; ml = ml->next) // read_file line 227
    {
        // read data written by mod file
        int type = ml->type;
        if (!coredata->mech_data->nrn_bbcore_read[type])
            continue;
        int icnt, dcnt;
        type = read_int(fp);
        icnt = read_int(fp);
        dcnt = read_int(fp);
        CoreMech *mech = coredata->map_type2mechptr[type];
        assert(ml == mech);
        if (icnt)
        {
            ml->icnt = icnt;
            mech->iArray = new int[icnt];//I array:即 Int Array，用于存储整数类型的数据
            read_array<int>(fp, mech->iArray, icnt); // 234行对应 tmls[i].iArray = F.read_vector<int>(icnt);
        }
        if (dcnt)
        {
            ml->dcnt = dcnt;
            mech->dArray = new double[dcnt];// D array:即 Double Array，用于存储浮点数类型的数据
            read_array<double>(fp, mech->dArray, dcnt);
        }
    }


    int n_vec_play_continuous = read_int(fp);
    coredata->vec_play_continuous_core.reserve(n_vec_play_continuous);
    for (int i = 0; i < n_vec_play_continuous; ++i)
    {
        VecPlayContinuous_Core item;
        item.vtype = read_int(fp);
        item.mtype = read_int(fp);
        item.ix = read_int(fp);
        int sz = read_int(fp);
        item.yvec.resize(sz);
        item.tvec.resize(sz);
        read_array_vector(fp, item.yvec, sz);
        read_array_vector(fp, item.tvec, sz);

        assert(item.vtype == VecPlayContinuous_Core::VecPlayContinuousType);//出自Phase2::set_vec_play这个函数，不知道为什么只有这一个类型，先assert上
        coredata->vec_play_continuous_core.push_back(std::move(item));
    }
    // F.record_checkpoint(); //TODO:尚未实现，并且也没看过这到底是干啥

    
    if (eof(fp))
    {
        for(auto &vecplay:coredata->vec_play_continuous_core){
            vecplay.print();
        }
        fclose(fp);
        return;
    }

    assert(read_int(fp) == n_vec_play_continuous);
    for (int i = 0; i < n_vec_play_continuous; ++i)
    {
        auto &vecPlay = coredata->vec_play_continuous_core[i];
        vecPlay.last_index = read_int(fp);
        vecPlay.discon_index = read_int(fp);
        vecPlay.ubound_index = read_int(fp);
    }
    coredata->patstim_index = read_int(fp);
    printf("patstim_index=%d\n", coredata->patstim_index);

    assert(read_int(fp) == -1);
    for (int i = 0; i < coredata->n_presyn; ++i)
    {
        coredata->preSynConditionEventFlags.push_back(read_int(fp));
        printf("preSynConditionEventFlags[%d]=%d\n", i, coredata->preSynConditionEventFlags[i]);
    }

    assert(read_int(fp) == -1);
    // restore_events(F);//TODO:

    assert(read_int(fp) == -1);
    // restore_events(F);//TODO:
    fclose(fp);
    for(auto &vecplay:coredata->vec_play_continuous_core){
        vecplay.print();
    }


}

void setup_cleanup()
{
}
