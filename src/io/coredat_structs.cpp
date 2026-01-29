#include <iostream>
#include <string>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <map>
#include <vector>
#include "coredat_structs.h"
#include "utils.h"
#include "neuron.h"
#include "vecdata.h"
#include "permute.h"
#include "permute_order.h"

using namespace std;
using namespace coreneuron;



CoreData::CoreData(int cid)
{
    id = cid;
    gidgroups = nullptr;
    imult = nullptr;
    mech_list = nullptr;
    mech_data = nullptr;
    netcon_srcgid = nullptr;
    idata = nullptr;
    _v_parent_index = nullptr;
    output_gid = nullptr;
    _actual_a = nullptr;
    _actual_b = nullptr;
    _actual_d = nullptr;
    _actual_rhs = nullptr;
    _actual_area = nullptr;
    _actual_v = nullptr;
    _actual_area = nullptr;
    _actual_diam = nullptr;
    weights = nullptr;
    pntprocs = nullptr;
    netcon_arr = nullptr;
    presyn_arr = nullptr;
    pnt2presyn_ix = nullptr;
    map_type2mechptr = nullptr;
    permute = nullptr;
    gap_transfer = nullptr;
}

CoreData::~CoreData()
{
    if (gidgroups)
    {
        delete[] gidgroups;
        gidgroups = nullptr;
    }
    if (imult)
    {
        delete[] imult;
        imult = nullptr;
    }
    if (netcon_srcgid)
    {
        delete[] netcon_srcgid;
        netcon_srcgid = nullptr;
    }
    if (idata)
    {
        delete[] idata;
        idata = nullptr;
    }
    if (_v_parent_index)
    {
        delete[] _v_parent_index;
        _v_parent_index = nullptr;
    }
    if (output_gid)
    {
        delete[] output_gid;
        output_gid = nullptr;
    }
    if (_actual_a)
    {
        delete[] _actual_a;
        _actual_a = nullptr;
    }
    if (_actual_b)
    {
        delete[] _actual_b;
        _actual_b = nullptr;
    }
    if (_actual_d)
    {
        delete[] _actual_d;
        _actual_d = nullptr;
    }
    if (_actual_rhs)
    {
        delete[] _actual_rhs;
        _actual_rhs = nullptr;
    }
    if (_actual_area)
    {
        delete[] _actual_area;
        _actual_area = nullptr;
    }
    if (_actual_v)
    {
        delete[] _actual_v;
        _actual_v = nullptr;
    }
    if (_actual_area)
    {
        delete[] _actual_area;
        _actual_area = nullptr;
    }
    if (_actual_diam)
    {
        delete[] _actual_diam;
        _actual_diam = nullptr;
    }
    if (weights)
    {
        delete[] weights;
        weights = nullptr;
    }

    //这段代码能把链表的每一个节点都删掉，因为链表每个节点肯定都存在这个数组中
    for(int i=0;i<mech_data->nmech_type;i++)
    {
        if(map_type2mechptr[i])
        {
            delete map_type2mechptr[i];
            map_type2mechptr[i] = nullptr;
        }
    }
    try_delete_arr(map_type2mechptr);




    if (pntprocs)
    {
        delete[] pntprocs;
        pntprocs = nullptr;
    }
    if (netcon_arr)
    {
        delete[] netcon_arr;
        netcon_arr = nullptr;
    }
    if (presyn_arr)
    {
        delete[] presyn_arr;
        presyn_arr = nullptr;
    }
    if (mech_data)
    {
        delete mech_data;
        mech_data = nullptr;
    }
    if (permute)
    {
        delete[] permute;
        permute = nullptr;
    }
}

CoreMech::CoreMech()
{
    type = -1;
    nodecount = 0;
    nodeindices = nullptr;
    pdata = nullptr;
    _data = nullptr;
    next = nullptr;
    icnt = 0;
    dcnt = 0;
    iArray = nullptr;
    dArray = nullptr;
    permute = nullptr;
}

CoreMech::~CoreMech()
{
    if (nodeindices)
    {
        delete[] nodeindices;
        nodeindices = nullptr;
    }
    if (pdata)
    {
        delete[] pdata;
        pdata = nullptr;
    }
    if (_data)
    {
        delete[] _data;
        _data = nullptr;
    }
    if (iArray)
    {
        delete[] iArray;
        iArray = nullptr;
    }
    if (dArray)
    {
        delete[] dArray;
        dArray = nullptr;
    }
    if (permute)
    {
        delete[] permute;
        permute = nullptr;
    }
}

CoreMechData::CoreMechData()
{
    nmech_type = 0;
    nrn_prop_param_size = nullptr;
    nrn_prop_dparam_size = nullptr;
    nrn_mech_data_layout = nullptr;
    pnt_map = nullptr;
    pnt_receive_size = nullptr;
    nrn_is_artificial = nullptr;
    nrn_artcell_qindex = nullptr;
    nrn_bbcore_read = nullptr;
}

CoreMechData::~CoreMechData()
{
    if (nrn_prop_param_size)
    {
        delete[] nrn_prop_param_size;
        nrn_prop_param_size = nullptr;
    }
    if (nrn_prop_dparam_size)
    {
        delete[] nrn_prop_dparam_size;
        nrn_prop_dparam_size = nullptr;
    }
    if (nrn_mech_data_layout)
    {
        delete[] nrn_mech_data_layout;
        nrn_mech_data_layout = nullptr;
    }
    if (pnt_map)
    {
        delete[] pnt_map;
        pnt_map = nullptr;
    }
    if (pnt_receive_size)
    {
        delete[] pnt_receive_size;
        pnt_receive_size = nullptr;
    }
    if (nrn_is_artificial)
    {
        delete[] nrn_is_artificial;
        nrn_is_artificial = nullptr;
    }
    if (nrn_artcell_qindex)
    {
        delete[] nrn_artcell_qindex;
        nrn_artcell_qindex = nullptr;
    }
    if (nrn_bbcore_read)
    {
        delete[] nrn_bbcore_read;
        nrn_bbcore_read = nullptr;
    }
}

