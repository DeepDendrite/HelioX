/*
Permute nodes.

To make gaussian elimination on gpu more efficient.

Permutation vector p[i] applied to a data vector, moves the data_original[i]
to data[p[i]].
That suffices for node properties such as area[i], a[i], b[i]. e.g.
  area[p[i]] <- area_original[i]

Notice that p on the left side is a forward permutation. On the right side
it serves as the inverse permutation.
area_original[i] <- area_permuted[p[i]]

but things
get a bit more complicated when the data is an integer index into the
original data.

For example:

parent[i] needs to be transformed so that
parent[p[i]] <- p[parent_original[i]] except that if parent_original[j] = -1
  then parent[p[j]] = -1

membrane mechanism nodelist ( a subset of nodes) needs to be at least
minimally transformed so that
nodelist_new[k] <- p[nodelist_original[k]]
This does not affect the order of the membrane mechanism property data.

However, computation is more efficient to permute (sort) nodelist_new so that
it follows as much as possible the permuted node ordering, ie in increasing
node order.  Consider this further mechanism specific nodelist permutation,
which is to be applied to the above nodelist_new, to be p_m, which has the same
size as nodelist. ie.
nodelist[p_m[k]] <- nodelist_new[k].

Notice the similarity to the parent case...
nodelist[p_m[k]] = p[nodelist_original[k]]

and now the membrane mechanism node data, does need to be permuted to have an
order consistent with the new nodelist. Since there are nm instances of the
mechanism each with sz data values (consider AoS layout).
The data permutation is
for k=[0:nm] for isz=[0:sz]
  data_m[p_m[k]*sz + isz] = data_m_original[k*sz + isz]

For an SoA layout the indexing is k + isz*nm (where nm may include padding).

A more complicated case is a mechanisms dparam array (nm instances each with
dsz values) Some of those values are indices into another mechanism (eg
pointers to ion properties) or voltage or area depending on the semantics of
the value. We can use the above data_m permutation but then need to update
the values according to the permutation of the object the value indexes into.
Consider the permutation of the target object to be p_t . Then a value
iold = pdata_m(k, isz) - data_t in AoS format
refers to k_t = iold % sz_t and isz_t = iold - k_t*sz_t
and for a target in SoA format isz_t = iold % nm_t and k_t = iold - isz_t*nm_t
ie k_t_new = p_m_t[k_t] so, for AoS, inew = k_t_new*sz_t + isz_t
or , for SoA, inew = k_t_new + isz_t*nm_t
so pdata_m(k, isz) = inew + data_t


*/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "read_coredat.h"
#include "coredat_structs.h"
#include "permute.h"
#include "legacy_index_utils.h"
#include "mechanism.h"

static bool is_node_pointer_target_type(int target_type) {
    return target_type == static_cast<int>(coreneuron::gap_idx_type::voltage) ||
           target_type == static_cast<int>(coreneuron::gap_idx_type::i_membrane_);
}

static int permute_node_index_canonical(int raw, int nnode, const int* node_permute_vec) {
    if (raw < 0) {
        return raw;
    }
    if (nnode <= 0) {
        return raw;
    }

    // NOTE:
    // - CoreNEURON stores some dparam indices as offsets into NrnThread::_data.
    // - HELIOX does not preserve that monolithic layout; we canonicalize node-referential
    //   semantics into direct node indices.
    // - For the legacy encoding "base + node_index", a robust decode is `node = raw % nnode`.
    int node_index = raw % nnode;
    if (node_index < 0 || node_index >= nnode) {
        return raw;
    }
    if (node_permute_vec) {
        node_index = node_permute_vec[node_index];
    }
    return node_index;
}

template <typename T>   //P:转换矩阵，data:被转换的数据
void permute(T* data, int cnt, int sz, int* p) {//mem_layout_util.cpp
    // data(p[icnt], isz) <- data(icnt, isz)
    // this does not change data, merely permutes it.
    // assert len(p) == cnt
    if (!p) 
    {
        return;
    }
    int n = cnt * sz;//设计的时候想支持二维数组，所以先转一维
    if (n < 1) 
    {
        return;
    }

    T* data_orig = new T[n];//拷贝一份原始数据
    for (int i = 0; i < n; ++i) {
        data_orig[i] = data[i];
    }

    for (int icnt = 0; icnt < cnt; ++icnt) {
        for (int isz = 0; isz < sz; ++isz) {
            // note that when layout==0, nrn_i_layout takes into account SoA padding.
            //int i = nrn_i_layout(icnt, cnt, isz, sz, layout);
            //int ip = nrn_i_layout(p[icnt], cnt, isz, sz, layout);

            /*对应的coreNeuron是
                将i和ip都进行如下转换：其中，IP调用的是icnt=p[icnt]
                case Layout::AoS:
                return icnt * sz + isz;
            */
            int i = icnt * sz + isz;    // data[cnt][sz]，算位置
            int ip = p[icnt] * sz + isz;//p是将第一维数据映射到其他位置的矩阵，只改变第一列数据
            data[ip] = data_orig[i];
        }
    }

    delete[] data_orig;
}

int* inverse_permute(int* p, int n) {//node_permute.cpp line 159
    int* pinv = new int[n];
    for (int i = 0; i < n; ++i) {
        pinv[p[i]] = i;
    }
    return pinv;
}

static void invert_permute(int* p, int n) {//node_permute.cpp line 167
    int* pinv = inverse_permute(p, n);
    for (int i = 0; i < n; ++i) {
        p[i] = pinv[i];
    }
    delete[] pinv;
}

/*void update_pdata_values(Memb_list* ml, int type, NrnThread& nt) {
    // assumes AoS to SoA transformation already made since we are using
    // nrn_i_layout to determine indices into both ml->pdata and into target data
    int psz = nrn_prop_dparam_size_[type];
    if (psz == 0) {
        return;
    }
    if (nrn_is_artificial_[type]) {
        return;
    }
    int* semantics = memb_func[type].dparam_semantics;
    if (!semantics) {
        return;
    }
    int* pdata = ml->pdata;
    int layout = nrn_mech_data_layout_[type];
    int cnt = ml->nodecount;
    // ml padding does not matter (but target padding does matter)

    // interesting semantics are -1 (area), -5 (pointer), -9 (diam), or 0-999 (ion variables)
    for (int i = 0; i < psz; ++i) {
        int s = semantics[i];
        if (s == -1) {                               // area
            int area0 = nt._actual_area - nt._data;  // includes padding if relevant
            int* p_target = nt._permute;
            for (int iml = 0; iml < cnt; ++iml) {
                int* pd = pdata + nrn_i_layout(iml, cnt, i, psz, layout);
                // *pd is the original integer into nt._data . Needs to be replaced
                // by the permuted value

                // This is ok whether or not area changed by padding?
                // since old *pd updated appropriately by earlier AoS to SoA
                // transformation
                int ix = *pd - area0;  // original integer into area array.
                nrn_assert((ix >= 0) && (ix < nt.end));
                int ixnew = p_target[ix];
                *pd = ixnew + area0;
            }
        } else if (s == -9) {                        // diam
            int diam0 = nt._actual_diam - nt._data;  // includes padding if relevant
            int* p_target = nt._permute;
            for (int iml = 0; iml < cnt; ++iml) {
                int* pd = pdata + nrn_i_layout(iml, cnt, i, psz, layout);
                // *pd is the original integer into nt._data . Needs to be replaced
                // by the permuted value

                // This is ok whether or not diam changed by padding?
                // since old *pd updated appropriately by earlier AoS to SoA
                // transformation
                int ix = *pd - diam0;  // original integer into actual_diam array.
                nrn_assert((ix >= 0) && (ix < nt.end));
                int ixnew = p_target[ix];
                *pd = ixnew + diam0;
            }
        } else if (s == -5) {  // assume pointer to membrane voltage
            int v0 = nt._actual_v - nt._data;
            // same as for area semantics
            int* p_target = nt._permute;
            for (int iml = 0; iml < cnt; ++iml) {
                int* pd = pdata + nrn_i_layout(iml, cnt, i, psz, layout);
                int ix = *pd - v0;  // original integer into area array.
                nrn_assert((ix >= 0) && (ix < nt.end));
                int ixnew = p_target[ix];
                *pd = ixnew + v0;
            }
        } else if (s >= 0 && s < 1000) {  // ion
            int etype = s;
            int elayout = nrn_mech_data_layout_[etype];
            Memb_list* eml = nt._ml_list[etype];
            int edata0 = eml->data - nt._data;
            int ecnt = eml->nodecount;
            int esz = nrn_prop_param_size_[etype];
            int* p_target = eml->_permute;
            for (int iml = 0; iml < cnt; ++iml) {
                int* pd = pdata + nrn_i_layout(iml, cnt, i, psz, layout);
                int ix = *pd - edata0;
                // from ix determine i_ecnt and i_esz (need to permute i_ecnt)
                int i_ecnt, i_esz, padded_ecnt;
                if (elayout == 1) {  // AoS
                    padded_ecnt = ecnt;
                    i_ecnt = ix / esz;
                    i_esz = ix % esz;
                } else {  // SoA
                    assert(elayout == 0);
                    padded_ecnt = nrn_soa_padded_size(ecnt, elayout);
                    i_ecnt = ix % padded_ecnt;
                    i_esz = ix / padded_ecnt;
                }
                int i_ecnt_new = p_target[i_ecnt];
                int ix_new = nrn_i_layout(i_ecnt_new, ecnt, i_esz, esz, elayout);
                *pd = ix_new + edata0;
            }
        }
    }
}*/

void node_permute(int* vec, int n, int* permute) {//node_permute, node_permute.cpp line 367
    for (int i = 0; i < n; ++i) {
        if (vec[i] >= 0) {
            vec[i] = permute[vec[i]];
        }
    }
}

void permute_ptr(int* vec, int n, int* p) {//node_permute.cpp line 388
    permute(vec, n, 1, p);
}

void permute_data(double* vec, int n, int* p) {//node_permute.cpp line 392
    permute(vec, n, 1, p);
}

void update_pdata_values(coreneuron::CoreMech* ml, coreneuron::CoreData* coredata) {
    if (!ml || !coredata || !coredata->mech_data) {
        return;
    }
    if (!ml->pdata) {
        return;
    }

    const int type = ml->type;
    const int dsz = coredata->mech_data->nrn_prop_dparam_size[type];
    if (dsz <= 0) {
        return;
    }

    // CoreNEURON-style fix-up after permutation:
    // - `pointer2type` stores the target type per POINTER instance (flattened in instance-major
    //   order, and only for POINTER slots).
    // - We must update `pdata` values that encode indices into permuted targets (node arrays or
    //   mechanism instance arrays) so they still refer to the same logical target after permute.
    //
    // After node/mechanism permutation, the stored indices must be updated so
    // they still refer to the same logical target.
    const auto& mech_name = coredata->mech_data->name_vec[type];
    const std::vector<int>* semantics = MechanismFactory::getInstance().getDparamSemantics(mech_name);

    const int nnode = coredata->end;
    int* node_permute_vec = coredata->permute; // maps original node index -> permuted node index

    // If we have dparam semantics, we can fix up non-POINTER dparam indices too (area/diam/ion).
    // This mirrors CoreNEURON's intent, but uses HELIOX's canonical encoding where node targets
    // are stored as direct node indices.
    const int inst_count = ml->nodecount;
    if (inst_count <= 0) {
        return;
    }
    if (semantics && static_cast<int>(semantics->size()) == dsz) {
        for (int inst = 0; inst < inst_count; ++inst) {
            for (int ip = 0; ip < dsz; ++ip) {
                const int s = semantics->at(ip);
                if (s == dpsem(DparamSemantics::pointer)) {
                    // POINTER slots require pointer2type metadata (semantics alone doesn't encode target type).
                    continue;
                }

                int& raw = ml->pdata[inst * dsz + ip];
                if (raw < 0) {
                    continue;
                }

                if (s == dpsem(DparamSemantics::area) || s == dpsem(DparamSemantics::diam)) {
                    raw = permute_node_index_canonical(raw, nnode, node_permute_vec);
                } else if (dpsem_is_ion_or_ionstyle(s)) {
                    const int target_type = dpsem_ion_mech_type(s);
                    if (target_type < 0 || target_type >= coredata->mech_data->nmech_type) {
                        continue;
                    }
                    coreneuron::CoreMech* target_ml = coredata->map_type2mechptr[target_type];
                    if (!target_ml) {
                        continue;
                    }

                    const auto& dims = coredata->mech_data->nrn_array_dims[target_type];
                    int target_inst = -1;
                    int var_index = -1;
                    int array_index = 0;
                    if (!dims.empty()) {
                        auto decoded = legacy2soaos_index(raw, dims);
                        target_inst = decoded[0];
                        var_index = decoded[1];
                        array_index = decoded[2];
                    } else {
                        const int sz = coredata->mech_data->nrn_prop_param_size[target_type];
                        if (sz > 0) {
                            target_inst = raw / sz;
                            var_index = raw % sz;
                            array_index = 0;
                        }
                    }

                    if (target_inst < 0 || target_inst >= target_ml->nodecount) {
                        continue;
                    }
                    if (target_ml->permute) {
                        target_inst = target_ml->permute[target_inst];
                    }

                    if (!dims.empty()) {
                        raw = soaos2legacy_index(target_inst, var_index, array_index, dims);
                    } else {
                        const int sz = coredata->mech_data->nrn_prop_param_size[target_type];
                        if (sz > 0) {
                            raw = target_inst * sz + var_index;
                        }
                    }
                } else {
                    // Other semantics (pntproc, cvodeieq, watch, etc.) are either stable under
                    // permutation or currently unused in HELIOX. Leave unchanged.
                }
            }
        }
    }

    // POINTER fix-up: requires pointer2type metadata (bbcore_write v1.8+).
    if (ml->pointer2type.empty()) {
        return;
    }

    // bbcore_write exports pointer2type in an instance-major order and only for POINTER slots:
    // for inst in nodecount { for each POINTER slot (in ascending dparam index) { push(type) } }.
    if (ml->pointer2type.size() % static_cast<size_t>(inst_count) != 0) {
        return;
    }
    const int ptr_slots = static_cast<int>(ml->pointer2type.size() / static_cast<size_t>(inst_count));
    if (ptr_slots <= 0) {
        return;
    }

    // Determine which dparam slots correspond to POINTERs for this mechanism.
    // We need the slot list because `pointer2type` is ordered by increasing dparam slot.
    std::vector<int> pointer_slots;
    pointer_slots.reserve(ptr_slots);
    if (semantics && static_cast<int>(semantics->size()) == dsz) {
        // CoreNEURON-style semantics: -5 == POINTER dparam slot.
        for (int i = 0; i < dsz; ++i) {
            if (semantics->at(i) == dpsem(DparamSemantics::pointer)) {
                pointer_slots.push_back(i);
            }
        }
        // Only accept the semantics-derived slot list if it matches the exported pointer2type arity.
        if (static_cast<int>(pointer_slots.size()) != ptr_slots) {
            pointer_slots.clear();
        }
    }

    if (pointer_slots.empty()) {
        if (const std::vector<int>* slots = MechanismFactory::getInstance().getPointerDparamSlots(mech_name);
            slots && static_cast<int>(slots->size()) == ptr_slots) {
            pointer_slots = *slots;
        } else {
            // Backward-compatible fallback: common nrnivmodl POINT_PROCESS layout places POINTERs at dparam[2..].
            const bool is_point_process = coredata->mech_data->pnt_map[type] > 0;
            if (!is_point_process) {
                // For non-point-process mechanisms, we cannot reliably infer POINTER slot indices without
                // additional semantic metadata. If you add a density mechanism with POINTERs, register its
                // dparam slot indices via REGISTER_POINTER_DPARAM_SLOTS or REGISTER_DPARAM_SEMANTICS.
                return;
            }
            for (int i = 0; i < ptr_slots; ++i) {
                pointer_slots.push_back(2 + i);
            }
        }
    }
    std::sort(pointer_slots.begin(), pointer_slots.end());

    // Keep pointer2type aligned with permuted instance order (ml->pdata was permuted already).
    // This permutes blocks of size ptr_slots per mechanism instance.
    permute(ml->pointer2type.data(), inst_count, ptr_slots, ml->permute);

    const bool debug_this_mech =
#ifdef DEBUG_PRINTF
        (::getenv("HELIOX_DEBUG_POINTER") != nullptr &&
         type >= 0 && type < static_cast<int>(coredata->mech_data->name_vec.size()) &&
         coredata->mech_data->name_vec[type] == "gapjunction_lr");
#else
        false;
#endif

    for (int inst = 0; inst < inst_count; ++inst) {
        for (int pslot = 0; pslot < ptr_slots; ++pslot) {
            const int ip = pointer_slots[pslot];
            if (ip < 0 || ip >= dsz) {
                continue;
            }

            int& raw = ml->pdata[inst * dsz + ip];
            if (raw < 0) {
                continue;
            }

            const int target_type = ml->pointer2type[inst * ptr_slots + pslot];

            if (debug_this_mech && inst < 8) {
                printf_debug("update_pdata_values[%s]: inst=%d dparam=%d raw(before)=%d target_type=%d\n",
                             coredata->mech_data->name_vec[type].c_str(), inst, ip, raw, target_type);
            }

            if (is_node_pointer_target_type(target_type)) {
                // Node-level targets: apply node permutation.
                int node_index = raw;
                if (node_permute_vec) {
                    if (node_index < 0 || node_index >= nnode) {
                        printf("update_pdata_values: invalid node_index=%d (raw=%d, nnode=%d)\n",
                               node_index, raw, nnode);
                        continue;
                    }
                    node_index = node_permute_vec[node_index];
                }

                // HELIOX canonical convention for node-level POINTERs: direct node index.
                raw = node_index;
            } else if (target_type > 0 && target_type < coredata->mech_data->nmech_type) {
                // Mechanism-level target:
                // Treat `raw` as a legacy flat index into that target mechanism's
                // instance-row layout, and apply the target mechanism's instance
                // permutation.
                const auto& dims = coredata->mech_data->nrn_array_dims[target_type];
                coreneuron::CoreMech* target_ml = coredata->map_type2mechptr[target_type];
                int target_inst = -1;
                int var_index = -1;
                int array_index = 0;
                if (!dims.empty()) {
                    auto decoded = legacy2soaos_index(raw, dims);
                    target_inst = decoded[0];
                    var_index = decoded[1];
                    array_index = decoded[2];
                } else {
                    // Pre-array_dims export fallback: treat each variable as scalar.
                    const int sz = coredata->mech_data->nrn_prop_param_size[target_type];
                    if (sz > 0) {
                        target_inst = raw / sz;
                        var_index = raw % sz;
                        array_index = 0;
                    }
                }

                if (target_ml && target_ml->permute) {
                    if (target_inst >= 0 && target_inst < target_ml->nodecount) {
                        target_inst = target_ml->permute[target_inst];
                    }
                }

                if (!dims.empty()) {
                    raw = soaos2legacy_index(target_inst, var_index, array_index, dims);
                } else {
                    const int sz = coredata->mech_data->nrn_prop_param_size[target_type];
                    if (sz > 0) {
                        raw = target_inst * sz + var_index;
                    }
                }
            } else {
                // Unknown/unhandled target type. Leave unchanged.
            }

            if (debug_this_mech && inst < 8) {
                printf_debug("update_pdata_values[%s]: inst=%d dparam=%d raw(after)=%d\n",
                             coredata->mech_data->name_vec[type].c_str(), inst, ip, raw);
            }
        }
    }
}

void permute_ml(coreneuron::CoreMech* ml, int type, coreneuron::CoreMechData* mdat, coreneuron::CoreData* coredata) {
    int sz = mdat->nrn_prop_param_size[type];
    int psz = mdat->nrn_prop_dparam_size[type];
    int layout = mdat->nrn_mech_data_layout[type];
    permute(ml->_data, ml->nodecount, sz, ml->permute);
    permute(ml->pdata, ml->nodecount, psz, ml->permute);

    // CoreNEURON calls `update_pdata_values(ml, type, nt)` here.
    // We mirror that structure so future semantics-driven support can slot in.
    update_pdata_values(ml, coredata);
}

int nrn_index_permute(int ix, int type, coreneuron::CoreMech* ml, coreneuron::CoreMechData* mdat) {
    int* p = ml->permute;
    if (!p) {
        return ix;
    }
    int sz = mdat->nrn_prop_param_size[type];
    int icnt = ix / sz;
    int isz = ix % sz;
    return p[icnt] * sz + isz;
    /*int layout = nrn_mech_data_layout_[type];
    if (layout == 1) {
        int sz = nrn_prop_param_size_[type];
        int i_cnt = ix / sz;
        int i_sz = ix % sz;
        return p[i_cnt] * sz + i_sz;
    } else {
        assert(layout == 0);
        int padded_cnt = nrn_soa_padded_size(ml->nodecount, layout);
        int i_cnt = ix % padded_cnt;
        int i_sz = ix / padded_cnt;
        return i_sz * padded_cnt + p[i_cnt];
    }*/
}

#if 0
static void pr(const char* s, int* x, int n) {
  printf("%s:", s);
  for (int i=0; i < n; ++i) {
    printf("  %d %d", i, x[i]);
  }
  printf("\n");
}

static void pr(const char* s, double* x, int n) {
  printf("%s:", s);
  for (int i=0; i < n; ++i) {
    printf("  %d %g", i, x[i]);
  }
  printf("\n");
}
#endif

// note that sort_indices has the sense of an inverse permutation in that
// the value of sort_indices[0] is the index with the smallest value in the
// indices array

static bool nrn_index_sort_cmp(const std::pair<int, int>& a, const std::pair<int, int>& b) {
    bool result = false;
    if (a.first < b.first) {
        result = true;
    } else if (a.first == b.first) {
        if (a.second < b.second) {
            result = true;
        }
    }
    return result;
}

int* nrn_index_sort(int* values, int n) {//node_permute.cpp line 443
    std::vector<std::pair<int, int> > vi(n);
    for (int i = 0; i < n; ++i) {
        vi[i].first = values[i];
        vi[i].second = i;
    }
    std::sort(vi.begin(), vi.end(), nrn_index_sort_cmp);
    int* sort_indices = new int[n];
    for (int i = 0; i < n; ++i) {
        sort_indices[i] = vi[i].second;
    }
    return sort_indices;
}

void permute_nodeindices(coreneuron::CoreMech* ml, int* p) {
    // nodeindices values are permuted according to p (that per se does
    //  not affect vec).

    node_permute(ml->nodeindices, ml->nodecount, p);

    // Then the new node indices are sorted by
    // increasing index. Instances using the same node stay in same
    // original relative order so that their contributions to rhs, d (if any)
    // remain in same order (except for gpu parallelism).
    // That becomes ml->_permute

    ml->permute = nrn_index_sort(ml->nodeindices, ml->nodecount);
    invert_permute(ml->permute, ml->nodecount);
    permute_ptr(ml->nodeindices, ml->nodecount, ml->permute);
}
