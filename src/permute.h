#pragma once
#include "coredat_structs.h"


int* inverse_permute(int* p, int n);
void node_permute(int* vec, int n, int* permute);
void permute_ptr(int* vec, int n, int* p);
void permute_data(double* vec, int n, int* p);
void permute_ml(coreneuron::CoreMech* ml, int type, coreneuron::CoreMechData* mdat, coreneuron::CoreData* coredata);
void permute_nodeindices(coreneuron::CoreMech* ml, int* p);

// CoreNEURON-style fix-up step after permutation:
// Some entries in `pdata` encode indices ("pointers") into other permuted arrays
// (e.g. node voltage). After permutation, the stored indices must be remapped so
// they still refer to the same original target.
//
// This function is intentionally narrow today (focused on point-process POINTERs),
// but is structured to match CoreNEURON's `update_pdata_values` call-site.
void update_pdata_values(coreneuron::CoreMech* ml, coreneuron::CoreData* coredata);
