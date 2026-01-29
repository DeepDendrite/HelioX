#pragma once

#include <array>
#include <numeric>
#include <vector>

// Helpers for interpreting "legacy" flattened indices that refer to mechanism
// variables laid out as a row per instance, and columns as a concatenation of
// (possibly-array) variables.
//
// This matches the encoding/decoding used in CoreNEURON export paths and is
// reused for VecPlay, gap-transfer, and POINTER resolution.

inline int legacy_row_width(const std::vector<int>& array_dims) {
    return std::accumulate(array_dims.begin(), array_dims.end(), 0);
}

// Input: legacy flat index + per-variable array sizes.
// Output: {instance_index, variable_index, array_index}.
inline std::array<int, 3> legacy2soaos_index(int legacy_index, const std::vector<int>& array_dims) {
    const int variable_count = static_cast<int>(array_dims.size());
    const int row_width = legacy_row_width(array_dims);

    const int column_index = legacy_index % row_width;
    const int instance_index = legacy_index / row_width;

    int variable_index = 0;
    int prefix_sum = 0;
    for (int k = 0; k < variable_count - 1; ++k) {
        if (column_index >= prefix_sum + array_dims[k]) {
            prefix_sum += array_dims[k];
            variable_index = k + 1;
        } else {
            break;
        }
    }
    const int array_index = column_index - prefix_sum;
    return {instance_index, variable_index, array_index};
}

inline int soaos2legacy_index(int instance_index,
                              int variable_index,
                              int array_index,
                              const std::vector<int>& array_dims) {
    const int row_width = legacy_row_width(array_dims);
    int prefix_sum = 0;
    for (int k = 0; k < variable_index; ++k) {
        prefix_sum += array_dims[k];
    }
    return instance_index * row_width + prefix_sum + array_index;
}

