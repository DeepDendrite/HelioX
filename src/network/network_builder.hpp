#pragma once

#include "biophysical/biophys_builder.hpp"
#include "neuron.h"

#include <string>
#include <vector>

namespace neurong_network {

enum class PreEndpointKind {
    RealCell = 0,
    ArtificialByLoc = 1,
};

// Single connection semantic unit used by the in-memory network frontend.
// Field names intentionally match the Python/front-end contract:
// - real pre connection:
//     [id], preGid, [preSection+preSectionLocation], threshold,
//     postGid, postSection, postSectionLocation, postMech, postParams{weight, delay}
// - artificial pre connection:
//     [id], [preGid], preMech+[preSection+preSectionLocation],
//     postGid, postSection, postSectionLocation, postMech, postParams{weight, delay}
struct ConnectionSpec {
    int id{0};
    // Global pre-source index in the spike vector:
    // - [0 .. n_presyn_real)   : real-cell threshold detectors (resolved per unique pre node)
    // - [n_presyn_real .. ...) : artificial-cell sources (e.g. VecStim/NetStim instances)
    int preCellId{-1};
    PreEndpointKind preKind{PreEndpointKind::RealCell};
    // Source gid before preCellId resolution.
    int preGid{-1};
    // Only used when preKind == ArtificialByLoc.
    std::string preMech{};
    std::string preSection{};
    double preSectionLocation{0.5};
    int postCellId{-1};
    bool postSectionIsIndex{false};
    int postSectionIndex{-1};
    std::string postSection{};
    double postSectionLocation{0.5};
    std::string postMech{};
    double threshold{10.0};
    double weight{0.0};
    double delay{0.0};
};

struct LoadOptions {
    double dt{0.025};
};

struct LoadResult {
    int n_presyn_real{0};
    int n_presyn_arti{0};
    int n_connections{0};
    int n_bound_connections{0};
};

void setup_spike_runtime(Mode mode,
                         NeuronGroupData& group,
                         const neurong_biophysical::CellTemplateMorphLayout& morph,
                         std::vector<ConnectionSpec>& connections);

LoadResult load_connections(const LoadOptions& opt,
                            Mode mode,
                            NeuronGroupData& group,
                            const neurong_biophysical::CellTemplateMorphLayout& morph,
                            const std::vector<ConnectionSpec>& connections);

}  // namespace neurong_network
