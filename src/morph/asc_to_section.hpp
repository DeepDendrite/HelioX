#pragma once

#include "swc_to_section.hpp"

#include <string>

namespace neurong_morph {

// Parse a Neurolucida ASC morphology file and build Morph sections/pt3d.
//
// Design notes:
// - Parsing/branch semantics follow MorphIO's ASC reader behavior for the core
//   neurite grammar (CellBody/Axon/Dendrite/Apical + split branches '|').
// - Top-level non-neurite blocks (ImageCoords, user lines, etc.) are ignored.
// - Child sections prepend parent terminal point when needed (MorphIO-equivalent)
//   so topology can be represented with NEURON-style section connections.
Morph load_asc_morphology(const std::string& asc_path);

}  // namespace neurong_morph
