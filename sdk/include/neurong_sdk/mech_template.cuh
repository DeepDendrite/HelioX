#pragma once

// Public SDK entrypoint for MechTemp-based mechanism authoring.
//
// The implementation still lives in NeuronG's template-internal headers so the
// runtime and external plugins share the same variable, tape, gradient, and
// adjoint machinery. External mechanism packages should include this wrapper
// instead of depending on the internal header layout directly.
#if __has_include(<neurong_template_internal/mech/mech_template.cuh>)
#include <neurong_template_internal/mech/mech_template.cuh>
#elif __has_include(<mech_template.cuh>)
#include <mech_template.cuh>
#else
#error "NeuronG MechTemp headers were not found. Build external mechanisms with neurong-mech-build or link NeuronG::MechTemplate."
#endif
