# `heliox_learn` (HELIOX Learning Framework, WIP)

This package is a small, **contract-focused** learning framework layer for HELIOX.

For simulation-only use (no learning), prefer the core wrapper layer:

```python
from heliox_core import HelioXManager
```

It is designed around a frontend/backend split:

- **NEURON** is the *model authoring frontend* (build sections/cells/mechanisms, then export).
- **HELIOX** is the *runtime backend* (forward simulation + optional learning kernels).

The framework goal is to make training look like calling an API (compile → train), while
encoding the tricky NEURON/HELIOX export contracts in one place.

## Quickstart (conceptual)

```python
from heliox_learn import CompileConfig, Runtime, RuntimeConfig, Trainer, TrainerConfig
from heliox_learn import compile_from_neuron

def build_model():
    # build in NEURON (sections, cells, NetCons...)
    ...

bundle = compile_from_neuron(
    export_dir="bundle/my_case",
    build_fn=build_model,
    cfg=CompileConfig(dt=0.1, v_init=-65, device="gpu", permute_type=3, overwrite=False),
)

rt = Runtime(bundle, cfg=RuntimeConfig(device="gpu", permute_type=3))

trainer = Trainer(runtime=rt, cfg=TrainerConfig(output_dir="runs/my_case", resume=True))
result = trainer.fit(task=..., epochs=50)
```

## Core contracts

### Export timing (row-index stability)

NEURON `_ref_*` variables embed `row=...` indices. These indices can change after
`h.finitialize()` or during export. Therefore:

- Do **not** parse/cache `_ref_* row=` indices before export.
- Export must happen only after NEURON has finalized the model (`finitialize`).
- Handle binding must be resolved after export+load (via bundle metadata).

This contract is enforced by `compile_from_neuron(...)`.

### No mixed runtime

Training hot loops should not step NEURON and HELIOX in the same run:

- NEURON is for build/export only.
- HELIOX is the runtime backend for forward simulation and learning.

Keeping this boundary clean is a key part of “productization”.

## Status

This is intentionally minimal and evolving:

- `ExportBundle` + `compile_from_neuron` define the export artifact.
- `Runtime` loads a bundle and executes in HELIOX.
- `Trainer` is a simple orchestration loop that delegates epoch logic to a `LearningTask`.

Older internal planning/roadmap notes have been removed to keep the repo clean; refer to Git history if you need the original proposals.
