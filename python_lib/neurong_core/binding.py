"""
neurong_core.binding
===================

Small utilities to bind NEURON objects to NEURONG variable handles.

Why this exists
---------------
- NEURON `_ref_*` handles embed a `row=...` index in their string representation.
- That `row` is the correct "linearized" index for CoreNEURON/NEURONG access, and it can
  change after `h.finitialize()` / `pc.nrnbbcore_write()`.
- Many applications want a *batch* of handles for high-throughput IO, without re-implementing
  the same fragile parsing logic.

This module intentionally stays small and "plumbing-only":
- It does NOT allocate gids or create NetCons (that is model/network semantics, not core IO).
- It does NOT decide *which* variables to bind (that remains application logic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class VariableDescriptor:
    """
    Backend descriptor for a NEURONG variable.

    This is not a user-facing object model. It is the normalized tuple passed
    from Python binding code to the C++ runtime:
    `(mech, var, index, array_index)`.

    Descriptor contract:
    - `mech == "global"` means `index` is a skeleton/original node index.
    - `mech != "global"` means `index` is an exported mechanism instance row.

    User code should not construct internal/permuted rows; mechanism
    variables should be bound from concrete NEURON object refs whenever
    possible.
    """

    mech: str
    var: str
    index: int
    array_index: int = 0

def try_parse_row_from_ref(ref: Any) -> Optional[int]:
    """Best-effort parse of `row=<int>` from a NEURON `_ref_*` handle."""
    try:
        match = re.search(r"row=(\d+)", str(ref))
    except Exception:
        return None
    if not match:
        return None
    return int(match.group(1))


def parse_row_from_ref(ref: Any) -> int:
    """Parse `row=<int>` from a NEURON `_ref_*` handle (or its string)."""
    row = try_parse_row_from_ref(ref)
    if row is None:
        raise RuntimeError(f"Failed to parse row= from ref: {ref!r}")
    return row


def extract_mech_name(mech: Any) -> str:
    """Best-effort mechanism name extraction from a NEURON point/range mechanism object."""
    try:
        hname = mech.hname()
        if "[" in hname:
            return hname.split("[")[0]
        return hname
    except Exception:
        pass

    # Fallback: parse from repr like `IClamp[0]` or `gapjunction_lr[12]`.
    repr_str = repr(mech)
    match = re.search(r"([A-Za-z][A-Za-z0-9_]*)\[", repr_str)
    if match:
        return match.group(1)

    # Final fallback: type name.
    obj_str = str(type(mech))
    match = re.search(r"'([A-Za-z][A-Za-z0-9_]*)'", obj_str)
    if match:
        return match.group(1)

    # Most permissive fallback: some range mechanisms stringify as just their mechanism name
    # (e.g. `slo1_unc2_lr`). In that case, use `str(mech)` as the "mech_name" key.
    name = str(mech).strip()
    if name:
        return name
    return repr_str


def resolve_variable_descriptor(
    obj: Any,
    var_name: str,
    array_index: int = 0,
    *,
    allow_segment_node_index_fallback: bool = True,
) -> VariableDescriptor:
    """
    Resolve a NEURON Segment/mechanism variable into the backend descriptor.

    This should run after `h.finitialize()` / `pc.nrnbbcore_write()`, when
    NEURON `_ref_*` strings expose the exported row values. It is intended for
    binding time only; hot reads/writes should use cached backend handles.
    """
    ref_name = f"_ref_{var_name}"

    # Segment (compartment) variable
    if hasattr(obj, "node_index"):
        try:
            ref = getattr(obj, ref_name)
            row = try_parse_row_from_ref(ref)
        except Exception:
            row = None
        if row is None:
            if not allow_segment_node_index_fallback:
                raise RuntimeError(f"Failed to parse row= from segment {ref_name}")
            row = int(obj.node_index())
        return VariableDescriptor("global", var_name, int(row), int(array_index))

    # Mechanism variable
    mech_name = extract_mech_name(obj)
    try:
        ref = getattr(obj, ref_name)
    except AttributeError as exc:
        raise ValueError(f"Mechanism {mech_name} does not have variable {var_name}") from exc
    ref_str = str(ref)

    # Hoc array refs show as "incomplete pointer to hoc array"
    if "incomplete pointer to hoc array" in ref_str:
        ref_elem = ref[array_index]
        row = parse_row_from_ref(ref_elem)
        return VariableDescriptor(mech_name, var_name, int(row), int(array_index))

    row = parse_row_from_ref(ref)
    return VariableDescriptor(mech_name, var_name, int(row), int(array_index))


def get_variable_handle_from_descriptor(manager: Any, descriptor: VariableDescriptor) -> int:
    """Bind a descriptor to a NEURONG variable handle."""
    client = manager.client if hasattr(manager, "client") else getattr(manager, "_client", manager)
    handle = int(
        client.get_variable_handle_with_array(
            descriptor.mech,
            descriptor.var,
            int(descriptor.index),
            int(descriptor.array_index),
        )
    )
    if handle < 0:
        raise RuntimeError(
            f"Failed to get handle for "
            f"{descriptor.mech}.{descriptor.var}[index={descriptor.index}][{descriptor.array_index}]"
        )
    return handle


def get_variable_handle(manager: Any, obj: Any, var_name: str, array_index: int = 0) -> int:
    """
    Bind a NEURON object variable to a NEURONG handle.

    Parameters
    ----------
    manager:
        `neurong_wrapper.NeurongManager` (re-exported as `neurong_core.NeurongManager`).
    obj:
        NEURON Segment or mechanism object.
    var_name:
        Variable name (e.g. "v", "amp", "pure_i").
    array_index:
        For hoc array variables, element index.
    """
    descriptor = resolve_variable_descriptor(obj, var_name, array_index=array_index)
    return get_variable_handle_from_descriptor(manager, descriptor)
