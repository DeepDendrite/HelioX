"""
heliox_core.binding
===================

Small utilities to bind NEURON objects to HELIOX variable handles.

Why this exists
---------------
- NEURON `_ref_*` handles embed a `row=...` index in their string representation.
- That `row` is the correct "linearized" index for CoreNEURON/HELIOX access, and it can
  change after `h.finitialize()` / `pc.nrnbbcore_write()`.
- Many applications want a *batch* of handles for high-throughput IO, without re-implementing
  the same fragile parsing logic.

This module intentionally stays small and "plumbing-only":
- It does NOT allocate gids or create NetCons (that is model/network semantics, not core IO).
- It does NOT decide *which* variables to bind (that remains application logic).
"""

from __future__ import annotations

import re
from typing import Any, Tuple


def parse_row_from_ref(ref: Any) -> int:
    """Parse `row=<int>` from a NEURON `_ref_*` handle (or its string)."""
    match = re.search(r"row=(\d+)", str(ref))
    if not match:
        raise RuntimeError(f"Failed to parse row= from ref: {ref!r}")
    return int(match.group(1))


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


def resolve_mech_and_row(obj: Any, var_name: str, array_index: int = 0) -> Tuple[str, int]:
    """
    Resolve `(mech_name, row)` for accessing `obj.var_name[array_index]` in HELIOX.

    Rules:
    - Segment variables use mech_name="global" and row parsed from `obj._ref_<var_name>`.
    - Mechanism variables use mech_name extracted from the mechanism, row parsed from
      `obj._ref_<var_name>` (or an element ref for hoc array refs).
    """
    ref_name = f"_ref_{var_name}"

    # Segment (compartment) variable
    if hasattr(obj, "node_index"):
        ref = getattr(obj, ref_name)
        return "global", parse_row_from_ref(ref)

    # Mechanism variable
    mech_name = extract_mech_name(obj)
    ref = getattr(obj, ref_name)
    ref_str = str(ref)

    # Hoc array refs show as "incomplete pointer to hoc array"
    if "incomplete pointer to hoc array" in ref_str:
        ref_elem = ref[array_index]
        return mech_name, parse_row_from_ref(ref_elem)

    return mech_name, parse_row_from_ref(ref)


def get_variable_handle(manager: Any, obj: Any, var_name: str, array_index: int = 0) -> int:
    """
    Bind a NEURON object variable to a HELIOX handle.

    Parameters
    ----------
    manager:
        `heliox_wrapper.HelioXManager` (re-exported as `heliox_core.HelioXManager`).
    obj:
        NEURON Segment or mechanism object.
    var_name:
        Variable name (e.g. "v", "amp", "pure_i").
    array_index:
        For hoc array variables, element index.
    """
    mech_name, row = resolve_mech_and_row(obj, var_name, array_index=array_index)
    handle = int(manager.client.get_variable_handle_with_array(mech_name, var_name, int(row), int(array_index)))
    if handle < 0:
        raise RuntimeError(f"Failed to get handle for {mech_name}.{var_name}[row={row}][{array_index}]")
    return handle
