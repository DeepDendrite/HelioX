from __future__ import annotations

"""
Neurong in-memory frontend helpers.

This module hosts descriptor-based wrappers used by pure neurong frontends.
It is intentionally separated from `NeurongManager` (NRN-export workflow).
"""

from typing import Dict, Optional

import neurong
import numpy as np


class _InMemArrayVarProxy:
    """
    Baseline-like array variable proxy backed by NeuronG handles.
    """

    def __init__(self, owner: "InMemObjWrapper", var_name: str, length: int):
        self._owner = owner
        self._var_name = str(var_name)
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._length)
            return [self._owner.get_var(self._var_name, idx) for idx in range(start, stop, step)]
        idx = int(item)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Array index out of range: {idx}, length={self._length}")
        return self._owner.get_var(self._var_name, idx)

    def __setitem__(self, item, value) -> None:
        if isinstance(item, slice):
            start, stop, step = item.indices(self._length)
            indices = list(range(start, stop, step))
            values = list(value)
            if len(indices) != len(values):
                raise ValueError(
                    f"Slice assignment size mismatch: expected {len(indices)}, got {len(values)}"
                )
            for idx, v in zip(indices, values):
                self._owner.set_var(self._var_name, float(v), idx)
            return
        idx = int(item)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Array index out of range: {idx}, length={self._length}")
        self._owner.set_var(self._var_name, float(value), idx)


class InMemObjWrapper:
    """
    Descriptor-based ObjWrapper for in-memory frontend.
    """

    def __init__(
        self,
        sim: object,
        mech_name: str,
        gid: int,
        section_name: str,
        loc: float,
        slot: int = 0,
        var_array_lengths: Optional[Dict[str, int]] = None,
    ):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_mech_name", str(mech_name))
        object.__setattr__(self, "_gid", int(gid))
        object.__setattr__(self, "_section_name", str(section_name))
        object.__setattr__(self, "_loc", float(InMemObjWrapper._normalize_loc(float(loc))))
        object.__setattr__(self, "_slot", int(slot))
        object.__setattr__(self, "_var_array_lengths", dict(var_array_lengths or {}))
        # Bind hot-path runtime methods once to avoid repeated attribute lookup.
        object.__setattr__(self, "_resolve_handle_by_loc", sim.get_variable_handle_with_array_by_loc)
        object.__setattr__(self, "_get_variable_by_handle", sim.get_variable_by_handle)
        object.__setattr__(self, "_set_variable_by_handle", sim.set_variable_by_handle)
        # Fast-path caches:
        # - scalars (array_index==0): var -> handle
        # - arrays  (array_index>0): var -> {array_index -> handle}
        object.__setattr__(self, "_scalar_handles", {})
        object.__setattr__(self, "_array_handles", {})

    @staticmethod
    def _normalize_loc(loc: float) -> float:
        x = float(loc)
        if not np.isfinite(x):
            raise ValueError(f"loc must be finite, got {loc}")
        if x < 0.0 or x > 1.0:
            raise ValueError(f"loc out of range (0 <= loc <= 1), got {loc}")
        return float(x)

    def get_handle(self, var_name: str, array_index: int = 0) -> int:
        var = str(var_name)
        arr = int(array_index)
        if arr == 0:
            cached = self._scalar_handles.get(var)
            if cached is not None:
                return cached
        else:
            by_var = self._array_handles.get(var)
            if by_var is not None:
                cached = by_var.get(arr)
                if cached is not None:
                    return cached

        handle = int(
            self._resolve_handle_by_loc(
                self._mech_name,
                var,
                self._gid,
                self._section_name,
                self._loc,
                arr,
                self._slot,
            )
        )
        if handle < 0:
            raise RuntimeError(
                "Failed to resolve variable handle: "
                f"mech={self._mech_name}, var={var}, gid={self._gid}, section={self._section_name}, "
                f"loc={self._loc}, slot={self._slot}, array_index={arr}"
            )
        if arr == 0:
            self._scalar_handles[var] = int(handle)
        else:
            by_var = self._array_handles.get(var)
            if by_var is None:
                by_var = {}
                self._array_handles[var] = by_var
            by_var[arr] = int(handle)
        return handle

    def get_var(self, var_name: str, array_index: int = 0) -> float:
        arr = int(array_index)
        handle = self.get_handle(var_name, arr)
        return float(self._get_variable_by_handle(handle))

    def set_var(self, var_name: str, value: float, array_index: int = 0) -> None:
        arr = int(array_index)
        handle = self.get_handle(var_name, arr)
        rc = int(self._set_variable_by_handle(handle, float(value)))
        if rc != 0:
            raise RuntimeError(
                f"Failed to set variable by handle: mech={self._mech_name}, var={var_name}, "
                f"gid={self._gid}, section={self._section_name}, loc={self._loc}, "
                f"slot={self._slot}, array_index={array_index}"
            )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        length = self._var_array_lengths.get(str(name))
        if length is not None and int(length) > 1:
            return _InMemArrayVarProxy(self, str(name), int(length))
        return self.get_var(str(name), 0)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        length = self._var_array_lengths.get(str(name))
        if length is not None and int(length) > 1:
            values = list(value)
            if len(values) != int(length):
                raise ValueError(
                    f"Array assignment size mismatch: var={name}, expected={int(length)}, got={len(values)}"
                )
            for idx, v in enumerate(values):
                self.set_var(str(name), float(v), idx)
            return
        self.set_var(str(name), float(value), 0)


# Morph helper aliases for pure neurong frontend flow.
section = neurong.section
load_swc_sections = neurong.load_swc_sections
load_asc_sections = neurong.load_asc_sections
sections_with_label = neurong.sections_with_label
delete_section = neurong.delete_section
delete_section_subtree = neurong.delete_section_subtree
delete_section_label = neurong.delete_section_label
section_distance_um = neurong.section_distance_um
section_diam_um_by_loc = neurong.section_diam_um_by_loc
section_segment_index_by_loc = neurong.section_segment_index_by_loc
