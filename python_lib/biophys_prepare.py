from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple


class BioValue:
    @staticmethod
    def fixed(value: object) -> float:
        return float(value)

    @staticmethod
    def uniform(low: object, high: object) -> Dict[str, object]:
        low_value = float(low)
        high_value = float(high)
        if high_value < low_value:
            raise ValueError("uniform random spec requires high >= low")
        return {
            "distribution": "uniform",
            "low": low_value,
            "high": high_value,
        }

    @staticmethod
    def normal(mean: object, std: object) -> Dict[str, object]:
        mean_value = float(mean)
        standard_deviation = float(std)
        if standard_deviation < 0.0:
            raise ValueError("normal random spec requires std >= 0")
        return {
            "distribution": "normal",
            "mean": mean_value,
            "std": standard_deviation,
        }


@dataclass(frozen=True)
class InsertOp:
    label: str
    mech: str
    params: Dict[str, object]


@dataclass(frozen=True)
class IonSpeciesOp:
    label: str
    ion: str
    model: str
    params: Dict[str, object]


@dataclass(frozen=True)
class SpecificCapacitanceOp:
    label: str
    value: object


@dataclass(frozen=True)
class ResistivityOp:
    label: str
    value: object


class BiophysPrepare:
    @staticmethod
    def _normalize_random_spec_dict(d: Dict[object, object]) -> Dict[str, object]:
        raw: Dict[str, object] = {str(k): v for (k, v) in d.items()}
        distribution = str(raw.get("distribution", "")).strip().lower()
        if not distribution:
            raise ValueError("random spec dict requires key 'distribution'")
        out: Dict[str, object] = {"distribution": distribution}
        if distribution == "fixed":
            if "value" not in raw:
                raise ValueError("fixed random spec requires key 'value'")
            out["value"] = float(raw["value"])
            return out
        if distribution == "uniform":
            if "low" not in raw or "high" not in raw:
                raise ValueError("uniform random spec requires keys 'low' and 'high'")
            out["low"] = float(raw["low"])
            out["high"] = float(raw["high"])
            if out["high"] < out["low"]:
                raise ValueError("uniform random spec requires high >= low")
            return out
        if distribution == "normal":
            if "mean" not in raw:
                raise ValueError("normal random spec requires key 'mean'")
            if "std" not in raw:
                raise ValueError("normal random spec requires key 'std'")
            out["mean"] = float(raw["mean"])
            out["std"] = float(raw["std"])
            if out["std"] < 0.0:
                raise ValueError("normal random spec requires std >= 0")
            return out
        raise ValueError("random distribution must be one of: fixed, uniform, normal")

    @staticmethod
    def _normalize_section_ref(section: object) -> Tuple[str, str, int]:
        sec_name = getattr(section, "name", None)
        sec_label = getattr(section, "label", None)
        sec_nseg = getattr(section, "nseg", None)

        if sec_name is None or sec_label is None:
            raise TypeError("section must provide 'name' and 'label' attributes")
        try:
            name = str(sec_name)
            label = str(sec_label)
        except Exception as exc:
            raise TypeError("failed to stringify section name/label") from exc
        if not name:
            raise ValueError("section name must be non-empty")
        if not label:
            raise ValueError("section label must be non-empty")

        if sec_nseg is None:
            nseg = 1
        else:
            try:
                nseg = int(sec_nseg)
            except Exception as exc:
                raise TypeError("section nseg must be an int-like value") from exc
            if nseg < 1:
                raise ValueError("section nseg must be >= 1")
        return name, label, nseg

    @staticmethod
    def _normalize_label(label: object) -> str:
        label_name = str(label)
        if not label_name:
            raise ValueError("label must be non-empty; use 'all' to target all labels")
        return label_name

    @staticmethod
    def _coerce_section_scalar_value(value: object) -> object:
        if isinstance(value, dict):
            if "distribution" in value:
                return BiophysPrepare._normalize_random_spec_dict(value)
            out: Dict[str, object] = {}
            for k, v in value.items():
                out[str(k)] = BiophysPrepare.LabelScope._coerce_section_map_entry(v)
            return out
        if isinstance(value, (list, tuple)):
            raise TypeError(
                "section property value must be scalar, random spec dict, or dict[section_name -> sequence|random]"
            )
        try:
            return float(value)
        except Exception as exc:
            raise TypeError(f"unsupported section property value type: {type(value)}") from exc

    class LabelScope:
        def __init__(self, prepare: "BiophysPrepare", label: str):
            object.__setattr__(self, "_prepare", prepare)
            object.__setattr__(self, "_label", BiophysPrepare._normalize_label(label))

        @staticmethod
        def _coerce_section_map_entry(value: object) -> object:
            if isinstance(value, dict):
                if "distribution" in value:
                    return BiophysPrepare._normalize_random_spec_dict(value)
                raise TypeError("section param map dict values must be sequence or random spec dict")
            if isinstance(value, (list, tuple)):
                seq = value
            elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                seq = list(value)
            else:
                raise TypeError("section param values must be sequences or random spec dict")
            return [float(x) for x in seq]

        @staticmethod
        def _coerce_param_value(value: object) -> object:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                if "distribution" in value:
                    return BiophysPrepare._normalize_random_spec_dict(value)
                out: Dict[str, object] = {}
                for k, v in value.items():
                    out[str(k)] = BiophysPrepare.LabelScope._coerce_section_map_entry(v)
                return out
            if isinstance(value, (list, tuple)):
                raise TypeError("flat list params are not supported; use dict[section_name -> list]")
            try:
                return float(value)
            except Exception as exc:
                raise TypeError(f"unsupported param value type: {type(value)}") from exc

        def insert(
            self,
            mech: str,
            params: Optional[Dict[str, object]] = None,
            **kwargs: object,
        ) -> "BiophysPrepare.LabelScope":
            merged: Dict[str, object] = {}
            if params is not None:
                merged.update({str(k): self._coerce_param_value(v) for (k, v) in params.items()})
            if kwargs:
                merged.update({str(k): self._coerce_param_value(v) for (k, v) in kwargs.items()})
            self._prepare.insert(self._label, mech, merged)
            return self

        def insert_inhomogeneous(
            self,
            mech: str,
            *,
            sections: Iterable[object],
            reference_gbar: object,
            exp_offset: object,
            exp_rate: object,
            exp_shift: object,
            exp_scale: object,
            apic_label: str = "apic",
            origin_label: str = "soma",
            params: Optional[Dict[str, object]] = None,
            param_name: str = "gbar",
            **kwargs: object,
        ) -> "BiophysPrepare.LabelScope":
            self._prepare.insert_inhomogeneous(
                self._label,
                mech,
                sections=sections,
                apic_label=apic_label,
                origin_label=origin_label,
                reference_gbar=reference_gbar,
                exp_offset=exp_offset,
                exp_rate=exp_rate,
                exp_shift=exp_shift,
                exp_scale=exp_scale,
                params=params,
                param_name=param_name,
                **kwargs,
            )
            return self

        def specificCapacitance(self, value: object) -> "BiophysPrepare.LabelScope":
            coerced = BiophysPrepare._coerce_section_scalar_value(value)
            self._prepare.specificCapacitance(self._label, coerced)
            return self

        def resistivity(self, value: object) -> "BiophysPrepare.LabelScope":
            coerced = BiophysPrepare._coerce_section_scalar_value(value)
            self._prepare.resistivity(self._label, coerced)
            return self

        def ion_species(
            self,
            ion: str,
            model: str,
            params: Optional[Dict[str, object]] = None,
            **kwargs: object,
        ) -> "BiophysPrepare.LabelScope":
            merged: Dict[str, object] = {}
            if params is not None:
                merged.update({str(k): self._coerce_param_value(v) for (k, v) in params.items()})
            if kwargs:
                merged.update({str(k): self._coerce_param_value(v) for (k, v) in kwargs.items()})
            self._prepare.ion_species(self._label, ion, model, merged)
            return self

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        def __enter__(self) -> "BiophysPrepare.LabelScope":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class SectionScope:
        def __init__(self, prepare: "BiophysPrepare", section: object):
            object.__setattr__(self, "_prepare", prepare)
            sec_name, sec_label, sec_nseg = BiophysPrepare._normalize_section_ref(section)
            object.__setattr__(self, "_section_name", sec_name)
            object.__setattr__(self, "_label", sec_label)
            object.__setattr__(self, "_nseg", sec_nseg)

        def _coerce_for_section(self, value: object) -> object:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                if "distribution" in value:
                    return {
                        str(self._section_name): BiophysPrepare._normalize_random_spec_dict(value),
                    }
                out: Dict[str, object] = {}
                for k, v in value.items():
                    out[str(k)] = BiophysPrepare.LabelScope._coerce_section_map_entry(v)
                return out
            if isinstance(value, (list, tuple)):
                vals = [float(x) for x in value]
                if len(vals) != int(self._nseg):
                    raise ValueError(
                        f"section param list length must equal nseg ({self._nseg}), got {len(vals)}"
                    )
                return {str(self._section_name): vals}
            try:
                scalar = float(value)
            except Exception as exc:
                raise TypeError(f"unsupported param value type: {type(value)}") from exc
            return {str(self._section_name): [scalar] * int(self._nseg)}

        def _coerce_keyed_param(self, key: str, value: object) -> object:
            key_lower = str(key).strip().lower()
            # Ion metadata are not range vars; builder expects scalars.
            if key_lower in {"ion_charge", "erev", "ion_conci", "ion_conco"}:
                if isinstance(value, dict):
                    if "distribution" in value:
                        if key_lower == "ion_charge":
                            raise TypeError("ion_charge cannot be random")
                        return BiophysPrepare._normalize_random_spec_dict(value)
                if isinstance(value, (dict, list, tuple)):
                    raise TypeError(f"{key} must be a scalar number")
                try:
                    return float(value)
                except Exception as exc:
                    raise TypeError(f"{key} must be a scalar number") from exc
            return self._coerce_for_section(value)

        def insert(
            self,
            mech: str,
            params: Optional[Dict[str, object]] = None,
            **kwargs: object,
        ) -> "BiophysPrepare.SectionScope":
            merged: Dict[str, object] = {}
            if params is not None:
                merged.update({str(k): self._coerce_keyed_param(str(k), v) for (k, v) in params.items()})
            if kwargs:
                merged.update({str(k): self._coerce_keyed_param(str(k), v) for (k, v) in kwargs.items()})
            self._prepare.insert(self._label, mech, merged)
            return self

        def specificCapacitance(self, value: object) -> "BiophysPrepare.SectionScope":
            coerced = self._coerce_for_section(value)
            if isinstance(coerced, str):
                raise TypeError("specificCapacitance does not accept string values")
            self._prepare.specificCapacitance(self._label, coerced)
            return self

        def resistivity(self, value: object) -> "BiophysPrepare.SectionScope":
            coerced = self._coerce_for_section(value)
            if isinstance(coerced, str):
                raise TypeError("resistivity does not accept string values")
            self._prepare.resistivity(self._label, coerced)
            return self

        def ion_species(
            self,
            ion: str,
            model: str,
            params: Optional[Dict[str, object]] = None,
            **kwargs: object,
        ) -> "BiophysPrepare.SectionScope":
            merged: Dict[str, object] = {}
            if params is not None:
                merged.update({str(k): self._coerce_keyed_param(str(k), v) for (k, v) in params.items()})
            if kwargs:
                merged.update({str(k): self._coerce_keyed_param(str(k), v) for (k, v) in kwargs.items()})
            self._prepare.ion_species(self._label, ion, model, merged)
            return self

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        def __enter__(self) -> "BiophysPrepare.SectionScope":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def __init__(self) -> None:
        self._inserts: List[InsertOp] = []
        self._ion_species: List[IonSpeciesOp] = []
        self._specific_capacitance: List[SpecificCapacitanceOp] = []
        self._resistivity: List[ResistivityOp] = []

    def for_label(self, label: str) -> "BiophysPrepare.LabelScope":
        return BiophysPrepare.LabelScope(self, BiophysPrepare._normalize_label(label))

    def for_section(self, section: object) -> "BiophysPrepare.SectionScope":
        return BiophysPrepare.SectionScope(self, section)

    def for_sections(self, sections: Iterable[object]) -> List["BiophysPrepare.SectionScope"]:
        return [self.for_section(sec) for sec in sections]

    def insert(
        self,
        label: str,
        mech: str,
        params: Optional[Dict[str, object]] = None,
    ) -> None:
        label_name = BiophysPrepare._normalize_label(label)
        mech_name = str(mech)
        merged: Dict[str, object] = {}
        if params is not None:
            merged.update({str(k): BiophysPrepare.LabelScope._coerce_param_value(v) for (k, v) in params.items()})
        self._inserts.append(
            InsertOp(
                label=label_name,
                mech=mech_name,
                params=dict(merged),
            )
        )

    @staticmethod
    def compute_apic_exp_distribution(
        sections: Iterable[object],
        *,
        reference_gbar: object,
        exp_offset: object,
        exp_rate: object,
        exp_shift: object,
        exp_scale: object,
        apic_label: str = "apic",
        origin_label: str = "soma",
    ) -> Dict[str, List[float]]:
        section_list = list(sections)
        if not section_list:
            return {}

        apic_label_name = str(apic_label)
        origin_label_name = str(origin_label)
        reference_gbar_value = float(reference_gbar)
        exp_offset_value = float(exp_offset)
        exp_rate_value = float(exp_rate)
        exp_shift_value = float(exp_shift)
        exp_scale_value = float(exp_scale)

        apic_sections: List[int] = []
        names: List[str] = []
        labels: List[str] = []
        nsegs: List[int] = []
        lengths: List[float] = []
        parents: List[object] = []
        parentxs: List[float] = []

        for idx, sec in enumerate(section_list):
            sec_name, sec_label, sec_nseg = BiophysPrepare._normalize_section_ref(sec)
            names.append(sec_name)
            labels.append(sec_label)
            nsegs.append(sec_nseg)
            try:
                sec_length = float(getattr(sec, "L_um"))
            except Exception as exc:
                raise TypeError("section must provide numeric L_um for inhomogeneous insert") from exc
            if sec_length < 0.0:
                raise ValueError("section L_um must be >= 0")
            lengths.append(sec_length)
            parent = getattr(sec, "parent", None)
            parents.append(parent)
            if parent is None:
                parentxs.append(1.0)
            else:
                try:
                    parentxs.append(float(getattr(sec, "parentx")))
                except Exception as exc:
                    raise TypeError("section must provide numeric parentx for inhomogeneous insert") from exc
            if sec_label == apic_label_name:
                apic_sections.append(idx)

        if not apic_sections:
            return {}

        name_to_idx = {sec_name: idx for idx, sec_name in enumerate(names)}

        origin_idx = apic_sections[0]
        for idx in apic_sections:
            parent_label = None
            parent = parents[idx]
            if parent is not None:
                parent_idx = name_to_idx.get(str(parent))
                if parent_idx is not None:
                    parent_label = labels[parent_idx]
            if parent_label != apic_label_name:
                origin_idx = idx
                if parent_label == origin_label_name:
                    break

        apic_children = {idx: [] for idx in apic_sections}
        for idx in apic_sections:
            parent = parents[idx]
            if parent is None:
                continue
            parent_idx = name_to_idx.get(str(parent))
            if parent_idx in apic_children:
                apic_children[parent_idx].append(idx)

        start_dist: Dict[int, float] = {origin_idx: 0.0}
        stack = [origin_idx]
        while stack:
            idx = stack.pop()
            base = start_dist[idx]
            parent_L = lengths[idx]
            for child_idx in apic_children[idx]:
                start_dist[child_idx] = base + parent_L * parentxs[child_idx]
                stack.append(child_idx)

        max_len = 0.0
        for idx in apic_sections:
            if apic_children[idx]:
                continue
            dist_end = start_dist.get(idx, 0.0) + lengths[idx]
            if dist_end > max_len:
                max_len = dist_end
        if max_len <= 0.0:
            max_len = lengths[origin_idx]

        out: Dict[str, List[float]] = {}
        for idx in apic_sections:
            nseg = nsegs[idx]
            if nseg <= 0:
                continue
            base = start_dist.get(idx, 0.0)
            seg_len = lengths[idx] / float(nseg)
            values: List[float] = []
            for j in range(nseg):
                if j == nseg - 1:
                    dist = base + seg_len * float(nseg)
                else:
                    dist = base + seg_len * (float(j) + 0.5)
                dist_norm = dist / max_len if max_len > 0.0 else 0.0
                value = (
                    exp_offset_value + exp_scale_value * math.exp(exp_rate_value * (dist_norm - exp_shift_value))
                ) * reference_gbar_value
                values.append(value)
            out[names[idx]] = values

        return out

    def insert_inhomogeneous(
        self,
        label: str,
        mech: str,
        *,
        sections: Iterable[object],
        reference_gbar: object,
        exp_offset: object,
        exp_rate: object,
        exp_shift: object,
        exp_scale: object,
        apic_label: str = "apic",
        origin_label: str = "soma",
        params: Optional[Dict[str, object]] = None,
        param_name: str = "gbar",
        **kwargs: object,
    ) -> None:
        label_name = BiophysPrepare._normalize_label(label)
        mech_name = str(mech)
        merged: Dict[str, object] = {}
        if params is not None:
            merged.update({str(k): BiophysPrepare.LabelScope._coerce_param_value(v) for (k, v) in params.items()})
        if kwargs:
            merged.update({str(k): BiophysPrepare.LabelScope._coerce_param_value(v) for (k, v) in kwargs.items()})

        param_key = str(param_name)
        if not param_key:
            raise ValueError("param_name must be non-empty")
        if param_key in merged:
            raise ValueError(f"param_name '{param_key}' must not be provided in params/kwargs")

        section_values = BiophysPrepare.compute_apic_exp_distribution(
            sections,
            reference_gbar=reference_gbar,
            exp_offset=exp_offset,
            exp_rate=exp_rate,
            exp_shift=exp_shift,
            exp_scale=exp_scale,
            apic_label=apic_label,
            origin_label=origin_label,
        )
        if not section_values:
            return

        merged[param_key] = BiophysPrepare.LabelScope._coerce_param_value(section_values)
        self._inserts.append(
            InsertOp(
                label=label_name,
                mech=mech_name,
                params=dict(merged),
            )
        )

    def ion_species(
        self,
        label: str,
        ion: str,
        model: str,
        params: Optional[Dict[str, object]] = None,
    ) -> None:
        label_name = BiophysPrepare._normalize_label(label)
        merged: Dict[str, object] = {}
        if params is not None:
            merged.update({str(k): BiophysPrepare.LabelScope._coerce_param_value(v) for (k, v) in params.items()})
        self._ion_species.append(
            IonSpeciesOp(
                label=label_name,
                ion=str(ion),
                model=str(model),
                params=dict(merged),
            )
        )

    def specificCapacitance(
        self,
        label: str,
        value: object,
    ) -> None:
        label_name = BiophysPrepare._normalize_label(label)
        coerced = BiophysPrepare._coerce_section_scalar_value(value)
        self._specific_capacitance.append(
            SpecificCapacitanceOp(
                label=label_name,
                value=coerced,
            )
        )

    def resistivity(
        self,
        label: str,
        value: object,
    ) -> None:
        label_name = BiophysPrepare._normalize_label(label)
        coerced = BiophysPrepare._coerce_section_scalar_value(value)
        self._resistivity.append(
            ResistivityOp(
                label=label_name,
                value=coerced,
            )
        )

    @staticmethod
    def _clone_section_scalar_value(value: object) -> object:
        if isinstance(value, dict):
            cloned: Dict[str, object] = {}
            for key, entry in value.items():
                if isinstance(entry, list):
                    cloned[str(key)] = list(entry)
                elif isinstance(entry, dict):
                    cloned[str(key)] = dict(entry)
                else:
                    cloned[str(key)] = entry
            return cloned
        return value

    def prepare_inserts(self) -> List[object]:
        out: List[object] = []
        for op in self._inserts:
            out.append((op.label, op.mech, dict(op.params)))
        return out

    def prepare_ion_species(self) -> List[object]:
        out: List[object] = []
        for op in self._ion_species:
            out.append((op.label, op.ion, op.model, dict(op.params)))
        return out

    def prepare_specificCapacitance(self) -> List[object]:
        out: List[object] = []
        for op in self._specific_capacitance:
            out.append((op.label, BiophysPrepare._clone_section_scalar_value(op.value)))
        return out

    def prepare_resistivity(self) -> List[object]:
        out: List[object] = []
        for op in self._resistivity:
            out.append((op.label, BiophysPrepare._clone_section_scalar_value(op.value)))
        return out


def _coerce_v_init_value(v_init: object) -> object:
    if isinstance(v_init, dict):
        return BiophysPrepare._normalize_random_spec_dict(v_init)
    try:
        return float(v_init)
    except Exception as exc:
        raise TypeError("v_init must be scalar or random spec dict") from exc


def build_bio_template(
    *,
    name: object,
    num_cells: object,
    sections: Iterable[object],
    biophys: BiophysPrepare,
    v_init: object | None = None,
) -> Dict[str, object]:
    template: Dict[str, object] = {
        "name": str(name),
        "num_cells": int(num_cells),
        "inserts": biophys.prepare_inserts(),
        "ion_species": biophys.prepare_ion_species(),
        "specificCapacitance": biophys.prepare_specificCapacitance(),
        "resistivity": biophys.prepare_resistivity(),
        "sections": list(sections),
    }
    if v_init is not None:
        template["v_init"] = _coerce_v_init_value(v_init)
    return template


__all__ = [
    "BioValue",
    "InsertOp",
    "IonSpeciesOp",
    "SpecificCapacitanceOp",
    "ResistivityOp",
    "BiophysPrepare",
    "build_bio_template",
]
