#!/usr/bin/env python3
"""
eworm_learn (sim-only) correctness check: NEURON vs HELIOX
==========================================================

Workflow (single model build, then split):
  NEURON build model -> export bbcore -> HELIOX load
  then run HELIOX and run NEURON separately and compare voltages.

This avoids building the model twice inside NEURON (which can cause conflicts).
"""

import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
from neuron import h, coreneuron
import time

# Make repo-local python_lib importable when running from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIB = os.path.abspath(os.path.join(_HERE, "..", "python_lib"))
if _PYLIB not in sys.path:
    sys.path.insert(0, _PYLIB)

from heliox_wrapper import HelioXManager


def _load_eworm_config(eworm_root: str) -> Tuple[dict, dict, dict, str]:
    config_file = os.path.join(eworm_root, "trial10", "000_circuit_search_config.json")
    connection_file = os.path.join(eworm_root, "trial10", "sample_#0_circuit_old.pkl")
    with open(config_file) as f:
        cfg = json.load(f)
    return cfg, cfg["config"], cfg["sim_config"], connection_file


def _make_net_config_subset(net_config: dict, keep_cell_names: List[str]) -> Tuple[dict, List[str]]:
    name2id = {v: k for k, v in net_config["cell_info"]["cells_name_dic"].items()}
    keep_ids = [name2id[n] for n in keep_cell_names if n in name2id]
    new_cfg = dict(net_config)
    new_cfg["cell_info"] = dict(new_cfg["cell_info"])
    new_cfg["cell_info"]["cells_id_sim"] = keep_ids
    return new_cfg, keep_ids


def _abs_dir_info(eworm_root: str, net_config: dict) -> dict:
    out = dict(net_config)
    out["dir_info"] = dict(out["dir_info"])
    for key in ("model_dir", "cell_param_dir"):
        if key in out["dir_info"]:
            out["dir_info"][key] = os.path.join(eworm_root, out["dir_info"][key])
    return out


def _build_cells(net_config: dict) -> Dict[str, object]:
    cells_id_sim = net_config["cell_info"]["cells_id_sim"]
    id2name = net_config["cell_info"]["cells_name_dic"]
    model_dir = net_config["dir_info"]["model_dir"]
    cell_param_dir = net_config["dir_info"]["cell_param_dir"]
    length_per_seg = net_config["cnt_info"]["length_per_seg"]

    mech_list = [
        "slo1_unc2_lr",
        "egl2_lr",
        "shl1_lr",
        "kqt3_lr",
        "unc2_lr",
        "kvs1_lr",
        "slo1_egl19_lr",
        "slo2_unc2_lr",
        "irk_lr",
        "egl36_lr",
        "egl19_lr",
        "cca1_lr",
        "shk1_lr",
        "slo2_egl19_lr",
        "nca_lr",
        "kcnl_lr",
        "cainternm_lr",
    ]

    cells: Dict[str, object] = {}
    for cell_id in cells_id_sim:
        cell_name = id2name[cell_id]
        h.load_file(model_dir + cell_name + ".hoc")
        cell = getattr(h, cell_name)()

        for section in cell.all:
            soma_flag = "Soma" in section.name()
            section.nseg = 1 if soma_flag else int(np.ceil(section.L / length_per_seg))

        with open(cell_param_dir + cell_name + ".json") as f:
            cell_param = json.load(f)

        sec = cell.Soma
        sec.Ra = cell_param["soma"]["Ra"]
        sec.cm = cell_param["soma"]["cm"]
        sec.insert("pas")
        for seg in sec:
            seg.pas.g = cell_param["soma"]["gpas"]
            seg.pas.e = cell_param["soma"]["epas"]

        for sec in cell.all:
            if "Soma" in sec.name():
                continue
            sec.Ra = cell_param["neurite"]["Ra"]
            sec.cm = cell_param["soma"]["cm"]
            sec.insert("pas")
            for seg in sec:
                seg.pas.g = cell_param["neurite"]["gpas"]
                seg.pas.e = cell_param["neurite"]["epas"]

        for m in mech_list:
            cell.Soma.insert(m)
        for seg in cell.Soma:
            seg.shl1_lr.gbshl1 = cell_param["soma"]["gbshl1"]
            seg.shk1_lr.gbshk1 = cell_param["soma"]["gbshk1"]
            seg.kvs1_lr.gbkvs1 = cell_param["soma"]["gbkvs1"]
            seg.egl2_lr.gbegl2 = cell_param["soma"]["gbegl2"]
            seg.egl36_lr.gbegl36 = cell_param["soma"]["gbegl36"]
            seg.kqt3_lr.gbkqt3 = cell_param["soma"]["gbkqt3"]
            seg.egl19_lr.gbegl19 = cell_param["soma"]["gbegl19"]
            seg.unc2_lr.gbunc2 = cell_param["soma"]["gbunc2"]
            seg.cca1_lr.gbcca1 = cell_param["soma"]["gbcca1"]
            seg.slo1_egl19_lr.gbslo1 = cell_param["soma"]["gbslo1_egl19"]
            seg.slo1_unc2_lr.gbslo1 = cell_param["soma"]["gbslo1_unc2"]
            seg.slo2_egl19_lr.gbslo2 = cell_param["soma"]["gbslo2_egl19"]
            seg.slo2_unc2_lr.gbslo2 = cell_param["soma"]["gbslo2_unc2"]
            seg.kcnl_lr.gbkcnl = cell_param["soma"]["gbkcnl"]
            seg.nca_lr.gbnca = cell_param["soma"]["gbnca"]
            seg.irk_lr.gbirk = cell_param["soma"]["gbirk"]

        cells[cell_id] = cell

    return cells


def _get_3dp_segment(cell, seg_id: int):
    seg_cnt = 0
    for section in cell.all:
        if seg_id < seg_cnt + section.nseg:
            loading_bar = np.linspace(0, 1, section.nseg + 1)
            loading_cnt = seg_id - seg_cnt
            return section((loading_bar[loading_cnt] + loading_bar[loading_cnt + 1]) / 2)
        seg_cnt += section.nseg
    raise IndexError(f"seg_id {seg_id} out of range")


def _wire_connections(
    cells: Dict[str, object],
    connection_file: str,
    input_cell_ids: List[str],
) -> Tuple[Dict[str, object], List[object]]:
    import pickle

    connection_info = pickle.load(open(connection_file, "rb"))
    synapse_list: List[object] = []

    # Create input clamps only for requested input cells (to avoid exporting a huge number of
    # unnecessary point processes).
    input_synlist: Dict[str, object] = {}
    for cell_id in input_cell_ids:
        if cell_id not in cells:
            continue
        cell = cells[cell_id]
        seg = _get_3dp_segment(cell, 0)
        stim = h.IClamp(seg)
        stim.amp = 0.0
        stim.delay = 0.0
        stim.dur = 1e9
        input_synlist[cell_id] = stim

    for post_cell_id in cells.keys():
        if post_cell_id not in connection_info:
            continue
        for pre_cell_id, pre_point, post_point, syntype, w in connection_info[post_cell_id]:
            if pre_cell_id not in cells:
                continue
            pre_seg = _get_3dp_segment(cells[pre_cell_id], int(pre_point))
            post_seg = _get_3dp_segment(cells[post_cell_id], int(post_point))
            if syntype == 0:
                gj = h.gapjunction_lr(post_seg)
                gj.w = float(w)
                gj._ref_vpre = pre_seg._ref_v
                synapse_list.append(gj)
            elif syntype == 1 or syntype == 2:
                syn = h.neuron_to_neuron_syn_lr(post_seg)
                syn.w = (float(w) * 1e-4) if syntype == 1 else (-float(w) * 1e-4)
                syn._ref_vpre = pre_seg._ref_v
                synapse_list.append(syn)
            else:
                raise ValueError(f"Unknown syntype={syntype}, expected 0/1/2")

    return input_synlist, synapse_list


def main():
    eworm_root = os.environ.get("EWORM_LEARN_SIMONLY", "$HOME/path/to/eworm_learn_simonly")
    device = os.environ.get("HELIOX_DEVICE", "cpu").lower()
    permute_type = int(os.environ.get("HELIOX_PERMUTE_TYPE", "3"))

    cfg, net_config, sim_cfg, connection_file = _load_eworm_config(eworm_root)

    dt = float(os.environ.get("DT", sim_cfg["dt"]))
    tstop = float(os.environ.get("TSTOP", sim_cfg["tstop"]))
    v_init = float(os.environ.get("V_INIT", sim_cfg["v_init"]))

    h.load_file("stdrun.hoc")
    h.cvode.active(0)
    h.cvode.cache_efficient(1)

    # Load the MOD library used by this model (NEURON side). This includes POINTER mechanisms.
    libnrnmech = os.path.join(eworm_root, "x86_64", "libnrnmech.so")
    h.nrn_load_dll(libnrnmech)

    # Full network simulation: build all cells in config.
    # Compare/record a selected set of output cells (can be "all").
    output_names_all = cfg["search_config"]["output_cell_names"]
    compare_n = int(os.environ.get("COMPARE_N", "0"))  # 0 => all
    if compare_n <= 0:
        output_names = list(output_names_all)
    else:
        output_names = output_names_all[:compare_n]
    print_per_cell = os.environ.get("PRINT_PER_CELL", "0") == "1"

    input_names = cfg["search_config"]["input_cell_names"]

    net_config = _abs_dir_info(eworm_root, net_config)
    keep_ids = net_config["cell_info"]["cells_id_sim"]

    # Build NEURON model once
    cells = _build_cells(net_config)

    # Wire full connectivity (only edges where pre is in our built set).
    with tempfile.TemporaryDirectory() as tmp:
        # Determine input cells (by name) that exist in this simulation.
        name2id = {v: k for k, v in net_config["cell_info"]["cells_name_dic"].items()}
        input_cell_ids = [name2id[n] for n in input_names if n in name2id and name2id[n] in cells]
        if not input_cell_ids:
            input_cell_ids = list(cells.keys())[: min(3, len(cells))]

        input_synlist, _syns = _wire_connections(cells, connection_file, input_cell_ids)

        # Assign GIDs (required for export)
        pc = h.ParallelContext()
        for gid, cell_id in enumerate(cells.keys()):
            soma = cells[cell_id].Soma
            pc.set_gid2node(gid, int(pc.id()))
            nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
            pc.cell(gid, nc)

        pc.setup_transfer()
        pc.set_maxstep(10)

        # Input current time series (shared for NEURON & HELIOX)
        nstep = int(tstop / dt)
        rng = np.random.default_rng(42)
        input_is = rng.normal(loc=0.0, scale=1e-3, size=(len(input_cell_ids), nstep)) + 0.03
        tvec = [float(i * dt) for i in range(nstep)]

        # NEURON: use Vector.play to avoid Python-per-step overhead.
        neuron_play_refs = []
        for i, cell_id in enumerate(input_cell_ids):
            stim = input_synlist[cell_id]
            ivec = h.Vector()
            tv = h.Vector()
            for tt, ii in zip(tvec, input_is[i].tolist()):
                tv.append(tt)
                ivec.append(ii)
            ivec.play(stim._ref_amp, tv, 1)
            neuron_play_refs.append((tv, ivec))

        # Decide which cell_ids to compare, based on output_names.
        output_name_set = set(output_names)
        compare_cell_ids = [
            cell_id
            for cell_id in keep_ids
            if cell_id in cells and net_config["cell_info"]["cells_name_dic"][cell_id] in output_name_set
        ]
        if not compare_cell_ids:
            raise RuntimeError("No output cells to compare after filtering by output_names.")

        # Record NEURON voltages (soma v) for compare cells
        v_neuron_vec = []
        for cell_id in compare_cell_ids:
            vec = h.Vector()
            vec.record(cells[cell_id].Soma(0.5)._ref_v)
            v_neuron_vec.append((cell_id, vec))

        # Export and load HELIOX (model split point)
        manager = HelioXManager()
        manager.set_default_device(device)
        manager.set_default_permute_type(permute_type)
        manager.dt = dt

        recorders = []
        for cell_id in compare_cell_ids:
            rec = manager.create_recorder(cells[cell_id].Soma(0.5), "v")
            recorders.append((cell_id, rec))

        vecplays = []
        for cell_id in input_cell_ids:
            vecplays.append((cell_id, manager.create_vecplay_wrapper(input_synlist[cell_id], "amp")))

        export_path = os.path.join(tmp, "eworm_export")
        coreneuron.enable = True
        manager.setup_and_load_model(export_path, dt=dt, v_init=v_init)

        # HELIOX: program same stimuli via VecPlay.
        for i, (cell_id, vp) in enumerate(vecplays):
            vp.play(tvec, input_is[i].tolist())

        t0 = time.perf_counter()
        manager.finitialize(v_init)
        manager.run(tstop)
        t_heliox = time.perf_counter() - t0
        v_heliox = {cell_id: np.asarray(rec.data, dtype=np.float64) for cell_id, rec in recorders}

        # NEURON reference run (classic NEURON, POINTER-safe)
        coreneuron.enable = False
        h.dt = dt
        t0 = time.perf_counter()
        h.finitialize(v_init)
        h.continuerun(tstop)
        t_neuron = time.perf_counter() - t0
        v_neuron = {cell_id: np.asarray(vec, dtype=np.float64) for cell_id, vec in v_neuron_vec}

        # Compare: align possible off-by-one sample at t=0
        diffs = []
        per_cell_stats: List[Tuple[str, float, float]] = []
        for cell_id in compare_cell_ids:
            if cell_id not in v_heliox or cell_id not in v_neuron:
                raise RuntimeError(f"Missing trace for cell_id={cell_id}")
            vn = v_neuron[cell_id]
            vg = v_heliox[cell_id]

            if len(vn) == len(vg) + 1:
                vn = vn[1:]
            elif len(vg) == len(vn) + 1:
                vg = vg[1:]

            m = min(len(vn), len(vg))
            if m == 0:
                continue
            d = np.abs(vn[:m] - vg[:m])
            diffs.append(d)
            name = net_config["cell_info"]["cells_name_dic"][cell_id]
            dmax = float(d.max())
            dmean = float(d.mean())
            per_cell_stats.append((name, dmax, dmean))
            if print_per_cell:
                print(f"{name}: max |Δv| = {dmax:.6g} mV, mean |Δv| = {dmean:.6g} mV")

        if not diffs:
            raise RuntimeError("No traces compared.")

        all_d = np.concatenate(diffs)
        print(f"OVERALL: max |Δv| = {all_d.max():.6g} mV, mean |Δv| = {all_d.mean():.6g} mV")
        print(f"TIMING: HELIOX run {t_heliox:.3f}s, NEURON run {t_neuron:.3f}s, tstop={tstop}ms dt={dt}ms")
        print(f"COMPARED: {len(per_cell_stats)} cells (PRINT_PER_CELL={'1' if print_per_cell else '0'})")

        # Show top offenders without flooding logs.
        if not print_per_cell and per_cell_stats:
            per_cell_stats.sort(key=lambda x: x[1], reverse=True)
            topk = min(10, len(per_cell_stats))
            print("TOP_MAX_DIFF:")
            for name, dmax, dmean in per_cell_stats[:topk]:
                print(f"  {name}: max={dmax:.6g} mV mean={dmean:.6g} mV")


if __name__ == "__main__":
    main()
