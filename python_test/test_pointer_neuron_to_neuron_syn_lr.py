#!/usr/bin/env python3
"""
POINTER correctness test (neuron_to_neuron_syn_lr): NEURON vs HELIOX
====================================================================

This test builds a small multi-cell network in NEURON using a POINT_PROCESS
mechanism that contains a POINTER (neuron_to_neuron_syn_lr.vpre).

Model is exported via nrnbbcore_write, loaded by HELIOX, then both simulators
are run and the voltages compared.

Why this test matters:
- Uses POINTER to presynaptic voltage for a stateful synapse (cnexp).
- Enables HELIOX permute_type=3 so node/instance order changes.
- Ensures POINTER resolution + state/current scheduling matches NEURON.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np

from neuron import h, coreneuron

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIB = os.path.abspath(os.path.join(_HERE, "..", "python_lib"))
if _PYLIB not in sys.path:
    sys.path.insert(0, _PYLIB)

from heliox_wrapper import HelioXManager


def _compile_mod(mod_path: str, build_dir: str) -> str:
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["nrnivmodl", mod_path], cwd=build_dir, check=True)
    lib = os.path.join(build_dir, "x86_64", "libnrnmech.so")
    if not os.path.exists(lib):
        raise FileNotFoundError(f"nrnivmodl output not found: {lib}")
    return lib


def _build_network(ncell: int = 3):
    g_pas = float(os.environ.get("PAS_G", "0.001"))
    e_pas = float(os.environ.get("PAS_E", "-65.0"))

    stim_amp = float(os.environ.get("STIM_AMP", "0.08"))  # nA
    stim_del = float(os.environ.get("STIM_DEL", "5.0"))   # ms
    stim_dur = float(os.environ.get("STIM_DUR", "40.0"))  # ms

    syn_w = float(os.environ.get("SYN_W", "1.0"))
    syn_g = float(os.environ.get("SYN_G", "4.9"))

    cells = []
    for i in range(ncell):
        soma = h.Section(name=f"soma_{i}")
        soma.L = 50
        soma.diam = 50
        soma.Ra = 100
        soma.insert("pas")
        soma.g_pas = g_pas
        soma.e_pas = e_pas
        cells.append(soma)

    # Stimulate cell 0
    stim = h.IClamp(cells[0](0.5))
    stim.amp = stim_amp
    stim.delay = stim_del
    stim.dur = stim_dur

    syns = []
    for i in range(ncell):
        syn = h.neuron_to_neuron_syn_lr(cells[i](0.5))
        syn.w = syn_w
        syn.g = syn_g
        syns.append(syn)

    # Chain/ring: syn on cell i reads vpre from cell (i-1), so stimulus on cell0 influences cell1.
    for i in range(ncell):
        pre = cells[(i - 1) % ncell]
        h.setpointer(pre(0.5)._ref_v, "vpre", syns[i])

    return cells, syns


def main():
    h.load_file("stdrun.hoc")
    h.cvode.active(0)
    h.cvode.cache_efficient(1)
    coreneuron.enable = True

    dt = 0.025
    tstop = 60.0
    permute_type = int(os.environ.get("HELIOX_PERMUTE_TYPE", "3"))
    device = os.environ.get("HELIOX_DEVICE", "cpu").lower()

    mod_src = os.path.join(_HERE, "modfiles", "neuron_to_neuron_syn_lr.mod")
    if not os.path.exists(mod_src):
        raise FileNotFoundError(mod_src)

    with tempfile.TemporaryDirectory() as tmp:
        mod_build = os.path.join(tmp, "nrnivmodl_neuron_to_neuron_syn_lr")
        libnrnmech = _compile_mod(mod_src, mod_build)
        h.nrn_load_dll(libnrnmech)

        cells, syns = _build_network(ncell=3)

        pc = h.ParallelContext()
        for gid, soma in enumerate(cells):
            pc.set_gid2node(gid, pc.id())
            nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
            pc.cell(gid, nc)

        t_neuron_vec = h.Vector()
        t_neuron_vec.record(h._ref_t)
        v_neuron_vec = []
        for soma in cells:
            vec = h.Vector()
            vec.record(soma(0.5)._ref_v)
            v_neuron_vec.append(vec)

        export_path = os.path.join(tmp, "ptr_neuron_to_neuron_syn_lr_export")
        pc.set_maxstep(10)
        pc.setup_transfer()
        h.dt = dt
        h.finitialize(-65)
        pc.nrnbbcore_write(export_path)

        manager = HelioXManager()
        manager.set_device(device)
        manager.set_permute_type(permute_type)
        manager.dt = dt

        monitors = [manager.create_monitor_wrapper(soma(0.5), "v") for soma in cells]
        manager.set_data_path(export_path)
        manager.load_model()
        for m in monitors:
            m._initialize()

        manager.finitialize(-65)
        manager.run(tstop)
        v_heliox = [np.asarray(m.data, dtype=np.float64) for m in monitors]

        # NEURON reference run (classic NEURON): CoreNEURON may abort with POINTER mechanisms.
        coreneuron.enable = False
        h.dt = dt
        h.finitialize(-65)
        h.continuerun(tstop)
        v_neuron = [np.asarray(v, dtype=np.float64) for v in v_neuron_vec]

        max_diffs = []
        for i in range(len(cells)):
            vn = v_neuron[i]
            vng = v_heliox[i]

            if len(vn) == len(vng) + 1:
                vn = vn[1:]
            elif len(vng) == len(vn) + 1:
                vng = vng[1:]

            min_len = min(len(vn), len(vng))
            diff = np.max(np.abs(vn[:min_len] - vng[:min_len]))
            max_diffs.append(diff)
            print(f"cell {i}: max |Δv| = {diff:.6g} mV")

        overall = max(max_diffs) if max_diffs else 0.0
        print(f"overall max |Δv| = {overall:.6g} mV")

        tol = 1e-6 if device == "cpu" else 1e-4
        if overall > tol:
            raise AssertionError(f"POINTER test failed: overall max diff {overall} > {tol}")

        print("PASS: NEURON vs HELIOX match within tolerance.")


if __name__ == "__main__":
    main()
