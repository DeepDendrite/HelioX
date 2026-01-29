#!/usr/bin/env python3
"""
POINTER correctness test (ion variable target): NEURON vs HELIOX
===============================================================

This test checks POINTERs that target ion variables (stored in ion mechanisms):
- enapre points to another cell's `ena` (na_ion reversal potential / erev)

We export the NEURON model via nrnbbcore_write, load it in HELIOX, run with
permute_type enabled, and compare voltage traces.
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

    cells = []
    for i in range(ncell):
        soma = h.Section(name=f"soma_{i}")
        soma.L = 50
        soma.diam = 50
        soma.Ra = 100
        soma.insert("pas")
        soma.g_pas = g_pas
        soma.e_pas = e_pas
        soma.insert("na_ion")
        cells.append(soma)

    # Make ion variable differ per cell so wrong pointers show up.
    ena_values = [40.0, 55.0, 70.0]
    for i, soma in enumerate(cells):
        soma(0.5).ena = ena_values[i % len(ena_values)]

    ionptrs = []
    for i in range(ncell):
        pp = h.ionptr_ng(cells[i](0.5))
        pp.g = 0.03
        ionptrs.append(pp)

    # Ring: each cell i reads enapre from cell (i+1)%ncell
    for i in range(ncell):
        j = (i + 1) % ncell
        h.setpointer(cells[j](0.5)._ref_ena, "enapre", ionptrs[i])

    return cells, ionptrs


def main():
    h.load_file("stdrun.hoc")
    h.cvode.active(0)
    h.cvode.cache_efficient(1)
    coreneuron.enable = True

    dt = 0.025
    tstop = 50.0
    permute_type = int(os.environ.get("HELIOX_PERMUTE_TYPE", "3"))
    device = os.environ.get("HELIOX_DEVICE", "cpu").lower()

    mod_src = os.path.join(_HERE, "modfiles", "ionptr.mod")
    if not os.path.exists(mod_src):
        raise FileNotFoundError(mod_src)

    with tempfile.TemporaryDirectory() as tmp:
        mod_build = os.path.join(tmp, "nrnivmodl_ionptr")
        libnrnmech = _compile_mod(mod_src, mod_build)
        h.nrn_load_dll(libnrnmech)

        cells, _pps = _build_network(ncell=3)

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

        export_path = os.path.join(tmp, "ptr_ionvar_export")
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
        v0_legacy = cells[0](0.5).node_index()
        manager.set_variable_value(-50.0, "global", "v", v0_legacy)
        manager.run(tstop)
        v_heliox = [np.asarray(m.data, dtype=np.float64) for m in monitors]

        coreneuron.enable = False
        h.dt = dt
        h.finitialize(-65)
        cells[0](0.5).v = -50.0
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
            raise AssertionError(f"POINTER ionvar test failed: overall max diff {overall} > {tol}")

        print("PASS: NEURON vs HELIOX match within tolerance.")


if __name__ == "__main__":
    main()
