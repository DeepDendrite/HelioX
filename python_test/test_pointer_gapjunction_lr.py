#!/usr/bin/env python3
"""
POINTER correctness test (gapjunction_lr): NEURON vs HELIOX
===========================================================

This test builds a small multi-cell network in NEURON using a POINT_PROCESS
mechanism that contains a POINTER (gapjunction_lr.vpre). The model is exported
with nrnbbcore_write, loaded by HELIOX, then both simulators are run and the
voltages compared.

Why this test matters:
- It uses multiple cells and multiple POINTER instances.
- It enables HELIOX permute_type=3 so node/instance order changes.
- If POINTER fix-up is wrong, the coupled voltages diverge quickly.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np

from neuron import h, coreneuron

# Make repo-local python_lib importable when running from repo root
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
    g_gap = float(os.environ.get("GJ_G", "0.1"))
    w_gap = float(os.environ.get("GJ_W", "1.0"))
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
        cells.append(soma)

    gaps = []
    for i in range(ncell):
        gj = h.gapjunction_lr(cells[i](0.5))
        gj.g = g_gap
        gj.w = w_gap
        gaps.append(gj)

    # Ring: cell i reads vpre from cell (i+1)%ncell
    for i in range(ncell):
        j = (i + 1) % ncell
        h.setpointer(cells[j](0.5)._ref_v, "vpre", gaps[i])

    return cells, gaps


def main():
    # NEURON setup
    h.load_file("stdrun.hoc")
    h.cvode.active(0)
    h.cvode.cache_efficient(1)
    # NOTE: CoreNEURON has limitations around POINTER ("not thread safe").
    # We export via nrnbbcore_write, but run the NEURON reference with CoreNEURON disabled
    # so POINTER semantics match classic NEURON.
    coreneuron.enable = True

    dt = 0.025
    tstop = 100.0
    permute_type = int(os.environ.get("HELIOX_PERMUTE_TYPE", "3"))
    device = os.environ.get("HELIOX_DEVICE", "cpu").lower()

    # Compile mod
    mod_src = os.path.join(_HERE, "modfiles", "gapjunction_lr.mod")
    if not os.path.exists(mod_src):
        raise FileNotFoundError(mod_src)

    with tempfile.TemporaryDirectory() as tmp:
        mod_build = os.path.join(tmp, "nrnivmodl_gapjunction_lr")
        libnrnmech = _compile_mod(mod_src, mod_build)
        h.nrn_load_dll(libnrnmech)

        # Build network with POINTERs
        cells, gaps = _build_network(ncell=3)

        # Setup GIDs for export
        pc = h.ParallelContext()
        for gid, soma in enumerate(cells):
            pc.set_gid2node(gid, pc.id())
            nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
            pc.cell(gid, nc)

        # Record NEURON voltages
        t_neuron_vec = h.Vector()
        t_neuron_vec.record(h._ref_t)
        v_neuron_vec = []
        for soma in cells:
            vec = h.Vector()
            vec.record(soma(0.5)._ref_v)
            v_neuron_vec.append(vec)

        # Export (t=0 state)
        export_path = os.path.join(tmp, "ptr_gapjunction_lr_export")
        pc.set_maxstep(10)
        pc.setup_transfer()
        h.dt = dt
        h.finitialize(-65)
        pc.nrnbbcore_write(export_path)

        # Run HELIOX
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
        # Initial condition perturbation (no extra stim mechanisms needed):
        # raise cell 0's voltage so current flows through POINTER-based gap junctions.
        v0_legacy = cells[0](0.5).node_index()
        manager.set_variable_value(-50.0, "global", "v", v0_legacy)
        manager.run(tstop)
        v_heliox = [np.asarray(m.data, dtype=np.float64) for m in monitors]

        # Run NEURON reference (classic NEURON): CoreNEURON may abort with POINTER mechanisms.
        coreneuron.enable = False
        h.dt = dt
        h.finitialize(-65)
        cells[0](0.5).v = -50.0
        h.continuerun(tstop)
        v_neuron = [np.asarray(v, dtype=np.float64) for v in v_neuron_vec]
        t_neuron = np.asarray(t_neuron_vec, dtype=np.float64)

        # Compare
        max_diffs = []
        for i in range(len(cells)):
            vn = v_neuron[i]
            vng = v_heliox[i]

            # NEURON typically records an initial sample at t=0 after finitialize.
            # HELIOX currently starts logging after the first step (t=dt).
            if len(vn) == len(vng) + 1:
                vn = vn[1:]
            elif len(vng) == len(vn) + 1:
                vng = vng[1:]

            min_len = min(len(vn), len(vng))
            diff = np.max(np.abs(vn[:min_len] - vng[:min_len]))
            max_diffs.append(diff)
            print(f"cell {i}: max |Δv| = {diff:.6g} mV")
            if os.environ.get("DEBUG_ALIGN", "0") == "1":
                print(f"  len(NEURON)={len(v_neuron[i])} len(HELIOX)={len(v_heliox[i])} len(comp)={min_len}")
                print(f"  head NEURON:  {vn[:5]}")
                print(f"  head HELIOX: {vng[:5]}")

        overall = max(max_diffs) if max_diffs else 0.0
        print(f"overall max |Δv| = {overall:.6g} mV")

        # GPU may introduce tiny floating-point differences; allow a slightly looser tolerance.
        tol = 1e-6 if device == "cpu" else 1e-4
        if overall > tol:
            raise AssertionError(f"POINTER test failed: overall max diff {overall} > {tol}")

        print("PASS: NEURON vs HELIOX match within tolerance.")


if __name__ == "__main__":
    main()
