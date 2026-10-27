"""pet_kokkos from Python: the goldens, batches, and the ASE calculator.

Run with the package installed: uv run --with pytest --with ase pytest python/tests
A model that is not installed skips its tests rather than being downloaded.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pet_kokkos
from pet_kokkos import _core

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden"
GOLDENS = sorted(GOLDEN.glob("*.json"))


def _model_of(path):
    return path.stem.rsplit("_", 1)[0]


_calcs = {}


def calculator(model):
    if _core.find_model(model) is None:
        pytest.skip(f"model {model} is not installed")
    if model not in _calcs:
        _calcs[model] = pet_kokkos.Calculator(model)
    return _calcs[model]


def load(path):
    g = json.loads(path.read_text())
    calc = calculator(_model_of(path))
    scale = g.get("model_metadata", {}).get("energy_scale")
    if scale is not None and abs(scale - calc._calc.energy_scale) > 1e-9 * max(1.0, abs(scale)):
        pytest.skip(f"{_model_of(path)} resolves to another checkpoint than the golden's")
    cell = np.array(g["cell"]).reshape(3, 3) if g["periodic"] else None
    return g, calc, (g["atomic_numbers"], np.array(g["positions"]).reshape(-1, 3), cell, g["pbc"])


@pytest.mark.parametrize("path", GOLDENS, ids=[p.stem for p in GOLDENS])
def test_golden(path):
    g, calc, (z, x, cell, pbc) = load(path)
    r = calc.compute(z, x, cell, pbc)
    assert abs(r["energy"] - g["total_energy"]) <= max(1e-3, 2e-6 * abs(g["total_energy"]))
    assert np.max(np.abs(r["forces"] - np.array(g["forces"]).reshape(-1, 3))) <= 1e-2
    assert r["energies"].shape == (len(z),)
    if g["periodic"] and g.get("stress"):
        ref = np.array(g["stress"]).reshape(3, 3)
        ref = 0.5 * (ref + ref.T)
        w = r["virial"] / g["volume"]  # xx yy zz xy xz yz
        mine = np.array([[w[0], w[3], w[4]], [w[3], w[1], w[5]], [w[4], w[5], w[2]]])
        assert np.max(np.abs(mine - ref)) <= 1e-4


def test_batch_matches_single():
    paths = [p for p in GOLDENS if _model_of(p) == "pet-mad-xs"]
    if not paths:
        pytest.skip("no pet-mad-xs goldens")
    loaded = [load(p) for p in paths]
    calc = loaded[0][1]
    single = [calc.compute(*s) for _, _, s in loaded]
    batch = calc.compute_batch([s for _, _, s in loaded])
    for a, b in zip(single, batch):
        assert abs(a["energy"] - b["energy"]) <= 1e-6 * max(1.0, abs(a["energy"]))
        assert np.max(np.abs(a["forces"] - b["forces"])) <= 1e-4


def test_energy_only():
    path = next((p for p in GOLDENS if _model_of(p) == "pet-mad-xs"), None)
    if path is None:
        pytest.skip("no pet-mad-xs goldens")
    _, calc, s = load(path)
    r = calc.compute(*s, forces=False)
    assert "forces" not in r and np.isfinite(r["energy"])


def test_ase_forces_and_stress_match_finite_differences():
    ase = pytest.importorskip("ase")
    from ase.build import bulk
    from pet_kokkos.ase import PETCalculator

    if _core.find_model("pet-mad-xs") is None:
        pytest.skip("model pet-mad-xs is not installed")
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.05, seed=1)
    atoms.set_cell(atoms.cell * [[1.0, 0.0, 0.0], [0.02, 1.0, 0.0], [0.0, -0.01, 1.0]], scale_atoms=True)
    atoms.calc = PETCalculator("pet-mad-xs")
    forces, stress = atoms.get_forces(), atoms.get_stress()
    # Before the finite differences, which leave their last strained step's results behind.
    assert atoms.calc.results["energies"].sum() == pytest.approx(atoms.get_potential_energy(), abs=1e-4)
    num_forces = atoms.calc.calculate_numerical_forces(atoms, d=1e-3)
    # A strain step below ~1e-4 drowns in the fp32 network's rounding (1e-5
    # gives ~5e-4 eV/A^3 of noise); at 1e-3 the two agree to ~1e-5.
    num_stress = atoms.calc.calculate_numerical_stress(atoms, d=1e-3)
    assert np.max(np.abs(forces - num_forces)) <= 5e-3
    assert np.max(np.abs(stress - num_stress)) <= 1e-4


def test_backend_and_closing():
    assert pet_kokkos.backend() in {"Serial", "OpenMP", "Threads", "Cuda", "HIP"}
    if _core.find_model("pet-mad-xs") is None:
        pytest.skip("model pet-mad-xs is not installed")
    calc = pet_kokkos.Calculator("pet-mad-xs")
    calc.close()
    with pytest.raises(RuntimeError):
        calc.compute([14], [[0.0, 0.0, 0.0]])


def test_no_leak_warning_when_a_calculator_outlives_the_interpreter():
    # As in IPython, whose Out history keeps calculators past atexit: an extra
    # reference that is never dropped.
    if _core.find_model("pet-mad-xs") is None:
        pytest.skip("model pet-mad-xs is not installed")
    script = (
        "import ctypes, pet_kokkos\n"
        "calc = pet_kokkos.Calculator('pet-mad-xs')\n"
        "ctypes.pythonapi.Py_IncRef(ctypes.py_object(calc))\n"
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "leaked" not in run.stderr, run.stderr
