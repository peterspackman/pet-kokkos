"""pet_kokkos.D3, and D3 added to a Calculator.

The references are s-dftd3's (tests/golden/d3/reference.json, from
tools/make_d3_golden.py), at its own cutoffs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pet_kokkos
from pet_kokkos import _core

REFERENCE = Path(__file__).resolve().parents[2] / "tests" / "golden" / "d3" / "reference.json"
CASES = json.loads(REFERENCE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['structure']}-{c['damping']}-s9={c['s9']}" for c in CASES])
def test_matches_s_dftd3(case):
    cut = case["cutoffs_bohr"]
    bohr = 0.529177210903
    d3 = pet_kokkos.D3(damping=case["damping"], **case["params"], s9=case["s9"], reference=True, precision="double",
                       cutoff=cut["pair"] * bohr, cn_cutoff=cut["cn"] * bohr, atm_cutoff=cut["atm"] * bohr)
    r = d3.compute(case["numbers"], case["positions"], case.get("cell"), case.get("pbc"))
    assert abs(r["energy"] - case["energy"]) <= 2e-9 * abs(case["energy"])
    f = np.asarray(case["forces"])
    assert np.abs(r["forces"] - f).max() <= 2e-8 * np.abs(f).max() + 1e-11
    if case.get("pbc"):
        w = np.asarray(case["virial"])
        assert np.abs(r["virial"] - w).max() <= 2e-7 * np.abs(w).max() + 1e-11
    assert abs(r["energies"].sum() - r["energy"]) <= 1e-12 * abs(r["energy"])


def test_names_and_parameters():
    assert "pbesol" in pet_kokkos.d3_functionals()
    assert "r2scan" not in pet_kokkos.d3_functionals("zero")
    d3 = pet_kokkos.D3("PBE", atm=True)
    p = d3.params
    assert (p["damping"], p["a1"], p["s8"], p["s9"]) == ("bj", 0.4289, 0.7875, 1.0)
    assert abs(p["a2"] - 4.4407) < 1e-12  # bohr, as published
    assert d3.cutoffs["cutoff"] == 12.0 and d3.cutoffs["tail"]
    assert d3.ghost_cutoff(exchange=True) == 12.0
    with pytest.raises(Exception):
        pet_kokkos.D3("no-such-functional")


def test_batch_is_each_alone():
    case = next(c for c in CASES if c["structure"] == "water8")
    mol = next(c for c in CASES if c["structure"] == "water_dimer")
    d3 = pet_kokkos.D3("pbe")
    a = d3.compute(case["numbers"], case["positions"], case["cell"], case["pbc"])
    b = d3.compute(mol["numbers"], mol["positions"])
    both = d3.compute_batch([(case["numbers"], case["positions"], case["cell"], case["pbc"]),
                             (mol["numbers"], mol["positions"], None, None)])
    assert both[0]["energy"] == a["energy"] and both[1]["energy"] == b["energy"]
    assert np.array_equal(both[0]["forces"], a["forces"]) and np.array_equal(both[1]["forces"], b["forces"])


def test_calculator_adds_d3():
    if _core.find_model("pet-mad-xs") is None:
        pytest.skip("model pet-mad-xs is not installed")
    case = next(c for c in CASES if c["structure"] == "water8")
    args = (case["numbers"], case["positions"], case["cell"], case["pbc"])
    plain = pet_kokkos.Calculator("pet-mad-xs")
    with_d3 = pet_kokkos.Calculator("pet-mad-xs", d3="pbesol")
    d3 = pet_kokkos.D3("pbesol")
    a, b, d = with_d3.compute(*args), plain.compute(*args), d3.compute(*args)
    assert abs(a["energy"] - b["energy"] - d["energy"]) < 1e-9
    assert np.abs(a["forces"] - b["forces"] - d["forces"]).max() < 1e-9
    assert np.abs(a["virial"] - b["virial"] - d["virial"]).max() < 1e-9
    assert np.abs(a["energies"] - b["energies"] - d["energies"]).max() < 1e-9
