#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["dftd3==1.7.0", "numpy"]
# ///
"""Reference DFT-D3 energies, forces and virials from s-dftd3 (the dftd3 package,
its Fortran library) for tests/test_d3.cpp: tests/golden/d3_reference.json.

Units in the file are pet-kokkos's (Angstrom, eV, eV/A; the virial is
W = sum v (x) dE/dv in Voigt order xx yy zz xy xz yz); cutoffs and a2 stay in
bohr, as s-dftd3 takes them. Conversions are CODATA 2018, as d3.cpp's.

Run:  uv run tools/make_d3_golden.py > tests/golden/d3_reference.json
"""

import json
import sys

import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam, ZeroDampingParam

BOHR = 0.529177210903
HARTREE = 27.211386245988


def structures():
    rng = np.random.default_rng(7)
    s = {}
    s["water_dimer"] = dict(
        numbers=[8, 1, 1, 8, 1, 1],
        positions=[[-1.551007, -0.114520, 0.0], [-1.934259, 0.762503, 0.0], [-0.599677, 0.040712, 0.0],
                   [1.350625, 0.111469, 0.0], [1.680398, -0.373741, -0.758561], [1.680398, -0.373741, 0.758561]])
    ring = []
    for k in range(6):
        t = np.pi / 3 * k
        ring.append([6, 1.397 * np.cos(t), 1.397 * np.sin(t), 0.0])
        ring.append([1, 2.481 * np.cos(t), 2.481 * np.sin(t), 0.0])
    s["benzene"] = dict(numbers=[a[0] for a in ring], positions=[a[1:] for a in ring])
    # cisplatin, square planar, rough geometry
    s["cisplatin"] = dict(
        numbers=[78, 17, 17, 7, 7, 1, 1, 1, 1, 1, 1],
        positions=[[0.0, 0.0, 0.0], [2.31, 0.0, 0.0], [0.0, 2.31, 0.0], [-2.05, 0.0, 0.0], [0.0, -2.05, 0.0],
                   [-2.40, 0.95, 0.10], [-2.40, -0.45, 0.85], [-2.40, -0.50, -0.82],
                   [0.95, -2.40, 0.10], [-0.45, -2.40, 0.85], [-0.50, -2.40, -0.82]])
    a = 3.567
    s["diamond2"] = dict(numbers=[6, 6], positions=[[0.0, 0.0, 0.0], [a / 4 + 0.05, a / 4, a / 4 - 0.03]],
                         cell=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]], pbc=[True] * 3)
    c = 3.615
    s["cu1"] = dict(numbers=[29], positions=[[0.1, 0.0, 0.0]],
                    cell=[[0, c / 2, c / 2], [c / 2, 0, c / 2], [c / 2, c / 2, 0]], pbc=[True] * 3)
    w = 3.165
    s["w1"] = dict(numbers=[74], positions=[[0.0, 0.0, 0.0]],
                   cell=[[-w / 2, w / 2, w / 2], [w / 2, -w / 2, w / 2], [w / 2, w / 2, -w / 2]], pbc=[True] * 3)
    # eight waters in a 6.2 A box, randomly oriented, slightly sheared cell
    L = 6.2
    mol = np.array([[0.0, 0.0, 0.0], [0.7572, 0.5859, 0.0], [-0.7572, 0.5859, 0.0]])
    pos, num = [], []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                w0, x, y, z = q
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w0), 2 * (x * z + y * w0)],
                              [2 * (x * y + z * w0), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w0)],
                              [2 * (x * z - y * w0), 2 * (y * z + x * w0), 1 - 2 * (x * x + y * y)]])
                ctr = (np.array([ix, iy, iz]) + 0.25 + 0.1 * rng.random(3)) * L / 2
                for k, p in enumerate(mol @ R.T + ctr):
                    pos.append(p.tolist())
                    num.append([8, 1, 1][k])
    s["water8"] = dict(numbers=num, positions=pos, cell=[[L, 0, 0], [0.3, L, 0], [0.2, -0.1, L]], pbc=[True] * 3)
    g = 2.46
    s["graphene"] = dict(numbers=[6, 6], positions=[[0.0, 0.0, 5.0], [g / 2, g / (2 * np.sqrt(3)) + 0.02, 5.03]],
                         cell=[[g, 0, 0], [g / 2, g * np.sqrt(3) / 2, 0], [0, 0, 10.0]], pbc=[True, True, False])
    return s


PBE_BJ = dict(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)
PBE_ZERO = dict(s6=1.0, s8=0.722, rs6=1.217, rs8=1.0)
B3LYP_BJ = dict(s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211)
B3LYP_ZERO = dict(s6=1.0, s8=1.703, rs6=1.261, rs8=1.0)
PBE0_BJ = dict(s6=1.0, s8=1.2177, a1=0.4145, a2=4.8593)
PBESOL_BJ = dict(s6=1.0, s8=2.9491, a1=0.4466, a2=6.1742)
STD = (95.0, 40.0, 40.0)  # pair, cn, atm (bohr)

CASES = [
    ("water_dimer", "pbe", "bj", PBE_BJ, 0.0, STD),
    ("water_dimer", "pbe", "zero", PBE_ZERO, 0.0, STD),
    ("water_dimer", "pbe", "bj", PBE_BJ, 1.0, STD),
    ("water_dimer", "pbe", "zero", PBE_ZERO, 1.0, STD),
    ("benzene", "b3lyp", "bj", B3LYP_BJ, 0.0, STD),
    ("benzene", "b3lyp", "zero", B3LYP_ZERO, 1.0, STD),
    ("cisplatin", "pbe0", "bj", PBE0_BJ, 1.0, STD),
    ("diamond2", "pbe", "bj", PBE_BJ, 0.0, STD),
    ("diamond2", "pbe", "zero", PBE_ZERO, 0.0, STD),
    ("diamond2", "pbe", "bj", PBE_BJ, 1.0, (95.0, 40.0, 10.0)),
    ("cu1", "pbesol", "bj", PBESOL_BJ, 0.0, STD),
    ("w1", "pbe", "zero", PBE_ZERO, 0.0, STD),
    ("water8", "pbe", "bj", PBE_BJ, 0.0, STD),
    ("water8", "pbe", "zero", PBE_ZERO, 1.0, (95.0, 40.0, 12.0)),
    ("graphene", "pbe", "bj", PBE_BJ, 0.0, STD),
]


def run(st, damping, par, s9, cut):
    num = np.array(st["numbers"])
    pos = np.array(st["positions"]) / BOHR
    cell = np.array(st["cell"]) / BOHR if "cell" in st else None
    pbc = np.array(st["pbc"]) if "pbc" in st else None
    m = DispersionModel(num, pos, cell, pbc)
    m.set_realspace_cutoff(cut[0], cut[2], cut[1])
    P = RationalDampingParam if damping == "bj" else ZeroDampingParam
    r = m.get_dispersion(P(s9=s9, **par), grad=True)
    W = r["virial"] * HARTREE
    return dict(energy=r["energy"] * HARTREE, forces=(-r["gradient"] * HARTREE / BOHR).tolist(),
                virial=[W[0, 0], W[1, 1], W[2, 2], 0.5 * (W[0, 1] + W[1, 0]), 0.5 * (W[0, 2] + W[2, 0]),
                        0.5 * (W[1, 2] + W[2, 1])])


def main():
    S = structures()
    cases = []
    for name, func, damping, par, s9, cut in CASES:
        st = S[name]
        ref = run(st, damping, par, s9, cut)
        cases.append(dict(structure=name, functional=func, damping=damping, params=par, s9=s9,
                          cutoffs_bohr=dict(pair=cut[0], cn=cut[1], atm=cut[2]), **st, **ref))
    json.dump(dict(source="s-dftd3 1.7.0 (pip dftd3), tools/make_d3_golden.py",
                   units=dict(length="angstrom", energy="eV", a2_and_cutoffs="bohr"), cases=cases),
              sys.stdout, indent=1)


if __name__ == "__main__":
    main()
