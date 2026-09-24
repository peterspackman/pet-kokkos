#!/usr/bin/env python3
"""LAMMPS's single point (log.lammps + <data>.forces) against pet-eval's on the
same structure (<data>.xyz): energy, per-atom energy, forces, pressure.

    compare.py DATA PET_EVAL MODEL [LOG]
"""
import json, subprocess, sys

data, pet_eval, model = sys.argv[1:4]
log = sys.argv[4] if len(sys.argv) > 4 else "log.lammps"
ref = json.loads(subprocess.run([pet_eval, model, data + ".xyz", "--forces", "--per-atom", "--json"],
                                capture_output=True, text=True, check=True).stdout)["frames"][0]

lines = open(log).read().splitlines()
h = next(i for i, l in enumerate(lines) if l.split()[:2] == ["Step", "PotEng"])
row = dict(zip(lines[h].split(), map(float, lines[h + 1].split())))
dump = open(data + ".forces").read().splitlines()
k = dump.index(next(l for l in dump if l.startswith("ITEM: ATOMS")))
atoms = [list(map(float, l.split())) for l in dump[k + 1:]]
n = len(atoms)

BAR = 1.602176634e6  # eV/A^3 in bar; LAMMPS pressure is minus the stress
dE = abs(row["PotEng"] - ref["energy"])
dF = max(abs(a[1 + c] - ref["forces"][i][c]) for i, a in enumerate(atoms) for c in range(3))
Fmax = max(abs(x) for f in ref["forces"] for x in f)
dPA = max(abs(a[4] - ref["per_atom_energy"][i]) for i, a in enumerate(atoms))
p_ref = [-s * BAR for s in ref["stress"]]  # Voigt xx yy zz xy xz yz
p_lmp = [row[k] for k in ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")]
dP = max(abs(a - b) for a, b in zip(p_lmp, p_ref))
print(f"{data}: {n} atoms")
print(f"  energy      LAMMPS {row['PotEng']:.10f}  pet-eval {ref['energy']:.10f}  |dE|/atom {dE / n:.2e} eV")
print(f"  per-atom E  max |d| {dPA:.2e} eV")
print(f"  forces      max |dF| {dF:.2e} eV/A  (max |F| {Fmax:.3f})")
print(f"  pressure    max |dP| {dP:.3g} bar  (Pxx LAMMPS {p_lmp[0]:.6g}, pet-eval {p_ref[0]:.6g})")
print(f"  per-atom virial sums to the global one: Pxx {row['v_pxx_atom']:.8g} vs {row['Pxx']:.8g}, "
      f"Pxy {row['v_pxy_atom']:.8g} vs {row['Pxy']:.8g}")
