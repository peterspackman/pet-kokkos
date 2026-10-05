#!/usr/bin/env python3
"""Write an orthogonal-cell extxyz frame as a LAMMPS data file (atom_style
atomic), optionally rattled, and the same structure back as extxyz so pet-eval
can evaluate exactly what LAMMPS does.

    xyz2data.py in.xyz out-prefix [rattle-A] [seed]
"""
import os, random, re, sys

src, out = sys.argv[1], sys.argv[2]
rattle = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
lines = open(src).read().splitlines()
n = int(lines[0])
lat = [float(x) for x in re.search(r'Lattice="([^"]+)"', lines[1]).group(1).split()]
assert all(abs(lat[i]) < 1e-12 for i in (1, 2, 3, 5, 6, 7)), "orthogonal cells only"
atoms = []
for l in lines[2:2 + n]:
    s, x, y, z = l.split()[:4]
    atoms.append((s, [float(x) + random.uniform(-rattle, rattle) for x in (x, y, z)]))
elements = sorted({s for s, _ in atoms})
with open(out + ".data", "w") as f:
    f.write(f"{os.path.basename(src)} rattled {rattle} A\n\n{n} atoms\n{len(elements)} atom types\n\n")
    for d, L in zip("xyz", (lat[0], lat[4], lat[8])):
        f.write(f"0.0 {L} {d}lo {d}hi\n")
    mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Si": 28.085}
    f.write("\nMasses\n\n")
    for t, e in enumerate(elements):
        f.write(f"{t + 1} {mass[e]}\n")
    f.write("\nAtoms # atomic\n\n")
    for i, (s, p) in enumerate(atoms):
        f.write(f"{i + 1} {elements.index(s) + 1} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")
with open(out + ".xyz", "w") as f:
    f.write(f"{n}\n{lines[1]}\n")
    for s, p in atoms:
        f.write(f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")
print(" ".join(elements))
