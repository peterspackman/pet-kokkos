"""An ASE calculator over pet_kokkos.Calculator.

>>> from pet_kokkos.ase import PETCalculator
>>> atoms.calc = PETCalculator("pet-mad-s")
>>> atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()

For MD or a relaxation of the same atoms, ``cache_neighbors=True`` (the
default here) keeps a Verlet cache of the neighbour topology between steps.
A charge and spin multiplicity, for models conditioned on them, are read from
``atoms.info["charge"]`` and ``atoms.info["spin"]``.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from . import Calculator as _PET

# pet-kokkos's Voigt order is xx yy zz xy xz yz; ASE's is xx yy zz yz xz xy.
_TO_ASE_VOIGT = [0, 1, 2, 5, 4, 3]


class PETCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model: str = "pet-mad-s", *, cache_neighbors: bool = True, **options):
        super().__init__()
        self.pet = _PET(model, cache_neighbors=cache_neighbors, **options)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        pbc = tuple(bool(p) for p in atoms.pbc)
        cell = np.asarray(atoms.cell.array, dtype=np.float64) if any(pbc) else None
        want_forces = any(p in properties for p in ("forces", "stress"))
        r = self.pet.compute(
            atoms.numbers,
            atoms.positions,
            cell,
            pbc,
            forces=want_forces,
            charge=int(atoms.info.get("charge", 0)),
            spin_multiplicity=int(atoms.info.get("spin", 1)),
        )
        self.results["energy"] = self.results["free_energy"] = float(r["energy"])
        self.results["energies"] = np.array(r["energies"])
        if want_forces:
            self.results["forces"] = np.array(r["forces"])
            if all(pbc) and "virial" in r:
                self.results["stress"] = np.asarray(r["virial"])[_TO_ASE_VOIGT] / atoms.get_volume()
