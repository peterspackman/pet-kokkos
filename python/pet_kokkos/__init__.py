"""pet-kokkos: PET machine-learned interatomic potentials on Kokkos.

>>> import pet_kokkos
>>> calc = pet_kokkos.Calculator("pet-mad-s")
>>> result = calc.compute(numbers, positions, cell, pbc=(True, True, True))
>>> result["energy"], result["forces"], result["virial"]

Units are the model's: Angstrom, eV, eV/A. ``virial`` is the symmetric
W = V * stress in Voigt order [xx, yy, zz, xy, xz, yz]. For ASE, see
``pet_kokkos.ase.PETCalculator``.

Everything runs in fp32 unless asked otherwise: ``gemm="tf32"`` and
``attention="tf32"`` run on TF32 tensor cores on recent NVIDIA GPUs (see the
README's "Faster modes"); both are process-wide.
"""

from __future__ import annotations

import atexit
import weakref
from typing import Iterable, Sequence

import numpy as np

from . import _core
from .models import data_dir, download, resolve

__all__ = ["Calculator", "backend", "download", "data_dir", "metatomic_plugin"]

_live: "weakref.WeakSet[Calculator]" = weakref.WeakSet()


def backend() -> str:
    """The Kokkos execution space this build computes on."""
    return _core.backend()


def metatomic_plugin() -> str | None:
    """The bundled metatomic plugin's path, when this build carries one."""
    from pathlib import Path

    path = Path(__file__).with_name("pet-kokkos-metatomic.so")
    return str(path) if path.exists() else None


def _arrays(numbers, positions, cell, pbc):
    numbers = np.ascontiguousarray(numbers, dtype=np.int32)
    positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    if cell is not None:
        cell = np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3)
    if pbc is None:
        pbc = (cell is not None,) * 3
    elif isinstance(pbc, (bool, np.bool_)):
        pbc = (bool(pbc),) * 3
    pbc = tuple(bool(p) for p in pbc)
    return numbers, positions, cell, pbc


class Calculator:
    """A loaded PET model. Reuse one across calls: it keeps its device memory,
    caches and recorded graphs between them.

    model: a published model's name (downloaded on first use) or a path prefix
        to ``<prefix>.json`` and ``<prefix>.safetensors``.
    gemm, attention: "native" (fp32) or "tf32", ... (process-wide; None leaves
        the current mode, fp32 unless set).
    cache_neighbors: keep a Verlet cache of the neighbour topology across calls,
        for MD and relaxations of the same atoms.
    """

    def __init__(
        self,
        model: str = "pet-mad-s",
        *,
        gemm: str | None = None,
        attention: str | None = None,
        device_neighbors: bool = True,
        cache_neighbors: bool = False,
        recompute: str | None = None,
        memory_budget_gib: float = 0.0,
        max_batch_atoms: int = 0,
    ):
        self.model = resolve(model)
        self._calc = _core.Calculator(
            self.model,
            gemm=gemm,
            attention=attention,
            device_neighbors=device_neighbors,
            cache_neighbors=cache_neighbors,
            recompute=recompute,
            memory_budget_gib=memory_budget_gib,
            max_batch_atoms=max_batch_atoms,
        )
        _live.add(self)

    def compute(self, numbers, positions, cell=None, pbc=None, *, forces=True, charge=0, spin_multiplicity=1) -> dict:
        """Energy (and forces and virial) of one structure.

        Returns a dict with ``energy`` (float), ``energies`` [N] and, with
        forces, ``forces`` [N, 3] and ``virial`` [6] (periodic structures).
        """
        z, x, c, p = _arrays(numbers, positions, cell, pbc)
        return self._calc.compute(z, x, c, p, forces, charge, spin_multiplicity)

    def compute_batch(self, structures: Iterable[Sequence], *, forces=True) -> list[dict]:
        """Several structures as one evaluation: each item is (numbers,
        positions, cell, pbc[, charge, spin_multiplicity]); one dict each."""
        items = []
        for s in structures:
            z, x, c, p = _arrays(*s[:4])
            q = int(s[4]) if len(s) > 4 else 0
            m = int(s[5]) if len(s) > 5 else 1
            items.append((z, x, c, p, q, m))
        return self._calc.compute_batch(items, forces)

    @property
    def cutoff(self) -> float:
        return self._calc.cutoff

    @property
    def atomic_types(self) -> list[int]:
        return self._calc.atomic_types

    def memory_report(self) -> str:
        return self._calc.memory_report()

    def close(self) -> None:
        """Free the model and its device memory now."""
        self._calc.close()


@atexit.register
def _shutdown() -> None:
    # Every model's device memory must go before Kokkos does.
    for calc in list(_live):
        calc.close()
    _core._finalize()
