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

from ._backend import core as _core
from .models import data_dir, download, resolve

__all__ = ["Calculator", "backend", "precision_modes", "download", "data_dir", "metatomic_plugin"]

_live: "weakref.WeakSet[Calculator]" = weakref.WeakSet()


def backend() -> str:
    """The Kokkos execution space this process computes on (Serial, OpenMP,
    Threads, Cuda, HIP): this build's."""
    return _core.backend()


def precision_modes() -> tuple[str, str]:
    """The process-wide (gemm, attention) modes: ("native", "native") unless a
    Calculator or PET_GEMM / PET_ATTN asked for faster ones."""
    return tuple(_core.precision_modes())


def metatomic_plugin() -> str | None:
    """The bundled metatomic plugin's path, when this build carries one."""
    from pathlib import Path

    path = Path(_core.__file__).with_name("pet-kokkos-metatomic.so")
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
    non_conservative: forces and virial from the model's direct heads (upet's
        ``non_conservative=True``): no backward pass, about twice as fast, but
        not the energy's gradient. Most published models have the heads; a
        model without them raises. Without a stress head the virial is zero.
    uncertainty: also return the model's LLPR energy uncertainty (upet's
        ``energy_uncertainty``, a standard deviation in eV) as
        ``energy_uncertainty`` and per atom as ``energies_uncertainty``, and,
        for a model with one, the ``energy_ensemble``. A model without one
        raises.
    smooth_cutoff: for a model whose edges taper with the cosine (pet-mols-s),
        a taper with the same ends whose first three derivatives vanish there.
        The cosine's second derivative jumps at both ends of the taper, so its
        Hessian jumps wherever a pair distance crosses one, which strain
        derivatives of force constants (quasi-harmonic free energies) trip
        over. It changes the model slightly; nothing for a bump-tapered model.
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
        non_conservative: bool = False,
        uncertainty: bool = False,
        smooth_cutoff: bool = False,
    ):
        self.model = resolve(model)
        self.non_conservative = non_conservative
        self._calc = _core.Calculator(
            self.model,
            gemm=gemm,
            attention=attention,
            device_neighbors=device_neighbors,
            cache_neighbors=cache_neighbors,
            recompute=recompute,
            memory_budget_gib=memory_budget_gib,
            max_batch_atoms=max_batch_atoms,
            non_conservative=non_conservative,
            uncertainty=uncertainty,
            smooth_cutoff=smooth_cutoff,
        )
        _live.add(self)

    def compute(self, numbers, positions, cell=None, pbc=None, *, forces=True, charge=0, spin_multiplicity=1) -> dict:
        """Energy (and forces and virial) of one structure.

        Returns a dict with ``energy`` (float), ``energies`` [N] and, with
        forces, ``forces`` [N, 3] and ``virial`` [6] (periodic structures);
        with ``uncertainty``, ``energy_uncertainty``, ``energies_uncertainty``
        [N] and ``energy_ensemble`` [members].
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

    def response(self, numbers, positions, cell=None, pbc=None, *, displacements=None, strains=None, charge=0,
                 spin_multiplicity=1) -> dict:
        """The derivatives of the forces and virial along K directions, exactly:
        a Hessian-vector product, with no finite step. Direction k moves the
        atoms by ``displacements[k]`` ([N, 3], A per unit t) and strains the
        structure by ``strains[k]`` ([3, 3]: positions and cell go to
        x (I + t e)^T); either may be None. Returns ``forces`` [K, N, 3]
        (d forces / dt), ``virial`` [K, 6] (d virial / dt, the virial's order)
        and ``base_forces`` [N, 3].

        One structure at a time; every published model. See
        ``pet_kokkos.properties`` for Hessians, vibrations and elastic tensors
        built on it.
        """
        z, x, c, p = _arrays(numbers, positions, cell, pbc)
        n = len(z)
        u = np.zeros((0, n, 3)) if displacements is None else np.ascontiguousarray(displacements, dtype=np.float64)
        e = np.zeros((0, 3, 3)) if strains is None else np.ascontiguousarray(strains, dtype=np.float64)
        u = u.reshape(-1, n, 3)
        e = e.reshape(-1, 3, 3)
        K = max(len(u), len(e))
        if len(u) == 0:
            u = np.zeros((K, n, 3))
        if len(e) == 0:
            e = np.zeros((K, 3, 3))
        return self._calc.response(z, x, c, p, u, e, charge, spin_multiplicity)

    @property
    def cutoff(self) -> float:
        return self._calc.cutoff

    @property
    def atomic_types(self) -> list[int]:
        return self._calc.atomic_types

    @property
    def has_non_conservative_forces(self) -> bool:
        return self._calc.has_non_conservative_forces

    @property
    def has_non_conservative_stress(self) -> bool:
        return self._calc.has_non_conservative_stress

    @property
    def has_uncertainty(self) -> bool:
        return self._calc.has_uncertainty

    @property
    def n_ensemble(self) -> int:
        return self._calc.n_ensemble

    def memory_report(self) -> str:
        return self._calc.memory_report()

    def close(self) -> None:
        """Free the model and its device memory now."""
        self._calc.close()


@atexit.register
def _shutdown() -> None:
    # Every model's device memory must go before Kokkos does. The wrappers may
    # outlive us (IPython's Out history holds them), so drop their cores here
    # too, or nanobind reports them as leaked.
    for calc in list(_live):
        calc.close()
        del calc._calc
    _core._finalize()
