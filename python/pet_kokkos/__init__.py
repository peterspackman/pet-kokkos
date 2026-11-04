"""pet-kokkos: PET machine-learned interatomic potentials on Kokkos.

>>> import pet_kokkos
>>> calc = pet_kokkos.Calculator("pet-mad-s")
>>> result = calc.compute(numbers, positions, cell, pbc=(True, True, True))
>>> result["energy"], result["forces"], result["virial"]

Units are the model's: Angstrom, eV, eV/A. ``virial`` is the symmetric
W = V * stress in Voigt order [xx, yy, zz, xy, xz, yz]. For ASE, see
``pet_kokkos.ase.PETCalculator``.

Grimme's D3 dispersion correction, on its own or added to a model:

>>> d3 = pet_kokkos.D3("pbesol")                      # rational (BJ) damping
>>> d3.compute(numbers, positions, cell, pbc=True)["energy"]
>>> calc = pet_kokkos.Calculator("pet-mad-s", d3="pbesol")

Everything runs in plain fp32 unless asked otherwise: ``gemm="3xtf32"`` and
``attention="3xtf32"`` keep fp32's force accuracy on the tensor cores of A100,
H100/H200 and B200, faster there; ``gemm="tf32"`` and ``attention="tf32"`` run on TF32 tensor cores on recent
NVIDIA GPUs, faster and less exact (see the README's "Faster modes"); both are
process-wide.
"""

from __future__ import annotations

import atexit
import weakref
from typing import Iterable, Sequence

import numpy as np

from ._backend import core as _core
from .models import data_dir, download, resolve

__all__ = ["Calculator", "D3", "d3_functionals", "backend", "precision_modes", "download", "data_dir",
           "metatomic_plugin"]

_live: "weakref.WeakSet" = weakref.WeakSet()


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


def d3_functionals(damping: str = "bj") -> list[str]:
    """The functionals with published D3 parameters for a damping, "bj" or "zero"."""
    return list(_core.d3_functionals(damping))


class D3:
    """Grimme's DFT-D3 dispersion correction (pet::D3), on the device, in double.

    functional: the DFT functional whose parameters to use ("pbe", "pbesol",
        "r2scan", "b3lyp", ...; ``d3_functionals()`` lists them). Which one
        suits a model is the model's training data's: PET-MAD was trained on
        PBEsol. A model trained on a dispersion-inclusive functional (the omol
        models, wB97M-V) already has dispersion: do not add D3 to it.
    damping: "bj" (rational, Becke-Johnson; the default) or "zero".
    atm: add the Axilrod-Teller-Muto three-body term (s9 = 1).
    s6, s8, a1, a2, rs6, rs8, alpha, s9: override parameters, in their
        published units (a2 in bohr). Without a functional, give them all.
    cutoff, cn_cutoff, atm_cutoff: where the pair, coordination-number and
        three-body sums stop (Angstrom; default 12, 8 and 8), each with a smooth
        taper of width *_taper ending there (default 2; 0 is a hard cutoff).
    tail: add the pair sum beyond the cutoff for 3D-periodic structures as a
        uniform continuum (energy, pressure and the forces through the CNs).
    reference: s-dftd3's cutoffs (95 bohr pairs, 40 bohr CN and three-body,
        hard, no tail): slow, for checking.
    mixed_precision: each pair's and triangle's arithmetic in float, the sums
        in double: several times faster on GPUs with slow double precision,
        within ~1e-7 relative of the default (all double).

    ``compute`` takes the arrays a Calculator does and returns ``energy``,
    ``energies`` [N] and, with forces, ``forces`` [N, 3] and ``virial`` [6].
    """

    def __init__(
        self,
        functional: str | None = None,
        damping: str = "bj",
        atm: bool = False,
        *,
        s6: float | None = None,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
        rs6: float | None = None,
        rs8: float | None = None,
        alpha: float | None = None,
        s9: float | None = None,
        cutoff: float | None = None,
        cutoff_taper: float | None = None,
        cn_cutoff: float | None = None,
        cn_taper: float | None = None,
        atm_cutoff: float | None = None,
        atm_taper: float | None = None,
        tail: bool = True,
        reference: bool = False,
        mixed_precision: bool = False,
    ):
        self.functional = functional
        self._d3 = _core.D3(
            functional, damping, atm, s6=s6, s8=s8, a1=a1, a2=a2, rs6=rs6, rs8=rs8, alpha=alpha, s9=s9,
            cutoff=cutoff, cutoff_taper=cutoff_taper, cn_cutoff=cn_cutoff, cn_taper=cn_taper,
            atm_cutoff=atm_cutoff, atm_taper=atm_taper, tail=tail, reference=reference,
            mixed_precision=mixed_precision,
        )
        _live.add(self)

    def compute(self, numbers, positions, cell=None, pbc=None, *, forces=True) -> dict:
        """D3's energy (and forces and virial) for one structure, as Calculator.compute."""
        z, x, c, p = _arrays(numbers, positions, cell, pbc)
        return self._d3.compute(z, x, c, p, forces)

    def compute_batch(self, structures: Iterable[Sequence], *, forces=True) -> list[dict]:
        """Several structures in one pass: each item is (numbers, positions, cell, pbc)."""
        return self._d3.compute_batch([_arrays(*s[:4]) for s in structures], forces)

    @property
    def params(self) -> dict:
        """damping, s6, s8, a1, a2 (bohr), rs6, rs8, alpha, s9."""
        return self._d3.params

    @property
    def cutoffs(self) -> dict:
        """cutoff, cutoff_taper, cn_cutoff, cn_taper, atm_cutoff, atm_taper (Angstrom), tail."""
        return self._d3.cutoffs

    def ghost_cutoff(self, exchange: bool = False) -> float:
        """How far past its own atoms an engine must supply ghosts (see pet::D3)."""
        return self._d3.ghost_cutoff(exchange)

    def close(self) -> None:
        """Free the device memory now."""
        self._d3.close()

    def __repr__(self) -> str:
        p = self.params
        return f"D3({self.functional!r}, damping={p['damping']!r}, s9={p['s9']})"


class Calculator:
    """A loaded PET model. Reuse one across calls: it keeps its device memory,
    caches and recorded graphs between them.

    model: a published model's name (downloaded on first use) or a path prefix
        to ``<prefix>.json`` and ``<prefix>.safetensors``.
    gemm, attention: "native" (fp32 on the CUDA cores), "3xtf32" (as accurate,
        on tensor cores) or "tf32", ... (process-wide; None leaves the current
        mode, native unless set).
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
    d3: add Grimme's D3 dispersion to every result: a functional's name
        ("pbesol"), with ``d3_damping`` and ``d3_atm``, or a ``D3`` with
        settings of its own. Off by default; see ``D3`` for which functional.
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
        d3: "str | D3 | None" = None,
        d3_damping: str = "bj",
        d3_atm: bool = False,
    ):
        self.model = resolve(model)
        if isinstance(d3, str):
            d3 = D3(d3, d3_damping, d3_atm)
        self.d3 = d3
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
            d3=None if d3 is None else d3._d3,
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
        built on it. With D3, D3's part is by central differences of its
        forces (it is double and smooth, so to ~1e-8).
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
    for obj in list(_live):
        obj.close()
        if hasattr(obj, "_calc"):
            del obj._calc
        if hasattr(obj, "_d3"):
            del obj._d3
    _core._finalize()
