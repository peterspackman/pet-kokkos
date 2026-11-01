"""Second-derivative properties: Hessians, vibrations, phonons, elastic tensors.

>>> from pet_kokkos import Calculator, properties
>>> calc = Calculator("pet-mad-s")
>>> vib = properties.vibrations(calc, atoms)            # cm^-1
>>> el = properties.elastic_tensor(calc, crystal)        # GPa
>>> ph = properties.phonons(calc, crystal, [2, 2, 2])    # a phonopy.Phonopy

Everything is built on one primitive, :func:`response`: the derivatives of the
forces and virial along a set of directions, each an atomic displacement or a
homogeneous strain. Two ways to get them:

- ``method="exact"``: a Hessian-vector product, differentiating the network's
  forward and backward along each direction (:meth:`Calculator.response`). No
  step, no truncation, and the translations' zero response is exact. Every
  published model (pet-mad, pet-omol, pet-mols).
- ``method="fd"``: central finite differences of the forces, every displaced
  structure of a property in one batched evaluation. Any model.

``method="auto"`` (the default) takes the exact product, and falls back to
finite differences for a model it cannot differentiate.

A structure is an ASE ``Atoms`` or a tuple ``(numbers, positions, cell, pbc)``
as :meth:`Calculator.compute` takes it.

Accuracy, measured against the exact product in an fp64 build (benzene, the CO2
crystal, silicon; pet-mad-s). The exact product in fp32 gives Hessians to
~3e-5 eV/A^2 and frequencies to ~0.0005 cm^-1, and is faster than either
stencil. Finite differences are limited by the stencil's truncation, not by
fp32 noise: ``points=2, step=0.0025`` (the default) gives frequencies to ~0.1
cm^-1 and elastic constants to ~0.02 GPa, ``points=4, step=0.005`` ~0.02 cm^-1
and ~0.01 GPa for twice the evaluations. Near soft or unstable modes the
relaxed-ion elastic tensor divides by small eigenvalues and those errors grow
(0.2-0.6 GPa on an unrelaxed cell). TF32 (or any lower-precision GEMM or
attention) puts its noise through the 1/step of a difference -- 1-20 cm^-1,
several GPa -- so it is refused unless asked for.

Units are the model's: Angstrom, eV; frequencies in cm^-1 (imaginary modes as
negative numbers), elastic constants in GPa, in the standard Voigt order xx yy
zz yz xz xy with engineering shear strains. (The virial is pet-kokkos's order,
xx yy zz xy xz yz, as :meth:`Calculator.compute` returns it.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from . import precision_modes

__all__ = ["Response", "response", "hessian", "Vibrations", "vibrations", "Elastic", "elastic_tensor",
           "phonons", "phonopy_forces"]

#: sqrt(eV / A^2 / amu) in cm^-1
EV_A2_AMU_TO_CM = 521.47091
#: eV / A^3 in GPa
EV_A3_TO_GPA = 160.21766208

# Central differences: (offset in steps, weight); the derivative is
# sum(weight * f(x + offset * step)) / step.
STENCILS = {
    2: ((1, 1 / 2), (-1, -1 / 2)),
    4: ((1, 2 / 3), (-1, -2 / 3), (2, -1 / 12), (-2, 1 / 12)),
    6: ((1, 3 / 4), (-1, -3 / 4), (2, -3 / 20), (-2, 3 / 20), (3, 1 / 60), (-3, -1 / 60)),
}

# pet-kokkos's virial order, and the standard Voigt order's place in it.
_PET_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_PET_FROM_VOIGT = [0, 1, 2, 5, 4, 3]


def _structure(s):
    "(numbers, positions, cell or None, pbc) from an ASE Atoms or a tuple."
    if hasattr(s, "get_positions"):
        pbc = tuple(bool(p) for p in s.pbc)
        cell = np.array(s.cell.array, dtype=np.float64) if any(pbc) else None
        return np.asarray(s.numbers), np.array(s.positions, dtype=np.float64), cell, pbc
    numbers, positions, cell, pbc = (list(s) + [None, None])[:4]
    positions = np.array(positions, dtype=np.float64).reshape(-1, 3)
    if cell is not None:
        cell = np.array(cell, dtype=np.float64).reshape(3, 3)
    if pbc is None:
        pbc = (cell is not None,) * 3
    elif isinstance(pbc, (bool, np.bool_)):
        pbc = (bool(pbc),) * 3
    return np.asarray(numbers), positions, cell, tuple(bool(p) for p in pbc)


def _check(calc, allow_reduced_precision):
    if getattr(calc, "non_conservative", False):
        raise ValueError(
            "pet_kokkos.properties: this Calculator has non_conservative=True; those forces are not the "
            "energy's gradient, so their derivatives are not a Hessian")
    gemm, attention = precision_modes()
    if not allow_reduced_precision and (gemm not in ("native", "ozaki") or attention != "native"):
        raise ValueError(
            f"pet_kokkos.properties: gemm={gemm}, attention={attention}: a finite difference turns their noise "
            "into errors of several cm^-1 and GPa. Use fp32 (native) for second derivatives, or pass "
            "allow_reduced_precision=True")


@dataclass
class Response:
    """Derivatives along K directions: ``forces`` [K, N, 3] (eV/A per unit of
    the direction) and ``virial`` [K, 6] (eV, pet-kokkos's order; zero when the
    structure is not periodic)."""

    forces: np.ndarray
    virial: np.ndarray


def response(calc, structure, displacements=None, strains=None, *, method: str = "auto", points: int = 2,
             step: float = 0.0025, strain_step: float = 5e-4, batch_atoms: int = 4096,
             allow_reduced_precision: bool = False) -> Response:
    """d(forces)/dt and d(virial)/dt along each direction x(t) = x + t d.

    displacements: [K, N, 3] atomic displacement directions (A per unit t).
    strains: [K', 3, 3] strain directions: positions and cell go to
        x (I + t e)^T. Their derivatives follow the displacements'.
    method: "exact" (the Hessian-vector product), "fd" or "auto" (exact where
        the model has it, else fd).
    points, step, strain_step: the central-difference stencil (2, 4 or 6
        points) and its steps for displacements (A) and strains; fd only.
    batch_atoms: atoms per batched evaluation for fd, and atoms times
        directions per product for exact (each direction keeps a copy of the
        network's activations).
    """
    _check(calc, allow_reduced_precision)
    if method not in ("auto", "exact", "fd"):
        raise ValueError("method must be auto, exact or fd")
    if points not in STENCILS:
        raise ValueError(f"points must be one of {sorted(STENCILS)}")
    numbers, x, cell, pbc = _structure(structure)
    n = len(numbers)
    periodic = cell is not None and any(pbc)
    disp = np.zeros((0, n, 3)) if displacements is None else np.asarray(displacements, dtype=np.float64).reshape(-1, n, 3)
    eps = np.zeros((0, 3, 3)) if strains is None else np.asarray(strains, dtype=np.float64).reshape(-1, 3, 3)
    if len(eps) and not periodic:
        raise ValueError("strain directions need a periodic structure")
    if method != "fd":
        try:
            return _exact(calc, (numbers, x, cell, pbc), disp, eps, batch_atoms)
        except RuntimeError as e:
            if method == "exact" or "Hessian-vector product" not in str(e):
                raise
    stencil = STENCILS[points]

    def structures() -> Iterator[tuple]:
        for d in disp:
            for o, _ in stencil:
                yield numbers, x + (o * step) * d, cell, pbc
        for e in eps:
            for o, _ in stencil:
                deform = np.eye(3) + (o * strain_step) * e
                yield numbers, x @ deform.T, cell @ deform.T, pbc

    K = len(disp) + len(eps)
    forces = np.zeros((K, n, 3))
    virial = np.zeros((K, 6))
    per_batch = max(1, batch_atoms // max(n, 1))
    it = structures()
    done = 0
    total = K * len(stencil)
    while done < total:
        batch = [next(it) for _ in range(min(per_batch, total - done))]
        for i, r in enumerate(calc.compute_batch(batch), start=done):
            k, j = divmod(i, len(stencil))
            w = stencil[j][1] / (step if k < len(disp) else strain_step)
            forces[k] += w * np.asarray(r["forces"])
            if periodic:
                virial[k] += w * np.asarray(r["virial"])
        done += len(batch)
    return Response(forces, virial)


def _exact(calc, structure, disp, eps, batch_atoms):
    numbers, x, cell, pbc = structure
    n = len(numbers)
    u = np.concatenate([disp, np.zeros((len(eps), n, 3))])
    e = np.concatenate([np.zeros((len(disp), 3, 3)), eps])
    K = len(u)
    forces = np.zeros((K, n, 3))
    virial = np.zeros((K, 6))
    chunk = max(1, batch_atoms // max(n, 1))
    for k in range(0, K, chunk):
        r = calc.response(numbers, x, cell, pbc, displacements=u[k : k + chunk], strains=e[k : k + chunk])
        forces[k : k + chunk] = r["forces"]
        virial[k : k + chunk] = r["virial"]
    return Response(forces, virial)


def _unit_displacements(n, atoms=None):
    atoms = range(n) if atoms is None else atoms
    out = []
    for a in atoms:
        for c in range(3):
            d = np.zeros((n, 3))
            d[a, c] = 1.0
            out.append(d)
    return np.array(out).reshape(-1, n, 3)


def _voigt_strains():
    "The six unit strains of the standard Voigt order, shears as engineering strains."
    out = np.zeros((6, 3, 3))
    for j, (a, b) in enumerate(_VOIGT):
        out[j, a, b] += 0.5 if a != b else 1.0
        out[j, b, a] += 0.5 if a != b else 0.0
    return out


def hessian(calc, structure, *, symmetrize: bool = True, **fd) -> np.ndarray:
    """The Hessian d2E/dx dx [3N, 3N] (eV/A^2), one displacement per
    coordinate. ``fd`` takes :func:`response`'s options (method, stencil)."""
    n = len(_structure(structure)[0])
    H = -response(calc, structure, _unit_displacements(n), **fd).forces.reshape(3 * n, 3 * n)
    return 0.5 * (H + H.T) if symmetrize else H


def _masses(numbers, masses):
    if masses is not None:
        return np.asarray(masses, dtype=np.float64)
    try:
        from ase.data import atomic_masses
    except ImportError as e:
        raise ImportError("pet_kokkos.properties needs masses=... or ASE installed for the standard ones") from e
    return atomic_masses[np.asarray(numbers)]


def _rigid_modes(positions, masses, rotations):
    """Orthonormal mass-weighted translations (and rotations) [3N, k]; a linear
    molecule has two rotations, so the basis is cut to its rank."""
    n = len(masses)
    sm = np.repeat(np.sqrt(masses), 3)
    vecs = []
    for a in range(3):
        v = np.zeros((n, 3))
        v[:, a] = 1.0
        vecs.append(v.ravel() * sm)
    if rotations:
        com = (positions * masses[:, None]).sum(0) / masses.sum()
        r = positions - com
        for a in range(3):
            e = np.zeros(3)
            e[a] = 1.0
            vecs.append(np.cross(e, r).ravel() * sm)
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, s > 1e-8 * s.max()]


@dataclass
class Vibrations:
    """Harmonic modes, lowest first. ``frequencies`` in cm^-1, imaginary ones
    negative; ``modes`` [n_modes, N, 3] the mass-weighted normalised
    eigenvectors; ``hessian`` [3N, 3N] eV/A^2; ``projected``: the rigid modes
    taken out (translations, and rotations for a molecule)."""

    frequencies: np.ndarray
    modes: np.ndarray
    hessian: np.ndarray
    masses: np.ndarray
    projected: str

    @property
    def energies(self) -> np.ndarray:
        "Mode energies hbar omega, eV."
        return self.frequencies * 1.239841984e-4


def vibrations(calc, structure, *, masses=None, project: str = "auto", hessian_matrix=None, **fd) -> Vibrations:
    """Harmonic frequencies and modes.

    project: "auto" (rotations and translations for a molecule, translations
        for a periodic cell: the Gamma point), "rotations", "translations" or
        "none". PET is not exactly rotationally invariant -- a molecule's
        rotations come out at a few to ~20 cm^-1 even in fp64 -- so leave the
        rotations projected for molecules. Projected modes are dropped.
    hessian_matrix: a Hessian already computed, to skip the evaluations.
    """
    numbers, x, cell, pbc = _structure(structure)
    n = len(numbers)
    m = _masses(numbers, masses)
    H = hessian(calc, structure, **fd) if hessian_matrix is None else np.asarray(hessian_matrix)
    if project == "auto":
        project = "translations" if any(pbc) else "rotations"
    if project not in ("rotations", "translations", "none"):
        raise ValueError("project must be auto, rotations, translations or none")
    sm = np.repeat(np.sqrt(m), 3)
    D = H / np.outer(sm, sm)
    D = 0.5 * (D + D.T)
    if project != "none":
        R = _rigid_modes(x, m, project == "rotations")
        # the vibrations' space: an orthonormal complement of the rigid modes
        q, _ = np.linalg.qr(np.hstack([R, np.eye(3 * n)]))
        B = q[:, R.shape[1] : 3 * n]
        w, v = np.linalg.eigh(B.T @ D @ B)
        v = B @ v
    else:
        w, v = np.linalg.eigh(D)
    freqs = np.sign(w) * np.sqrt(np.abs(w)) * EV_A2_AMU_TO_CM
    return Vibrations(freqs, v.T.reshape(-1, n, 3), H, m, project)


@dataclass
class Elastic:
    """``C`` the relaxed-ion (or, with relaxed_ions=False, clamped-ion) elastic
    tensor [6, 6] in GPa, standard Voigt order; ``clamped`` the clamped-ion
    tensor; ``internal_strain`` [3N, 6] d2E/dx de (eV/A); ``hessian`` [3N, 3N];
    ``asymmetry`` the largest |C - C^T| (GPa) before symmetrising, a noise
    estimate; ``volume`` A^3."""

    C: np.ndarray
    clamped: np.ndarray
    internal_strain: np.ndarray
    hessian: np.ndarray | None
    asymmetry: float
    volume: float

    @property
    def bulk_modulus(self) -> float:
        "Voigt bulk modulus, GPa."
        c = self.C
        return float((c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9)


def elastic_tensor(calc, structure, *, relaxed_ions: bool = True, rcond: float = 1e-8, **fd) -> Elastic:
    """The elastic tensor of a periodic structure at its geometry, from one
    batch: C = C0 - L^T Phi^+ L / V, C0 the clamped-ion tensor (virial response
    to strain), Phi the Gamma-point Hessian and L the internal-strain tensor
    (force response to strain). No relaxations, but the atoms should be at a
    minimum (forces ~0) for C to mean the relaxed-ion tensor; a cell under
    stress gets the stress-dependent terms of d2E/de2 as they are.

    rcond: Phi's eigenvalues below rcond * its largest are treated as zero
        modes, beyond the three translations always removed.
    """
    numbers, x, cell, pbc = _structure(structure)
    if cell is None or not all(pbc):
        raise ValueError("elastic_tensor needs a structure periodic in all three directions")
    n = len(numbers)
    V = abs(np.linalg.det(cell))
    disp = _unit_displacements(n) if relaxed_ions else None
    r = response(calc, structure, disp, _voigt_strains(), **fd)
    K = 0 if disp is None else len(disp)
    dW = r.virial[K:][:, _PET_FROM_VOIGT]  # [strain j, virial i] in standard Voigt
    C0 = dW.T / V
    L = -r.forces[K:].reshape(6, 3 * n).T  # d2E/dx de_j
    H = None
    C = C0.copy()
    if relaxed_ions:
        H = -r.forces[:K].reshape(3 * n, 3 * n)
        H = 0.5 * (H + H.T)
        # the three translations, then whatever else is (numerically) soft
        t = np.zeros((3 * n, 3))
        for a in range(3):
            t[a::3, a] = 1 / np.sqrt(n)
        P = np.eye(3 * n) - t @ t.T
        w, v = np.linalg.eigh(P @ H @ P)
        order = np.argsort(np.abs(w))
        keep = order[3:]
        if w[keep].min() < -1e-4 * np.abs(w).max():
            warnings.warn(
                f"pet_kokkos.properties.elastic_tensor: the Hessian has negative eigenvalues (down to "
                f"{w[keep].min():.3g} eV/A^2): the atoms are not at a minimum, and the relaxed-ion tensor "
                "divides by those modes. Relax the structure first.", stacklevel=2)
        keep = keep[np.abs(w[keep]) > rcond * np.abs(w).max()]
        Hinv = (v[:, keep] / w[keep]) @ v[:, keep].T
        C = C0 - L.T @ Hinv @ L / V
    asym = float(np.abs(C - C.T).max()) * EV_A3_TO_GPA
    sym = lambda a: 0.5 * (a + a.T) * EV_A3_TO_GPA  # noqa: E731
    return Elastic(sym(C), sym(C0), L, H, asym, V)


def phonopy_forces(calc, phonon, *, step: float | None = None, **fd) -> np.ndarray:
    """Sets ``phonon.forces`` for a phonopy.Phonopy whose displacements are
    generated, and returns them.

    Rather than evaluating the displaced supercells phonopy made, each of its
    displacements u is taken as a direction and the force derivative along it
    computed with :func:`response`'s stencil: phonopy gets F = (dF/du) |u|, the
    displaced cell's forces less the reference's and the stencil's truncation
    error. So generate displacements without plus-minus pairs (they are
    redundant here); their length only sets the scale, the stencil's own
    ``step`` the accuracy.
    """
    sc = phonon.supercell
    structure = (sc.numbers, sc.positions, sc.cell, (True, True, True))
    dirs, lengths = [], []
    for d in phonon.dataset["first_atoms"]:
        u = np.zeros((len(sc.numbers), 3))
        u[d["number"]] = d["displacement"]
        lengths.append(np.linalg.norm(d["displacement"]))
        dirs.append(u / lengths[-1])
    if step is not None:
        fd["step"] = step
    r = response(calc, structure, np.array(dirs), **fd)
    forces = r.forces * np.array(lengths)[:, None, None]
    phonon.forces = forces
    return forces


def phonons(calc, structure, supercell_matrix, *, primitive_matrix="auto", symprec: float = 1e-5,
            distance: float = 0.01, **fd):
    """A phonopy.Phonopy with its force constants: phonopy's symmetry-reduced
    displacements of the supercell, their force derivatives from
    :func:`phonopy_forces` (one batch), then produce_force_constants. Ask the
    object for band structures, DOS, thermal properties."""
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as e:
        raise ImportError("pet_kokkos.properties.phonons needs phonopy (pip install phonopy)") from e
    numbers, x, cell, pbc = _structure(structure)
    if cell is None or not all(pbc):
        raise ValueError("phonons needs a structure periodic in all three directions")
    unit = PhonopyAtoms(numbers=numbers, cell=cell, positions=x)
    sm = np.asarray(supercell_matrix)
    sm = np.diag(sm) if sm.ndim == 1 else sm
    phonon = Phonopy(unit, supercell_matrix=sm, primitive_matrix=primitive_matrix, symprec=symprec)
    phonon.generate_displacements(distance=distance, is_plusminus=False)
    phonopy_forces(calc, phonon, **fd)
    phonon.produce_force_constants()
    return phonon
