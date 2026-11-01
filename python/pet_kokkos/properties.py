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

:class:`HessianOperator` wraps the product as a matrix-free operator for
Krylov methods (scipy's ``cg``, ``eigsh``, Newton-CG); ``elastic_tensor`` uses
it, by block CG, for cells of 100 atoms and more.

A structure is a tuple ``(numbers, positions, cell, pbc)`` as
:meth:`Calculator.compute` takes it, a chmpy ``System``, ``Crystal`` or
``Molecule``, or anything else with ``numbers``, ``positions``, ``cell`` and
``pbc`` attributes. Nothing here needs ASE.

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
several GPa -- so finite differences refuse it unless asked. The exact product
has no step to amplify it: in TF32 it gives frequencies to ~0.1-0.4 cm^-1 and
elastic constants to ~0.02 GPa, about fp32 2-point differences' accuracy, so
it is allowed.

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

__all__ = ["Response", "response", "hessian", "HessianOperator", "exp_preconditioner", "Vibrations", "vibrations",
           "Elastic", "elastic_tensor", "phonons", "phonopy_forces", "symmetry_operations", "symmetrize_hessian",
           "symmetrize_elastic"]

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
    """(numbers, positions, cell or None, pbc) from a tuple, a chmpy System,
    Crystal or Molecule, or any object with numbers/positions/cell/pbc."""
    if not isinstance(s, (tuple, list)):
        if type(s).__module__.startswith("chmpy") and not hasattr(s, "numbers"):
            from chmpy.calc.system import System  # a Crystal or a Molecule

            s = System.from_crystal(s) if hasattr(s, "space_group") else System.from_molecule(s)
        cell = getattr(s, "cell", None)
        cell = None if cell is None else np.asarray(getattr(cell, "array", cell), dtype=np.float64)
        pbc = getattr(s, "pbc", None)
        if cell is not None and not cell.any():  # chmpy's isolated systems carry a zero cell
            cell = None
        s = (s.numbers, s.positions, cell, None if pbc is None else tuple(bool(p) for p in np.atleast_1d(pbc)))
    numbers, positions, cell, pbc = (list(s) + [None, None])[:4]
    positions = np.array(positions, dtype=np.float64).reshape(-1, 3)
    if cell is not None:
        cell = np.array(cell, dtype=np.float64).reshape(3, 3)
    if pbc is None:
        pbc = (cell is not None,) * 3
    elif isinstance(pbc, (bool, np.bool_)) or len(pbc) == 1:
        pbc = (bool(np.atleast_1d(pbc)[0]),) * 3
    pbc = tuple(bool(p) for p in pbc)
    if not any(pbc):
        cell = None
    return np.asarray(numbers), positions, cell, pbc


def _check(calc):
    if getattr(calc, "non_conservative", False):
        raise ValueError(
            "pet_kokkos.properties: this Calculator has non_conservative=True; those forces are not the "
            "energy's gradient, so their derivatives are not a Hessian")


def _check_fd_precision(allow_reduced_precision):
    gemm, attention = precision_modes()
    if not allow_reduced_precision and (gemm not in ("native", "ozaki") or attention != "native"):
        raise ValueError(
            f"pet_kokkos.properties: gemm={gemm}, attention={attention}: a finite difference turns their noise "
            "into errors of several cm^-1 and GPa (the exact product does not: use method='exact', or fp32 "
            "(native), or pass allow_reduced_precision=True)")


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
    batch_atoms: atoms per batched evaluation for fd. The exact product sizes
        its own passes by memory (each direction keeps a copy of the network's
        activations).
    """
    _check(calc)
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
            return _exact(calc, (numbers, x, cell, pbc), disp, eps)
        except RuntimeError as e:
            if method == "exact" or "Hessian-vector product" not in str(e):
                raise
    _check_fd_precision(allow_reduced_precision)
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


def _exact(calc, structure, disp, eps):
    numbers, x, cell, pbc = structure
    n = len(numbers)
    u = np.concatenate([disp, np.zeros((len(eps), n, 3))])
    e = np.concatenate([np.zeros((len(disp), 3, 3)), eps])
    r = calc.response(numbers, x, cell, pbc, displacements=u, strains=e)
    return Response(np.asarray(r["forces"]), np.asarray(r["virial"]))


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


# ---- crystal symmetry --------------------------------------------------------------
#
# PET is not exactly equivariant, so a crystal's second derivatives break its
# symmetry slightly (silicon's cubic constants spread by ~0.5 GPa, its optical
# triplet by ~1.5 cm^-1). Averaging over the space group restores it, as
# phonopy does for its force constants. Opt-in: it changes the numbers. The
# same breaking moves atoms during an unconstrained relaxation: X23 crystals
# relaxed from their symmetric cells left their space groups by 0.01-0.05 A, so
# relax keeping the space group (chmpy's relax(crystal, calc, symmetry=True)
# varies the asymmetric unit, so it cannot break it) when it is to be kept.


def _symprec(symmetry):
    return 1e-3 if symmetry is True else float(symmetry)


def symmetry_operations(structure, symprec: float = 1e-3):
    """The structure's space-group operations as (R, perm): R the Cartesian
    rotation [3, 3] and perm[i] the atom that atom i goes to. Needs spglib."""
    import spglib

    numbers, x, cell, pbc = _structure(structure)
    if cell is None or not all(pbc):
        raise ValueError("crystal symmetry needs a structure periodic in all three directions")
    frac = np.linalg.solve(cell.T, x.T).T
    ops = spglib.get_symmetry((cell, frac, numbers), symprec=symprec)
    if ops is None:
        raise RuntimeError("spglib found no symmetry")
    out = []
    Lt = cell.T  # x = Lt f
    for W, w in zip(ops["rotations"], ops["translations"]):
        R = Lt @ W @ np.linalg.inv(Lt)
        img = frac @ W.T + w
        d = img[:, None, :] - frac[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=2)
        perm = np.argmin(dist, axis=1)
        if dist[np.arange(len(perm)), perm].max() > 10 * symprec or len(set(perm)) != len(perm):
            raise RuntimeError("a symmetry operation did not map the atoms onto each other")
        out.append((R, perm))
    return out


def symmetrize_hessian(H, structure, symprec: float = 1e-3):
    """The Gamma-point Hessian averaged over the space group:
    H_ij -> mean over g of R^T H_{g(i) g(j)} R."""
    n = len(_structure(structure)[0])
    Hb = np.asarray(H).reshape(n, 3, n, 3)
    acc = np.zeros_like(Hb)
    ops = symmetry_operations(structure, symprec)
    for R, perm in ops:
        acc += np.einsum("ka,ikjl,lb->iajb", R, Hb[perm][:, :, perm], R)
    return (acc / len(ops)).reshape(3 * n, 3 * n)


def _voigt_to_tensor(C):
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT):
        for J, (k, l) in enumerate(_VOIGT):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    T[a, b, c, d] = C[I, J]
    return T


def symmetrize_elastic(C, structure, symprec: float = 1e-3):
    """A Voigt elastic tensor averaged over the crystal's point group."""
    T = _voigt_to_tensor(np.asarray(C))
    ops = symmetry_operations(structure, symprec)
    acc = np.zeros_like(T)
    for R, _ in ops:
        acc += np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, T)
    acc /= len(ops)
    return np.array([[acc[i, j, k, l] for (k, l) in _VOIGT] for (i, j) in _VOIGT])


def hessian(calc, structure, *, symmetrize: bool = True, symmetry=False, **fd) -> np.ndarray:
    """The Hessian d2E/dx dx [3N, 3N] (eV/A^2), one displacement per
    coordinate. ``fd`` takes :func:`response`'s options (method, stencil).
    symmetrize: (H + H^T)/2. symmetry: True (or a spglib tolerance, default
    1e-3 A) to average over the crystal's space group."""
    n = len(_structure(structure)[0])
    H = -response(calc, structure, _unit_displacements(n), **fd).forces.reshape(3 * n, 3 * n)
    if symmetrize:
        H = 0.5 * (H + H.T)
    return symmetrize_hessian(H, structure, _symprec(symmetry)) if symmetry else H


def _masses(numbers, masses):
    if masses is not None:
        return np.asarray(masses, dtype=np.float64)
    from ._elements import MASSES

    return np.array(MASSES, dtype=np.float64)[np.asarray(numbers)]


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


def vibrations(calc, structure, *, masses=None, project: str = "auto", hessian_matrix=None, symmetry=False,
               **fd) -> Vibrations:
    """Harmonic frequencies and modes.

    project: "auto" (rotations and translations for a molecule, translations
        for a periodic cell: the Gamma point), "rotations", "translations" or
        "none". PET is not exactly rotationally invariant -- a molecule's
        rotations come out at a few to ~20 cm^-1 even in fp64 -- so leave the
        rotations projected for molecules. Projected modes are dropped.
    hessian_matrix: a Hessian already computed, to skip the evaluations.
    symmetry: average the Hessian over the crystal's space group (see
        :func:`hessian`).
    """
    numbers, x, cell, pbc = _structure(structure)
    n = len(numbers)
    m = _masses(numbers, masses)
    H = hessian(calc, structure, **fd) if hessian_matrix is None else np.asarray(hessian_matrix)
    if symmetry:
        H = symmetrize_hessian(H, structure, _symprec(symmetry))
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


# ---- matrix-free: the Hessian as an operator --------------------------------------


class HessianOperator:
    """The Hessian H as a linear operator on [3N] vectors (atom-major, as
    positions.ravel()), each product an exact Hessian-vector product: a block of
    k vectors is one call with k directions. No [3N, 3N] matrix is formed, so
    Krylov methods -- scipy's ``eigsh``, ``lobpcg``, ``cg``, Newton-CG -- need
    only as many products as they take iterations.

    mass_weighted: D = M^-1/2 H M^-1/2 instead (vibrational frequencies are
        sqrt of its eigenvalues).
    project: "none", "translations" or "rotations" (both rigid motions): the
        operator P H P on the space orthogonal to them (mass-weighted when D).
    method: as :func:`response` ("auto", "exact", "fd").

    ``n_products`` counts the products taken; ``as_linear_operator()`` gives a
    scipy LinearOperator.
    """

    def __init__(self, calc, structure, *, mass_weighted: bool = False, masses=None, project: str = "none",
                 method: str = "auto", **fd):
        numbers, x, cell, pbc = _structure(structure)
        self.structure = (numbers, x, cell, pbc)
        self.calc, self.method, self.fd = calc, method, fd
        self.n = len(numbers)
        self.shape = (3 * self.n, 3 * self.n)
        self.dtype = np.dtype(np.float64)
        self.n_products = 0
        m = _masses(numbers, masses) if (mass_weighted or project == "rotations") else np.ones(self.n)
        self.sm = np.repeat(np.sqrt(m), 3) if mass_weighted else None
        if project not in ("none", "translations", "rotations"):
            raise ValueError("project must be none, translations or rotations")
        # the rigid motions span the same space whatever point the rotations are
        # about, so in plain coordinates unit masses give an orthonormal basis of it
        self.rigid = None
        if project != "none":
            self.rigid = _rigid_modes(x, m if mass_weighted else np.ones(self.n), project == "rotations")

    def _project(self, V):
        return V if self.rigid is None else V - self.rigid @ (self.rigid.T @ V)

    def matmat(self, V):
        V = self._project(np.asarray(V, dtype=np.float64).reshape(self.shape[0], -1))
        k = V.shape[1]
        U = V / self.sm[:, None] if self.sm is not None else V
        r = response(self.calc, self.structure, U.T.reshape(k, self.n, 3), method=self.method, **self.fd)
        HU = -r.forces.reshape(k, 3 * self.n).T
        self.n_products += k
        return self._project(HU / self.sm[:, None] if self.sm is not None else HU)

    def matvec(self, v):
        return self.matmat(np.asarray(v).reshape(-1, 1))[:, 0]

    def __matmul__(self, V):
        V = np.asarray(V)
        return self.matvec(V) if V.ndim == 1 else self.matmat(V)

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(self.shape, matvec=self.matvec, matmat=self.matmat, rmatvec=self.matvec,
                              rmatmat=self.matmat, dtype=self.dtype)


def _pairs(numbers, x, cell, pbc, r_cut):
    """Every pair (i, j, d) closer than r_cut, periodic images included (each
    pair both ways), by a sweep over the lattice shifts that can reach."""
    shifts = [np.zeros(3)]
    if cell is not None:
        heights = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
        reach = [int(np.ceil(r_cut / h)) if p else 0 for h, p in zip(heights, pbc)]
        g = np.mgrid[-reach[0] : reach[0] + 1, -reach[1] : reach[1] + 1, -reach[2] : reach[2] + 1].reshape(3, -1).T
        shifts = g @ cell
    I, J, Dd = [], [], []
    for t in shifts:
        for i0 in range(0, len(x), 512):  # rows in blocks: N x N distances at once would not fit
            d = np.linalg.norm(x[None, :, :] + t - x[i0 : i0 + 512, None, :], axis=2)
            i, j = np.nonzero((d < r_cut) & (d > 1e-8))
            I.append(i + i0), J.append(j), Dd.append(d[i, j])
    return np.concatenate(I), np.concatenate(J), np.concatenate(Dd)


def exp_preconditioner(structure, *, mu: float = 1.0, A: float = 3.0, r_cut: float | None = None,
                       c_stab: float = 0.1):
    """The "Exp" preconditioner (Packwood et al., J. Chem. Phys. 144, 164109
    (2016)) as a sparse [3N, 3N] matrix: a pair Hessian mu sum_ij exp(-A (r_ij /
    r_nn - 1)) (e_i - e_j)(e_i - e_j)^T (x) I_3 over pairs within r_cut (2 r_nn,
    r_nn the nearest-neighbour distance), plus c_stab mu I. A cheap
    approximation of the Hessian's shape, for preconditioning Krylov solves on
    it. Needs scipy."""
    import scipy.sparse as sp

    numbers, x, cell, pbc = _structure(structure)
    n = len(numbers)
    _, _, d0 = _pairs(numbers, x, cell, pbc, 4.0)
    r_nn = float(d0.min()) if len(d0) else 1.0
    r_cut = 2.0 * r_nn if r_cut is None else r_cut
    i, j, d = _pairs(numbers, x, cell, pbc, r_cut)
    c = -mu * np.exp(-A * (d / r_nn - 1.0))
    diag = np.zeros(n)
    np.add.at(diag, i, -c)
    rows = np.concatenate([i, np.arange(n)])
    cols = np.concatenate([j, np.arange(n)])
    vals = np.concatenate([c, diag + c_stab * mu])
    P1 = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return sp.kron(P1, sp.identity(3), format="csc")


def _scaled_preconditioner(op, structure, P):
    """P scaled to op's size (mu from one Rayleigh quotient of a smooth vector),
    and its factorisation's solve, in op's coordinates."""
    from scipy.sparse.linalg import splu
    import scipy.sparse as sp

    n = op.n
    rng = np.random.default_rng(1)
    _, x, _, _ = _structure(structure)
    v = np.sin(x @ rng.normal(size=3) * 0.7).repeat(3) * rng.normal(size=3 * n)  # smooth-ish
    if op.sm is not None:
        S = sp.diags(1.0 / op.sm)
        P = S @ P @ S
    v = op._project(v.reshape(-1, 1))[:, 0]
    mu = float(v @ op.matvec(v)) / float(v @ (P @ v))
    lu = splu((abs(mu) * P).tocsc())
    return lambda R: op._project(lu.solve(np.asarray(R).reshape(op.shape[0], -1)))


class _NotConverged(RuntimeError):
    pass


def _block_pcg(op, B, solve, rtol, maxiter):
    """Block preconditioned CG for op X = B (op symmetric positive definite on
    its range), one product of B's width per iteration; the search block is
    re-orthonormalised each step so a converged column cannot break it. Raises
    _NotConverged on negative curvature or at maxiter."""
    X = np.zeros_like(B)
    R = B.copy()
    Z = solve(R)
    Pk = Z
    bnorm = np.linalg.norm(B, axis=0)
    for it in range(1, maxiter + 1):
        Pk, _ = np.linalg.qr(Pk)
        Q = op.matmat(Pk)
        PQ = 0.5 * (Pk.T @ Q + Q.T @ Pk)
        if np.linalg.eigvalsh(PQ).min() <= 0.0:
            raise _NotConverged(f"negative curvature at iteration {it}: the Hessian is not positive definite")
        alpha = np.linalg.solve(PQ, Pk.T @ R)
        X += Pk @ alpha
        R -= Q @ alpha
        if (np.linalg.norm(R, axis=0) <= rtol * np.maximum(bnorm, 1e-300)).all():
            return X, it
        Z = solve(R)
        beta = -np.linalg.solve(PQ, Q.T @ Z)
        Pk = Z + Pk @ beta
    raise _NotConverged(f"block CG did not reach rtol={rtol} in {maxiter} iterations")


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
    n_products: int = 0  # Hessian-vector products (or difference directions) taken
    iterations: int = 0

    @property
    def bulk_modulus(self) -> float:
        "Voigt bulk modulus, GPa."
        c = self.C
        return float((c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9)


def elastic_tensor(calc, structure, *, relaxed_ions: bool = True, rcond: float = 1e-8, solver: str = "auto",
                   rtol: float = 1e-6, maxiter: int = 300, symmetry=False, **fd) -> Elastic:
    """The elastic tensor of a periodic structure at its geometry, from one
    batch: C = C0 - L^T Phi^+ L / V, C0 the clamped-ion tensor (virial response
    to strain), Phi the Gamma-point Hessian and L the internal-strain tensor
    (force response to strain). No relaxations, but the atoms should be at a
    minimum (forces ~0) for C to mean the relaxed-ion tensor; a cell under
    stress gets the stress-dependent terms of d2E/de2 as they are.

    rcond: Phi's eigenvalues below rcond * its largest are treated as zero
        modes, beyond the three translations always removed.
    solver: "direct" builds Phi (3N products) and inverts it; "cg" solves Phi X
        = L by block preconditioned CG (the Exp preconditioner; needs scipy),
        6 products per iteration and never forming Phi. CG took 25-40
        iterations on molecular crystals of 12-384 atoms, so it wins from ~100
        atoms; "auto" (the default) takes it from there, falling back to
        direct, with a warning, if Phi turns out not positive definite (atoms
        not at a minimum) or CG does not converge. ``n_products`` and
        ``iterations`` on the result say what it took.
    rtol, maxiter: CG's convergence (relative residual per column) and limit.
    symmetry: True (or a spglib tolerance) to average C and C0 over the
        crystal's point group; ``asymmetry`` stays the raw tensor's.
    """
    numbers, x, cell, pbc = _structure(structure)
    if cell is None or not all(pbc):
        raise ValueError("elastic_tensor needs a structure periodic in all three directions")
    n = len(numbers)
    V = abs(np.linalg.det(cell))
    if solver not in ("auto", "direct", "cg"):
        raise ValueError("solver must be auto, direct or cg")
    if solver == "auto":
        solver = "cg" if n >= 100 else "direct"
    if solver == "cg" and relaxed_ions:
        r = response(calc, structure, None, _voigt_strains(), **fd)
        C0 = r.virial[:, _PET_FROM_VOIGT].T / V
        L = -r.forces.reshape(6, 3 * n).T
        op = HessianOperator(calc, structure, project="translations", **fd)
        try:
            solve = _scaled_preconditioner(op, structure, exp_preconditioner(structure))
        except ImportError:
            solve = op._project
        try:
            X, its = _block_pcg(op, op._project(L), solve, rtol, maxiter)
        except _NotConverged as e:
            warnings.warn(f"pet_kokkos.properties.elastic_tensor: {e}; building the Hessian instead", stacklevel=2)
        else:
            C = C0 - L.T @ X / V
            asym = float(np.abs(C - C.T).max()) * EV_A3_TO_GPA
            return Elastic(_finish(C, structure, symmetry), _finish(C0, structure, symmetry), L, None, asym, V,
                           n_products=op.n_products + 6, iterations=its)
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
    return Elastic(_finish(C, structure, symmetry), _finish(C0, structure, symmetry), L, H, asym, V,
                   n_products=K + 6)


def _finish(C, structure, symmetry):
    "eV/A^3 to GPa, symmetric, and averaged over the point group if asked."
    C = 0.5 * (C + C.T) * EV_A3_TO_GPA
    return symmetrize_elastic(C, structure, _symprec(symmetry)) if symmetry else C


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
