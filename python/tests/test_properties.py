"""pet_kokkos.properties: Hessians, vibrations, elastic tensors, phonons.

The finite differences' accuracy against an fp64 reference was measured when
these were written (the module docstring); these hold the invariants and the
stencils' agreement with each other.
"""

import numpy as np
import pytest

import pet_kokkos
from pet_kokkos import _core, properties

_calcs = {}


def calculator(model):
    if _core.find_model(model) is None:
        pytest.skip(f"model {model} is not installed")
    if model not in _calcs:
        _calcs[model] = pet_kokkos.Calculator(model)
    return _calcs[model]


# Structures as (numbers, positions, cell, pbc) tuples: nothing here needs ASE.


def silicon(a=5.43):
    "Diamond silicon, the 8-atom conventional cell."
    f = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cell = a * np.eye(3)
    return np.full(8, 14), np.vstack([f, f + 0.25]) @ cell, cell, True


def silicon_primitive(a=5.43):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return np.full(2, 14), np.array([[0, 0, 0], [0.25 * a] * 3]), cell, True


def rattled(structure, sigma, seed):
    z, x, cell, pbc = structure
    return z, x + np.random.default_rng(seed).normal(0.0, sigma, x.shape), cell, pbc


def n_atoms(structure):
    return len(structure[0])


def relaxed_benzene(calc):
    from scipy.optimize import minimize

    t = np.arange(6) * np.pi / 3
    ring = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1)
    z = np.array([6] * 6 + [1] * 6)
    x = np.vstack([1.395 * ring, 2.482 * ring])

    def energy(v):
        r = calc.compute(z, v.reshape(-1, 3))
        return r["energy"], -np.asarray(r["forces"]).ravel()

    out = minimize(energy, x.ravel(), jac=True, method="L-BFGS-B", options={"gtol": 1e-5, "maxiter": 2000})
    return z, out.x.reshape(-1, 3), None, False


def test_translation_has_no_response():
    calc = calculator("pet-mad-s")
    si = silicon()
    d = np.zeros((1, n_atoms(si), 3))
    d[0, :, 0] = 1.0
    r = properties.response(calc, si, d)
    assert np.abs(r.forces).max() < 1e-3
    assert np.abs(r.virial).max() < 1e-3


def test_hessian_symmetric_and_acoustic_sum():
    calc = calculator("pet-mad-s")
    si = silicon()
    H = properties.hessian(calc, si, symmetrize=False)
    assert np.abs(H - H.T).max() < 2e-3 * np.abs(H).max()
    rows = H.reshape(3 * n_atoms(si), n_atoms(si), 3).sum(axis=1)  # sum over displaced atoms
    assert np.abs(rows).max() < 2e-3 * np.abs(H).max()


def test_vibrations_benzene_stencils_agree():
    calc = calculator("pet-mols-s")
    a = relaxed_benzene(calc)
    v2 = properties.vibrations(calc, a, method="fd")
    v4 = properties.vibrations(calc, a, method="fd", points=4, step=0.005)
    assert v2.projected == "rotations"
    assert len(v2.frequencies) == 3 * n_atoms(a) - 6
    assert v2.frequencies.min() > 300  # benzene's lowest mode is ~400 cm^-1; no soft rigid modes
    assert np.abs(v2.frequencies - v4.frequencies).max() < 0.5
    assert v2.modes.shape == (3 * n_atoms(a) - 6, n_atoms(a), 3)


def test_elastic_silicon_cubic():
    calc = calculator("pet-mad-s")
    el = properties.elastic_tensor(calc, silicon(), method="fd")
    C = el.C
    assert 100 < C[0, 0] < 200 and 30 < C[0, 1] < 100 and 40 < C[3, 3] < 100
    # cubic, up to PET's own breaking of the symmetry (~0.5 GPa: it is not exactly
    # equivariant), which no stencil changes
    assert np.allclose(np.diag(C)[:3], C[0, 0], rtol=1e-2)
    assert np.allclose(np.diag(C)[3:], C[3, 3], rtol=1e-2)
    assert np.abs(C[:3, 3:]).max() < 1.0
    assert el.asymmetry < 0.1
    el4 = properties.elastic_tensor(calc, silicon(), method="fd", points=4, step=0.005, strain_step=1e-3)
    assert np.abs(el.C - el4.C).max() < 0.1
    # the internal strain relaxation softens C44 only, in diamond
    assert el.clamped[3, 3] > C[3, 3] + 5 and abs(el.clamped[0, 0] - C[0, 0]) < 0.1


def test_exact_matches_finite_differences():
    calc = calculator("pet-mad-s")
    si = rattled(silicon(), 0.05, seed=3)
    H = properties.hessian(calc, si, method="exact", symmetrize=False)
    H4 = properties.hessian(calc, si, method="fd", points=4, step=0.005, symmetrize=False)
    # the exact Hessian is symmetric, and a translation's response is zero, to rounding
    assert np.abs(H - H.T).max() < 1e-4
    assert np.abs(H.reshape(3 * n_atoms(si), n_atoms(si), 3).sum(axis=1)).max() < 1e-4
    assert np.abs(H - H4).max() < 5e-3 * np.abs(H).max()
    e = properties.elastic_tensor(calc, silicon(), method="exact")
    e4 = properties.elastic_tensor(calc, silicon(), method="fd", points=4,
                                   step=0.005, strain_step=1e-3)
    assert np.abs(e.C - e4.C).max() < 0.05


def test_hessian_operator():
    calc = calculator("pet-mad-s")
    si = rattled(silicon(), 0.03, seed=5)
    H = properties.hessian(calc, si, method="exact", symmetrize=False)
    op = properties.HessianOperator(calc, si)
    V = np.random.default_rng(0).normal(size=(3 * n_atoms(si), 4))
    assert np.abs(op @ V - H @ V).max() < 1e-4 * np.abs(H).max()
    assert op.n_products == 4
    # projected: a translation goes to zero, and so does anything's translation part
    pr = properties.HessianOperator(calc, si, project="translations")
    t = np.tile([1.0, 0.0, 0.0], n_atoms(si))
    assert np.abs(pr @ t).max() < 1e-10
    assert np.abs((pr @ V).reshape(n_atoms(si), 3, -1).sum(axis=0)).max() < 1e-8


def test_elastic_cg_matches_direct():
    pytest.importorskip("scipy")
    calc = calculator("pet-mad-s")
    si = silicon()
    direct = properties.elastic_tensor(calc, si, solver="direct")
    cg = properties.elastic_tensor(calc, si, solver="cg")
    assert np.abs(cg.C - direct.C).max() < 1e-3
    assert cg.hessian is None and 0 < cg.iterations < 100
    assert direct.n_products == 3 * n_atoms(si) + 6


def test_crystal_symmetry():
    pytest.importorskip("spglib")
    calc = calculator("pet-mad-s")
    si = silicon()
    C = properties.elastic_tensor(calc, si, symmetry=True).C
    assert np.allclose(np.diag(C)[:3], C[0, 0], atol=1e-6) and np.allclose(np.diag(C)[3:], C[3, 3], atol=1e-6)
    assert np.abs(C[:3, 3:]).max() < 1e-6 and abs(C[0, 1] - C[1, 2]) < 1e-6
    v = properties.vibrations(calc, si, symmetry=True)
    assert np.ptp(np.sort(v.frequencies)[-3:]) < 1e-6  # the optical triplet, degenerate again
    H = properties.hessian(calc, si, symmetry=True)
    assert np.abs(properties.symmetrize_hessian(H, si) - H).max() < 1e-10


def test_exact_residual_featurizer():
    calc = calculator("pet-mols-s")  # residual featurizer, LayerNorm, SiLU, cosine cutoff
    a = relaxed_benzene(calc)
    v = properties.vibrations(calc, a, method="exact")
    v4 = properties.vibrations(calc, a, method="fd", points=4, step=0.005)
    assert np.abs(v.frequencies - v4.frequencies).max() < 0.1


def test_phonons_match_gamma_vibrations():
    pytest.importorskip("phonopy")
    calc = calculator("pet-mad-s")
    si = silicon()
    prim = silicon_primitive()
    ph = properties.phonons(calc, prim, [2, 2, 2])
    ph.run_qpoints([[0, 0, 0]])
    gamma = np.sort(ph.qpoints.frequencies[0]) * 33.35641  # THz -> cm^-1
    vib = properties.vibrations(calc, si)  # the conventional cell's Gamma point
    optical = np.sort(vib.frequencies)[-3:]
    assert np.abs(gamma[:3]).max() < 1.0
    # phonopy symmetrises the force constants; the raw Hessian's triplet is split
    # by ~1.5 cm^-1, PET not being exactly equivariant: compare the means
    assert abs(gamma[3:].mean() - optical.mean()) < 0.5
    assert np.ptp(optical) < 3.0


def test_chmpy_structures():
    chmpy = pytest.importorskip("chmpy")  # noqa: F841
    from chmpy.calc.system import System

    calc = calculator("pet-mad-s")
    z, x, cell, _ = silicon()
    H = properties.hessian(calc, System(z, x, cell))
    assert np.allclose(H, properties.hessian(calc, silicon()))
    zb, xb, _, _ = relaxed_benzene(calculator("pet-mols-s"))
    v = properties.vibrations(calculator("pet-mols-s"), System(zb, xb))  # isolated: zero cell, no pbc
    assert v.projected == "rotations" and len(v.frequencies) == 3 * len(zb) - 6


def test_refuses_non_conservative():
    calc = calculator("pet-mad-s")
    if not calc.has_non_conservative_forces:
        pytest.skip("model has no non-conservative heads")
    nc = pet_kokkos.Calculator("pet-mad-s", non_conservative=True)
    with pytest.raises(ValueError, match="non_conservative"):
        properties.hessian(nc, silicon())
    nc.close()


def test_smooth_cutoff():
    calculator("pet-mols-s")  # skips when not installed
    plain = pet_kokkos.Calculator("pet-mols-s")
    smooth = pet_kokkos.Calculator("pet-mols-s", smooth_cutoff=True)
    z, x, _, _ = relaxed_benzene(plain)
    a, b = plain.compute(z, x), smooth.compute(z, x)
    assert abs(a["energy"] - b["energy"]) < 1e-3 * len(z)  # the model barely moves
    assert np.abs(np.asarray(a["forces"]) - np.asarray(b["forces"])).max() < 0.02
    # and its Hessian no longer depends on the step across the taper
    H1 = properties.hessian(smooth, (z, x), method="fd", points=4, step=1e-3)
    H2 = properties.hessian(smooth, (z, x), method="fd", points=4, step=4e-3)
    He = properties.hessian(smooth, (z, x), method="exact")
    assert np.abs(H1 - He).max() < 5e-3 and np.abs(H2 - He).max() < 5e-2
    plain.close(), smooth.close()
