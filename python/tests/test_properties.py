"""pet_kokkos.properties: Hessians, vibrations, elastic tensors, phonons.

The finite differences' accuracy against an fp64 reference was measured when
these were written (the module docstring); these hold the invariants and the
stencils' agreement with each other.
"""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.build import bulk, molecule  # noqa: E402

import pet_kokkos  # noqa: E402
from pet_kokkos import _core, properties  # noqa: E402

_calcs = {}


def calculator(model):
    if _core.find_model(model) is None:
        pytest.skip(f"model {model} is not installed")
    if model not in _calcs:
        _calcs[model] = pet_kokkos.Calculator(model)
    return _calcs[model]


def relaxed_benzene(calc):
    from ase.optimize import BFGS

    from pet_kokkos.ase import PETCalculator

    a = molecule("C6H6")
    a.calc = PETCalculator(calc.model)
    BFGS(a, logfile=None).run(fmax=1e-4, steps=1000)
    a.calc = None
    return a


def silicon():
    return bulk("Si", "diamond", a=5.43, cubic=True)


def test_translation_has_no_response():
    calc = calculator("pet-mad-s")
    si = silicon()
    d = np.zeros((1, len(si), 3))
    d[0, :, 0] = 1.0
    r = properties.response(calc, si, d)
    assert np.abs(r.forces).max() < 1e-3
    assert np.abs(r.virial).max() < 1e-3


def test_hessian_symmetric_and_acoustic_sum():
    calc = calculator("pet-mad-s")
    si = silicon()
    H = properties.hessian(calc, si, symmetrize=False)
    assert np.abs(H - H.T).max() < 2e-3 * np.abs(H).max()
    rows = H.reshape(3 * len(si), len(si), 3).sum(axis=1)  # sum over displaced atoms
    assert np.abs(rows).max() < 2e-3 * np.abs(H).max()


def test_vibrations_benzene_stencils_agree():
    calc = calculator("pet-mols-s")
    a = relaxed_benzene(calc)
    v2 = properties.vibrations(calc, a, method="fd")
    v4 = properties.vibrations(calc, a, method="fd", points=4, step=0.005)
    assert v2.projected == "rotations"
    assert len(v2.frequencies) == 3 * len(a) - 6
    assert v2.frequencies.min() > 300  # benzene's lowest mode is ~400 cm^-1; no soft rigid modes
    assert np.abs(v2.frequencies - v4.frequencies).max() < 0.5
    assert v2.modes.shape == (3 * len(a) - 6, len(a), 3)


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
    si = silicon()
    si.rattle(0.05, seed=3)
    H = properties.hessian(calc, si, method="exact", symmetrize=False)
    H4 = properties.hessian(calc, si, method="fd", points=4, step=0.005, symmetrize=False)
    # the exact Hessian is symmetric, and a translation's response is zero, to rounding
    assert np.abs(H - H.T).max() < 1e-4
    assert np.abs(H.reshape(3 * len(si), len(si), 3).sum(axis=1)).max() < 1e-4
    assert np.abs(H - H4).max() < 5e-3 * np.abs(H).max()
    e = properties.elastic_tensor(calc, bulk("Si", "diamond", a=5.43, cubic=True), method="exact")
    e4 = properties.elastic_tensor(calc, bulk("Si", "diamond", a=5.43, cubic=True), method="fd", points=4,
                                   step=0.005, strain_step=1e-3)
    assert np.abs(e.C - e4.C).max() < 0.05


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
    prim = bulk("Si", "diamond", a=5.43)
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


def test_refuses_non_conservative():
    calc = calculator("pet-mad-s")
    if not calc.has_non_conservative_forces:
        pytest.skip("model has no non-conservative heads")
    nc = pet_kokkos.Calculator("pet-mad-s", non_conservative=True)
    with pytest.raises(ValueError, match="non_conservative"):
        properties.hessian(nc, silicon())
    nc.close()
