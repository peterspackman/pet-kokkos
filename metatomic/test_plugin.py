"""The pet-kokkos metatomic plugin, driven through metatomic's Python API.

    PET_METATOMIC_PLUGIN=build/metatomic/pet-kokkos-metatomic.so pytest metatomic/test_plugin.py

Needs metatomic (core, with its Python package), metatensor, numpy, ase and
pytest; the golden models (pet-mad-xs, pet-mad-xs-v1.6, pet-mols-s-v1.1) and the
charge-conditioned pet-omol-s where pet-kokkos looks for models; and pet-eval
from the same build (apps/pet-eval beside metatomic/), or PET_EVAL. A missing
model fails, it is not skipped.

The references are tests/golden -- metatrain's PET through metatomic's ASE
calculator, held to test_golden.cpp's tolerances with the GEMMs in native fp32
-- and, for charge and spin, pet-eval.
"""

import json
import os
import pathlib
import subprocess

import ase
import ase.neighborlist
import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import metatomic as mta


ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
MODELS = ["pet-mad-xs", "pet-mad-xs-v1.6", "pet-mols-s-v1.1"]
CASES = ["crystal", "disordered", "molecule"]
OPTIONS = {"gemm": "native"}

ENERGY = mta.Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)
ENERGY_NO_GRADIENTS = mta.Quantity(name="energy", unit="eV", sample_kind="system")
ATOM_ENERGY = mta.Quantity(name="energy", unit="eV", sample_kind="atom")


def energy_tolerance(reference):
    return max(1e-3, 2e-6 * abs(reference))


FORCE_TOLERANCE = 1e-2  # eV/A
STRESS_TOLERANCE = 1e-4  # eV/A^3


@pytest.fixture(scope="module", autouse=True)
def plugin():
    path = os.environ.get("PET_METATOMIC_PLUGIN")
    if not path:
        raise RuntimeError("set PET_METATOMIC_PLUGIN to the built pet-kokkos-metatomic.so")
    mta.load_plugin(path)
    metatensor.register_external_data_wrapper(
        "metatensor::SimpleDataArray", metatensor.ExternalCpuArray
    )


_models = {}


def model(name, **options):
    key = (name, tuple(sorted(options.items())))
    if key not in _models:
        _models[key] = mta.load_model(name, {**OPTIONS, **options}, plugin_name="pet-kokkos")
    return _models[key]


def golden(name, case):
    with open(GOLDEN / f"{name}_{case}.json") as f:
        return json.load(f)


def make_system(atoms, cutoff, charge=None, spin=None):
    """A metatomic System for `atoms`, with the full pair list at `cutoff`."""
    system = mta.System(
        "Angstrom",
        np.asarray(atoms.numbers, dtype=np.int32),
        np.asarray(atoms.positions, dtype=np.float64),
        np.asarray(atoms.cell[:], dtype=np.float64),
        np.asarray(atoms.pbc, dtype=bool),
    )

    i, j, S, D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    samples = np.column_stack([i, j, S]).astype(np.int32)
    pairs = TensorBlock(
        values=D.reshape(-1, 3, 1).astype(np.float64),
        samples=Labels(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    options = mta.PairListOptions(cutoff=cutoff, full_list=True, strict=False)
    system.add_pairs(options, pairs)

    for name, value in (("charge", charge), ("spin_multiplicity", spin)):
        if value is None:
            continue
        block = TensorBlock(
            values=np.array([[float(value)]]),
            samples=Labels(["system"], np.array([[0]], dtype=np.int32)),
            components=[],
            properties=Labels.range(name, 1),
        )
        system.add_custom_data(name, TensorMap(Labels(["_"], np.array([[0]])), [block]))
    return system


def golden_atoms(g):
    return ase.Atoms(
        numbers=g["atomic_numbers"],
        positions=g["positions"],
        cell=g["cell"],
        pbc=g["pbc"],
    )


def system_for(name, atoms, **kwargs):
    m = model(name)
    cutoff = m.requested_pair_lists()[0].cutoff
    if m.requested_inputs():
        kwargs.setdefault("charge", 0)
        kwargs.setdefault("spin", 1)
    return make_system(atoms, cutoff, **kwargs)


def evaluate(name, systems, outputs, selected_atoms=None, check=True):
    return mta.execute_model(
        model(name),
        systems,
        outputs,
        selected_atoms=selected_atoms,
        check_consistency=check,
    )


def test_capabilities():
    m = model("pet-mad-xs")
    capabilities = m.capabilities()
    assert capabilities.length_unit.lower() == "angstrom"
    assert capabilities.dtype == "float64"
    # A CUDA build reads and writes device memory too, and says so first.
    assert capabilities.supported_devices in (["cpu"], ["cuda", "cpu"])
    assert 1 in capabilities.atomic_types and 8 in capabilities.atomic_types
    # Two message-passing layers and the adaptive cutoff, 7.5 A each.
    assert capabilities.interaction_range == pytest.approx(3 * 7.5)

    (pairs,) = m.requested_pair_lists()
    assert pairs.full_list
    assert pairs.cutoff == pytest.approx(7.5)
    assert m.requested_inputs() == []

    assert model("pet-mols-s-v1.1").requested_inputs() == []
    inputs = model("pet-omol-s").requested_inputs()
    assert [q.name for q in inputs] == ["charge", "spin_multiplicity"]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("case", CASES)
def test_golden(name, case):
    g = golden(name, case)
    atoms = golden_atoms(g)
    energy, atom_energy = evaluate(name, [system_for(name, atoms)], [ENERGY, ATOM_ENERGY])

    block = energy.block()
    e = block.values[0, 0]
    assert abs(e - g["total_energy"]) <= energy_tolerance(g["total_energy"])

    per_atom = atom_energy.block().values[:, 0]
    assert abs(per_atom.sum() - e) <= 1e-6 * max(1.0, abs(e))
    np.testing.assert_allclose(per_atom, g["per_atom_energies"], atol=energy_tolerance(g["total_energy"]))

    forces = -block.gradient("positions").values[:, :, 0]
    np.testing.assert_allclose(forces, g["forces"], atol=FORCE_TOLERANCE)

    if g["periodic"]:
        virial = block.gradient("strain").values[0, :, :, 0]
        np.testing.assert_allclose(virial, virial.T, atol=1e-10)
        # The reference symmetrized, as test_golden.cpp does: autograd's
        # antisymmetric part is noise, and large enough to dominate.
        reference = np.array(g["stress"])
        reference = 0.5 * (reference + reference.T)
        np.testing.assert_allclose(virial / g["volume"], reference, atol=STRESS_TOLERANCE)


def test_batch_matches_single():
    name = "pet-mad-xs"
    systems = [golden_atoms(golden(name, case)) for case in CASES]
    batch = evaluate(name, [system_for(name, a) for a in systems], [ENERGY])[0].block()
    gradients = batch.gradient("positions")
    for s, atoms in enumerate(systems):
        single = evaluate(name, [system_for(name, atoms)], [ENERGY])[0].block()
        assert batch.values[s, 0] == pytest.approx(single.values[0, 0], abs=1e-9)
        rows = gradients.samples.values[:, 1] == s
        np.testing.assert_allclose(
            gradients.values[rows], single.gradient("positions").values, atol=1e-9
        )
        np.testing.assert_allclose(
            batch.gradient("strain").values[s], single.gradient("strain").values[0], atol=1e-9
        )


def test_selected_atoms():
    """A selection's energy is its atoms' share; selections that split the
    atoms add up to the whole, gradients included."""
    name = "pet-mad-xs"
    atoms = golden_atoms(golden(name, "disordered"))
    n = len(atoms)
    whole = evaluate(name, [system_for(name, atoms)], [ENERGY, ATOM_ENERGY])
    per_atom = whole[1].block().values[:, 0]

    rng = np.random.default_rng(7)
    first = np.sort(rng.choice(n, size=n // 3, replace=False))
    rest = np.setdiff1d(np.arange(n), first)

    parts = []
    for chosen in (first, rest):
        selection = Labels(
            ["system", "atom"],
            np.column_stack([np.zeros_like(chosen), chosen]).astype(np.int32),
        )
        energy, atom_energy = evaluate(
            name, [system_for(name, atoms)], [ENERGY, ATOM_ENERGY], selected_atoms=selection
        )
        block = energy.block()
        assert block.values[0, 0] == pytest.approx(per_atom[chosen].sum(), abs=1e-6)
        np.testing.assert_array_equal(atom_energy.block().samples.values[:, 1], chosen)
        np.testing.assert_allclose(atom_energy.block().values[:, 0], per_atom[chosen], atol=1e-6)
        # The gradient is still with respect to every atom.
        assert block.gradient("positions").values.shape[0] == n
        parts.append(block)

    total = whole[0].block()
    assert parts[0].values[0, 0] + parts[1].values[0, 0] == pytest.approx(total.values[0, 0], abs=1e-6)
    for g in ("positions", "strain"):
        np.testing.assert_allclose(
            parts[0].gradient(g).values + parts[1].gradient(g).values,
            total.gradient(g).values,
            atol=1e-5,
        )


def test_system_without_selected_atoms():
    """A system the selection leaves out gets no energy row, as metatomic
    expects; the others keep theirs, with gradients that point at them."""
    name = "pet-mad-xs"
    molecule = golden_atoms(golden(name, "molecule"))
    crystal = golden_atoms(golden(name, "crystal"))
    systems = [system_for(name, molecule), system_for(name, crystal)]
    chosen = np.array([0, 3, 5])
    selection = Labels(
        ["system", "atom"], np.column_stack([np.ones_like(chosen), chosen]).astype(np.int32)
    )
    energy, atom_energy = evaluate(name, systems, [ENERGY, ATOM_ENERGY], selected_atoms=selection)

    block = energy.block()
    np.testing.assert_array_equal(block.samples.values, [[1]])
    gradient = block.gradient("positions")
    assert gradient.values.shape[0] == len(crystal)
    np.testing.assert_array_equal(gradient.samples.values[:, :2], [[0, 1]] * len(crystal))
    assert block.gradient("strain").values.shape[0] == 1

    alone = evaluate(name, [system_for(name, crystal)], [ATOM_ENERGY])[0].block().values[:, 0]
    assert block.values[0, 0] == pytest.approx(alone[chosen].sum(), abs=1e-6)
    np.testing.assert_array_equal(atom_energy.block().samples.values, selection.values)


def test_energy_only():
    name = "pet-mad-xs"
    atoms = golden_atoms(golden(name, "crystal"))
    g = golden(name, "crystal")
    (energy,) = evaluate(name, [system_for(name, atoms)], [ENERGY_NO_GRADIENTS])
    assert abs(energy.block().values[0, 0] - g["total_energy"]) <= energy_tolerance(g["total_energy"])
    assert energy.block().gradients_list() == []


def pet_eval(name, xyz, charge, spin):
    exe = os.environ.get("PET_EVAL")
    if not exe:
        exe = pathlib.Path(os.environ["PET_METATOMIC_PLUGIN"]).resolve().parent.parent / "apps" / "pet-eval"
    out = subprocess.run(
        [str(exe), name, str(xyz), "-f", "--json", "--charge", str(charge), "--spin", str(spin)],
        env={**os.environ, "PET_GEMM": "native"},
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)["frames"][0]


def test_charge_and_spin(tmp_path):
    name = "pet-omol-s"
    water = ase.Atoms(
        "OH2", positions=[[0, 0, 0.11926], [0, 0.76323, -0.47704], [0, -0.76323, -0.47704]]
    )
    xyz = tmp_path / "water.xyz"
    water.write(xyz)

    energies = []
    for charge, spin in ((0, 1), (1, 2)):
        reference = pet_eval(name, xyz, charge, spin)
        (energy,) = evaluate(name, [system_for(name, water, charge=charge, spin=spin)], [ENERGY])
        block = energy.block()
        assert block.values[0, 0] == pytest.approx(reference["energy"], abs=1e-6)
        np.testing.assert_allclose(
            -block.gradient("positions").values[:, :, 0], reference["forces"], atol=1e-6
        )
        energies.append(block.values[0, 0])
    assert abs(energies[1] - energies[0]) > 1e-3

    # Both in one call: each system keeps its own charge and spin.
    systems = [system_for(name, water, charge=0, spin=1), system_for(name, water, charge=1, spin=2)]
    (energy,) = evaluate(name, systems, [ENERGY_NO_GRADIENTS])
    np.testing.assert_allclose(energy.block().values[:, 0], energies, atol=1e-6)


def test_options():
    with pytest.raises(mta.MetatomicError, match="unknown pet-kokkos option 'colour'"):
        mta.load_model("pet-mad-xs", {"colour": "blue"}, plugin_name="pet-kokkos")
    with pytest.raises(mta.MetatomicError, match="recompute"):
        mta.load_model("pet-mad-xs", {"recompute": "sometimes"}, plugin_name="pet-kokkos")
    with pytest.raises(mta.MetatomicError, match="could not load the model"):
        mta.load_model("not-a-pet-model", plugin_name="pet-kokkos")

    # A recompute tier changes the memory held, not the answer.
    atoms = golden_atoms(golden("pet-mad-xs", "disordered"))
    reference = evaluate("pet-mad-xs", [system_for("pet-mad-xs", atoms)], [ENERGY])[0].block()
    layers = mta.execute_model(
        model("pet-mad-xs", recompute="layers"),
        [system_for("pet-mad-xs", atoms)],
        [ENERGY],
        check_consistency=True,
    )[0].block()
    assert layers.values[0, 0] == pytest.approx(reference.values[0, 0], abs=1e-6)
    np.testing.assert_allclose(
        layers.gradient("positions").values, reference.gradient("positions").values, atol=1e-6
    )


def test_unit_conversion():
    """metatomic converts the outputs to the units the engine asks for."""
    name = "pet-mad-xs"
    atoms = golden_atoms(golden(name, "crystal"))
    ev = evaluate(name, [system_for(name, atoms)], [ENERGY_NO_GRADIENTS])[0].block().values[0, 0]
    kj = mta.Quantity(name="energy", unit="kJ/mol", sample_kind="system")
    (energy,) = evaluate(name, [system_for(name, atoms)], [kj])
    assert energy.block().values[0, 0] == pytest.approx(ev * 96.48533212, rel=1e-6)


# --- systems in device memory --------------------------------------------------
#
# A CUDA build takes systems whose arrays are on the GPU (here torch tensors)
# without copying them to the host, and leaves its outputs there. Run whenever
# the plugin says it can; they need PyTorch with CUDA then, and fail without it.
#
# Without metatomic's consistency checks: they build their expected labels on
# the CPU and cannot compare them with a device block's, whose labels metatensor
# requires on the device (test_device_consistency_checks).


def cuda_build():
    return "cuda" in model("pet-mad-xs").capabilities().supported_devices


needs_cuda = pytest.mark.skipif(
    "not cuda_build()", reason="the plugin was not built for CUDA"
)


@pytest.fixture(scope="module")
def torch():
    import torch

    assert torch.cuda.is_available(), "a CUDA build's device tests need PyTorch with CUDA"
    metatensor.register_external_data_wrapper("pet_kokkos::KokkosArray", TorchCudaArray)
    return torch


class TorchCudaArray:
    """The plugin's device arrays as torch tensors, sharing their memory.

    metatensor's ExternalCudaArray does the same through CuPy, or else hands
    torch an object that refuses torch's stream argument; here torch gets the
    DLPack capsule itself. The plugin's export has already synchronized.
    """

    def __new__(cls, mts_array, parent):
        import ctypes

        import torch
        from ctypes_dlpack import DLDevice, DLManagedTensorVersioned, DLPackArray, DLPackVersion
        from metatensor._status import check_status

        pointer = ctypes.POINTER(DLManagedTensorVersioned)()
        status = mts_array.as_dlpack(
            mts_array.ptr, ctypes.byref(pointer), DLDevice(device_type=2, device_id=0), None,
            DLPackVersion(major=1, minor=0),
        )
        check_status(status)
        capsule = DLPackArray(pointer).__dlpack__(stream=None, max_version=(1, 0))
        tensor = torch.from_dlpack(capsule)
        tensor._parent = parent
        return tensor


def torch_range(torch, name, n, device):
    return Labels([name], torch.arange(n, dtype=torch.int32, device=device).reshape(-1, 1))


def evaluate_device(name, systems, outputs, selected_atoms=None):
    return evaluate(name, systems, outputs, selected_atoms=selected_atoms, check=False)


def device_system(torch, name, atoms, order=None, charge=None, spin=None):
    """`atoms` as a System in CUDA memory, its pairs too (metatensor wants a
    block's labels where its values are); `order` reorders the pairs."""
    m = model(name)
    options = m.requested_pair_lists()[0]
    cuda = torch.device("cuda")
    system = mta.System(
        "Angstrom",
        torch.tensor(atoms.numbers, dtype=torch.int32, device=cuda),
        torch.tensor(atoms.positions, dtype=torch.float64, device=cuda),
        torch.tensor(atoms.cell[:], dtype=torch.float64, device=cuda),
        torch.tensor(atoms.pbc, dtype=torch.bool, device=cuda),
    )
    i, j, S, D = ase.neighborlist.neighbor_list("ijSD", atoms, options.cutoff)
    if order is not None:
        i, j, S, D = i[order(len(i))], j[order(len(i))], S[order(len(i))], D[order(len(i))]
    samples = torch.tensor(np.column_stack([i, j, S]), dtype=torch.int32, device=cuda)
    pairs = TensorBlock(
        values=torch.tensor(D.reshape(-1, 3, 1), dtype=torch.float64, device=cuda),
        samples=Labels(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            samples,
        ),
        components=[torch_range(torch, "xyz", 3, samples.device)],
        properties=torch_range(torch, "distance", 1, samples.device),
    )
    system.add_pairs(options, pairs)
    if m.requested_inputs():
        for key, value in (("charge", charge or 0), ("spin_multiplicity", spin or 1)):
            block = TensorBlock(
                values=torch.tensor([[float(value)]], dtype=torch.float64, device=cuda),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32, device=cuda)),
                components=[],
                properties=torch_range(torch, key, 1, cuda),
            )
            system.add_custom_data(
                key, TensorMap(Labels(["_"], torch.tensor([[0]], dtype=torch.int32, device=cuda)), [block])
            )
    return system


def on_gpu(array):
    return getattr(array, "is_cuda", False)


def as_numpy(array):
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


def same(device_map, host_map, gradients=("positions", "strain"), atol=1e-10):
    d, h = device_map.block(), host_map.block()
    assert on_gpu(d.values) and on_gpu(d.samples.values)
    np.testing.assert_array_equal(as_numpy(d.samples.values), as_numpy(h.samples.values))
    np.testing.assert_allclose(as_numpy(d.values), as_numpy(h.values), atol=atol, rtol=0)
    for g in gradients:
        dg, hg = d.gradient(g), h.gradient(g)
        assert on_gpu(dg.values)
        np.testing.assert_array_equal(as_numpy(dg.samples.values), as_numpy(hg.samples.values))
        np.testing.assert_allclose(as_numpy(dg.values), as_numpy(hg.values), atol=atol, rtol=0)


@needs_cuda
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("case", CASES)
def test_device_matches_host(torch, name, case):
    atoms = golden_atoms(golden(name, case))
    host = evaluate(name, [system_for(name, atoms)], [ENERGY, ATOM_ENERGY])
    device = evaluate_device(
        name, [device_system(torch, name, atoms)], [ENERGY, ATOM_ENERGY]
    )
    same(device[0], host[0])
    same(device[1], host[1], gradients=())


@needs_cuda
def test_device_pairs_in_any_order(torch):
    """The pairs need not come grouped by atom; scattered ones give the same
    answer, and the same one every time."""
    name = "pet-mad-xs"
    atoms = golden_atoms(golden(name, "disordered"))
    grouped = evaluate_device(name, [device_system(torch, name, atoms)], [ENERGY])[0]
    rng = np.random.default_rng(3)
    permutation = rng.permutation(10**6)
    shuffle = lambda n: permutation[permutation < n]  # noqa: E731
    first = evaluate_device(name, [device_system(torch, name, atoms, order=shuffle)], [ENERGY])[0]
    again = evaluate_device(name, [device_system(torch, name, atoms, order=shuffle)], [ENERGY])[0]
    same(first, again, atol=0)
    np.testing.assert_allclose(
        as_numpy(first.block().values), as_numpy(grouped.block().values), atol=1e-6
    )
    np.testing.assert_allclose(
        as_numpy(first.block().gradient("positions").values),
        as_numpy(grouped.block().gradient("positions").values),
        atol=1e-5,
    )


@needs_cuda
def test_device_selected_atoms(torch):
    name = "pet-mad-xs"
    molecule = golden_atoms(golden(name, "molecule"))
    crystal = golden_atoms(golden(name, "disordered"))
    rng = np.random.default_rng(11)
    chosen = np.sort(rng.choice(len(crystal), size=len(crystal) // 2, replace=False))[::-1]
    selection = Labels(
        ["system", "atom"], np.column_stack([np.ones_like(chosen), chosen]).astype(np.int32)
    )
    host = evaluate(
        name, [system_for(name, molecule), system_for(name, crystal)], [ENERGY, ATOM_ENERGY],
        selected_atoms=selection,
    )
    device = evaluate_device(
        name,
        [device_system(torch, name, molecule), device_system(torch, name, crystal)],
        [ENERGY, ATOM_ENERGY],
        selected_atoms=selection,
    )
    same(device[0], host[0])
    same(device[1], host[1], gradients=())


@needs_cuda
@pytest.mark.xfail(
    strict=True,
    raises=mta.MetatomicError,
    reason="metatomic wants custom data on the system's device, but validates it "
    "with labels it makes on the CPU, so device systems can not carry a charge yet",
)
def test_device_charge_and_spin(torch):
    name = "pet-omol-s"
    water = ase.Atoms(
        "OH2", positions=[[0, 0, 0.11926], [0, 0.76323, -0.47704], [0, -0.76323, -0.47704]]
    )
    for charge, spin in ((0, 1), (1, 2)):
        host = evaluate(name, [system_for(name, water, charge=charge, spin=spin)], [ENERGY])
        device = evaluate_device(
            name, [device_system(torch, name, water, charge=charge, spin=spin)], [ENERGY]
        )
        same(device[0], host[0])


@needs_cuda
def test_device_unit_conversion(torch):
    """metatomic scales the outputs in place, on the device."""
    name = "pet-mad-xs"
    atoms = golden_atoms(golden(name, "crystal"))
    ev = evaluate_device(name, [device_system(torch, name, atoms)], [ENERGY_NO_GRADIENTS])[0]
    kj = mta.Quantity(name="energy", unit="kJ/mol", sample_kind="system")
    (energy,) = evaluate_device(name, [device_system(torch, name, atoms)], [kj])
    assert on_gpu(energy.block().values)
    assert as_numpy(energy.block().values)[0, 0] == pytest.approx(
        as_numpy(ev.block().values)[0, 0] * 96.48533212, rel=1e-6
    )


@needs_cuda
def test_device_consistency_checks(torch):
    """metatomic's checks cannot yet look at device outputs. When this starts
    failing, they can: run the device tests with them."""
    atoms = golden_atoms(golden("pet-mad-xs", "crystal"))
    with pytest.raises(mta.MetatomicError, match="on different devices"):
        evaluate("pet-mad-xs", [device_system(torch, "pet-mad-xs", atoms)], [ENERGY])
