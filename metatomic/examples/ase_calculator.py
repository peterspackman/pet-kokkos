"""An ASE calculator over metatomic's C API, for any metatomic plugin model.

metatomic's own ASE calculator (metatomic_ase) takes TorchScript models only,
for now. This one loads a plugin model with metatomic.load_model and calls
metatomic.execute_model, with neighbour lists from vesin:

    import metatomic
    metatomic.load_plugin("build-cuda/metatomic/pet-kokkos-metatomic.so")
    atoms.calc = PluginCalculator("pet-mad-xs", plugin="pet-kokkos", device="cuda")

With device="cuda" the system and its pairs are handed over as CUDA tensors and
the results come back as CUDA tensors (pet-kokkos reads and writes them in
place); only the energy, forces and stress ASE asks for are copied to the host.
Needs metatensor, metatomic (core), vesin, numpy, and torch for "cuda".
"""

import ctypes

import metatensor
import numpy as np
import vesin
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from metatensor import Labels, TensorBlock, TensorMap

import metatomic as mta


class _TorchCudaArray:
    """A metatensor array from a plugin, in CUDA memory, as a torch tensor that
    shares it (through the DLPack capsule, which torch takes as is)."""

    def __new__(cls, mts_array, parent):
        import torch
        from ctypes_dlpack import DLDevice, DLManagedTensorVersioned, DLPackArray, DLPackVersion
        from metatensor._status import check_status

        pointer = ctypes.POINTER(DLManagedTensorVersioned)()
        status = mts_array.as_dlpack(
            mts_array.ptr,
            ctypes.byref(pointer),
            DLDevice(device_type=2, device_id=torch.cuda.current_device()),
            None,
            DLPackVersion(major=1, minor=0),
        )
        check_status(status)
        tensor = torch.from_dlpack(DLPackArray(pointer).__dlpack__(stream=None, max_version=(1, 0)))
        tensor._parent = parent
        return tensor


_registered = False


def _register_wrappers():
    global _registered
    if not _registered:
        metatensor.register_external_data_wrapper("metatensor::SimpleDataArray", metatensor.ExternalCpuArray)
        metatensor.register_external_data_wrapper("pet_kokkos::KokkosArray", _TorchCudaArray)
        _registered = True


class PluginCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(self, model, plugin=None, options=None, device="cpu", charge=0, spin_multiplicity=1, **kwargs):
        super().__init__(**kwargs)
        _register_wrappers()
        self.model = mta.load_model(model, options or {}, plugin_name=plugin)
        capabilities = self.model.capabilities()
        if device not in capabilities.supported_devices:
            raise ValueError(f"the model runs on {capabilities.supported_devices}, not {device}")
        self.device = device
        (self.pairs,) = self.model.requested_pair_lists()
        self.inputs = {q.name for q in self.model.requested_inputs()}
        self.charge, self.spin = charge, spin_multiplicity
        self.neighbors = vesin.NeighborList(cutoff=self.pairs.cutoff, full_list=self.pairs.full_list)
        if device == "cuda":
            import torch

            self._torch = torch
            self._cuda = torch.device("cuda")

    def _array(self, values, dtype):
        if self.device == "cpu":
            return np.ascontiguousarray(values, dtype=dtype)
        torch_dtype = {np.int32: self._torch.int32, np.float64: self._torch.float64, bool: self._torch.bool}[dtype]
        return self._torch.as_tensor(np.asarray(values), dtype=torch_dtype, device=self._cuda)

    def _labels(self, names, values):
        return Labels(names, self._array(np.asarray(values, dtype=np.int32).reshape(-1, len(names)), np.int32))

    def _system(self, atoms):
        cell = atoms.cell[:] if atoms.pbc.any() else np.zeros((3, 3))
        system = mta.System(
            "Angstrom",
            self._array(atoms.numbers, np.int32),
            self._array(atoms.positions, np.float64),
            self._array(cell, np.float64),
            self._array(atoms.pbc, bool),
        )
        i, j, S, D = self.neighbors.compute(atoms.positions, cell, atoms.pbc, "ijSD")
        pairs = TensorBlock(
            values=self._array(D.reshape(-1, 3, 1), np.float64),
            samples=self._labels(
                ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                np.column_stack([i, j, S]),
            ),
            components=[self._labels(["xyz"], [0, 1, 2])],
            properties=self._labels(["distance"], [0]),
        )
        system.add_pairs(self.pairs, pairs)
        for name, value in (("charge", self.charge), ("spin_multiplicity", self.spin)):
            if name in self.inputs:
                # On the host: metatomic cannot validate custom data on a device yet.
                block = TensorBlock(
                    values=np.array([[float(value)]]),
                    samples=Labels(["system"], np.array([[0]], dtype=np.int32)),
                    components=[],
                    properties=Labels.range(name, 1),
                )
                system.add_custom_data(name, TensorMap(Labels(["_"], np.array([[0]])), [block]))
        return system

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        periodic = atoms.pbc.all()
        gradients = ["positions"] + (["strain"] if periodic else [])
        outputs = [mta.Quantity(name="energy", unit="eV", sample_kind="system", gradients=gradients)]
        if "energies" in properties:
            outputs.append(mta.Quantity(name="energy", unit="eV", sample_kind="atom"))
        # metatomic's consistency checks can not read device outputs yet.
        results = mta.execute_model(
            self.model, [self._system(atoms)], outputs, check_consistency=self.device == "cpu"
        )

        def host(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

        block = results[0].block()
        self.results["energy"] = float(host(block.values)[0, 0])
        self.results["forces"] = -host(block.gradient("positions").values)[:, :, 0]
        if periodic:
            virial = host(block.gradient("strain").values)[0, :, :, 0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / atoms.get_volume())
        if "energies" in properties:
            self.results["energies"] = host(results[1].block().values)[:, 0]
