#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "metatrain",
#     "upet",
#     "metatomic-torch",
#     "ase",
#     "huggingface_hub",
# ]
# ///
"""Generate a "golden" reference output for a metatrain PET model.

Produces a reproducible JSON file with total energy, per-atom energies, forces
and (for periodic cells) stress for a few small deterministic structures, so the
C++/Kokkos PET forward pass can be validated against them.

Evaluation path
---------------
    metatrain.utils.io.load_model(ckpt / hf-name)  -> metatrain PET wrapper
    wrapper.export()                               -> metatomic AtomisticModel
    metatomic.torch.ase_calculator.MetatomicCalculator(model, variants=...)
        .compute_energy(atoms, compute_forces_and_stresses=True, per_atom=True)

The `--variant pbe0` flag maps to `variants={"energy": "pbe0"}`, selecting the
`energy/pbe0` head. The total energy INCLUDES the composition (per-species
reference) energies and the output scaler, matching the converted model JSON:
  E_total = energy_scale * sum_i(E_backbone_i) + sum_i(composition_energy[Z_i])
  F       = energy_scale * F_backbone
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np


def download_upet_checkpoint(model_name: str) -> str:
    """Resolve a named upet model to a local checkpoint path.

    `model_name` is the full name including the size suffix, e.g. "pet-mad-xs"
    or "pet-omat-s". Everything before the last "-" is the model family and the
    last segment is the size, which is how `lab-cosmo/upet` names its files.

    upet's own resolver is used where it is available, because it is what knows
    which versions exist and how "latest" maps onto one; the public
    `upet_resolve_model` is the fallback, since the underscore-prefixed helper
    is not API.
    """
    from huggingface_hub import hf_hub_download

    family, _, size = model_name.rpartition("-")
    if not family:
        raise SystemExit(f"model name '{model_name}' has no size suffix (e.g. pet-mad-xs)")

    try:
        from upet._models import _resolve_and_download_checkpoint

        _, _, path = _resolve_and_download_checkpoint(family, size, "latest")
        return path
    except ImportError:
        pass

    from upet._models import upet_resolve_model

    resolved_size, version = upet_resolve_model(model=family, requested_size=size)
    return hf_hub_download(
        repo_id="lab-cosmo/upet",
        filename=f"{family}-{resolved_size}-v{version}.ckpt",
        subfolder="models",
    )


def load_full_model(model_name: str = None, ckpt: str = None):
    """Load the full metatomic AtomisticModel (composition + scaler included)."""
    import metatomic.torch  # noqa: F401  (registers torchscript classes)
    from metatrain.utils.io import load_model

    if ckpt:
        path = ckpt
    else:
        path = download_upet_checkpoint(model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        wrapper = load_model(path)
        atomistic = wrapper.export() if hasattr(wrapper, "export") else wrapper
    return atomistic


def build_structures():
    """Deterministic test cases: name -> ase.Atoms (nothing random)."""
    import ase

    cases = {}

    # (a) Tiny non-periodic molecule: water (H2O). No-PBC / no-ghost path.
    cases["molecule"] = ase.Atoms(
        symbols="OHH",
        positions=[
            [0.00000, 0.00000, 0.11926],
            [0.00000, 0.76323, -0.47704],
            [0.00000, -0.76323, -0.47704],
        ],
        cell=[0.0, 0.0, 0.0],
        pbc=[False, False, False],
    )

    # (b) Small periodic crystal: bulk diamond carbon (8-atom conventional cell).
    a = 3.567
    scaled = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
    cases["crystal"] = ase.Atoms(
        symbols="C8", scaled_positions=scaled, cell=[a, a, a], pbc=True
    )

    # (c) Disordered/rattled bulk: deterministic sinusoidal displacement so
    # neighbor distances spread across the cutoff taper zone.
    rattled = ase.Atoms(
        symbols="C8", scaled_positions=scaled, cell=[a, a, a], pbc=True
    )
    pos = rattled.get_positions()
    for i in range(len(pos)):
        pos[i, 0] += 0.25 * np.sin(1.0 * i + 0.0)
        pos[i, 1] += 0.25 * np.sin(2.0 * i + 1.0)
        pos[i, 2] += 0.25 * np.sin(3.0 * i + 2.0)
    rattled.set_positions(pos)
    cases["disordered"] = rattled

    return cases


def evaluate(calculator, atoms):
    result = calculator.compute_energy(
        atoms, compute_forces_and_stresses=True, per_atom=True
    )

    def _to_list(x):
        return np.asarray(x, dtype=float).tolist()

    total_energy = float(np.asarray(result["energy"]).reshape(-1)[0])
    per_atom = result.get("energies")
    forces = result.get("forces")
    stress = result.get("stress")
    virial = result.get("virial")
    try:
        vol = float(abs(np.linalg.det(np.asarray(atoms.get_cell(), dtype=float))))
    except Exception:
        vol = None
    return {
        "result_keys": list(result.keys()),
        "total_energy": total_energy,
        "per_atom_energies": _to_list(per_atom) if per_atom is not None else None,
        "forces": _to_list(forces) if forces is not None else None,
        "stress": _to_list(stress) if stress is not None else None,
        "virial": _to_list(virial) if virial is not None else None,
        "volume": vol,
    }


def make_golden(model_label, calculator, case_name, atoms, out_dir, variant):
    ev = evaluate(calculator, atoms)
    pbc = [bool(p) for p in atoms.get_pbc()]

    golden = {
        "model": model_label,
        "variant": variant,
        "case": case_name,
        "description": (
            f"Golden reference for metatrain PET model '{model_label}'"
            f"{f' (variant {variant})' if variant else ''}, case '{case_name}'. "
            f"Energy in eV, positions in Angstrom, forces in eV/Angstrom. Total "
            f"energy INCLUDES composition energies and the output scaler."
        ),
        "evaluation_path": (
            "metatrain.utils.io.load_model(ckpt).export() -> metatomic "
            "AtomisticModel; MetatomicCalculator(model, variants={'energy': "
            f"'{variant}'}}).compute_energy(atoms, "
            "compute_forces_and_stresses=True, per_atom=True)"
        ),
        "units": {"energy": "eV", "length": "Angstrom", "forces": "eV/Angstrom"},
        "atomic_numbers": [int(z) for z in atoms.get_atomic_numbers()],
        "positions": np.asarray(atoms.get_positions(), dtype=float).tolist(),
        "cell": np.asarray(atoms.get_cell(), dtype=float).tolist(),
        "pbc": pbc,
        "periodic": any(pbc),
        "n_atoms": len(atoms),
        "total_energy": ev["total_energy"],
        "per_atom_energies": ev["per_atom_energies"],
        "forces": ev["forces"],
        "stress": ev["stress"],
        "virial": ev["virial"],
        "volume": ev["volume"],
        "result_keys": ev["result_keys"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_label}_{case_name}.json"
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2)

    max_force = (
        float(np.max(np.abs(np.asarray(ev["forces"]))))
        if ev["forces"] is not None else float("nan")
    )
    print(
        f"  [{case_name}] n_atoms={len(atoms)} E={ev['total_energy']:.6f} eV "
        f"max|F|={max_force:.6f} eV/Ang -> {out_path}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden PET reference output for C++ validation."
    )
    parser.add_argument("--model", default=None, help="named upet model, e.g. pet-mad-xs")
    parser.add_argument("--ckpt", default=None, help="local metatrain .ckpt path")
    parser.add_argument("--variant", default=None,
                        help="energy variant, e.g. 'pbe0' -> variants={'energy':'pbe0'}")
    parser.add_argument("--label", default=None,
                        help="output filename stem (default derived from model/ckpt)")
    parser.add_argument("--out-dir", default=None,
                        help="output dir (default: the tests/golden next to this script)")
    parser.add_argument("--case", default="all",
                        choices=["all", "molecule", "crystal", "disordered"])
    args = parser.parse_args()

    if (args.model is None) == (args.ckpt is None):
        parser.error("exactly one of --model / --ckpt is required")

    label = args.label or (
        Path(args.ckpt).stem if args.ckpt else args.model
    )
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(__file__).resolve().parents[1] / "tests" / "golden")

    print(f"Loading full atomistic model: {label} ...")
    atomistic = load_full_model(model_name=args.model, ckpt=args.ckpt)

    from metatomic.torch.ase_calculator import MetatomicCalculator

    variants = {"energy": args.variant} if args.variant else {}
    calculator = MetatomicCalculator(atomistic, variants=variants, device="cpu")

    structures = build_structures()
    if args.case != "all":
        structures = {args.case: structures[args.case]}

    print(f"Evaluating golden structures (variant={args.variant}):")
    written = [
        make_golden(label, calculator, name, atoms, out_dir, args.variant)
        for name, atoms in structures.items()
    ]

    print("\nDone. Golden files written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
