#!/usr/bin/env python3
"""Re-emit a PET golden with the reference evaluated in float64.

The shipped goldens are float32 end to end (every stored value is exactly
representable as a float32), so for the pbe0 model -- whose per-atom energies are
~-1036 eV because the composition energies are full atomic DFT totals -- one fp32
ulp is 1.2e-4 eV per atom, ~1e-3 eV on the total. That is the floor the C++ is
being measured against, and it is ~100x coarser than for pet-mad-xs purely because
pet-mad-xs energies are ~100x smaller.

Output uses the same schema as tools/make_golden.py so tests/pet_smoke.cpp can
consume it unchanged.
"""
import argparse, json, warnings
import numpy as np
import torch
import ase

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True, help="existing golden, for geometry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    args = ap.parse_args()

    from metatrain.utils.io import load_model
    from metatomic.torch.ase_calculator import MetatomicCalculator

    g = json.load(open(args.golden))
    atoms = ase.Atoms(numbers=g["atomic_numbers"], positions=g["positions"],
                      cell=g["cell"], pbc=g["pbc"])

    pet = load_model(args.ckpt).to(getattr(torch, args.dtype))
    kw = {"variants": {"energy": args.variant}} if args.variant else {}
    calc = MetatomicCalculator(pet.export(), device="cpu", **kw)
    res = calc.compute_energy(atoms, compute_forces_and_stresses=g["periodic"] or True,
                              per_atom=True)

    out = dict(g)  # keep geometry/metadata
    out["description"] = (f"float64 re-evaluation of {args.golden} "
                          f"(model dtype {args.dtype})")
    out["reference_dtype"] = args.dtype
    out["total_energy"] = float(np.asarray(res["energy"], dtype=np.float64).reshape(-1)[0])
    out["per_atom_energies"] = np.asarray(res["energies"], dtype=np.float64).reshape(-1).tolist()
    if "forces" in res:
        out["forces"] = np.asarray(res["forces"], dtype=np.float64).reshape(-1, 3).tolist()
    if g.get("periodic") and "stress" in res and res["stress"] is not None:
        out["stress"] = np.asarray(res["stress"], dtype=np.float64).reshape(3, 3).tolist()
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}  total_energy={out['total_energy']!r}")


if __name__ == "__main__":
    main()
