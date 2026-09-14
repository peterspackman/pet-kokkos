#!/usr/bin/env python3
"""Dump per-atom network energies and per-layer node features for a PET model on a
small structure, as an order-independent ground truth for the C++/Kokkos forward.

    E_atom_total = energy_scale * E_atom_network + composition[Z]
=>  E_atom_network = (E_atom_total - composition[Z]) / energy_scale

Per-atom node features `node_features_list[L]` (shape [n_atoms, d_node]) are
captured via forward hooks on each CartesianTransformer; being atom-indexed they
line up with the C++ regardless of neighbor ordering.

Writes <out>.json with: atomic_numbers, per_atom_network, per_layer_node_features.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import metatomic.torch  # noqa: F401
import ase

warnings.filterwarnings("ignore")


def water():
    return ase.Atoms(
        symbols="OHH",
        positions=[[0, 0, 0.11926], [0, 0.76323, -0.47704], [0, -0.76323, -0.47704]],
        cell=[0, 0, 0], pbc=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="pbe0")
    ap.add_argument("--model-json", required=True, help="converted <model>.json (scale+comp)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from metatrain.utils.io import load_model
    from metatomic.torch.ase_calculator import MetatomicCalculator

    pet = load_model(args.ckpt)           # metatrain PET module
    atomistic = pet.export()

    # capture per-layer node outputs (output_node_embeddings) from each gnn layer
    captured = []
    handles = []
    for layer in pet.gnn_layers:
        def hook(mod, inp, out, store=captured):
            store.append(out[0].detach().cpu().numpy())  # [n_atoms, d_node]
        handles.append(layer.register_forward_hook(hook))

    calc = MetatomicCalculator(atomistic, variants={"energy": args.variant}, device="cpu")
    atoms = water()
    captured.clear()
    res = calc.compute_energy(atoms, compute_forces_and_stresses=False, per_atom=True)
    per_atom_total = np.asarray(res["energies"], dtype=float).reshape(-1)
    for h in handles:
        h.remove()

    meta = json.load(open(args.model_json))
    scale = meta["energy_scale"]
    comp = meta["composition_energies"]
    s2i = meta["species_to_index"]
    Z = atoms.get_atomic_numbers()
    per_atom_net = [(per_atom_total[i] - comp[s2i[int(Z[i])]]) / scale for i in range(len(Z))]

    out = {
        "atomic_numbers": [int(z) for z in Z],
        "positions": atoms.get_positions().tolist(),
        "energy_scale": scale,
        "per_atom_total": per_atom_total.tolist(),
        "per_atom_network": per_atom_net,
        "total_network": float(sum(per_atom_net)),
        "n_gnn_layers": len(captured),
        "per_layer_node_features": [c.tolist() for c in captured],
        "node_feature_dim": captured[0].shape[1] if captured else None,
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")
    print(f"  per_atom_total   = {per_atom_total.tolist()}")
    print(f"  per_atom_network = {per_atom_net}")
    print(f"  captured {len(captured)} gnn-layer node outputs, dim={out['node_feature_dim']}")


if __name__ == "__main__":
    main()
