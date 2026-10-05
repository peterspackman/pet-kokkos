#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "ase",
#     "metatrain",
#     "upet",
#     "metatomic-torch",
#     "huggingface_hub",
#     "safetensors",
# ]
# ///
# Wall-clock comparison with upet: one energy and forces evaluation of one
# structure on each side, neighbour list included, best of --repeat after two
# warmup calls (both sides allocate on the first).
#
#   uv run tools/bench_vs_upet.py --model pet-mad-s --xyz cell.xyz \
#       [--pet-eval build-cuda/apps/pet-eval] [--model-dir build-models]
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def bench_upet(ckpt_path, atoms, repeat, device):
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model
    import numpy as np
    import torch

    model = load_model(str(ckpt_path)).export()
    calc = MetatomicCalculator(model, device=device)

    # Each call gets its own tiny displacement: ASE's calculators cache on the
    # system and would otherwise return the cached answer without evaluating.
    rng = np.random.default_rng(0)
    base = atoms.get_positions()

    def once(k):
        a = atoms.copy()
        a.set_positions(base + 1e-6 * rng.standard_normal(base.shape))
        a.calc = calc
        e = float(a.get_potential_energy())
        f = a.get_forces()
        if device == "cuda":
            torch.cuda.synchronize()
        return e, f

    once(0)  # warmup: allocator, kernel selection, any lazy init
    once(1)
    best = float("inf")
    for k in range(repeat):
        t0 = time.perf_counter()
        e, f = once(k + 2)
        best = min(best, time.perf_counter() - t0)
    return best, e, f


def bench_ours(pet_eval, model_prefix, xyz, repeat, env=None):
    e = dict(os.environ)
    e.update(env or {})
    cmd = [str(pet_eval), "--forces", "--repeat", str(repeat), model_prefix, str(xyz)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=e).stdout
    best = None
    energy = None
    for line in out.splitlines():
        if line.startswith("best of"):
            best = float(line.split(":")[1].split("s")[0])
        if line.startswith("frame 0:"):
            energy = float(line.split("E =")[1].split("eV")[0])
    return best, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pet-mad-s")
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--pet-eval", default="build-cuda/apps/pet-eval")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-dir", default="build-models")
    a = ap.parse_args()

    from ase.io import read
    from convert_pet import download_upet_checkpoint

    atoms = read(a.xyz)
    ckpt = download_upet_checkpoint(a.model)

    t_up, e_up, f_up = bench_upet(ckpt, atoms, a.repeat, a.device)
    env = {"PET_MODEL_DIR": str(Path(a.model_dir).resolve())}
    t_us, e_us = bench_ours(a.pet_eval, a.model, a.xyz, a.repeat, env)

    n = len(atoms)
    print(json.dumps({
        "model": a.model, "n_atoms": n, "device": a.device,
        "upet_s": round(t_up, 4), "pet_kokkos_s": round(t_us, 4) if t_us else None,
        "speedup": round(t_up / t_us, 2) if t_us else None,
        "upet_energy": e_up, "pet_kokkos_energy": e_us,
        "d_energy_per_atom": abs(e_up - e_us) / n if (e_us is not None) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
