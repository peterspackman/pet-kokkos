"""Which compiled core this process uses.

The CPU core ships in pet_kokkos itself. A GPU package -- pet-kokkos-cuda12,
pet-kokkos-rocm6, installed as the extras pet-kokkos[cuda12], ... -- carries
its own, and is used when installed. PET_KOKKOS_DEVICE=cpu, cuda or rocm
chooses explicitly. A GPU core that is installed but cannot load (no driver,
say) falls back to the CPU one with a warning, unless it was asked for.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings

GPU_PACKAGES = ["pet_kokkos_cuda12", "pet_kokkos_rocm6"]


def _load():
    want = os.environ.get("PET_KOKKOS_DEVICE", "").strip().lower()
    if want == "cpu":
        return importlib.import_module("pet_kokkos._core")
    for name in GPU_PACKAGES:
        if want and want not in name:
            continue
        if importlib.util.find_spec(name) is None:
            continue
        try:
            return importlib.import_module(f"{name}._core")
        except ImportError as e:
            if want:
                raise
            warnings.warn(f"pet_kokkos: {name} is installed but did not load ({e}); using the CPU core", RuntimeWarning)
    if want:
        raise ImportError(f"pet_kokkos: PET_KOKKOS_DEVICE={want}, but no such GPU package is installed")
    return importlib.import_module("pet_kokkos._core")


core = _load()
