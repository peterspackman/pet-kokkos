"""scikit-build-core metadata for pet-kokkos's wheel variants.

PET_KOKKOS_VARIANT (unset for the CPU wheel) picks a GPU build: its wheel is
still pet-kokkos, versioned with PEP 440's local label (0.1.0+cu12), as
PyTorch's are, and depends on the libraries that build loads. GPU wheels are
not for PyPI, which takes no local versions: they are served from the GitHub
release (tools/wheel_index.py). The CMake side of each variant is
pyproject.toml's overrides.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# variant -> what the wheel needs installed beside it. The CUDA runtime and
# cuBLAS are NVIDIA's pip wheels (PyTorch's and JAX's), found through the
# module's RPATH; the lower bounds are what PyTorch's CUDA 12.4+ and 13.0
# wheels pin, so the two install together. ROCm is the system's.
VARIANTS = {
    "": [],
    "cu12": ["nvidia-cuda-runtime-cu12>=12.4", "nvidia-cublas-cu12>=12.4"],
    "cu13": ["nvidia-cuda-runtime>=13.0,<14", "nvidia-cublas>=13.0,<14"],
    "rocm6": [],
    "rocm7": [],
}


def variant() -> str:
    v = os.environ.get("PET_KOKKOS_VARIANT", "").strip()
    if v not in VARIANTS:
        raise ValueError(f"PET_KOKKOS_VARIANT={v!r}: not one of {sorted(VARIANTS)}")
    return v


def dynamic_metadata(field: str, settings: dict | None = None):
    if field == "version":
        text = Path("CMakeLists.txt").read_text()
        version = re.search(r"project\(\s*pet_kokkos\s+VERSION\s+([0-9.]+)", text).group(1)
        return f"{version}+{variant()}" if variant() else version
    if field == "dependencies":
        return ["numpy>=1.24", *VARIANTS[variant()]]
    raise ValueError(f"pet_variant: no metadata for {field}")
