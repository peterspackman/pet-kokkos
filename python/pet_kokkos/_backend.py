"""The compiled core, with a useful error when a GPU build cannot load it.

Each wheel carries one core: the CPU one, or a GPU variant's (version
0.1.0+cu12, +rocm6, ...), which needs that GPU's driver and libraries.
"""

from __future__ import annotations

from importlib import metadata


def _variant() -> str:
    try:
        return metadata.version("pet-kokkos").partition("+")[2]
    except metadata.PackageNotFoundError:  # a build tree, not an installed wheel
        return ""


try:
    from . import _core as core
except ImportError as e:
    v = _variant()
    if not v:
        raise
    need = {
        "cu": "an NVIDIA driver (libcuda.so.1) and the nvidia-* CUDA wheels",
        "rocm": "ROCm installed, its libraries (libamdhip64, rocBLAS, hipBLASLt) on the library path",
    }["cu" if v.startswith("cu") else "rocm"]
    raise ImportError(
        f"pet_kokkos {metadata.version('pet-kokkos')} is a GPU build ({v}) and needs {need}: {e}. "
        "For the CPU build: uv pip install --reinstall pet-kokkos --index-url https://pypi.org/simple"
    ) from e
