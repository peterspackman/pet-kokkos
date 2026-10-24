"""Finding PET models, and fetching the published ones.

A model is two files, ``<name>.json`` and ``<name>.safetensors``. A name is
looked for where pet-kokkos looks (``model_search_dirs()``); a published one
that is not found is downloaded from the Hugging Face repository below into
``$XDG_DATA_HOME/pet/models`` (else ``~/.local/share/pet/models``), which is on
that search path.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from ._backend import core as _core

REPOSITORY = "peterspackman/pet-kokkos-models"
_URL = "https://huggingface.co/{repo}/resolve/main/{file}"


def data_dir() -> Path:
    """Where downloaded models go."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "pet" / "models"


def download(name: str, directory: str | os.PathLike | None = None) -> Path:
    """Download a published model into ``directory`` (default ``data_dir()``).

    Returns the path prefix (``<directory>/<name>``). Each file is written to a
    temporary name and moved into place, so an interrupted download leaves
    nothing half-written behind.
    """
    target = Path(directory) if directory is not None else data_dir()
    target.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".safetensors"):
        dest = target / f"{name}{suffix}"
        if dest.exists():
            continue
        url = _URL.format(repo=REPOSITORY, file=f"{name}{suffix}")
        with urllib.request.urlopen(url) as response, tempfile.NamedTemporaryFile(
            dir=target, prefix=f".{name}{suffix}.", delete=False
        ) as tmp:
            shutil.copyfileobj(response, tmp)
        os.chmod(tmp.name, 0o644)  # not the temporary file's 0600: a model directory may be shared
        os.replace(tmp.name, dest)
    return target / name


def resolve(spec: str, fetch: bool = True) -> str:
    """A model name or path prefix, as the C++ library will find it.

    A path prefix (with a separator, or naming existing files) is returned as
    is. A name not found locally is downloaded when ``fetch`` is true.
    """
    if os.sep in spec or "/" in spec or Path(f"{spec}.json").exists():
        return spec
    if _core.find_model(spec) is not None:
        return spec
    if not fetch:
        raise FileNotFoundError(
            f"PET model '{spec}' not found; looked in: " + ", ".join(_core.model_search_dirs())
        )
    return str(download(spec))
