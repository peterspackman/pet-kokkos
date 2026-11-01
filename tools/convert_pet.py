#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "metatrain",
#     "upet",
#     "metatomic-torch",
#     "huggingface_hub",
#     "safetensors",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Convert a upet PET model into a libtorch-free checkpoint for pet-kokkos.

Emits two files:
  <out>.json         hyperparameters, species map, composition energies, energy scale
  <out>.safetensors  all float weight tensors (PyTorch state_dict names, row-major)

The C++/Kokkos reader (pet-kokkos) consumes these directly; no libtorch needed
at evaluation time.

Two model sources are supported:

  * a named upet model downloaded from HuggingFace (lab-cosmo/upet), the latest
    version or a given one:
      uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs
      uv run tools/convert_pet.py --model pet-mols-s --version 1.1.0 --out models/pet-mols-s-v1.1

  * a local metatrain checkpoint (.ckpt), optionally selecting one of several
    energy *variants* trained on the same backbone (e.g. a model with both an
    `energy` (PBE) and an `energy/pbe0` head):
      uv run tools/convert_pet.py --ckpt my-model.ckpt --variant pbe0 --out models/my-model

`--variant pbe0` selects the `energy/pbe0` output (the same thing
`MetatomicCalculator(model, variants={"energy": "pbe0"})` does at eval time) and
renames its head weights to the canonical `energy` names the C++ loader expects,
so a single conservative-energy path is emitted regardless of which variant was
chosen.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file

# Weight-name prefixes we keep (the backbone). The selected energy head and the
# non-conservative force/stress heads are kept by HEAD_ROOTS below; the scaler
# and additive_models are read separately, everything else is dropped.
KEEP_PREFIXES = (
    "gnn_layers.",
    "combination_norms.",
    "combination_mlps.",
    "node_embedders.",
    "edge_embedder.",
    "system_conditioning.",
)
# Head sub-trees: kept only for the selected targets.
HEAD_ROOTS = ("node_heads", "edge_heads", "node_last_layers", "edge_last_layers")

# Adaptive-cutoff schemes the C++ evaluator implements; the loader in
# src/checkpoint.cpp refuses any other, and this refuses them earlier, with a
# message that says what to do.
SUPPORTED_ADAPTIVE_METHODS = {"grid", "solver"}


def head_target(name: str, targets) -> str | None:
    """The selected target whose head `name` belongs to, if any."""
    for root in HEAD_ROOTS:
        for t in targets:
            if name.startswith(root + "." + t + "."):
                return t
    return None


def keep_weight(name: str, targets) -> bool:
    return name.startswith(KEEP_PREFIXES) or head_target(name, targets) is not None


def canonical_name(name: str, renames: dict) -> str:
    """Rename a selected head's weights to the names the C++ loader spells.

    The loader hardcodes 'energy', 'nc_force' and 'nc_stress'. When a variant
    such as 'energy/pbe0' is selected, its state_dict keys look like
    `node_last_layers.energy/pbe0.0.energy/pbe0___0.weight`; rewriting the target
    substring to 'energy' yields `node_last_layers.energy.0.energy___0.weight`,
    which is exactly the layout the base 'energy' target would have produced.
    The non-conservative heads, `non_conservative_force` (or `..._forces`, as
    some models spell it) and `non_conservative_stress`, become `nc_force` and
    `nc_stress` the same way. Backbone weights carry no target name and pass
    through.
    """
    t = head_target(name, renames)
    return name if t is None else name.replace(t, renames[t])


def find_nc_targets(sd, variant):
    """The non-conservative heads that go with the selected energy, as
    {upstream name: canonical name}. A variant's come from that variant, as in
    upet's calculator."""
    suffix = f"/{variant}" if variant else ""
    found = {}
    for canonical, names in (("nc_force", ("non_conservative_force", "non_conservative_forces")),
                             ("nc_stress", ("non_conservative_stress",))):
        for n in names:
            if any(k.startswith(f"node_heads.{n}{suffix}.") for k in sd):
                found[n + suffix] = canonical
                break
    return found


def extract_scale_vector(model, target: str, n_species: int):
    """A target's scale: one number, or one per species index."""
    sc = model.scaler
    inner = getattr(sc, "model", sc)
    block = inner.scales[target].block(0)
    vals = [float(x) for x in block.values.reshape(-1)]
    if len(vals) == 1:
        return vals
    # Per atom type, rows labelled by species index 0..n-1 -- the order the
    # composition energies and pet-kokkos's species indices use.
    labels = [int(x) for x in block.samples.values[:, 0]]
    if len(vals) != n_species or labels != list(range(n_species)):
        raise SystemExit(f"{target}: unexpected per-type scale layout {block.samples.names} "
                         f"({len(vals)} rows for {n_species} species)")
    return vals


def load_raw_model(ckpt_path: str):
    """A local .ckpt as metatrain loads it: the PET module, or the wrapper
    around it (an LLPRUncertaintyModel, for a model with uncertainties)."""
    import metatomic.torch  # noqa: F401  (registers torchscript classes)
    from metatrain.utils.io import load_model

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return load_model(ckpt_path)


def unwrap(model):
    """The PET module itself, out of any wrapper (composition and scaler
    attached; it exposes .additive_models, .scaler, .atomic_types, plus the
    network itself -- see pet_backbone())."""
    if not (hasattr(model, "gnn_layers") or hasattr(model, "backend")):
        model = getattr(model, "model", model)
    return model


def load_local_model(ckpt_path: str):
    """Load a full metatrain PET model from a local .ckpt (composition+scaler)."""
    return unwrap(load_raw_model(ckpt_path))


def pet_backbone(model):
    """The module carrying the network weights and the species map.

    metatrain <= 2026.1 put `gnn_layers`, `edge_embedder`, ... and
    `species_to_species_index` directly on the PET module. 2026.4 moved them
    into a `PETBackend` submodule, which also puts a `backend.` prefix on every
    weight name in the state_dict. Both layouts are read here, and
    `strip_backend_prefix` below undoes the rename, so a model converted from
    either metatrain produces byte-identical output.
    """
    return getattr(model, "backend", model)


def strip_backend_prefix(name: str) -> str:
    """Undo metatrain 2026.4's `backend.` state_dict prefix.

    The names beneath it are unchanged, so removing it keeps the C++ loader,
    which spells these names literally, working with checkpoints from either.
    """
    return name[len("backend."):] if name.startswith("backend.") else name


def download_upet_checkpoint(model_name: str, version: str = "latest") -> str:
    """Resolve a named upet model, at `version` ("1.5.0", or "latest"), to a
    local checkpoint path.

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

        _, _, path = _resolve_and_download_checkpoint(family, size, version)
        return path
    except ImportError:
        pass

    from upet._models import upet_resolve_model

    resolved_size, latest = upet_resolve_model(model=family, requested_size=size)
    if version == "latest":
        version = latest
    return hf_hub_download(
        repo_id="lab-cosmo/upet",
        filename=f"{family}-{resolved_size}-v{version}.ckpt",
        subfolder="models",
    )


def load_named_model(model_name: str, version: str = "latest"):
    """Load a named upet model: download its checkpoint, then load it through
    metatrain so the raw PET backbone (with composition + scaler attached) is
    what comes back -- not the TorchScripted AtomisticModel `upet.get_upet`
    hands out, whose internals this converter cannot reach."""
    return load_local_model(download_upet_checkpoint(model_name, version))


def extract_composition(model, target: str, n_species: int):
    """Per-species-index composition (reference) energies, length n_species."""
    cm = model.additive_models[0]
    inner = getattr(cm, "model", cm)
    block = inner.weights[target].block(0)
    vals = block.values.reshape(block.values.shape[0], -1)[:, 0].double()
    assert vals.shape[0] == n_species, (vals.shape, n_species)
    return [float(x) for x in vals]


def extract_energy_scale(model, target: str) -> float:
    sc = model.scaler
    inner = getattr(sc, "model", sc)
    block = inner.scales[target].block(0)
    return float(block.values.reshape(-1)[0])


def extract_llpr(raw, target: str):
    """The LLPR energy uncertainty of an uncertainty-wrapped model, or None.

    metatrain's sigma is multiplier * |L^-1 f|, f the last-layer features and L
    the Cholesky factor of their covariance. L^-1 is taken here, in float64:
    for pet-mad-s the factor's condition number is ~4e8, and the C++ side then
    needs only a triangular matrix-vector product. The ensemble's linear layer
    (members x features, no bias) comes along when there is one.
    """
    if not hasattr(raw, "llpr_ensemble_layers"):
        return None
    name = "energy_uncertainty" if target == "energy" else f"mtt::aux::{target}_uncertainty"
    if name not in raw.supported_outputs():
        return None
    L = raw._get_cholesky(name).detach().double()
    multiplier = float(raw._get_multiplier(name).reshape(-1)[0])
    if not math.isfinite(multiplier):
        return None  # never calibrated
    inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], dtype=torch.float64), upper=False)
    meta = {"multiplier": multiplier, "features": int(L.shape[0])}
    tensors = {"llpr.energy.inv_cholesky": inv.contiguous()}
    layer = raw.llpr_ensemble_layers[target] if target in raw.llpr_ensemble_layers else None
    if layer is not None:
        if layer.bias is not None:
            raise SystemExit("LLPR ensemble layer with a bias: not implemented")
        tensors["llpr.energy.ensemble"] = layer.weight.detach().double().contiguous()
        meta["ensemble"] = int(layer.weight.shape[0])
    return {"meta": {"energy": meta}, "tensors": tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="named upet model (downloaded from HuggingFace)")
    ap.add_argument("--version", default="latest",
                    help="with --model: which published version, e.g. 1.5.0")
    ap.add_argument("--ckpt", default=None,
                    help="path to a local metatrain .ckpt")
    ap.add_argument("--variant", default=None,
                    help="energy variant to select, e.g. 'pbe0' -> 'energy/pbe0'")
    ap.add_argument("--out", required=True, help="output path stem (without extension)")
    args = ap.parse_args()

    if (args.model is None) == (args.ckpt is None):
        ap.error("exactly one of --model / --ckpt is required")

    target = "energy" if not args.variant else f"energy/{args.variant}"
    source = args.ckpt or args.model

    raw = load_raw_model(args.ckpt or download_upet_checkpoint(args.model, args.version))
    model = unwrap(raw)
    model.eval()

    h = dict(model.hypers)
    d_pet = int(h["d_pet"])
    num_heads = int(h["num_heads"])
    assert d_pet % num_heads == 0, "d_pet must be divisible by num_heads"

    featurizer = str(h["featurizer_type"])
    num_gnn = int(h["num_gnn_layers"])
    num_readout = 1 if featurizer == "feedforward" else num_gnn

    atomic_types = [int(z) for z in model.atomic_types]
    n_species = len(atomic_types)

    # species_to_index: indexed by atomic number Z (0..max); -1 = unsupported.
    species_to_index = [int(x) for x in pet_backbone(model).species_to_species_index]

    # sanity: confirm the requested target exists in the state_dict heads.
    sd = {strip_backend_prefix(k): v for k, v in model.state_dict().items()}
    if not any(k.startswith("node_heads." + target + ".") for k in sd):
        avail = sorted({
            k.split(".")[1] for k in sd if k.startswith("node_heads.")
        })
        ap.error(f"target '{target}' not found; available energy targets: {avail}")

    nna = h.get("num_neighbors_adaptive", None)

    # Refuse what the C++ evaluator does not implement rather than convert a
    # model that would load and return wrong energies: the adaptive-cutoff
    # schemes choose different per-atom cutoffs.
    adaptive_method = str(h.get("adaptive_cutoff_method", "grid"))
    if nna is not None and adaptive_method not in SUPPORTED_ADAPTIVE_METHODS:
        ap.error(
            f"this model uses adaptive_cutoff_method='{adaptive_method}', which "
            f"pet-kokkos does not implement (it has: "
            f"{', '.join(sorted(SUPPORTED_ADAPTIVE_METHODS))}).\n"
            "Converting it anyway would produce a model that loads and returns "
            "silently wrong energies and forces."
        )

    meta = {
        "format_version": 1,
        "architecture": "pet",
        "source_model": source,
        "target": target,
        "hypers": {
            "d_pet": d_pet,
            "d_head": int(h["d_head"]),
            "d_node": int(h["d_node"]),
            "d_feedforward": int(h["d_feedforward"]),
            "num_heads": num_heads,
            "head_dim": d_pet // num_heads,
            "num_attention_layers": int(h["num_attention_layers"]),
            "num_gnn_layers": num_gnn,
            "num_readout_layers": num_readout,
            "normalization": str(h["normalization"]),
            "activation": str(h["activation"]),
            "transformer_type": str(h["transformer_type"]),
            "featurizer_type": featurizer,
            "attention_temperature": float(h.get("attention_temperature", 1.0)),
            "cutoff": float(h["cutoff"]),
            "cutoff_width": float(h["cutoff_width"]),
            "cutoff_function": str(h["cutoff_function"]),
            "num_neighbors_adaptive": (None if nna is None else float(nna)),
            "adaptive_cutoff_method": adaptive_method,
            # Absent upstream (metatrain before the split) -> mirror cutoff_width,
            # which is what that metatrain used for both tapers.
            "cutoff_width_adaptive": float(
                h.get("cutoff_width_adaptive", h["cutoff_width"])
            ),
            "system_conditioning": bool(h.get("system_conditioning", False)),
            "max_charge": int(h.get("max_charge", 10)),
            "max_spin_multiplicity": int(h.get("max_spin_multiplicity", 10)),
            "zbl": bool(h.get("zbl", False)),
            "long_range_enabled": bool(h.get("long_range", {}).get("enable", False)),
        },
        "length_unit": (getattr(model, "dataset_info", None)
                        and model.dataset_info.length_unit) or "angstrom",
        "energy_unit": "eV",
        "energy_scale": extract_energy_scale(model, target),
        "n_species": n_species,
        "atomic_types": atomic_types,
        "species_to_index": species_to_index,
        "composition_energies": extract_composition(model, target, n_species),
    }
    nc = find_nc_targets(sd, args.variant)
    if nc:
        meta["non_conservative"] = {
            canonical[3:]: {"source": name, "scale": extract_scale_vector(model, name, n_species)}
            for name, canonical in nc.items()
        }
    renames = {target: "energy", **nc}
    llpr = extract_llpr(raw, target)
    if llpr:
        meta["uncertainty"] = llpr["meta"]

    # Collect kept float weights (contiguous, fp32), renaming variant heads.
    tensors = {}
    dropped = []
    for name, t in sd.items():
        if not torch.is_floating_point(t):
            dropped.append(name)
            continue
        if not keep_weight(name, renames):
            dropped.append(name)
            continue
        tensors[canonical_name(name, renames)] = t.detach().to(torch.float32).contiguous()

    if llpr:
        tensors.update(llpr["tensors"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Appended, not with_suffix: a stem like pet-mols-s-v1.0 has a dot of its own.
    json_path = out.parent / (out.name + ".json")
    st_path = out.parent / (out.name + ".safetensors")

    json_path.write_text(json.dumps(meta, indent=2))
    save_file(tensors, str(st_path))

    print(f"wrote {json_path}")
    print(f"wrote {st_path}  ({len(tensors)} tensors)")
    print(f"  source={source}  target={target}")
    print(f"  d_pet={d_pet} d_node={h['d_node']} heads={num_heads} "
          f"gnn_layers={num_gnn} attn_layers={h['num_attention_layers']} "
          f"readout_layers={num_readout}")
    print(f"  featurizer={featurizer} norm={h['normalization']} "
          f"act={h['activation']} type={h['transformer_type']}"
          + ("  system_conditioning=on" if h.get("system_conditioning") else ""))
    print(f"  cutoff={h['cutoff']} fn={h['cutoff_function']} width={h['cutoff_width']} "
          f"adaptive={nna} ({adaptive_method})  energy_scale={meta['energy_scale']:.6f}")
    if nc:
        print(f"  non-conservative heads: {', '.join(f'{n} -> {c}' for n, c in nc.items())}")
    if llpr:
        u = llpr["meta"]["energy"]
        print(f"  LLPR energy uncertainty: {u['features']} features, multiplier {u['multiplier']:.6g}"
              + (f", {u['ensemble']}-member ensemble" if u.get("ensemble") else ""))
    print(f"  n_species={n_species}  dropped {len(dropped)} non-kept tensors")


if __name__ == "__main__":
    main()
