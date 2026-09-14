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
# ///
"""Convert a upet PET model into a libtorch-free checkpoint for pet-kokkos.

Emits two files:
  <out>.json         hyperparameters, species map, composition energies, energy scale
  <out>.safetensors  all float weight tensors (PyTorch state_dict names, row-major)

The C++/Kokkos reader (pet-kokkos) consumes these directly; no libtorch needed
at evaluation time.

Two model sources are supported:

  * a named upet model downloaded from HuggingFace (lab-cosmo/upet):
      uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs

  * a local metatrain checkpoint (.ckpt), optionally selecting one of several
    energy *variants* trained on the same backbone (e.g. a model with both an
    `energy` (PBE) and an `energy/pbe0` head):
      uv run tools/convert_pet.py --ckpt pbe0-pet.ckpt --variant pbe0 \
                                  --out models/pbe0-pet

`--variant pbe0` selects the `energy/pbe0` output (the same thing
`MetatomicCalculator(model, variants={"energy": "pbe0"})` does at eval time) and
renames its head weights to the canonical `energy` names the C++ loader expects,
so a single conservative-energy path is emitted regardless of which variant was
chosen.
"""
import argparse
import json
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file

# Weight-name prefixes we keep (conservative energy path + backbone). Everything
# under the non_conservative_* heads, the scaler, and additive_models is handled
# separately or dropped.
KEEP_PREFIXES = (
    "gnn_layers.",
    "combination_norms.",
    "combination_mlps.",
    "node_embedders.",
    "edge_embedder.",
)
# Head sub-trees: keep only the selected energy target.
HEAD_ROOTS = ("node_heads", "edge_heads", "node_last_layers", "edge_last_layers")

# Adaptive-cutoff schemes the C++ evaluator implements. Keep in step with
# parse_acm() in include/pet/checkpoint.hpp -- this is the Python half of the
# same gate, and it is the half that gives an actionable message.
SUPPORTED_ADAPTIVE_METHODS = {"grid"}


def keep_weight(name: str, target: str) -> bool:
    if name.startswith(KEEP_PREFIXES):
        return True
    for root in HEAD_ROOTS:
        if name.startswith(root + "." + target + "."):
            return True
    return False


def canonical_name(name: str, target: str) -> str:
    """Rename the selected target's head weights to the canonical 'energy' names.

    The C++ loader hardcodes the target name 'energy'. When a non-default variant
    such as 'energy/pbe0' is selected, its state_dict keys look like
    `node_last_layers.energy/pbe0.0.energy/pbe0___0.weight`; rewriting the target
    substring to 'energy' yields `node_last_layers.energy.0.energy___0.weight`,
    which is exactly the layout the base 'energy' target would have produced.
    Backbone weights never contain the target substring, so they pass through.
    """
    if target == "energy":
        return name
    return name.replace(target, "energy")


def load_local_model(ckpt_path: str):
    """Load a full metatrain PET model from a local .ckpt (composition+scaler)."""
    import metatomic.torch  # noqa: F401  (registers torchscript classes)
    from metatrain.utils.io import load_model

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = load_model(ckpt_path)
    # metatrain hands back the PET module directly (it exposes .additive_models,
    # .scaler, .atomic_types, plus the network itself -- see pet_backbone()).
    if not (hasattr(model, "gnn_layers") or hasattr(model, "backend")):
        model = getattr(model, "model", model)
    return model


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

    The names beneath it are unchanged, so removing it is the whole migration --
    and it keeps the C++ loader, which spells these names literally, working
    against checkpoints from either metatrain. (This is the fragility the v2
    model format's tensor manifest is meant to end; see PLAN.md section 3.)
    """
    return name[len("backend."):] if name.startswith("backend.") else name


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


def load_named_model(model_name: str):
    """Load a named upet model: download its checkpoint, then load it through
    metatrain so the raw PET backbone (with composition + scaler attached) is
    what comes back -- not the TorchScripted AtomisticModel `upet.get_upet`
    hands out, whose internals this converter cannot reach."""
    return load_local_model(download_upet_checkpoint(model_name))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="named upet model (downloaded from HuggingFace)")
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

    if args.ckpt:
        model = load_local_model(args.ckpt)
    else:
        model = load_named_model(args.model)
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

    # Refuse, here, anything the C++ evaluator does not implement -- a converted
    # model that loads and returns a wrong number is far worse than one that
    # refuses to convert. metatrain's adaptive_cutoff_method default changed from
    # "grid" to "solver", and the two choose different per-atom cutoffs, so this
    # is not a cosmetic difference: it is an energy.
    adaptive_method = str(h.get("adaptive_cutoff_method", "grid"))
    if nna is not None and adaptive_method not in SUPPORTED_ADAPTIVE_METHODS:
        ap.error(
            f"this model uses adaptive_cutoff_method='{adaptive_method}', which "
            f"pet-kokkos does not implement (it has: "
            f"{', '.join(sorted(SUPPORTED_ADAPTIVE_METHODS))}).\n"
            "Converting it anyway would produce a model that loads and returns "
            "silently wrong energies and forces. Retrain or re-export with "
            "adaptive_cutoff_method='grid', or wait for solver support "
            "(see PLAN.md section 5)."
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

    # Collect kept float weights (contiguous, fp32), renaming variant heads.
    tensors = {}
    dropped = []
    for name, t in sd.items():
        if not torch.is_floating_point(t):
            dropped.append(name)
            continue
        if not keep_weight(name, target):
            dropped.append(name)
            continue
        tensors[canonical_name(name, target)] = t.detach().to(torch.float32).contiguous()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    st_path = out.with_suffix(".safetensors")

    json_path.write_text(json.dumps(meta, indent=2))
    save_file(tensors, str(st_path))

    print(f"wrote {json_path}")
    print(f"wrote {st_path}  ({len(tensors)} tensors)")
    print(f"  source={source}  target={target}")
    print(f"  d_pet={d_pet} d_node={h['d_node']} heads={num_heads} "
          f"gnn_layers={num_gnn} attn_layers={h['num_attention_layers']} "
          f"readout_layers={num_readout}")
    print(f"  featurizer={featurizer} norm={h['normalization']} "
          f"act={h['activation']} type={h['transformer_type']}")
    print(f"  cutoff={h['cutoff']} fn={h['cutoff_function']} width={h['cutoff_width']} "
          f"adaptive={nna} ({adaptive_method})  energy_scale={meta['energy_scale']:.6f}")
    print(f"  n_species={n_species}  dropped {len(dropped)} non-kept tensors")


if __name__ == "__main__":
    main()
