#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "metatrain",
#     "upet",
#     "metatomic-torch",
#     "huggingface_hub",
# ]
# ///
"""Build a `num_attention_layers > 1` PET checkpoint from a single-layer one.

A small model with several attention layers, for the goldens: the published
models that have them are large. metatrain rebuilds the network from
`model_hypers`, so raising
`num_attention_layers` produces a genuine A-block Transformer, and copying block
0's weights into the new blocks gives it valid parameters. The result is a real
metatrain model: it loads, evaluates through metatrain's own code path, and can
therefore be used as a REFERENCE rather than a self-comparison.

Identical blocks still exercise the stack, since A identical blocks in sequence
are not one block, but not per-block weight indexing: `--distinct` perturbs each
copy so that reading block 0's weights for every block would fail.

    uv run tools/make_multilayer_checkpoint.py pet-mad-xs out.ckpt --layers 2
    uv run tools/convert_pet.py --ckpt out.ckpt --out models/pet-a2
    uv run tools/make_golden.py --ckpt out.ckpt --label pet-a2
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_pet import download_upet_checkpoint  # noqa: E402


def duplicate_blocks(state_dict, n_layers: int, jitter: float) -> int:
    """Copy every `trans.layers.0.*` tensor into blocks 1..n_layers-1."""
    added = {}
    for k, v in state_dict.items():
        if ".trans.layers.0." not in k:
            continue
        for a in range(1, n_layers):
            t = v.clone()
            if jitter and t.is_floating_point():
                # Deterministic per-block perturbation, so the blocks are not
                # interchangeable and an implementation that used block 0's
                # weights everywhere would produce a different answer.
                g = torch.Generator().manual_seed(hash((k, a)) & 0x7FFFFFFF)
                t += jitter * t.std() * torch.randn(t.shape, generator=g, dtype=t.dtype)
            added[k.replace(".trans.layers.0.", f".trans.layers.{a}.")] = t
    state_dict.update(added)
    return len(added)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="named upet model, or a path to a local .ckpt")
    ap.add_argument("out", help="output checkpoint path")
    ap.add_argument("--layers", type=int, default=2, help="num_attention_layers to produce")
    ap.add_argument("--distinct", type=float, default=0.0, metavar="FRAC",
                    help="perturb each duplicated block by this fraction of its own "
                         "standard deviation, so the blocks are not identical")
    args = ap.parse_args()

    if args.layers < 2:
        ap.error("--layers must be at least 2")

    src = args.model if Path(args.model).exists() else download_upet_checkpoint(args.model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ck = torch.load(src, map_location="cpu", weights_only=False)

    # upet checkpoints wrap the PET model in an ensemble container; the network's
    # own hypers and weights live one level down.
    inner = ck.get("wrapped_model_checkpoint", ck)
    hypers = inner["model_data"]["model_hypers"]
    have = int(hypers["num_attention_layers"])
    if have != 1:
        ap.error(f"expected a 1-attention-layer model, found {have}")
    if str(hypers.get("featurizer_type")) != "feedforward":
        ap.error(f"only the feedforward featurizer is handled; found "
                 f"{hypers.get('featurizer_type')}")
    hypers["num_attention_layers"] = args.layers

    total = 0
    # Both the inner model and the outer wrapper keep a state dict, and
    # load_model would read a stale single-block copy; without a wrapper they
    # are the same object.
    holders = [inner] if inner is ck else [inner, ck]
    for obj in holders:
        for key in ("model_state_dict", "best_model_state_dict"):
            sd = obj.get(key)
            if not isinstance(sd, dict):
                continue
            n = duplicate_blocks(sd, args.layers, args.distinct)
            if n:
                print(f"  {key}: +{n} tensors")
                total += n
    if total == 0:
        ap.error("found no trans.layers.0 weights to duplicate")

    torch.save(ck, args.out)
    print(f"wrote {args.out}  (num_attention_layers = {args.layers}"
          + (f", jitter = {args.distinct}" if args.distinct else "") + ")")


if __name__ == "__main__":
    main()
