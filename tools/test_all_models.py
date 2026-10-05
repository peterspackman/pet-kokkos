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
"""Convert every published upet model and check pet-kokkos against upet itself.

The goldens in tests/golden/ cover four architectures by hand. This covers the
whole published catalogue: for each checkpoint on `lab-cosmo/upet` it converts
the model, evaluates the same structures with metatomic (i.e. the reference
implementation, through PyTorch) and with pet-kokkos, and reports the deviation.

Run it against each new release of upet or metatrain: a model that starts
failing here usually means an architecture setting has changed, and the report
says which model.

    uv run tools/test_all_models.py                    # small models (<= 150 MB)
    uv run tools/test_all_models.py --all              # everything, ~15 GB
    uv run tools/test_all_models.py --models pet-mad-s pet-omat-xs
    uv run tools/test_all_models.py --report report.md

Needs `pet-eval` built; pass --pet-eval if it is not in ./build*/apps/.
Downloads are cached by huggingface_hub, and conversions are cached in the work
directory, so a re-run only repeats the evaluation.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Deviation above which a model is reported as a mismatch rather than a pass.
# The network runs in fp32 under the default mixed precision, so agreement is
# bounded by fp32 -- relative, because these models' total energies span three
# orders of magnitude (a few eV for a molecule, thousands for a composition-heavy
# periodic cell). An absolute bound would be vacuous at one end and impossible at
# the other.
ENERGY_RTOL = 2e-6
FORCE_ATOL = 1e-3   # eV/Angstrom; forces are small numbers, so absolute here
FORCE_RTOL = 1e-4   # ... unless they are large, in which case relative
STRESS_ATOL = 1e-4  # eV/Angstrom^3


@dataclass
class Result:
    name: str
    status: str = "pending"     # ok | mismatch | unsupported | error
    detail: str = ""
    arch: dict = field(default_factory=dict)
    d_energy: float = 0.0       # worst relative energy deviation
    d_force: float = 0.0        # worst absolute force deviation, eV/Angstrom
    d_stress: float = 0.0       # worst absolute stress deviation, eV/Angstrom^3
    seconds: float = 0.0
    conditioned: bool = False   # also checked at a non-default charge/spin


def list_checkpoints():
    """Every <model>-<size>-v<version>.ckpt on lab-cosmo/upet, newest first."""
    from huggingface_hub import HfApi
    from packaging.version import Version

    pattern = re.compile(r"^(?P<model>pet-[\w]+)-(?P<size>xs|s|m|l|xl)-v(?P<version>[\d.]+)\.ckpt$")
    info = HfApi().model_info("lab-cosmo/upet", files_metadata=True)
    out = []
    for f in info.siblings:
        if not f.rfilename.startswith("models/"):
            continue
        m = pattern.match(f.rfilename[len("models/"):])
        if not m:
            continue
        out.append({
            "name": f"{m['model']}-{m['size']}",
            "version": Version(m["version"]),
            "filename": f.rfilename[len("models/"):],
            "mb": (f.size or 0) / 1e6,
        })
    # One entry per model+size: the newest version, which is what `upet` itself
    # resolves and therefore what a user actually gets.
    best = {}
    for e in out:
        cur = best.get(e["name"])
        if cur is None or e["version"] > cur["version"]:
            best[e["name"]] = e
    return sorted(best.values(), key=lambda e: (e["mb"], e["name"]))


def build_structures():
    """A molecule, a crystal and a rattled crystal, as in the shipped goldens."""
    import ase
    import numpy as np

    cases = {}
    cases["molecule"] = ase.Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.11926], [0.0, 0.76323, -0.47704], [0.0, -0.76323, -0.47704]],
        cell=np.zeros((3, 3)), pbc=False,
    )
    a = 3.567
    cases["crystal"] = ase.Atoms(
        "C8",
        scaled_positions=[
            [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ],
        cell=[a, a, a], pbc=True,
    )
    rattled = cases["crystal"].copy()
    rng = np.random.default_rng(0)
    rattled.positions += rng.normal(0.0, 0.15, rattled.positions.shape)
    cases["disordered"] = rattled
    return cases


def reference_values(ckpt_path, structures, charge=0, spin=1):
    """Evaluate with metatomic -- the reference implementation, via PyTorch.

    `charge`/`spin` go into atoms.info, which is where metatomic's ASE
    calculator reads the electronic state for a conditioned model. A model
    without system_conditioning ignores them, exactly as pet-kokkos does.
    """
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model

    model = load_model(str(ckpt_path)).export()
    calc = MetatomicCalculator(model, device="cpu")
    out = {}
    for name, atoms in structures.items():
        a = atoms.copy()
        a.info["charge"] = charge
        a.info["spin"] = spin
        a.calc = calc
        entry = {"energy": float(a.get_potential_energy()), "forces": a.get_forces().tolist()}
        if a.pbc.any():
            s = a.get_stress(voigt=False)
            # Symmetrize: a stress tensor is symmetric, but autograd's is not
            # exactly, and its antisymmetric part is noise. pet-kokkos returns
            # the symmetrized one, so comparing against a raw off-diagonal would
            # measure that noise instead of this library.
            sym = 0.5 * (s + s.T)
            entry["stress"] = [sym[0, 0], sym[1, 1], sym[2, 2],
                               sym[0, 1], sym[0, 2], sym[1, 2]]
        out[name] = entry
    return out


def write_xyz(path, structures):
    from ase.io import write
    write(str(path), [structures[k] for k in structures], format="extxyz")


def run_pet_eval(pet_eval, model_prefix, xyz, extra=()):
    cmd = [str(pet_eval), str(model_prefix), str(xyz), "--forces", "--json", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip().split("\n")[-1])
    return json.loads(p.stdout)


def compare(ref, got, names):
    """Worst deviations across every structure. Returns (dE_rel, dF, dStress)."""
    de = df = ds = 0.0
    for i, name in enumerate(names):
        r, g = ref[name], got["frames"][i]
        scale = max(abs(r["energy"]), 1.0)
        de = max(de, abs(g["energy"] - r["energy"]) / scale)
        for a, (fr, fg) in enumerate(zip(r["forces"], g["forces"])):
            for c in range(3):
                df = max(df, abs(fg[c] - fr[c]))
        if "stress" in r and "stress" in g:
            for v in range(6):
                ds = max(ds, abs(g["stress"][v] - r["stress"][v]))
    return de, df, ds


def force_ok(d_force, ref, names):
    """Relative-or-absolute, against the largest force in the reference."""
    fmax = 0.0
    for name in names:
        for f in ref[name]["forces"]:
            fmax = max(fmax, max(abs(c) for c in f))
    return d_force <= FORCE_ATOL or d_force <= FORCE_RTOL * max(fmax, 1e-30)


def process(entry, args, structures, xyz, workdir):
    from convert_pet import download_upet_checkpoint  # noqa: E402

    res = Result(name=entry["name"])
    prefix = workdir / entry["name"]

    try:
        ckpt = download_upet_checkpoint(entry["name"])
    except Exception as e:
        res.status, res.detail = "error", f"download: {type(e).__name__}: {e}"
        return res

    # --- convert -------------------------------------------------------------
    if not (prefix.parent.joinpath(prefix.name + ".json").exists() and
            prefix.parent.joinpath(prefix.name + ".safetensors").exists()) or args.force:
        cmd = [sys.executable, str(Path(__file__).parent / "convert_pet.py"),
               "--ckpt", str(ckpt), "--out", str(prefix)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout).strip().split("\n")[-1]
            # The converter refuses, by name, architectures the evaluator does
            # not implement: reported as unsupported, not as an error.
            res.status = "unsupported" if "does not implement" in msg or "not found" in msg else "error"
            res.detail = msg.replace("convert_pet.py: error: ", "")
            return res

    # --- architecture --------------------------------------------------------
    try:
        info = run_pet_eval(args.pet_eval, prefix, xyz, extra=["--info"])
        res.arch = info
    except Exception as e:
        res.status, res.detail = "unsupported", str(e).replace("pet-eval: ", "")
        return res

    # --- reference + ours ----------------------------------------------------
    try:
        ref = reference_values(ckpt, structures)
    except Exception as e:
        res.status, res.detail = "error", f"reference: {type(e).__name__}: {e}"
        return res

    try:
        got = run_pet_eval(args.pet_eval, prefix, xyz)
    except Exception as e:
        res.status, res.detail = "error", f"pet-eval: {e}"
        return res

    names = list(structures)
    res.seconds = got.get("seconds", 0.0)
    res.d_energy, res.d_force, res.d_stress = compare(ref, got, names)

    # A conditioned model is also checked away from the neutral-singlet default.
    # The default alone would not exercise the embedding lookup: charge 0 lands
    # at index max_charge and spin 1 at index 0, so an off-by-one or a missing
    # offset would still land inside the table and produce a plausible number.
    if res.arch.get("system_conditioning") or got.get("system_conditioning"):
        res.conditioned = True
        try:
            ref_q = reference_values(ckpt, structures, charge=1, spin=2)
            got_q = run_pet_eval(args.pet_eval, prefix, xyz, extra=["--charge", "1", "--spin", "2"])
            de, df, ds = compare(ref_q, got_q, names)
            res.d_energy = max(res.d_energy, de)
            res.d_force = max(res.d_force, df)
            res.d_stress = max(res.d_stress, ds)
            # And the state has to actually do something, or the comparison
            # above is comparing two copies of the unconditioned answer.
            moved = abs(got_q["frames"][0]["energy"] - got["frames"][0]["energy"])
            if moved < 1e-9:
                res.status = "mismatch"
                res.detail = "charge/spin changed nothing; conditioning is not wired up"
                return res
        except Exception as e:
            res.status, res.detail = "error", f"conditioned: {e}"
            return res

    ok = (res.d_energy <= ENERGY_RTOL and force_ok(res.d_force, ref, names)
          and res.d_stress <= STRESS_ATOL)
    res.status = "ok" if ok else "mismatch"
    return res


def render(results, args):
    def arch_summary(a):
        if not a:
            return "-"
        bits = [a.get("featurizer", "?")[:4], a.get("transformer", "?"),
                f"G{a.get('num_gnn_layers', '?')}", f"A{a.get('num_attention_layers', '?')}",
                f"d{a.get('d_pet', '?')}"]
        if a.get("num_neighbors_adaptive", -1) > 0:
            bits.append(a.get("adaptive_cutoff_method", "?"))
        if a.get("system_conditioning"):
            bits.append("cond")
        return " ".join(bits)

    mark = {"ok": "PASS", "mismatch": "FAIL", "unsupported": "SKIP", "error": "ERR "}
    lines = []
    w = max((len(r.name) for r in results), default=10)
    lines.append(f"{'':5s} {'model'.ljust(w)}  {'architecture':34s} "
                 f"{'dE/E':>9s} {'dF':>9s} {'dStress':>9s}")
    lines.append("-" * (5 + w + 2 + 34 + 31))
    for r in results:
        if r.status in ("ok", "mismatch"):
            nums = f"{r.d_energy:9.1e} {r.d_force:9.1e} {r.d_stress:9.1e}"
        else:
            nums = " " * 29
        lines.append(f"{mark[r.status]:5s} {r.name.ljust(w)}  {arch_summary(r.arch):34s} {nums}")
        if r.detail:
            lines.append(f"      {'':{w}s}  -> {r.detail[:100]}")

    counts = {k: sum(1 for r in results if r.status == k) for k in mark}
    lines.append("")
    lines.append(f"{counts['ok']} pass, {counts['mismatch']} mismatch, "
                 f"{counts['unsupported']} unsupported, {counts['error']} error "
                 f"(of {len(results)})")
    text = "\n".join(lines)

    if args.report:
        Path(args.report).write_text(
            "# pet-kokkos vs upet\n\n"
            "Every published `lab-cosmo/upet` model (newest version of each size), converted\n"
            "and evaluated against metatomic's own evaluation of the same checkpoint.\n\n"
            "```\n" + text + "\n```\n")
        print(f"\nwrote {args.report}")
    return text


def find_pet_eval():
    here = Path(__file__).resolve().parents[1]
    for p in sorted(here.glob("build*/apps/pet-eval")):
        return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="*", default=None,
                    help="only these model names (e.g. pet-mad-s pet-omat-xs)")
    ap.add_argument("--all", action="store_true",
                    help="include the large checkpoints (~15 GB of downloads)")
    ap.add_argument("--max-mb", type=float, default=150.0,
                    help="skip checkpoints larger than this unless --all (default 150)")
    ap.add_argument("--workdir", default=None,
                    help="where converted models go (default: ./build-models)")
    ap.add_argument("--pet-eval", default=None, help="path to the pet-eval binary")
    ap.add_argument("--report", default=None, help="also write a markdown report here")
    ap.add_argument("--force", action="store_true", help="re-convert even if cached")
    ap.add_argument("--list", action="store_true", help="list checkpoints and exit")
    args = ap.parse_args()

    entries = list_checkpoints()
    if args.models:
        wanted = set(args.models)
        entries = [e for e in entries if e["name"] in wanted]
        missing = wanted - {e["name"] for e in entries}
        if missing:
            ap.error(f"unknown model(s): {', '.join(sorted(missing))}")
    elif not args.all:
        entries = [e for e in entries if e["mb"] <= args.max_mb]

    if args.list:
        for e in entries:
            print(f"  {e['name']:16s} {e['filename']:28s} {e['mb']:8.1f} MB")
        print(f"\n{len(entries)} checkpoints, "
              f"{sum(e['mb'] for e in entries) / 1000:.1f} GB")
        return 0

    args.pet_eval = Path(args.pet_eval) if args.pet_eval else find_pet_eval()
    if not args.pet_eval or not args.pet_eval.exists():
        ap.error("pet-eval not found; build it or pass --pet-eval")

    workdir = Path(args.workdir or Path(__file__).resolve().parents[1] / "build-models")
    workdir.mkdir(parents=True, exist_ok=True)
    # The converted models are a model search path in their own right, so
    # pet-eval finds them by name.
    os.environ["PET_MODEL_DIR"] = str(workdir)

    structures = build_structures()
    xyz = workdir / "_cases.xyz"
    write_xyz(xyz, structures)

    print(f"pet-eval : {args.pet_eval}")
    print(f"workdir  : {workdir}")
    print(f"models   : {len(entries)} "
          f"({sum(e['mb'] for e in entries) / 1000:.1f} GB of checkpoints)\n")

    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['name']} ({e['mb']:.0f} MB) ... ", end="", flush=True)
        try:
            r = process(e, args, structures, xyz, workdir)
        except Exception:
            r = Result(name=e["name"], status="error",
                       detail=traceback.format_exc().strip().split("\n")[-1])
        results.append(r)
        print(r.status + (f" ({r.detail[:60]})" if r.detail else ""))

    print("\n" + render(results, args))
    # A mismatch or an error is a failure; an architecture we knowingly do not
    # support is not.
    return 1 if any(r.status in ("mismatch", "error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
