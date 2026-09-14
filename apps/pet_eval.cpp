// pet-eval -- evaluate a PET model on structures from an extended-XYZ file.
//
//   pet-eval MODEL structure.xyz [options]
//
// MODEL is a model name to look up on the search path (pet-eval --models lists
// where it looks), or a path prefix naming <prefix>.json and
// <prefix>.safetensors.
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/io.hpp"

namespace {

void usage() {
  std::puts(
      "pet-eval -- evaluate a PET machine-learning potential\n"
      "\n"
      "usage: pet-eval MODEL STRUCTURE.xyz [options]\n"
      "       pet-eval --models\n"
      "\n"
      "  MODEL          model name (searched for) or path prefix naming\n"
      "                 <prefix>.json and <prefix>.safetensors\n"
      "  STRUCTURE.xyz  extended XYZ; every frame in the file is evaluated\n"
      "\n"
      "options:\n"
      "  -f, --forces         compute forces (and, for periodic frames, the virial)\n"
      "  -b, --batch          evaluate all frames in one pass instead of one at a time\n"
      "  -o, --output FILE    write the frames back out as extxyz, with forces\n"
      "      --per-atom       print per-atom energies\n"
      "      --host-neighbors build the neighbour list on the host\n"
      "      --repeat N       evaluate N times and report the best wall time\n"
      "      --info           print the model's properties and exit\n"
      "      --models         list the model search path and exit\n"
      "  -h, --help           this message\n"
      "\n"
      "Units are the model's own: Angstrom, eV, eV/Angstrom. The virial is the\n"
      "symmetric W = V*sigma in Voigt order [xx yy zz xy xz yz].");
}

// Voigt [xx, yy, zz, xy, xz, yz] -> the 3x3 stress, printed row by row.
void print_stress(const std::vector<double>& virial, int b, double volume) {
  if (virial.size() < static_cast<std::size_t>(b + 1) * 6 || volume <= 0.0) return;
  const double* w = &virial[static_cast<std::size_t>(b) * 6];
  const double s[9] = {w[0] / volume, w[3] / volume, w[4] / volume,
                       w[3] / volume, w[1] / volume, w[5] / volume,
                       w[4] / volume, w[5] / volume, w[2] / volume};
  std::puts("  stress (eV/A^3):");
  for (int r = 0; r < 3; ++r)
    std::printf("    % .8e  % .8e  % .8e\n", s[r * 3], s[r * 3 + 1], s[r * 3 + 2]);
}

double cell_volume(const pet::System& s) {
  const double* c = s.cell.data();
  return std::fabs(c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                   c[2] * (c[3] * c[7] - c[4] * c[6]));
}

struct Args {
  std::string model_spec, structure_path, output_path;
  bool want_forces = false, batch = false, per_atom = false;
  bool host_neighbors = false, info_only = false;
  int repeat = 1;
};

// Parse argv. Returns false when the program should stop; `rc` is the exit code
// and is 0 for the cases that legitimately end here (--help, --models).
bool parse_args(int argc, char** argv, Args& a, int& rc) {
  rc = 0;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "pet-eval: %s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "-h" || s == "--help") {
      usage();
      return false;
    } else if (s == "--models") {
      std::puts("PET model search path (first match wins; a directory must hold");
      std::puts("both <name>.json and <name>.safetensors):");
      for (const auto& d : pet::model_search_dirs()) std::printf("  %s\n", d.c_str());
      return false;
    } else if (s == "-f" || s == "--forces") {
      a.want_forces = true;
    } else if (s == "-b" || s == "--batch") {
      a.batch = true;
    } else if (s == "--per-atom") {
      a.per_atom = true;
    } else if (s == "--host-neighbors") {
      a.host_neighbors = true;
    } else if (s == "--info") {
      a.info_only = true;
    } else if (s == "-o" || s == "--output") {
      a.output_path = next("--output");
      a.want_forces = true;
    } else if (s == "--repeat") {
      a.repeat = std::max(1, std::atoi(next("--repeat").c_str()));
    } else if (!s.empty() && s[0] == '-') {
      std::fprintf(stderr, "pet-eval: unknown option '%s' (try --help)\n", s.c_str());
      rc = 2;
      return false;
    } else {
      positional.push_back(s);
    }
  }

  if (positional.empty()) {
    usage();
    rc = 2;
    return false;
  }
  a.model_spec = positional[0];
  if (positional.size() > 1) a.structure_path = positional[1];
  if (a.structure_path.empty() && !a.info_only) {
    std::fputs("pet-eval: need a structure file (or --info)\n", stderr);
    rc = 2;
    return false;
  }
  return true;
}

void print_model_info(const Args& a, const pet::Calculator& calc) {
  const pet::Hypers& h = calc.hypers();
  std::printf("model            : %s\n", a.model_spec.c_str());
  std::printf("cutoff           : %g %s\n", h.cutoff, calc.length_unit().c_str());
  std::printf("energy unit      : %s\n", calc.energy_unit().c_str());
  std::printf("featurizer       : %s\n",
              h.featurizer_type == pet::FeaturizerType::Residual ? "residual" : "feedforward");
  std::printf("transformer      : %s\n",
              h.transformer_type == pet::TransformerType::PreLN ? "PreLN" : "PostLN");
  std::printf("normalization    : %s\n",
              h.normalization == pet::Normalization::RMSNorm ? "RMSNorm" : "LayerNorm");
  std::printf("activation       : %s\n",
              h.activation == pet::Activation::SwiGLU ? "SwiGLU" : "SiLU");
  std::printf("cutoff function  : %s%s\n",
              h.cutoff_function == pet::CutoffFunction::Bump ? "Bump" : "Cosine",
              h.adaptive() ? " (adaptive)" : "");
  std::printf("dims             : d_pet=%d d_node=%d d_ff=%d heads=%d\n", h.d_pet, h.d_node,
              h.d_feedforward, h.num_heads);
  std::printf("layers           : %d GNN x %d attention, %d readout\n", h.num_gnn_layers,
              h.num_attention_layers, h.num_readout_layers);
  std::printf("species          : %zu (Z =", calc.atomic_types().size());
  for (int z : calc.atomic_types()) std::printf(" %d", z);
  std::puts(")");
}

// The whole program, in a scope of its own. Everything that owns a Kokkos::View
// is destroyed when this returns -- which has to happen BEFORE
// Kokkos::finalize(), and is the reason this is a function rather than the body
// of main with early returns in it.
int run(const Args& a) {
  pet::Options opts;
  opts.device_neighbors = !a.host_neighbors;
  pet::Calculator calc(a.model_spec, opts);

  if (a.info_only) {
    print_model_info(a, calc);
    return 0;
  }

  std::vector<pet::System> frames = pet::read_extxyz(a.structure_path);
  if (frames.empty()) {
    std::fputs("pet-eval: no frames in the structure file\n", stderr);
    return 2;
  }

  // Reject an unsupported element here rather than deep inside the neighbour
  // build, where the message would name a species index nobody can decode.
  for (std::size_t f = 0; f < frames.size(); ++f)
    for (int z : frames[f].atomic_numbers)
      if (!calc.supports(z)) {
        std::fprintf(stderr,
                     "pet-eval: frame %zu contains %s (Z=%d), which this model does not support\n",
                     f, pet::element_symbol(z), z);
        return 1;
      }

  pet::Results r;
  double best_seconds = 1e300;
  for (int k = 0; k < a.repeat; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    if (a.batch) {
      r = calc.compute(frames, a.want_forces);
    } else {
      pet::Results all;
      for (const auto& s : frames) {
        pet::Results one = calc.compute(s, a.want_forces);
        all.energy.push_back(one.energy[0]);
        all.n_atoms.push_back(s.n_atoms);
        all.forces.insert(all.forces.end(), one.forces.begin(), one.forces.end());
        all.per_atom_energy.insert(all.per_atom_energy.end(), one.per_atom_energy.begin(),
                                   one.per_atom_energy.end());
        all.virial.insert(all.virial.end(), one.virial.begin(), one.virial.end());
      }
      r = std::move(all);
    }
    Kokkos::fence();
    const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
    best_seconds = std::min(best_seconds, dt.count());
  }

  std::size_t foff = 0, eoff = 0;
  int total_atoms = 0;
  for (std::size_t b = 0; b < frames.size(); ++b) {
    const pet::System& s = frames[b];
    total_atoms += s.n_atoms;
    const double e = r.energy[b];
    std::printf("frame %zu: N=%d  E = %.10f eV  (%.10f eV/atom)\n", b, s.n_atoms, e,
                e / std::max(1, s.n_atoms));

    if (a.per_atom && r.per_atom_energy.size() >= eoff + static_cast<std::size_t>(s.n_atoms))
      for (int i = 0; i < s.n_atoms; ++i)
        std::printf("    %-3s % .10f eV\n", pet::element_symbol(s.atomic_numbers[i]),
                    r.per_atom_energy[eoff + i]);
    eoff += s.n_atoms;

    if (a.want_forces && r.forces.size() >= foff + static_cast<std::size_t>(s.n_atoms) * 3) {
      double max_f = 0.0;
      for (int i = 0; i < s.n_atoms * 3; ++i)
        max_f = std::max(max_f, std::fabs(r.forces[foff + i]));
      std::printf("  max|F| = %.8e eV/A\n", max_f);
      if (s.pbc[0] || s.pbc[1] || s.pbc[2])
        print_stress(r.virial, static_cast<int>(b), cell_volume(s));
    }
    foff += static_cast<std::size_t>(s.n_atoms) * 3;
  }

  if (a.repeat > 1)
    std::printf("\nbest of %d: %.4f s for %zu frames / %d atoms (%.4f ms/atom)\n", a.repeat,
                best_seconds, frames.size(), total_atoms,
                1e3 * best_seconds / std::max(1, total_atoms));

  if (!a.output_path.empty()) {
    std::ofstream out(a.output_path);
    if (!out) {
      std::fprintf(stderr, "pet-eval: cannot write '%s'\n", a.output_path.c_str());
      return 1;
    }
    std::size_t off = 0;
    for (std::size_t b = 0; b < frames.size(); ++b) {
      const double e = r.energy[b];
      pet::write_extxyz(out, frames[b], r.forces.empty() ? nullptr : &r.forces[off], &e);
      off += static_cast<std::size_t>(frames[b].n_atoms) * 3;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  int rc = 0;
  // Argument parsing happens before Kokkos comes up: --help and --models should
  // not pay for a device context, and a usage error should not need one either.
  if (!parse_args(argc, argv, args, rc)) return rc;

  Kokkos::initialize(argc, argv);
  try {
    rc = run(args);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "pet-eval: %s\n", ex.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
