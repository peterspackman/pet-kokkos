// pet-eval -- evaluate a PET model on structures from an extended-XYZ file.
//
//   pet-eval MODEL structure.xyz [options]
//
// MODEL is a model name to look up on the search path (pet-eval --models lists
// where it looks), or a path prefix naming <prefix>.json and
// <prefix>.safetensors.
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <utility>
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
      "      --json           emit results as JSON on stdout (for scripts)\n"
      "      --charge N       total charge per frame (conditioned models only)\n"
      "      --spin N         spin multiplicity 2S+1 per frame (conditioned models)\n"
      "      --per-atom       print per-atom energies\n"
      "      --host-neighbors build the neighbour list on the host\n"
      "      --builtin-neighbors  use the built-in O(N^2) search, not vesin\n"
      "      --repeat N       evaluate N times and report the best wall time\n"
      "      --memory         report the device scratch pool and its largest buffers\n"
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
  bool host_neighbors = false, builtin_neighbors = false, info_only = false, json = false;
  bool report_memory = false;
  int repeat = 1;
  int charge = 0;
  int spin = 1;
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
    } else if (s == "--builtin-neighbors") {
      a.builtin_neighbors = true;
    } else if (s == "--info") {
      a.info_only = true;
    } else if (s == "--json") {
      a.json = true;
    } else if (s == "-o" || s == "--output") {
      a.output_path = next("--output");
      a.want_forces = true;
    } else if (s == "--charge") {
      a.charge = std::atoi(next("--charge").c_str());
    } else if (s == "--spin") {
      a.spin = std::atoi(next("--spin").c_str());
    } else if (s == "--memory") {
      a.report_memory = true;
    } else if (s == "--repeat") {
      a.repeat = std::max(1, std::atoi(next("--repeat").c_str()));
    } else if (s.rfind("--kokkos-", 0) == 0) {
      continue;  // Kokkos::initialize's own
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

// The model's properties as (key, JSON value) pairs, for --info in either form.
std::vector<std::pair<std::string, std::string>> model_fields(const Args& a, const pet::Calculator& calc) {
  const pet::Hypers& h = calc.hypers();
  auto str = [](const char* v) { return std::string("\"") + v + "\""; };
  auto num = [](double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return std::string(buf);
  };
  auto flag = [](bool v) { return std::string(v ? "true" : "false"); };
  using pet::FeaturizerType, pet::TransformerType, pet::Normalization, pet::Activation;
  return {
      {"model", str(a.model_spec.c_str())},
      {"featurizer", str(h.featurizer_type == FeaturizerType::Residual ? "residual" : "feedforward")},
      {"transformer", str(h.transformer_type == TransformerType::PreLN ? "PreLN" : "PostLN")},
      {"normalization", str(h.normalization == Normalization::RMSNorm ? "RMSNorm" : "LayerNorm")},
      {"activation", str(h.activation == Activation::SwiGLU ? "SwiGLU" : "SiLU")},
      {"cutoff_function", str(h.cutoff_function == pet::CutoffFunction::Bump ? "Bump" : "Cosine")},
      {"cutoff", num(h.cutoff)},
      {"cutoff_width", num(h.cutoff_width)},
      {"cutoff_width_adaptive", num(h.cutoff_width_adaptive)},
      {"num_neighbors_adaptive", num(h.num_neighbors_adaptive)},
      {"adaptive_cutoff_method",
       str(h.adaptive_cutoff_method == pet::AdaptiveCutoffMethod::Solver ? "solver" : "grid")},
      {"d_pet", num(h.d_pet)},
      {"d_node", num(h.d_node)},
      {"d_head", num(h.d_head)},
      {"d_feedforward", num(h.d_feedforward)},
      {"num_heads", num(h.num_heads)},
      {"num_gnn_layers", num(h.num_gnn_layers)},
      {"num_attention_layers", num(h.num_attention_layers)},
      {"num_readout_layers", num(h.num_readout_layers)},
      {"vesin", flag(pet::vesin_available())},
      {"system_conditioning", flag(h.system_conditioning)},
      {"max_charge", num(h.max_charge)},
      {"max_spin_multiplicity", num(h.max_spin_multiplicity)},
      {"n_species", num((double) calc.atomic_types().size())},
  };
}

void print_model_info(const Args& a, const pet::Calculator& calc) {
  const auto fields = model_fields(a, calc);
  if (a.json) {
    std::printf("{\n");
    for (std::size_t i = 0; i < fields.size(); ++i)
      std::printf("  \"%s\": %s%s\n", fields[i].first.c_str(), fields[i].second.c_str(),
                  i + 1 < fields.size() ? "," : "");
    std::printf("}\n");
    return;
  }
  for (auto [k, v] : fields) {
    if (v.front() == '"') v = v.substr(1, v.size() - 2);
    std::printf("%-24s %s\n", k.c_str(), v.c_str());
  }
  std::printf("%-24s", "species (Z)");
  for (int z : calc.atomic_types()) std::printf(" %d", z);
  std::puts("");
}

// Results as JSON, at full precision: this is what gets diffed against a
// reference.
void print_json(const Args& a, const pet::Calculator& calc,
                const std::vector<pet::System>& frames, const pet::Results& r,
                double best_seconds) {
  auto num = [](double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return std::string(buf);
  };

  std::printf("{\n  \"model\": \"%s\",\n", a.model_spec.c_str());
  std::printf("  \"cutoff\": %s,\n", num(calc.cutoff()).c_str());
  std::printf("  \"energy_unit\": \"%s\",\n", calc.energy_unit().c_str());
  std::printf("  \"length_unit\": \"%s\",\n", calc.length_unit().c_str());
  std::printf("  \"seconds\": %s,\n", num(best_seconds).c_str());
  std::printf("  \"vesin\": %s,\n", pet::vesin_available() ? "true" : "false");
  std::printf("  \"system_conditioning\": %s,\n",
              calc.hypers().system_conditioning ? "true" : "false");
  std::printf("  \"charge\": %d,\n  \"spin_multiplicity\": %d,\n", a.charge, a.spin);
  std::printf("  \"frames\": [\n");

  std::size_t foff = 0, eoff = 0;
  for (std::size_t b = 0; b < frames.size(); ++b) {
    const pet::System& s = frames[b];
    std::printf("    {\n      \"n_atoms\": %d,\n", s.n_atoms);
    std::printf("      \"energy\": %s", num(r.energy[b]).c_str());

    if (a.per_atom && r.per_atom_energy.size() >= eoff + static_cast<std::size_t>(s.n_atoms)) {
      std::printf(",\n      \"per_atom_energy\": [");
      for (int i = 0; i < s.n_atoms; ++i)
        std::printf("%s%s", i ? ", " : "", num(r.per_atom_energy[eoff + i]).c_str());
      std::printf("]");
    }
    eoff += s.n_atoms;

    if (a.want_forces && r.forces.size() >= foff + static_cast<std::size_t>(s.n_atoms) * 3) {
      std::printf(",\n      \"forces\": [");
      for (int i = 0; i < s.n_atoms; ++i) {
        std::printf("%s[", i ? ", " : "");
        for (int c = 0; c < 3; ++c)
          std::printf("%s%s", c ? ", " : "", num(r.forces[foff + i * 3 + c]).c_str());
        std::printf("]");
      }
      std::printf("]");
      // Voigt [xx, yy, zz, xy, xz, yz], the symmetric virial W = V*sigma.
      if ((s.pbc[0] || s.pbc[1] || s.pbc[2]) &&
          r.virial.size() >= (b + 1) * 6) {
        const double vol = cell_volume(s);
        std::printf(",\n      \"volume\": %s", num(vol).c_str());
        std::printf(",\n      \"virial\": [");
        for (int v = 0; v < 6; ++v)
          std::printf("%s%s", v ? ", " : "", num(r.virial[b * 6 + v]).c_str());
        std::printf("]");
        if (vol > 0.0) {
          std::printf(",\n      \"stress\": [");
          for (int v = 0; v < 6; ++v)
            std::printf("%s%s", v ? ", " : "", num(r.virial[b * 6 + v] / vol).c_str());
          std::printf("]");
        }
      }
    }
    foff += static_cast<std::size_t>(s.n_atoms) * 3;
    std::printf("\n    }%s\n", b + 1 < frames.size() ? "," : "");
  }
  std::printf("  ]\n}\n");
}

// Everything owning a View is destroyed when this returns, before
// Kokkos::finalize.
int run(const Args& a) {
  pet::Options opts;
  opts.device_neighbors = !a.host_neighbors;
  opts.neighbors = a.builtin_neighbors ? pet::Options::Neighbors::Builtin
                                       : pet::Options::Neighbors::Vesin;
  pet::Calculator calc(a.model_spec, opts);

  if (a.info_only) {
    print_model_info(a, calc);
    return 0;
  }

  std::vector<pet::System> frames = pet::read_extxyz(a.structure_path);
  // A model without system_conditioning ignores them: warn, don't fail.
  if (a.charge != 0 || a.spin != 1) {
    if (!calc.hypers().system_conditioning)
      std::fprintf(stderr,
                   "pet-eval: warning: --charge/--spin given, but this model was not "
                   "trained with system_conditioning; they have no effect.\n");
    for (auto& s : frames) {
      s.charge = a.charge;
      s.spin_multiplicity = a.spin;
    }
  }
  if (frames.empty()) {
    std::fputs("pet-eval: no frames in the structure file\n", stderr);
    return 2;
  }

  // Here, where the message can name the element.
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

  if (a.json) {
    print_json(a, calc, frames, r, best_seconds);
    return 0;
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

  if (a.report_memory) {
    const std::size_t pool = calc.workspace_bytes();
    std::printf("\ndevice scratch pool: %.2f GiB\n", double(pool) / (1024.0 * 1024.0 * 1024.0));
    // Grouped by label with the numbers collapsed ("ck_node_*"), so one buffer
    // per layer shows as one row with a count.
    const auto rows = calc.workspace_breakdown();
    std::map<std::string, std::pair<std::size_t, int>> fam;
    for (const auto& [label, bytes] : rows) {
      std::string k;
      bool in_num = false;
      for (char c : label) {
        if (c >= '0' && c <= '9') {
          if (!in_num) k += '*';
          in_num = true;
        } else {
          k += c;
          in_num = false;
        }
      }
      auto& e = fam[k];
      e.first += bytes;
      e.second += 1;
    }
    std::vector<std::pair<std::string, std::pair<std::size_t, int>>> fv(fam.begin(), fam.end());
    std::sort(fv.begin(), fv.end(), [](const auto& a, const auto& b) {
      return a.second.first > b.second.first;
    });
    std::size_t shown = 0;
    for (std::size_t i = 0; i < fv.size() && i < 18; ++i) {
      if (fv[i].second.first < pool / 200) break;
      std::printf("  %-34s %4d x %8.1f MiB\n", fv[i].first.c_str(), fv[i].second.second,
                  double(fv[i].second.first) / (1024.0 * 1024.0));
      shown += fv[i].second.first;
    }
    std::printf("  %-34s      %10.1f MiB\n", "(everything else)",
                double(pool - shown) / (1024.0 * 1024.0));
  }

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
  // Before Kokkos: --help and a usage error need no device.
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
