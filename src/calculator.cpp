#include "pet/calculator.hpp"

#include "pet/checkpoint.hpp"
#include "pet/device_geometry.hpp"
#include "pet/device_neighbors.hpp"
#include "pet/gemm.hpp"
#include "pet/model.hpp"
#include "pet/neighbors.hpp"

#include <Kokkos_Core.hpp>
// After Kokkos_Core.hpp, which is what defines KOKKOS_ENABLE_HIP. nvcc pulls in
// the CUDA runtime implicitly; hipcc is not as reliable about it, and
// device_memory() below needs hipMemGetInfo.
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef PET_KOKKOS_MODEL_DIR
#define PET_KOKKOS_MODEL_DIR "."
#endif

namespace pet {

namespace {

constexpr const char* kDefaultModel = "pet-mad-xs";

// Choosing the batch width.
//
// Two independent ceilings bind, and the batch is the smaller of them.
//
// MEMORY. The backward holds saved activations shaped [N*M, width], so the
// footprint is linear in EDGE SLOTS (atoms x neighbours), not in atoms. That
// distinction is the whole problem: bytes-per-atom is not a property of the
// model -- an isolated 8-atom molecule has ~7 neighbours per atom and a crystal
// atom ~45, so a figure measured on one is meaningless for the other. Bytes per
// edge slot IS a model constant, measured at 57.8 KiB for the feedforward model
// and 104.5 KiB for the residual one, and it predicts the observed failures
// exactly: the residual model at 4096 atoms and M=45 wants
// 184320 slots x 104.5 KiB = 18.4 GiB on a 16 GiB card, and it duly aborts.
//
// The budget is (free + what the pool already holds), since the pool's current
// allocation is reusable capacity rather than a loss, times a headroom factor.
// The headroom covers three things: buffers reallocate one at a time while the
// old one is still live, so the peak sits one buffer above the steady state;
// fragmentation; and the next batch's M may exceed the last one's.
//
// THROUGHPUT. Memory alone would allow the feedforward model ~12k atoms, but its
// cost per round stops being flat well before that -- measured 0.108 s/structure
// at 4096 atoms against 0.118 at 8192, and no completion at 16384. That knee is
// about how much work one launch can keep resident, so it caps edge slots too,
// not atoms.
constexpr double kMemHeadroom = 0.8;
// Share of the WHOLE card a batch may occupy. Leaves room for everything else
// resident (model weights, optimizer state, the CUDA context) and, critically,
// stops the grow-only pool from citing its own past growth as justification for
// more.
constexpr double kMemCardFraction = 0.7;
constexpr long kMaxEdgeSlots = 131072;  // throughput knee, ~2x the measured optimum
// Below this the fixed-size buffers dominate and the per-slot figure is noise.
constexpr long kMinSampleSlots = 4096;

void device_memory(std::size_t budget_override, std::size_t& free_b, std::size_t& total_b) {
  free_b = total_b = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) free_b = total_b = 0;
#elif defined(KOKKOS_ENABLE_HIP)
  // Without this an AMD build silently takes the 4 GiB fallback below, which on
  // a 64 GB MI250X GCD sizes the batch at a sixteenth of what fits -- not a
  // failure, just a quietly tiny batch, which is the worst kind of performance
  // bug to spot.
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) free_b = total_b = 0;
#endif
  // An explicit budget comes FIRST so a smaller card can be simulated on a
  // larger one.
  if (budget_override > 0) {
    free_b = total_b = budget_override;
  }
  if (total_b == 0) free_b = total_b = std::size_t(4) * 1024u * 1024u * 1024u;
}

// A directory provides a model only if BOTH halves are there -- a stray .json
// next to missing weights should keep searching, not fail at load time.
bool model_dir_has(const std::string& dir, const std::string& name, std::string& json_out,
                   std::string& weights_out) {
  const std::string j = dir + "/" + name + ".json";
  const std::string s = dir + "/" + name + ".safetensors";
  std::error_code ec;
  if (!std::filesystem::exists(j, ec) || !std::filesystem::exists(s, ec)) return false;
  json_out = j;
  weights_out = s;
  return true;
}

void find_named_model(const std::string& name, std::string& json_out, std::string& weights_out) {
  const auto dirs = model_search_dirs();
  for (const auto& d : dirs)
    if (model_dir_has(d, name, json_out, weights_out)) return;

  // Report every location tried: a failed model lookup on a batch node is
  // otherwise near-impossible to diagnose from a log file.
  std::string msg = "PET model '" + name +
                    "' not found. Model weights ship separately from the source tree -- set "
                    "PET_MODEL_DIR to a directory containing " +
                    name + ".json and " + name + ".safetensors.\nSearched:";
  for (const auto& d : dirs) msg += "\n  " + d;
  throw std::runtime_error(msg);
}

}  // namespace

std::vector<std::string> model_search_dirs() {
  std::vector<std::string> dirs;
  auto add = [&dirs](std::string d) {
    if (!d.empty()) dirs.push_back(std::move(d));
  };

  // 1. PET_MODEL_DIR: an explicit ':'-separated list, wins over everything.
  if (const char* e = std::getenv("PET_MODEL_DIR")) {
    const std::string s(e);
    for (std::size_t p = 0; p <= s.size();) {
      const std::size_t q = s.find(':', p);
      add(s.substr(p, (q == std::string::npos ? s.size() : q) - p));
      if (q == std::string::npos) break;
      p = q + 1;
    }
  }
  // 2. The working directory, so a self-contained run directory just works.
  add("models");
  add(".");
  // 3. Per-user data directory.
  if (const char* x = std::getenv("XDG_DATA_HOME"); x && x[0])
    add(std::string(x) + "/pet/models");
  else if (const char* h = std::getenv("HOME"); h && h[0])
    add(std::string(h) + "/.local/share/pet/models");
  // 4. The build-time source tree: developer convenience for anyone who drops
  //    checkpoints into models/ (git-ignored).
  add(PET_KOKKOS_MODEL_DIR);
  return dirs;
}

void resolve_model(const std::string& spec, std::string& json_out, std::string& weights_out) {
  if (spec.empty()) {
    find_named_model(kDefaultModel, json_out, weights_out);
    return;
  }
  // A spec containing a path separator, or naming an existing <spec>.json, is a
  // path prefix. Otherwise it is a model name to search for.
  std::error_code ec;
  if (spec.find('/') != std::string::npos || std::filesystem::exists(spec + ".json", ec)) {
    json_out = spec + ".json";
    weights_out = spec + ".safetensors";
    return;
  }
  find_named_model(spec, json_out, weights_out);
}

// ---------------------------------------------------------------------------

struct Calculator::Impl {
  Checkpoint ckpt;
  PetModel model;
  Options opts;

  // Persistent neighbour-build scratch, reused across calls so a stepping loop
  // pays no per-round device (re)allocation for the geometry staging, the raw
  // COO edge list, or the NEF build. Mirrors the model's own workspace.
  Workspace nbr_ws;
  EdgeMap nbr_edge_map{16};
  int nbr_m_high = 0;
  NefCache cache;

  std::vector<int> atomic_types;

  Impl(const std::string& json_path, const std::string& weights_path, Options o)
      : ckpt(json_path, weights_path), model(ckpt), opts(o) {
    for (int z = 0; z < (int) ckpt.species_to_index.size(); ++z)
      if (ckpt.species_to_index[z] >= 0) atomic_types.push_back(z);
    // Applied at construction because the cuBLAS handle's math mode is fixed
    // when the handle is first created, which is at the first GEMM. Process-
    // global, so the last Calculator constructed wins -- see gemm.hpp.
    if (o.allow_tf32) set_tf32(true);
  }
};

Calculator::Calculator(const std::string& spec, Options opts) {
  std::string j, w;
  resolve_model(spec, j, w);
  if (!Kokkos::is_initialized())
    throw std::runtime_error(
        "pet::Calculator: Kokkos must be initialized before constructing a model.");
  impl_ = std::make_unique<Impl>(j, w, opts);
}

Calculator::Calculator(const std::string& model_json, const std::string& model_weights,
                       Options opts) {
  if (!Kokkos::is_initialized())
    throw std::runtime_error(
        "pet::Calculator: Kokkos must be initialized before constructing a model.");
  impl_ = std::make_unique<Impl>(model_json, model_weights, opts);
}

Calculator::~Calculator() = default;
Calculator::Calculator(Calculator&&) noexcept = default;
Calculator& Calculator::operator=(Calculator&&) noexcept = default;

const Hypers& Calculator::hypers() const { return impl_->ckpt.hypers; }
double Calculator::cutoff() const { return impl_->ckpt.hypers.cutoff; }
const std::vector<int>& Calculator::atomic_types() const { return impl_->atomic_types; }
const std::string& Calculator::length_unit() const { return impl_->ckpt.length_unit; }
const std::string& Calculator::energy_unit() const { return impl_->ckpt.energy_unit; }
Options& Calculator::options() { return impl_->opts; }
const Options& Calculator::options() const { return impl_->opts; }

bool Calculator::supports(int atomic_number) const {
  const auto& m = impl_->ckpt.species_to_index;
  return atomic_number >= 0 && atomic_number < (int) m.size() && m[atomic_number] >= 0;
}

DeviceGeom Calculator::stage(int n_atoms_total, int n_struct) const {
  return stage_geometry_views(impl_->nbr_ws, n_atoms_total, n_struct);
}

Results Calculator::compute(const System& system, bool compute_forces) const {
  // Single-structure path: the host neighbour-list path. It is what the goldens
  // validate, it returns the virial in the host EnergyResult directly, and it
  // is the only path that fills per_atom_energy.
  Results out;
  const int N = system.n_atoms;
  out.energy.resize(1);
  out.n_atoms.assign(1, N);

  EdgeData ed = build_edge_data(system, impl_->ckpt.hypers, impl_->ckpt.species_to_index);

  if (compute_forces) {
    std::vector<double> forces;
    EnergyResult r = impl_->model.energy_forces(ed, forces);
    out.energy[0] = r.total;
    out.per_atom_energy = r.per_atom;
    out.forces = std::move(forces);
    out.struct_id.assign(N, 0);
    out.virial.assign(r.virial, r.virial + 6);
  } else {
    EnergyResult r = impl_->model.energy(ed);
    out.energy[0] = r.total;
    out.per_atom_energy = r.per_atom;
  }
  return out;
}

Results Calculator::compute(const std::vector<System>& systems, bool compute_forces) const {
  const int B = static_cast<int>(systems.size());
  if (B == 0) return {};
  if (B == 1) return compute(systems[0], compute_forces);

  BatchResult br;
  if (impl_->opts.device_neighbors) {
    DeviceEdgeData dev = build_device_batch(
        systems, impl_->ckpt.hypers, impl_->ckpt.species_to_index, impl_->model.probes(),
        impl_->model.n_probes(), impl_->nbr_ws, impl_->nbr_edge_map, impl_->nbr_m_high,
        impl_->opts.cache_neighbors ? &impl_->cache : nullptr);
    br = impl_->model.energy_forces_batch(dev, compute_forces);
  } else {
    br = impl_->model.energy_forces_batch(systems, compute_forces);
  }

  Results out;
  out.energy.resize(B);
  out.n_atoms.resize(B);
  for (int b = 0; b < B; ++b) out.n_atoms[b] = systems[b].n_atoms;

  {
    auto h_e = Kokkos::create_mirror_view(br.energy);
    Kokkos::deep_copy(h_e, br.energy);
    for (int b = 0; b < B; ++b) out.energy[b] = h_e(b);
  }

  if (compute_forces && br.forces.extent(0) > 0) {
    const int Ntot = static_cast<int>(br.forces.extent(0));
    out.forces.resize(static_cast<std::size_t>(Ntot) * 3);
    auto h_f = Kokkos::create_mirror_view(br.forces);
    Kokkos::deep_copy(h_f, br.forces);
    for (int i = 0; i < Ntot; ++i)
      for (int c = 0; c < 3; ++c) out.forces[i * 3 + c] = h_f(i, c);

    auto h_id = Kokkos::create_mirror_view(br.struct_id);
    Kokkos::deep_copy(h_id, br.struct_id);
    out.struct_id.resize(Ntot);
    for (int i = 0; i < Ntot; ++i) out.struct_id[i] = h_id(i);

    if (br.virial.extent(0) > 0) {
      out.virial.resize(static_cast<std::size_t>(B) * 6);
      auto h_w = Kokkos::create_mirror_view(br.virial);
      Kokkos::deep_copy(h_w, br.virial);
      for (int b = 0; b < B; ++b)
        for (int v = 0; v < 6; ++v) out.virial[b * 6 + v] = h_w(b, v);
    }
  }
  return out;
}

BatchResult Calculator::compute_device(const DeviceGeom& geom, bool compute_forces) const {
  DeviceEdgeData dev = build_nef_device(
      geom, impl_->ckpt.hypers, impl_->model.probes(), impl_->model.n_probes(), impl_->nbr_ws,
      impl_->nbr_edge_map, impl_->nbr_m_high,
      impl_->opts.cache_neighbors ? &impl_->cache : nullptr);
  return impl_->model.energy_forces_batch(dev, compute_forces);
}

int Calculator::recommended_batch_atoms() const {
  if (impl_->opts.max_batch_atoms > 0) return impl_->opts.max_batch_atoms;

  // First batch of a run: no neighbour list has been built, so M is unknown.
  // These are measured floors on a 16 GiB card -- the feedforward model runs
  // comfortably at 4096 atoms, the residual one aborts above 2048.
  const int bootstrap =
      (impl_->ckpt.hypers.featurizer_type == FeaturizerType::Residual) ? 2048 : 4096;

  const std::size_t pool = impl_->model.workspace_bytes();
  const long slots = impl_->model.peak_edge_slots();
  if (pool == 0 || slots < kMinSampleSlots) return bootstrap;

  const double bytes_per_slot = double(pool) / double(slots);
  const int m = std::max(1, impl_->model.peak_max_neighbors());
  std::size_t free_b = 0, total_b = 0;
  device_memory(impl_->opts.memory_budget_bytes, free_b, total_b);

  // Bounded by the CARD, not by (free + pool) alone.
  //
  // (free + pool) on its own is a feedback loop: the workspace is a grow-only
  // high-water pool, so whatever one chunk overshoots by is still held when the
  // next chunk asks, and it comes back as a licence to grow again. Measured on a
  // 16 GiB card, four consecutive chunks: free 2.38 -> 1.42 -> 0.74 -> 0.00 GiB
  // while the pool ratcheted 9.99 -> 14.77 GiB, and once free reached zero the
  // cost per structure-evaluation went from 6.1 ms to 82.2 ms. That one effect
  // was 11.5x of the residual model's total runtime.
  //
  // Capping against the physical card breaks the loop, because the bound stops
  // being a function of how much the pool has already taken.
  const double budget =
      std::min(double(free_b) + double(pool), double(total_b) * kMemCardFraction) * kMemHeadroom;

  const long by_memory = static_cast<long>(budget / (bytes_per_slot * m));
  const long by_throughput = kMaxEdgeSlots / m;
  long atoms = std::min(by_memory, by_throughput);
  atoms = std::min<long>(std::max<long>(atoms, 256), 65536);
  return static_cast<int>(atoms);
}

}  // namespace pet
