#include "pet/calculator.hpp"

#include "pet/checkpoint.hpp"
#include "pet/device_geometry.hpp"
#include "pet/device_neighbors.hpp"
#include "pet/gemm.hpp"
#include "pet/model.hpp"
#include "pet/neighbors.hpp"

#include <Kokkos_Core.hpp>
// After Kokkos_Core.hpp, which defines KOKKOS_ENABLE_HIP; for hipMemGetInfo.
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
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

// The batch width is the smaller of two ceilings. Memory: the backward's saves
// are [atoms x neighbours, width], so bytes per edge slot is the model constant
// (~58 KiB feedforward, ~105 KiB residual), not bytes per atom. Throughput: past
// ~kMaxEdgeSlots a bigger batch stops being cheaper per structure.
//
// The memory budget is a share of the card (the pool is grow-only, so counting
// what it already holds as free would let it ratchet itself up forever) with
// headroom for reallocation peaks, fragmentation and a larger next M.
constexpr double kMemHeadroom = 0.8;
constexpr double kMemCardFraction = 0.7;
constexpr long kMaxEdgeSlots = 131072;
constexpr long kMinSampleSlots = 4096;  // below this, fixed-size buffers dominate the sample

void device_memory(const Options& o, std::size_t& free_b, std::size_t& total_b) {
  const std::size_t budget_override = o.memory_budget_bytes;
  free_b = total_b = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) free_b = total_b = 0;
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) free_b = total_b = 0;
#endif
  if (budget_override > 0) free_b = total_b = budget_override;  // also simulates a smaller card
  if (total_b == 0) free_b = total_b = std::size_t(4) * 1024u * 1024u * 1024u;
  if (o.device_share > 1) free_b /= o.device_share, total_b /= o.device_share;
}

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

  std::string msg = "PET model '" + name +
                    "' not found. Model weights ship separately from the source tree -- set "
                    "PET_MODEL_DIR to a directory containing " +
                    name + ".json and " + name + ".safetensors.\nSearched:";
  for (const auto& d : dirs) msg += "\n  " + d;
  throw std::runtime_error(msg);
}

// Edges within the cutoff before the adaptive cutoff prunes them: every pair for
// a molecule, the cutoff sphere at the cell's mean density for anything periodic.
double raw_edges_estimate(const System& s, double rc) {
  const double n = s.n_atoms;
  if (!(s.pbc[0] || s.pbc[1] || s.pbc[2])) return n * (n - 1);
  const auto& c = s.cell;
  const double vol = std::fabs(c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                               c[2] * (c[3] * c[7] - c[4] * c[6]));
  return vol > 0 ? n * n / vol * (4.0 / 3.0) * M_PI * rc * rc * rc : n * (n - 1);
}

}  // namespace

std::vector<std::string> model_search_dirs() {
  std::vector<std::string> dirs;
  auto add = [&dirs](std::string d) {
    if (!d.empty()) dirs.push_back(std::move(d));
  };

  const char* e = std::getenv("PET_MODEL_DIR");
  const std::string env = e ? e : "";
  for (std::size_t p = 0, q; p <= env.size(); p = q + 1) {
    q = std::min(env.find(':', p), env.size());
    add(env.substr(p, q - p));
  }
  add("models");
  add(".");
  if (const char* x = std::getenv("XDG_DATA_HOME"); x && x[0])
    add(std::string(x) + "/pet/models");
  else if (const char* h = std::getenv("HOME"); h && h[0])
    add(std::string(h) + "/.local/share/pet/models");
  add(PET_KOKKOS_MODEL_DIR);
  return dirs;
}

void resolve_model(const std::string& spec, std::string& json_out, std::string& weights_out) {
  if (spec.empty()) {
    find_named_model(kDefaultModel, json_out, weights_out);
    return;
  }
  // A path prefix if it has a '/' or <spec>.json exists; else a name.
  std::error_code ec;
  if (spec.find('/') != std::string::npos || std::filesystem::exists(spec + ".json", ec)) {
    json_out = spec + ".json";
    weights_out = spec + ".safetensors";
    return;
  }
  find_named_model(spec, json_out, weights_out);
}

struct Calculator::Impl {
  Checkpoint ckpt;
  PetModel model;
  Options opts;

  // Neighbour-build scratch, kept across calls like the model's own.
  Workspace nbr_ws;
  EdgeMap nbr_edge_map{16};
  int nbr_m_high = 0;
  NefCache cache;
  EdgeSession md;  // an engine's list (compute_edges, set_neighbors)

  std::vector<int> atomic_types;

  Impl(const std::string& json_path, const std::string& weights_path, Options o)
      : ckpt(json_path, weights_path), model(ckpt), opts(o) {
    for (int z = 0; z < (int) ckpt.species_to_index.size(); ++z)
      if (ckpt.species_to_index[z] >= 0) atomic_types.push_back(z);
    if (o.allow_tf32) set_tf32(true);
    std::size_t free_b = 0, total_b = 0;
    device_memory(o, free_b, total_b);
    // PET_GRAPHS=0 is for profilers that fence around every kernel.
    const char* g = std::getenv("PET_GRAPHS");
    model.set_graphs(o.graphs && !(g && g[0] == '0'));
    model.set_memory_policy(std::size_t(double(total_b) * kMemCardFraction * kMemHeadroom), o.recompute);
    // Process-global, as the two searches are meant to agree.
    neighbor_backend() = (o.neighbors == Options::Neighbors::Vesin) ? NeighborBackend::Vesin
                                                                    : NeighborBackend::Builtin;
  }
};

Calculator::Calculator(const std::string& spec, Options opts) {
  std::string j, w;
  resolve_model(spec, j, w);
  *this = Calculator(j, w, opts);
}

Calculator::Calculator(const std::string& model_json, const std::string& model_weights,
                       Options opts) {
  if (!Kokkos::is_initialized())
    throw std::runtime_error("pet::Calculator: Kokkos must be initialized before constructing a model.");
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
double Calculator::energy_scale() const { return impl_->ckpt.energy_scale; }
Options& Calculator::options() { return impl_->opts; }
const Options& Calculator::options() const { return impl_->opts; }

bool Calculator::supports(int atomic_number) const {
  const auto& m = impl_->ckpt.species_to_index;
  return atomic_number >= 0 && atomic_number < (int) m.size() && m[atomic_number] >= 0;
}

DeviceGeom Calculator::stage(const std::vector<System>& systems) const {
  impl_->cache = NefCache{};
  return stage_systems(impl_->nbr_ws, systems, impl_->ckpt.species_to_index);
}

Results Calculator::compute(const System& system, bool compute_forces) const {
  // Small structures stay on the host path, where the device build's fixed cost
  // would dominate; it is also the path the goldens pin to 1e-9.
  if (impl_->opts.device_neighbors && raw_edges_estimate(system, cutoff()) >= kDeviceSingleMinRawEdges)
    return compute_batch({system}, compute_forces);

  const int N = system.n_atoms;
  EdgeData ed = build_edge_data(system, impl_->ckpt.hypers, impl_->ckpt.species_to_index);
  Results out;
  out.n_atoms = {N};
  EnergyResult r;
  if (compute_forces) {
    r = impl_->model.energy_forces(ed, out.forces);
    out.struct_id.assign(N, 0);
    out.virial.assign(r.virial, r.virial + 6);
  } else {
    r = impl_->model.energy(ed);
  }
  out.energy = {r.total};
  out.per_atom_energy = std::move(r.per_atom);
  return out;
}

Results Calculator::compute(const std::vector<System>& systems, bool compute_forces) const {
  if (systems.empty()) return {};
  if (systems.size() == 1) return compute(systems[0], compute_forces);
  return compute_batch(systems, compute_forces);
}

// A device view into a row-major host vector.
template <class T, class V>
static void to_host(std::vector<T>& out, const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const std::size_t n0 = h.extent(0), n1 = h.extent(1);
  out.resize(n0 * n1);
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t c = 0; c < n1; ++c) {
      if constexpr (V::rank == 1) out[i] = h(i);
      else out[i * n1 + c] = h(i, c);
    }
}

Results Calculator::compute_batch(const std::vector<System>& systems, bool compute_forces) const {
  auto& I = *impl_;
  BatchResult br;
  if (I.opts.device_neighbors) {
    DeviceEdgeData dev =
        build_device_batch(systems, I.ckpt.hypers, I.ckpt.species_to_index, I.model.probes(),
                           I.model.n_probes(), I.nbr_ws, I.nbr_edge_map, I.nbr_m_high,
                           I.opts.cache_neighbors ? &I.cache : nullptr);
    br = I.model.energy_forces_batch(dev, compute_forces);
  } else {
    br = I.model.energy_forces_batch(systems, compute_forces);
  }

  Results out;
  for (const auto& s : systems) out.n_atoms.push_back(s.n_atoms);
  to_host(out.energy, br.energy);
  to_host(out.per_atom_energy, br.per_atom);
  if (compute_forces) {
    to_host(out.forces, br.forces);
    to_host(out.struct_id, br.struct_id);
    to_host(out.virial, br.virial);
  }
  return out;
}

// An evaluation on the session's list, back in the engine's order.
Results Calculator::session_results(const DeviceEdgeData& dev, bool compute_forces, bool edge_gradients) const {
  auto& I = *impl_;
  const BatchResult br = I.model.energy_forces_batch(dev, compute_forces);
  Results out;
  out.n_atoms = {dev.n_atoms};
  to_host(out.energy, br.energy);
  to_host(out.per_atom_energy, br.per_atom);
  if (!compute_forces) return out;
  to_host(out.forces, br.forces);
  to_host(out.virial, br.virial);
  out.struct_id.assign(dev.n_atoms, 0);
  if (!edge_gradients) return out;
  // A half list's pair gets both directions: g(i -> j) - g(j -> i).
  const EdgeSession& s = I.md;
  out.edge_gradient.assign(std::size_t(s.n_pairs) * 3, 0.0);
  if (br.edge_grad.extent(0) > 0) {
    std::vector<double> g;
    to_host(g, br.edge_grad);
    for (std::size_t e = 0; e < s.src.size(); ++e)
      for (int c = 0; c < 3; ++c) out.edge_gradient[std::size_t(s.src[e]) * 3 + c] += s.dir[e] * g[e * 3 + c];
  }
  return out;
}

Results Calculator::compute_edges(const EdgeListView& edges, bool compute_forces, bool edge_gradients) const {
  auto& I = *impl_;
  I.md.set(edges, I.ckpt.species_to_index);
  const DeviceEdgeData& dev =
      I.md.step(edges.positions, false, edges.cell, I.ckpt.hypers, I.model.probes(), I.model.n_probes(), false);
  return session_results(dev, compute_forces, edge_gradients);
}

void Calculator::set_neighbors(const EdgeListView& list) { impl_->md.set(list, impl_->ckpt.species_to_index); }

Results Calculator::compute_step(const double* positions, const double* cell, bool compute_forces,
                                 bool edge_gradients) const {
  auto& I = *impl_;
  const DeviceEdgeData& dev =
      I.md.step(positions, false, cell, I.ckpt.hypers, I.model.probes(), I.model.n_probes(), I.opts.md_fixed_shapes);
  return session_results(dev, compute_forces, edge_gradients);
}

Calculator::Totals Calculator::compute_step(const DeviceArrays& a, const double* cell) const {
  using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
  auto& I = *impl_;
  const DeviceEdgeData& dev = I.md.step(a.positions, true, cell, I.ckpt.hypers, I.model.probes(),
                                        I.model.n_probes(), I.opts.md_fixed_shapes);
  const BatchResult br = I.model.energy_forces_batch(dev, true);
  const int N = dev.n_atoms;
  using Out = Kokkos::View<double**, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryUnmanaged>;
  auto forces = br.forces;
  const Out f(a.forces, N, 3);
  Kokkos::parallel_for(
      "md_add_forces", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
        for (int c = 0; c < 3; ++c) f(i, c) += forces(i, c);
      });
  if (a.per_atom_energy) {
    auto e = br.per_atom;
    const Kokkos::View<double*, MemSpace, Kokkos::MemoryUnmanaged> out(a.per_atom_energy, N);
    Kokkos::parallel_for(
        "md_add_energy", RangePolicy(0, N), KOKKOS_LAMBDA(int i) { out(i) += e(i); });
  }
  // Each directed edge's sym(v (x) g), half to each end, gathered per atom: its
  // own edges, their partners (the edges into it), and the partnerless edges
  // into it.
  if (a.per_atom_virial && br.edge_grad.extent(0) > 0) {
    auto g = br.edge_grad;
    auto v = dev.raw_vec;
    auto roff = dev.raw_off, rrev = dev.raw_reverse, ooff = dev.orphan_off, oedge = dev.orphan_edge;
    const bool orphans = oedge.extent(0) > 0;
    const double scale = 0.5 * a.virial_scale;
    const Out w(a.per_atom_virial, N, 6);
    Kokkos::parallel_for(
        "md_atom_virial", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
          double s[6] = {0, 0, 0, 0, 0, 0};
          auto add = [&](int e) {
            s[0] += v(e, 0) * g(e, 0), s[1] += v(e, 1) * g(e, 1), s[2] += v(e, 2) * g(e, 2);
            s[3] += 0.5 * (v(e, 0) * g(e, 1) + v(e, 1) * g(e, 0));
            s[4] += 0.5 * (v(e, 0) * g(e, 2) + v(e, 2) * g(e, 0));
            s[5] += 0.5 * (v(e, 1) * g(e, 2) + v(e, 2) * g(e, 1));
          };
          for (int e = roff(i); e < roff(i + 1); ++e) {
            add(e);
            if (rrev(e) >= 0) add(rrev(e));
          }
          if (orphans)
            for (int k = ooff(i); k < ooff(i + 1); ++k) add(oedge(k));
          for (int k = 0; k < 6; ++k) w(i, k) += scale * s[k];
        });
  }
  Totals t;
  auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), br.energy);
  auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), br.virial);
  t.energy = e(0);
  for (int k = 0; k < 6; ++k) t.virial[k] = w(0, k);
  return t;
}

double Calculator::ghost_cutoff() const {
  const Hypers& h = impl_->ckpt.hypers;
  return (h.num_gnn_layers + (h.adaptive() ? 1 : 0)) * h.cutoff;
}

BatchResult Calculator::compute_device(const DeviceGeom& geom, bool compute_forces) const {
  DeviceEdgeData dev = build_nef_device(
      geom, impl_->ckpt.hypers, impl_->model.probes(), impl_->model.n_probes(), impl_->nbr_ws,
      impl_->nbr_edge_map, impl_->nbr_m_high,
      impl_->opts.cache_neighbors ? &impl_->cache : nullptr);
  return impl_->model.energy_forces_batch(dev, compute_forces);
}

std::size_t Calculator::workspace_bytes() const { return impl_->model.workspace_bytes(); }

std::vector<std::pair<std::string, std::size_t>> Calculator::workspace_breakdown() const {
  return impl_->model.workspace_breakdown();
}

int Calculator::recommended_batch_atoms() const {
  if (impl_->opts.max_batch_atoms > 0) return impl_->opts.max_batch_atoms;

  // Before the first evaluation M is unknown: measured safe sizes on 16 GiB.
  const int bootstrap =
      (impl_->ckpt.hypers.featurizer_type == FeaturizerType::Residual) ? 2048 : 4096;

  const std::size_t pool = impl_->model.workspace_bytes();
  const long slots = impl_->model.peak_edge_slots();
  if (pool == 0 || slots < kMinSampleSlots) return bootstrap;

  const double bytes_per_slot = double(pool) / double(slots);
  const int m = std::max(1, impl_->model.peak_max_neighbors());
  std::size_t free_b = 0, total_b = 0;
  device_memory(impl_->opts, free_b, total_b);

  const double budget =
      std::min(double(free_b) + double(pool), double(total_b) * kMemCardFraction) * kMemHeadroom;

  const long by_memory = static_cast<long>(budget / (bytes_per_slot * m));
  const long by_throughput = kMaxEdgeSlots / m;
  long atoms = std::min(by_memory, by_throughput);
  atoms = std::min<long>(std::max<long>(atoms, 256), 65536);
  return static_cast<int>(atoms);
}

}  // namespace pet
