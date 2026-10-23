// Bit-exact reproducibility, run to run. A relaxation is chaotic: last-bit
// force noise becomes different relaxed structures. So no float atomics feed the
// energy: neighbour slots follow the edge list, force gradients are gathered per
// atom, and per-structure sums are ordered. The model is compared against
// itself, with TF32 off.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/device_cell_list.hpp"
#include "pet/gemm.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// To the bit: a tolerance would pass the bug this exists to catch.
void require_bit_identical(const pet::Results& a, const pet::Results& b, const char* what) {
  INFO(what);
  REQUIRE(a.energy.size() == b.energy.size());
  for (std::size_t i = 0; i < a.energy.size(); ++i) REQUIRE(a.energy[i] == b.energy[i]);
  REQUIRE(a.forces.size() == b.forces.size());
  for (std::size_t i = 0; i < a.forces.size(); ++i) REQUIRE(a.forces[i] == b.forces[i]);
  REQUIRE(a.virial.size() == b.virial.size());
  for (std::size_t i = 0; i < a.virial.size(); ++i) REQUIRE(a.virial[i] == b.virial[i]);
  REQUIRE(a.per_atom_energy.size() == b.per_atom_energy.size());
  for (std::size_t i = 0; i < a.per_atom_energy.size(); ++i)
    REQUIRE(a.per_atom_energy[i] == b.per_atom_energy[i]);
}

constexpr int kRepeats = 4;

}  // namespace

TEST_CASE("repeated evaluation is bit-identical", "[model][determinism]") {
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) {
      WARN("model '" << model << "' is not installed; skipping its determinism check");
      continue;
    }
    pet::Calculator calc(found->first, found->second);
    // The process-global setting, which PET_GEMM could have changed.
    REQUIRE(pet::gemm_mode() == pet::GemmMode::Native);

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results first = calc.compute(g.system, true);
        for (int k = 1; k < kRepeats; ++k)
          require_bit_identical(first, calc.compute(g.system, true), "repeat evaluation");
      }
    }
  }
}

TEST_CASE("a batch gives each structure the same answer as evaluating it alone",
          "[model][determinism]") {
  // Edges never cross structures, so batching must not change any atom's energy
  // or force. Both sides run on the host neighbour path (these structures are
  // small), so this tests the concatenation, not the builders. Not to the bit:
  // the batch pads to its own M and sums per segment, a different fp32 order.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Options host_nef;
    host_nef.device_neighbors = false;
    pet::Calculator calc(found->first, found->second, host_nef);

    std::vector<Golden> goldens;
    for (const auto& path : golden_paths(model)) {
      Golden g = load_golden(path);
      if (g.periodic) goldens.push_back(std::move(g));  // batch path assumes cells
    }
    if (goldens.size() < 2) continue;

    DYNAMIC_SECTION(model << " / batch of " << goldens.size()) {
      std::vector<pet::System> systems;
      for (const auto& g : goldens) systems.push_back(g.system);
      const pet::Results batched = calc.compute(systems, true);

      REQUIRE(batched.energy.size() == goldens.size());
      std::size_t foff = 0;
      for (std::size_t b = 0; b < goldens.size(); ++b) {
        const pet::Results alone = calc.compute(goldens[b].system, true);
        INFO("structure " << b << " (" << goldens[b].name << ")");

        const double de = std::abs(batched.energy[b] - alone.energy[0]);
        INFO("|dE| = " << de << " eV on E = " << alone.energy[0]);
        CHECK(de <= 1e-5 * worst(1.0, std::abs(alone.energy[0])));

        double max_f = 0.0, max_df = 0.0;
        for (std::size_t i = 0; i < alone.forces.size(); ++i) {
          max_f = worst(max_f, std::abs(alone.forces[i]));
          max_df = worst(max_df, std::abs(batched.forces[foff + i] - alone.forces[i]));
        }
        // Alone takes the host neighbour path and the batch the device one, and
        // the vendor BLAS picks its kernels by size: fp32 noise, which a model
        // with several attention layers amplifies (~2e-5 on an MI250X).
        INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
        CHECK(max_df <= 1e-4 * std::max(max_f, 1.0));
        foff += alone.forces.size();
      }
    }
  }
}

TEST_CASE("a batch split to fit memory gives the answer of the whole batch",
          "[model][determinism]") {
  // compute() runs a batch too big for the device as consecutive pieces: by
  // max_batch_atoms, and by a memory budget so small every structure goes alone.
  // The pieces must come back in order, with struct_id renumbered, each structure
  // as the whole batch had it (to fp32 noise: a piece pads to its own M).
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    if (!found) continue;

    std::vector<pet::System> systems;
    for (int copy = 0; copy < 3; ++copy)
      for (const auto& path : golden_paths(model)) {
        Golden g = load_golden(path);
        if (g.periodic) systems.push_back(g.system);
      }
    if (systems.size() < 4) continue;
    int most = 0;
    for (const auto& s : systems) most = std::max(most, s.n_atoms);

    const pet::Results whole = pet::Calculator(found->first, found->second).compute(systems, true);
    pet::Options by_atoms, by_memory;
    by_atoms.max_batch_atoms = 2 * most;
    by_memory.memory_budget_bytes = 1;

    for (const auto& [what, opts] : {std::pair{"max_batch_atoms", by_atoms}, std::pair{"memory budget", by_memory}}) {
      DYNAMIC_SECTION(model << " / split by " << what) {
        const pet::Results split = pet::Calculator(found->first, found->second, opts).compute(systems, true);
        REQUIRE(split.n_atoms == whole.n_atoms);
        REQUIRE(split.struct_id == whole.struct_id);
        REQUIRE(split.virial.size() == whole.virial.size());
        REQUIRE(split.forces.size() == whole.forces.size());
        for (std::size_t b = 0; b < systems.size(); ++b) {
          INFO("structure " << b);
          CHECK(std::abs(split.energy[b] - whole.energy[b]) <= 1e-5 * worst(1.0, std::abs(whole.energy[b])));
        }
        double max_f = 0.0, max_df = 0.0;
        for (std::size_t i = 0; i < whole.forces.size(); ++i) {
          max_f = worst(max_f, std::abs(whole.forces[i]));
          max_df = worst(max_df, std::abs(split.forces[i] - whole.forces[i]));
        }
        INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
        CHECK(max_df <= 1e-4 * std::max(max_f, 1.0));
      }
    }
  }
}

TEST_CASE("a reused Calculator gives the same batched answer every time",
          "[model][determinism]") {
  // Repeated evaluation, with and without forces, must not move: state carried
  // between calls (an unzeroed counter, say) shows up here.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Calculator calc(found->first, found->second);

    std::vector<pet::System> systems;
    for (const auto& path : golden_paths(model)) {
      Golden g = load_golden(path);
      if (g.periodic) systems.push_back(g.system);
    }
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      for (bool forces : {false, true}) {
        const char* what = forces ? "with forces" : "energy only";
        INFO(what);
        const pet::Results first = calc.compute(systems, forces);
        REQUIRE(first.energy.size() == systems.size());
        // Catches that failure even if the first call already had it.
        for (std::size_t b = 0; b < systems.size(); ++b) {
          const double per_atom = first.energy[b] / std::max(1, systems[b].n_atoms);
          INFO("structure " << b << ": " << per_atom << " eV/atom");
          CHECK(std::abs(per_atom) > 1e-6);
        }
        for (int k = 0; k < 3; ++k) {
          const pet::Results again = calc.compute(systems, forces);
          for (std::size_t i = 0; i < first.energy.size(); ++i) {
            INFO("repeat " << k << ", structure " << i);
            CHECK(first.energy[i] == again.energy[i]);
          }
          REQUIRE(first.forces.size() == again.forces.size());
          for (std::size_t i = 0; i < first.forces.size(); ++i)
            CHECK(first.forces[i] == again.forces[i]);
        }
      }
    }
  }
}

TEST_CASE("recomputing activations in the backward changes nothing", "[model][determinism]") {
  // Every recompute tier re-runs the same kernels on the same data: bit-identical.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    auto calc = [&](pet::Recompute r) {
      pet::Options o;
      o.recompute = r;
      return std::make_unique<pet::Calculator>(found->first, found->second, o);
    };
    auto keep = calc(pet::Recompute::Never), wide = calc(pet::Recompute::Wide),
         layers = calc(pet::Recompute::Layers);
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results ref = keep->compute(g.system, true);
        require_bit_identical(ref, wide->compute(g.system, true), "Recompute::Wide");
        require_bit_identical(ref, layers->compute(g.system, true), "Recompute::Layers");
      }
    }
  }
}

TEST_CASE("a replayed CUDA graph gives the eager answer", "[model][determinism]") {
  // Eager, recording and replaying evaluations all equal one without graphs.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Options eager;
    eager.graphs = false;
    pet::Calculator a(found->first, found->second, eager), b(found->first, found->second);
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results ref = a.compute(g.system, true);
        for (int k = 0; k < kRepeats; ++k) require_bit_identical(ref, b.compute(g.system, true), "graph");
      }
    }
  }
}

namespace {

// Each structure's energy, forces and virial from stage + compute_device -- the
// path a relaxation or MD driver steps -- for the leading `lead` of `systems`
// (all of them when 0).
struct Device {
  std::vector<double> energy;
  std::vector<std::vector<double>> forces, virial;
};

Device device_eval(const pet::Calculator& calc, const std::vector<pet::System>& systems, int lead = 0) {
  pet::DeviceGeom g = calc.stage(systems);
  const int B = lead > 0 ? lead : static_cast<int>(systems.size());
  int n = 0;
  for (int b = 0; b < B; ++b) n += systems[b].n_atoms;
  g.B = B, g.Ntot = n;
  const pet::BatchResult r = calc.compute_device(g, true);
  const auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.energy);
  const auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.forces);
  const auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.virial);
  Device d;
  for (int b = 0, off = 0; b < B; off += systems[b++].n_atoms) {
    d.energy.push_back(he(b));
    std::vector<double> f, w;
    for (int i = 0; i < systems[b].n_atoms; ++i)
      for (int c = 0; c < 3; ++c) f.push_back(hf(off + i, c));
    for (int v = 0; v < 6; ++v) w.push_back(hw(b, v));
    d.forces.push_back(std::move(f)), d.virial.push_back(std::move(w));
  }
  return d;
}

// Structure b of `got` against `want`'s structure 0: to the bit where the build
// promises it (batch_invariant_gemms), else to the fp32 noise of a vendor BLAS
// picking its kernels by size.
void require_same_structure(const Device& got, int b, const Device& want, const char* what) {
  INFO(what << ", structure " << b);
  const bool bits = pet::batch_invariant_gemms();
  const double e = want.energy[0], de = std::abs(got.energy[b] - e);
  INFO("|dE| = " << de << " eV on E = " << e);
  if (bits) REQUIRE(got.energy[b] == e);
  else CHECK(de <= 1e-5 * worst(1.0, std::abs(e)));
  double max_f = 0.0, max_df = 0.0;
  for (std::size_t i = 0; i < want.forces[0].size(); ++i) {
    max_f = worst(max_f, std::abs(want.forces[0][i]));
    max_df = worst(max_df, std::abs(got.forces[b][i] - want.forces[0][i]));
    if (bits) REQUIRE(got.forces[b][i] == want.forces[0][i]);
  }
  INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
  if (!bits) CHECK(max_df <= 1e-4 * std::max(max_f, 1.0));
  for (int v = 0; v < 6; ++v)
    if (bits) REQUIRE(got.virial[b][v] == want.virial[0][v]);
}

// s repeated three times along its first lattice vector.
pet::System tripled(const pet::System& s) {
  pet::System t = s;
  t.n_atoms = 3 * s.n_atoms;
  t.atomic_numbers.clear(), t.positions.clear();
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < s.n_atoms; ++i) {
      t.atomic_numbers.push_back(s.atomic_numbers[i]);
      for (int d = 0; d < 3; ++d) t.positions.push_back(s.positions[3 * i + d] + k * s.cell[d]);
    }
  for (int d = 0; d < 3; ++d) t.cell[d] *= 3;
  return t;
}

// The periodic goldens tripled along one axis, each in copies moved apart by a
// few hundredths of an Angstrom, until the batch passes kCellListMinAtoms: so
// the batch takes the cell list, with two bins along that axis at pet-mols-s's
// 4.5 A, while each structure alone is small enough for the brute force (24
// atoms x 3 x 5 x 5 images).
std::vector<pet::System> rattled_batch(const std::string& model) {
  std::vector<pet::System> base;
  for (const auto& path : golden_paths(model)) {
    Golden g = load_golden(path);
    if (g.periodic) base.push_back(tripled(g.system));
  }
  std::vector<pet::System> out;
  int atoms = 0;
  for (int copy = 0; !base.empty() && atoms < 2 * pet::kCellListMinAtoms; ++copy)
    for (const auto& s : base) {
      pet::System r = s;
      for (std::size_t i = 0; i < r.positions.size(); ++i)
        r.positions[i] += 0.03 * std::sin(1.7 * double(i) + 2.3 * copy);
      atoms += r.n_atoms;
      out.push_back(std::move(r));
    }
  return out;
}

}  // namespace

TEST_CASE("a structure's answer does not depend on the batch it is in", "[model][determinism]") {
  // A relaxation packs the structures still running to the front, so a
  // structure's batch-mates and the batch's size change under it; a chaotic
  // optimiser turns last-bit differences into different minima. Alone, in the
  // whole batch, in a leading part of it and with the batch reversed, each
  // structure must get the same answer: to the bit where batch_invariant_gemms.
  for (const auto& model : golden_models()) {
    const pet::Calculator* calc = shared_calculator(model);
    if (!calc) continue;
    const std::vector<pet::System> batch = rattled_batch(model);
    if (batch.size() < 4) continue;
    DYNAMIC_SECTION(model << " / " << batch.size() << " structures") {
      INFO("batch_invariant_gemms() = " << pet::batch_invariant_gemms());
      std::vector<Device> alone;
      for (const auto& s : batch) alone.push_back(device_eval(*calc, {s}));
      const Device whole = device_eval(*calc, batch);
      const int lead = static_cast<int>(batch.size()) / 3;
      const Device front = device_eval(*calc, batch, lead);
      const std::vector<pet::System> reversed(batch.rbegin(), batch.rend());
      const Device back = device_eval(*calc, reversed);
      const int B = static_cast<int>(batch.size());
      for (int b = 0; b < B; ++b) {
        require_same_structure(whole, b, alone[b], "whole batch");
        if (b < lead) require_same_structure(front, b, alone[b], "leading part");
        require_same_structure(back, B - 1 - b, alone[b], "reversed");
      }
    }
  }
}

TEST_CASE("the GEMM's tile does not change the answer", "[model][determinism]") {
  // The batch invariance above rests on this: every CUTLASS tile runs the same
  // per-element FMA chain, so the tile -- chosen by the GEMM's size -- is free.
  if (!pet::batch_invariant_gemms()) {
    WARN("this build's GEMMs are not batch-invariant; skipping the tile check");
    return;
  }
  struct Unpin {
    ~Unpin() { pet::detail::set_cutlass_tile(-1); }
  } unpin;
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    // No graphs: a replayed graph would run the tile it was recorded with.
    pet::Options opts;
    opts.graphs = false;
    const pet::Calculator calc(found->first, found->second, opts);
    const std::vector<pet::System> batch = rattled_batch(model);
    if (batch.empty()) continue;
    DYNAMIC_SECTION(model) {
      pet::detail::set_cutlass_tile(-1);
      const Device chosen = device_eval(calc, batch);
      for (int tile : {0, 1, 2}) {
        INFO("tile " << tile);
        pet::detail::set_cutlass_tile(tile);
        const Device pinned = device_eval(calc, batch);
        for (std::size_t b = 0; b < batch.size(); ++b) {
          REQUIRE(pinned.energy[b] == chosen.energy[b]);
          REQUIRE(pinned.forces[b] == chosen.forces[b]);
          REQUIRE(pinned.virial[b] == chosen.virial[b]);
        }
      }
    }
  }
}
