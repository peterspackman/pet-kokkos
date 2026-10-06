// Engine-supplied neighbour lists (Calculator::compute_edges), the way an MD
// engine calls PET: owned atoms plus ghosts, its own full neighbour list, forces
// on ghosts summed back to their owners. That must reproduce the periodic
// evaluation, and the edge gradients must fold back into the forces and virial.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// A periodic structure as an engine's domain: its atoms owned, every image within
// `reach` of them a ghost, and neighbour lists out to `list_cut` -- only for the
// atoms within `listed` of an owned atom, which is all the contract asks for.
struct Domain {
  std::vector<double> pos;
  std::vector<int> z, owner, off{0}, nbr;
  std::vector<double> depth;  // distance to the nearest owned atom
  int n_local = 0;
};

Domain make_domain(const pet::System& s, double reach, double listed, double list_cut) {
  Domain d;
  d.n_local = s.n_atoms;
  for (int i = 0; i < s.n_atoms; ++i) {
    d.pos.insert(d.pos.end(), {s.positions[3 * i], s.positions[3 * i + 1], s.positions[3 * i + 2]});
    d.z.push_back(s.atomic_numbers[i]), d.owner.push_back(i), d.depth.push_back(0.0);
  }
  const auto rng = pet::detail::image_ranges(s, reach);
  for (int a = -rng[0]; a <= rng[0]; ++a)
    for (int b = -rng[1]; b <= rng[1]; ++b)
      for (int c = -rng[2]; c <= rng[2]; ++c) {
        if (!a && !b && !c) continue;
        for (int j = 0; j < s.n_atoms; ++j) {
          double p[3];
          for (int x = 0; x < 3; ++x)
            p[x] = s.positions[3 * j + x] + a * s.cell[x] + b * s.cell[3 + x] + c * s.cell[6 + x];
          double m2 = 1e300;
          for (int i = 0; i < s.n_atoms; ++i) {
            double r2 = 0;
            for (int x = 0; x < 3; ++x) r2 += (p[x] - s.positions[3 * i + x]) * (p[x] - s.positions[3 * i + x]);
            m2 = std::min(m2, r2);
          }
          if (m2 > reach * reach) continue;
          d.pos.insert(d.pos.end(), {p[0], p[1], p[2]});
          d.z.push_back(s.atomic_numbers[j]), d.owner.push_back(j), d.depth.push_back(std::sqrt(m2));
        }
      }
  const int n = d.z.size();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n && d.depth[i] <= listed; ++j) {
      double r2 = 0;
      for (int x = 0; x < 3; ++x) r2 += (d.pos[3 * j + x] - d.pos[3 * i + x]) * (d.pos[3 * j + x] - d.pos[3 * i + x]);
      if (j != i && r2 <= list_cut * list_cut) d.nbr.push_back(j);
    }
    d.off.push_back(d.nbr.size());
  }
  return d;
}

pet::EdgeListView view(const Domain& d) {
  pet::EdgeListView v;
  v.n_atoms = d.z.size(), v.n_local = d.n_local;
  v.positions = d.pos.data(), v.atomic_numbers = d.z.data(), v.offsets = d.off.data(), v.neighbors = d.nbr.data();
  return v;
}

// Each listed pair's vector r_j + shift . cell - r_i.
std::vector<double> vectors(const pet::EdgeListView& v) {
  std::vector<double> r;
  for (int i = 0; i < v.n_atoms; ++i)
    for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e)
      for (int c = 0; c < 3; ++c) {
        double x = v.positions[3 * v.neighbors[e] + c] - v.positions[3 * i + c];
        if (v.shifts)
          for (int k = 0; k < 3; ++k) x += v.shifts[3 * e + k] * v.cell[3 * k + c];
        r.push_back(x);
      }
  return r;
}

// Forces and virial folded from edge gradients, as an engine would; a half
// list's gradient already holds both directions.
void fold(const pet::EdgeListView& v, const std::vector<double>& g, std::vector<double>& f, double w[6]) {
  f.assign(std::size_t(v.n_atoms) * 3, 0.0);
  const std::vector<double> vec = vectors(v);
  double w9[9] = {};
  for (int i = 0; i < v.n_atoms; ++i)
    for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e) {
      const int j = v.neighbors[e];
      double r[3];
      for (int c = 0; c < 3; ++c) r[c] = vec[3 * e + c];
      for (int c = 0; c < 3; ++c) f[3 * i + c] += g[3 * e + c], f[3 * j + c] -= g[3 * e + c];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) w9[3 * a + b] += r[a] * g[3 * e + b];
    }
  const double v6[6] = {w9[0], w9[4], w9[8], 0.5 * (w9[1] + w9[3]), 0.5 * (w9[2] + w9[6]), 0.5 * (w9[5] + w9[7])};
  std::copy(v6, v6 + 6, w);
}

double max_abs(const std::vector<double>& a) {
  double m = 0;
  for (double x : a) m = worst(m, std::fabs(x));
  return m;
}

}  // namespace

TEST_CASE("an engine's owned and ghost atoms reproduce the periodic evaluation", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    // Its own, so a domain's large workspace goes with it.
    auto owned = std::make_unique<pet::Calculator>(found->first, found->second);
    auto* calc = owned.get();
    const Golden* g = nullptr;
    Golden gold;
    for (const auto& path : golden_paths(model))
      if ((gold = load_golden(path)).periodic) {
        g = &gold;
        break;
      }
    if (!g) continue;

    // The ghosts reach every message-passing layer's cutoff: thousands of atoms
    // for a 7.5 A model, minutes on a CPU. The exchange test covers the same
    // physics with a single cutoff of ghosts.
    if (kHostBackend && calc->ghost_cutoff() > 10.0) {
      WARN(model << ": a " << calc->ghost_cutoff() << " A ghost shell is skipped on a host backend");
      continue;
    }

    DYNAMIC_SECTION(model << " / " << g->name) {
      const pet::System& s = g->system;
      const double gc = calc->ghost_cutoff(), rc = calc->cutoff();
      const Domain d = make_domain(s, gc, gc - rc, rc + 0.5);
      const pet::EdgeListView v = view(d);
      const pet::Results r = calc->compute_edges(v, true), ref = calc->compute(s, true);

      // Energy: the owned atoms'.
      INFO(d.z.size() - d.n_local << " ghosts; E " << r.energy[0] << " vs " << ref.energy[0]);
      CHECK(std::fabs(r.energy[0] - ref.energy[0]) <= 1e-5 * std::fabs(ref.energy[0]));

      // Forces: every copy of an atom summed onto it.
      std::vector<double> f(std::size_t(s.n_atoms) * 3, 0.0);
      for (std::size_t a = 0; a < d.owner.size(); ++a)
        for (int c = 0; c < 3; ++c) f[3 * d.owner[a] + c] += r.forces[3 * a + c];
      double df = 0;
      for (std::size_t i = 0; i < f.size(); ++i) df = worst(df, std::fabs(f[i] - ref.forces[i]));
      INFO("max |dF| " << df << " of max |F| " << max_abs(ref.forces));
      CHECK(df <= 1e-4 * std::max(1.0, max_abs(ref.forces)));

      double dw = 0;
      for (int k = 0; k < 6; ++k) dw = worst(dw, std::fabs(r.virial[k] - ref.virial[k]));
      INFO("max |dW| " << dw << " of max |W| " << max_abs(ref.virial));
      CHECK(dw <= 1e-4 * std::max(1.0, max_abs(ref.virial)));

      // The edge gradients are what the forces and virial are folded from.
      std::vector<double> ff;
      double wf[6];
      fold(v, r.edge_gradient, ff, wf);
      double dff = 0, dwf = 0;
      for (std::size_t i = 0; i < ff.size(); ++i) dff = worst(dff, std::fabs(ff[i] - r.forces[i]));
      for (int k = 0; k < 6; ++k) dwf = worst(dwf, std::fabs(wf[k] - r.virial[k]));
      INFO("fold: max |dF| " << dff << ", max |dW| " << dwf);
      CHECK(dff <= 1e-9 * std::max(1.0, max_abs(r.forces)));
      CHECK(dwf <= 1e-9 * std::max(1.0, max_abs(r.virial)));
    }
  }
}

TEST_CASE("a periodic engine's edges with shifts reproduce the periodic evaluation", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    auto* calc = shared_calculator(model);
    if (!calc) continue;
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      if (!g.periodic) continue;
      DYNAMIC_SECTION(model << " / " << g.name) {
        // The host search's list, as a periodic engine without ghosts would give
        // it: every atom owned, images told apart by their shifts.
        const auto raw = pet::detail::build_raw_edges(g.system, calc->cutoff() + 0.5);
        std::vector<int> off(g.system.n_atoms + 1, 0), nbr, sh;
        for (const auto& e : raw) ++off[e.i + 1], nbr.push_back(e.j), sh.insert(sh.end(), {e.sa, e.sb, e.sc});
        for (int i = 0; i < g.system.n_atoms; ++i) off[i + 1] += off[i];
        pet::EdgeListView v;
        v.n_atoms = g.system.n_atoms;
        v.positions = g.system.positions.data(), v.atomic_numbers = g.system.atomic_numbers.data();
        v.offsets = off.data(), v.neighbors = nbr.data(), v.shifts = sh.data(), v.cell = g.system.cell.data();
        const pet::Results r = calc->compute_edges(v, true), ref = calc->compute(g.system, true);
        CHECK(std::fabs(r.energy[0] - ref.energy[0]) <= 1e-5 * std::fabs(ref.energy[0]));
        double df = 0;
        for (std::size_t i = 0; i < r.forces.size(); ++i) df = worst(df, std::fabs(r.forces[i] - ref.forces[i]));
        INFO("max |dF| " << df);
        CHECK(df <= 1e-4 * std::max(1.0, max_abs(ref.forces)));
      }
      break;
    }
  }
}

namespace {

// A periodic structure's list as a periodic engine keeps it: every atom owned,
// pairs out to cutoff + skin with their shifts, full or half.
struct PeriodicList {
  std::vector<int> off, nbr, sh;
  bool half = false;
  pet::EdgeListView view(const pet::System& s) const {
    pet::EdgeListView v;
    v.n_atoms = s.n_atoms;
    v.positions = s.positions.data(), v.atomic_numbers = s.atomic_numbers.data(), v.cell = s.cell.data();
    v.offsets = off.data(), v.neighbors = nbr.data(), v.shifts = sh.data(), v.half = half;
    return v;
  }
};

PeriodicList periodic_list(const pet::System& s, double reach, bool half) {
  PeriodicList l;
  l.half = half;
  l.off.assign(s.n_atoms + 1, 0);
  for (const auto& e : pet::detail::build_raw_edges(s, reach)) {
    // Half: each pair from one end, the later atom or the positive image.
    const bool first = e.i < e.j || (e.i == e.j && std::make_tuple(e.sa, e.sb, e.sc) > std::make_tuple(0, 0, 0));
    if (half && !first) continue;
    ++l.off[e.i + 1], l.nbr.push_back(e.j), l.sh.insert(l.sh.end(), {e.sa, e.sb, e.sc});
  }
  for (int i = 0; i < s.n_atoms; ++i) l.off[i + 1] += l.off[i];
  return l;
}

const Golden* first_periodic(const std::string& model, Golden& store) {
  for (const auto& path : golden_paths(model))
    if ((store = load_golden(path)).periodic) return &store;
  return nullptr;
}

// Largest difference over the reference's scale, floored at 1: small forces
// carry fp32 noise of about 1e-6 whatever their size.
double rel(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d = worst(d, std::fabs(a[i] - b[i])), m = worst(m, std::fabs(b[i]));
  return d / std::max(m, 1.0);
}

}  // namespace

TEST_CASE("a half list gives what the full list does", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    auto* calc = shared_calculator(model);
    Golden store;
    const Golden* g = calc ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    DYNAMIC_SECTION(model) {
      const pet::System& s = g->system;
      const PeriodicList full = periodic_list(s, calc->cutoff() + 0.5, false), half = periodic_list(s, calc->cutoff() + 0.5, true);
      const pet::Results a = calc->compute_edges(full.view(s)), b = calc->compute_edges(half.view(s));
      INFO("energy rel " << rel(b.energy, a.energy) << ", forces rel " << rel(b.forces, a.forces));
      CHECK(rel(b.energy, a.energy) <= 1e-6);
      CHECK(rel(b.forces, a.forces) <= 1e-4);
      std::vector<double> f;
      double w[6];
      fold(half.view(s), b.edge_gradient, f, w);
      INFO("half fold: forces rel " << rel(f, b.forces));
      CHECK(rel(f, b.forces) <= 1e-9);
    }
  }
}

TEST_CASE("a stepping engine's list follows its atoms, graphs or not", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    Golden store;
    const Golden* g = found ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    for (int f = 0; f < 2; ++f) {
      const bool fixed = f == 1;
      DYNAMIC_SECTION(model << (fixed ? " / fixed shapes" : "")) {
        pet::Options graphs, eager;
        graphs.md_fixed_shapes = eager.md_fixed_shapes = fixed;
        eager.graphs = false;
        pet::Calculator stepper(found->first, found->second, graphs), plain(found->first, found->second, eager),
            fresh(found->first, found->second);
        pet::System s = g->system;
        const PeriodicList list = periodic_list(s, stepper.cutoff() + 1.0, true);  // a 1 A skin
        stepper.set_neighbors(list.view(s));
        plain.set_neighbors(list.view(s));
        // Small moves, then a 3% compression that brings new pairs inside the
        // cutoff -- with fixed shapes, past the capacity the first step chose.
        const pet::System s0 = s;
        for (int step = 0; step < 6; ++step) {
          const double scale = step == 4 ? 0.97 : 1.0;
          for (std::size_t i = 0; i < s.positions.size(); ++i)
            s.positions[i] = scale * s0.positions[i] + 0.02 * std::sin(1.7 * i + step);
          for (int k = 0; k < 9; ++k) s.cell[k] = scale * s0.cell[k];
          const pet::Results a = stepper.compute_step(s.positions.data(), s.cell.data(), true, true);
          const pet::Results b = plain.compute_step(s.positions.data(), s.cell.data(), true, true);
          const pet::Results c = fresh.compute_edges(list.view(s));
          INFO("step " << step << ": vs a fresh list, energy rel " << rel(a.energy, c.energy) << ", forces rel "
                       << rel(a.forces, c.forces));
          CHECK(rel(a.energy, c.energy) <= 1e-6);
          CHECK(rel(a.forces, c.forces) <= 1e-4);
          CHECK(rel(a.edge_gradient, c.edge_gradient) <= 1e-4);
          // A replayed graph is the same computation to the bit.
          CHECK(a.energy == b.energy);
          CHECK(a.forces == b.forces);
        }
      }
    }
  }
}

TEST_CASE("a device-resident engine gets what the host one does", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    Golden store;
    const Golden* g = found ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    DYNAMIC_SECTION(model) {
      pet::Calculator host(found->first, found->second), device(found->first, found->second);
      const pet::System& s = g->system;
      const int n = s.n_atoms;
      const PeriodicList list = periodic_list(s, host.cutoff() + 1.0, true);
      host.set_neighbors(list.view(s));
      device.set_neighbors(list.view(s));
      const pet::Results r = host.compute_step(s.positions.data(), s.cell.data(), true, true);

      using D2 = Kokkos::View<double**, Kokkos::LayoutRight, pet::MemSpace>;
      D2 x("x", n, 3), f("f", n, 3), w("w", n, 6);
      Kokkos::View<double*, pet::MemSpace> e("e", n);
      Kokkos::deep_copy(x, Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(s.positions.data(), n, 3));
      pet::Calculator::DeviceArrays a;
      a.positions = x.data(), a.forces = f.data(), a.per_atom_energy = e.data(), a.per_atom_virial = w.data();
      const pet::Calculator::Totals t = device.compute_step(a, s.cell.data());
      auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
      auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e);
      auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);

      CHECK(t.energy == r.energy[0]);
      for (int k = 0; k < 6; ++k) CHECK(t.virial[k] == r.virial[k]);
      // No forces pointer: the energy alone, without the backward pass.
      pet::Calculator::DeviceArrays eo;
      eo.positions = x.data();
      CHECK(device.compute_step(eo, s.cell.data()).energy == r.energy[0]);
      std::vector<double> fd(hf.data(), hf.data() + 3 * n), ed(he.data(), he.data() + n);
      CHECK(fd == r.forces);
      CHECK(ed == r.per_atom_energy);

      // The per-atom virial: the host's edge gradients split per pair, and in
      // sum the virial.
      std::vector<double> want(6 * n, 0.0), sum(6, 0.0);
      const pet::EdgeListView v = list.view(s);
      const std::vector<double> vec = vectors(v);
      for (int i = 0; i < n; ++i)
        for (int k = v.offsets[i]; k < v.offsets[i + 1]; ++k) {
          const double *d = &vec[3 * k], *G = &r.edge_gradient[3 * k];
          const double p[6] = {d[0] * G[0], d[1] * G[1], d[2] * G[2], 0.5 * (d[0] * G[1] + d[1] * G[0]),
                               0.5 * (d[0] * G[2] + d[2] * G[0]), 0.5 * (d[1] * G[2] + d[2] * G[1])};
          for (int c = 0; c < 6; ++c) want[6 * i + c] += 0.5 * p[c], want[6 * v.neighbors[k] + c] += 0.5 * p[c];
        }
      std::vector<double> got(hw.data(), hw.data() + 6 * n);
      for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) sum[c] += got[6 * i + c];
      INFO("per-atom virial rel " << rel(got, want) << ", its sum vs the virial rel " << rel(sum, r.virial));
      CHECK(rel(got, want) <= 1e-9);
      CHECK(rel(sum, r.virial) <= 1e-9);
    }
  }
}

namespace {

// Other ranks, played in-process: every ghost is a periodic image of an owned
// atom, so its owner is that atom, and the partner of an edge i -> g is the edge
// owner(g) -> (an image of i) with the opposite vector.
class ImageExchange : public pet::Exchange {
 public:
  explicit ImageExchange(const Domain& d) : owner_(d.owner), n_local_(d.n_local) {
    struct Edge { int i, j; double v[3]; };
    std::vector<Edge> remote;
    for (int i = 0; i < d.n_local; ++i)
      for (int e = d.off[i]; e < d.off[i + 1]; ++e)
        if (d.nbr[e] >= d.n_local) {
          Edge x{i, d.nbr[e], {}};
          for (int c = 0; c < 3; ++c) x.v[c] = d.pos[3 * x.j + c] - d.pos[3 * i + c];
          remote.push_back(x);
        }
    partner_.assign(remote.size(), -1);
    for (std::size_t r = 0; r < remote.size(); ++r)
      for (std::size_t q = 0; q < remote.size(); ++q) {
        const Edge &a = remote[r], &b = remote[q];
        if (b.i != owner_[a.j] || owner_[b.j] != a.i) continue;
        if (std::fabs(a.v[0] + b.v[0]) + std::fabs(a.v[1] + b.v[1]) + std::fabs(a.v[2] + b.v[2]) < 1e-9) partner_[r] = q;
      }
  }
  void atoms_forward(pet::RView1D a) override {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    for (std::size_t g = n_local_; g < owner_.size(); ++g) h(g) = h(owner_[g]);
    Kokkos::deep_copy(a, h);
  }
  void atoms_reverse(pet::RView1D a) override {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    for (std::size_t g = n_local_; g < owner_.size(); ++g) h(owner_[g]) += h(g), h(g) = 0;
    Kokkos::deep_copy(a, h);
  }
  void set_live(pet::IView1D live) override {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), live);
    live_.assign(h.data(), h.data() + h.extent(0));
    row_.assign(partner_.size(), -1);
    for (std::size_t l = 0; l < live_.size(); ++l) row_[live_[l]] = l;
  }
  // Live row l is remote edge live_[l]; its partner's row is the partner edge's.
  void edges(pet::View2D out, pet::View2D in) override {
    auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    auto i = Kokkos::create_mirror_view(in);
    for (std::size_t l = 0; l < live_.size(); ++l) {
      const int q = partner_[live_[l]], lq = q >= 0 ? row_[q] : -1;
      for (std::size_t c = 0; c < o.extent(1); ++c) i(l, c) = lq >= 0 && !mute ? o(lq, c) : 0;
    }
    Kokkos::deep_copy(in, i);
  }
  bool paired() const { return std::find(partner_.begin(), partner_.end(), -1) == partner_.end(); }
  bool mute = false;  // drop the edge rows: what a ghost shell without the exchange would see

 private:
  std::vector<int> owner_, partner_;
  std::vector<int> live_, row_;  // this evaluation's live remote edges, and each one's row
  std::size_t n_local_;
};

}  // namespace

TEST_CASE("ranks that exchange messages reproduce the periodic evaluation", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    Golden store;
    const Golden* g = found ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    pet::Calculator calc(found->first, found->second);
    if (calc.hypers().featurizer_type != pet::FeaturizerType::FeedForward) continue;
    DYNAMIC_SECTION(model) {
      const pet::System& s = g->system;
      const double rc = calc.cutoff();
      // Ghosts one cutoff (and a skin) deep, and neighbour lists for owned atoms only.
      const Domain d = make_domain(s, rc + 0.5, 0.0, rc + 0.5);
      ImageExchange x(d);
      REQUIRE(x.paired());
      pet::EdgeListView v = view(d);
      v.exchange = &x;
      calc.set_neighbors(v);
      const pet::Results r = calc.compute_step(d.pos.data(), nullptr, true, true), ref = calc.compute(s, true);

      INFO(d.z.size() - d.n_local << " ghosts; E " << r.energy[0] << " vs " << ref.energy[0]);
      CHECK(std::fabs(r.energy[0] - ref.energy[0]) <= 1e-5 * std::fabs(ref.energy[0]));
      std::vector<double> f(std::size_t(s.n_atoms) * 3, 0.0);
      for (std::size_t a = 0; a < d.owner.size(); ++a)
        for (int c = 0; c < 3; ++c) f[3 * d.owner[a] + c] += r.forces[3 * a + c];
      INFO("forces rel " << rel(f, ref.forces) << ", virial rel " << rel(r.virial, ref.virial));
      CHECK(rel(f, ref.forces) <= 1e-4);
      CHECK(rel(r.virial, ref.virial) <= 1e-4);

      // And the exchange is what makes it so.
      x.mute = true;
      const pet::Results m = calc.compute_step(d.pos.data(), nullptr, true, false);
      INFO("without the edge rows, E " << m.energy[0]);
      CHECK(std::fabs(m.energy[0] - ref.energy[0]) > 1e-3 * std::fabs(ref.energy[0]));
    }
  }
}

namespace {

// A domain's list as LAMMPS's KOKKOS package holds it on the device: ilist,
// numneigh by atom, and a 2-D neighbour array, column-major (as on CUDA) and
// with a bit above the index that the mask must drop.
struct DeviceList {
  using IV = Kokkos::View<int*, pet::MemSpace>;
  IV ilist, numneigh, neighbors, z, image_of;
  Kokkos::View<double*, pet::MemSpace> pos;
  int n_all = 0, width = 0;
  static constexpr int kSpecial = 1 << 30, kMask = (1 << 29) - 1;

  explicit DeviceList(const Domain& d, int centres) {
    n_all = d.z.size();
    for (int i = 0; i < centres; ++i) width = std::max(width, d.off[i + 1] - d.off[i]);
    std::vector<int> il(centres), nn(n_all, 0), nb(std::size_t(n_all) * std::max(width, 1), -1);
    for (int i = 0; i < centres; ++i) {
      il[i] = i, nn[i] = d.off[i + 1] - d.off[i];
      for (int k = 0; k < nn[i]; ++k) nb[i + std::size_t(k) * n_all] = d.nbr[d.off[i] + k] | kSpecial;
    }
    ilist = upload(il), numneigh = upload(nn), neighbors = upload(nb), z = upload(d.z), image_of = upload(d.owner);
    pos = Kokkos::View<double*, pet::MemSpace>("pos", d.pos.size());
    Kokkos::deep_copy(pos, Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(d.pos.data(), d.pos.size()));
  }
  pet::DeviceEdgeListView view(int n_atoms, int n_local, int centres) const {
    pet::DeviceEdgeListView v;
    v.n_atoms = n_atoms, v.n_local = n_local, v.n_centres = centres;
    v.ilist = ilist.data(), v.numneigh = numneigh.data(), v.neighbors = neighbors.data();
    v.stride_i = 1, v.stride_k = n_all, v.mask = kMask;
    v.atomic_numbers = z.data();
    return v;
  }
  static IV upload(const std::vector<int>& h) {
    IV d("v", h.size());
    Kokkos::deep_copy(d, Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h.data(), h.size()));
    return d;
  }
};

void require_same(const pet::Results& a, const pet::Results& b) {
  REQUIRE(a.energy == b.energy);
  REQUIRE(a.forces == b.forces);
  REQUIRE(a.virial == b.virial);
  REQUIRE(a.per_atom_energy == b.per_atom_energy);
}

}  // namespace

TEST_CASE("a list set on the device gives what the same list set from the host does", "[model][edges]") {
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    Golden store;
    const Golden* g = found ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    pet::Calculator calc(found->first, found->second);
    const pet::System& s = g->system;
    const double rc = calc.cutoff();
    const Domain d = make_domain(s, rc + 0.5, 0.0, rc + 0.5);  // lists for the owned atoms only
    const DeviceList dl(d, d.n_local);

    DYNAMIC_SECTION(model << " / images") {
      // The host side: each ghost replaced by its owner, with the shift between them.
      double inv[9];
      const double* c = s.cell.data();
      const double det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
      const double adj[9] = {c[4] * c[8] - c[5] * c[7], c[2] * c[7] - c[1] * c[8], c[1] * c[5] - c[2] * c[4],
                             c[5] * c[6] - c[3] * c[8], c[0] * c[8] - c[2] * c[6], c[2] * c[3] - c[0] * c[5],
                             c[3] * c[7] - c[4] * c[6], c[1] * c[6] - c[0] * c[7], c[0] * c[4] - c[1] * c[3]};
      for (int k = 0; k < 9; ++k) inv[k] = adj[k] / det;
      std::vector<int> nbr, shifts;
      for (int i = 0; i < d.n_local; ++i)
        for (int e = d.off[i]; e < d.off[i + 1]; ++e) {
          const int j = d.nbr[e], o = d.owner[j];
          double r[3];
          for (int x = 0; x < 3; ++x) r[x] = d.pos[3 * j + x] - d.pos[3 * o + x];
          nbr.push_back(o);
          for (int x = 0; x < 3; ++x)
            shifts.push_back((int) std::lround(r[0] * inv[x] + r[1] * inv[3 + x] + r[2] * inv[6 + x]));
        }
      pet::EdgeListView hv;
      hv.n_atoms = d.n_local, hv.positions = d.pos.data(), hv.atomic_numbers = d.z.data();
      hv.offsets = d.off.data(), hv.neighbors = nbr.data(), hv.shifts = shifts.data(), hv.cell = c;
      calc.set_neighbors(hv);
      const pet::Results host = calc.compute_step(d.pos.data(), c, true, true);

      pet::DeviceEdgeListView dv = dl.view(d.n_local, -1, d.n_local);
      dv.image_of = dl.image_of.data(), dv.positions = dl.pos.data(), dv.cell = c;
      calc.set_neighbors(dv);
      const pet::Results device = calc.compute_step(d.pos.data(), c, true, true);
      require_same(host, device);
      REQUIRE(host.edge_gradient == device.edge_gradient);
    }

    DYNAMIC_SECTION(model << " / exchange") {
      if (calc.hypers().featurizer_type != pet::FeaturizerType::FeedForward) continue;
      ImageExchange x(d);
      pet::EdgeListView hv = view(d);
      hv.exchange = &x;
      calc.set_neighbors(hv);
      const pet::Results host = calc.compute_step(d.pos.data(), nullptr, true, false);

      pet::DeviceEdgeListView dv = dl.view(d.z.size(), d.n_local, d.n_local);
      dv.exchange = &x;
      calc.set_neighbors(dv);
      const pet::Results device = calc.compute_step(d.pos.data(), nullptr, true, false);
      require_same(host, device);
    }
  }
}

TEST_CASE("a device-resident engine's recorded steps are its eager ones", "[model][edges]") {
  // Fixed shapes record the whole step (geometry, network, totals) as one graph
  // and replay it: every step must be the same shapes run eagerly, to the bit,
  // through moves, a compression that outgrows the capacities, and a new cell.
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    Golden store;
    const Golden* g = found ? first_periodic(model, store) : nullptr;
    if (!g) continue;
    DYNAMIC_SECTION(model) {
      pet::Options fixed, eager;
      fixed.md_fixed_shapes = eager.md_fixed_shapes = true;
      eager.graphs = false;
      pet::Calculator a(found->first, found->second, fixed), b(found->first, found->second, eager);
      pet::System s = g->system;
      const int n = s.n_atoms;
      const PeriodicList list = periodic_list(s, a.cutoff() + 1.0, true);
      a.set_neighbors(list.view(s));
      b.set_neighbors(list.view(s));
      using D2 = Kokkos::View<double**, Kokkos::LayoutRight, pet::MemSpace>;
      D2 x("x", n, 3), fa("fa", n, 3), fb("fb", n, 3);
      const pet::System s0 = s;
      for (int step = 0; step < 8; ++step) {
        const double scale = step == 5 ? 0.97 : 1.0;
        for (std::size_t i = 0; i < s.positions.size(); ++i)
          s.positions[i] = scale * s0.positions[i] + 0.02 * std::sin(1.3 * i + step);
        for (int k = 0; k < 9; ++k) s.cell[k] = scale * s0.cell[k];
        Kokkos::deep_copy(x, Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace,
                                          Kokkos::MemoryUnmanaged>(s.positions.data(), n, 3));
        Kokkos::deep_copy(fa, 0.0), Kokkos::deep_copy(fb, 0.0);
        pet::Calculator::DeviceArrays da, db;
        da.positions = db.positions = x.data();
        da.forces = fa.data(), db.forces = fb.data();
        const pet::Calculator::Totals ta = a.compute_step(da, s.cell.data()), tb = b.compute_step(db, s.cell.data());
        auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fa);
        auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fb);
        INFO("step " << step);
        CHECK(ta.energy == tb.energy);
        for (int k = 0; k < 6; ++k) CHECK(ta.virial[k] == tb.virial[k]);
        bool same = true;
        for (int i = 0; i < n; ++i)
          for (int c = 0; c < 3; ++c) same = same && ha(i, c) == hb(i, c);
        CHECK(same);
      }
    }
  }
}
