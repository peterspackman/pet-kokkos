// Engine-supplied neighbour lists (Calculator::compute_edges), the way an MD
// engine calls PET: owned atoms plus ghosts, its own full neighbour list, forces
// on ghosts summed back to their owners. That must reproduce the periodic
// evaluation, and the edge gradients must fold back into the forces and virial.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
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

// Forces and virial folded from edge gradients, as an engine would.
void fold(const pet::EdgeListView& v, const std::vector<double>& g, std::vector<double>& f, double w[6]) {
  f.assign(std::size_t(v.n_atoms) * 3, 0.0);
  double w9[9] = {};
  for (int i = 0; i < v.n_atoms; ++i)
    for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e) {
      const int j = v.neighbors[e];
      double r[3];
      for (int c = 0; c < 3; ++c)
        r[c] = v.vectors ? v.vectors[3 * e + c] : v.positions[3 * j + c] - v.positions[3 * i + c];
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
  for (const auto& model : golden_models()) {
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
  for (const auto& model : golden_models()) {
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
        std::vector<double> vec;
        for (const auto& e : raw) {
          ++off[e.i + 1], nbr.push_back(e.j);
          sh.insert(sh.end(), {e.sa, e.sb, e.sc}), vec.insert(vec.end(), {e.vx, e.vy, e.vz});
        }
        for (int i = 0; i < g.system.n_atoms; ++i) off[i + 1] += off[i];
        pet::EdgeListView v;
        v.n_atoms = g.system.n_atoms;
        v.positions = g.system.positions.data(), v.atomic_numbers = g.system.atomic_numbers.data();
        v.offsets = off.data(), v.neighbors = nbr.data(), v.vectors = vec.data(), v.shifts = sh.data();
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
