// D3 in a Calculator (Options::d3): every entry point adds exactly what the
// D3 alone gives -- the model's part is the same bits with D3 on or off, so
// the difference is D3's to ~1e-10 -- including an engine's views of a
// periodic structure: its images (shifts), owned atoms and a ghost shell, and
// ghosts one cutoff deep with an exchange.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/d3.hpp"
#include "pet/neighbors.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

pet::D3Settings settings(bool tail) {
  pet::D3Settings s = pet::D3Settings::functional("pbesol", pet::D3Damping::BJ, true);
  s.cutoffs.pair = 8.0, s.cutoffs.cn = 6.0, s.cutoffs.atm = 5.0;  // short: ghost shells stay small
  s.cutoffs.tail = tail;
  return s;
}

struct Pair {
  std::unique_ptr<pet::Calculator> plain, with;
};
Pair calculators(const std::string& model, bool tail) {
  const auto found = find_model(model);
  if (!found) return {};
  pet::Options o;
  Pair p;
  p.plain = std::make_unique<pet::Calculator>(found->first, found->second, o);
  o.d3 = settings(tail);
  p.with = std::make_unique<pet::Calculator>(found->first, found->second, o);
  return p;
}

// a - b against d, entry by entry: what D3 added against D3 alone.
void require_added(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& d,
                   const char* what, double tol = 1e-9) {
  INFO(what);
  REQUIRE(a.size() == b.size());
  REQUIRE(a.size() == d.size());
  double err = 0, scale = 0;
  for (std::size_t k = 0; k < a.size(); ++k) err = worst(err, std::fabs(a[k] - b[k] - d[k])), scale = std::max(scale, std::fabs(d[k]));
  INFO("max error " << err << " of " << scale);
  CHECK(err <= tol * std::max(scale, 1e-2));
}

const Golden* periodic_golden(const std::string& model, Golden& store) {
  for (const auto& path : golden_paths(model))
    if ((store = load_golden(path)).periodic) return &store;
  return nullptr;
}

// Every image of a periodic structure's atoms within `reach` of them, as an
// engine's ghosts, with neighbour lists to `list_cut` for the atoms within
// `listed` of an owned one.
struct Domain {
  std::vector<double> pos;
  std::vector<int> z, owner, off{0}, nbr;
  int n_local = 0;
};
Domain make_domain(const pet::System& s, double reach, double listed, double list_cut) {
  Domain d;
  d.n_local = s.n_atoms;
  std::vector<double> depth;
  for (int i = 0; i < s.n_atoms; ++i) {
    d.pos.insert(d.pos.end(), {s.positions[3 * i], s.positions[3 * i + 1], s.positions[3 * i + 2]});
    d.z.push_back(s.atomic_numbers[i]), d.owner.push_back(i), depth.push_back(0.0);
  }
  const auto rng = pet::detail::image_ranges(s, reach);
  for (int a = -rng[0]; a <= rng[0]; ++a)
    for (int b = -rng[1]; b <= rng[1]; ++b)
      for (int c = -rng[2]; c <= rng[2]; ++c) {
        if (!a && !b && !c) continue;
        for (int j = 0; j < s.n_atoms; ++j) {
          double p[3];
          for (int x = 0; x < 3; ++x) p[x] = s.positions[3 * j + x] + a * s.cell[x] + b * s.cell[3 + x] + c * s.cell[6 + x];
          double m2 = 1e300;
          for (int i = 0; i < s.n_atoms; ++i) {
            double r2 = 0;
            for (int x = 0; x < 3; ++x) r2 += (p[x] - s.positions[3 * i + x]) * (p[x] - s.positions[3 * i + x]);
            m2 = std::min(m2, r2);
          }
          if (m2 > reach * reach) continue;
          d.pos.insert(d.pos.end(), {p[0], p[1], p[2]});
          d.z.push_back(s.atomic_numbers[j]), d.owner.push_back(j), depth.push_back(std::sqrt(m2));
        }
      }
  const int n = d.z.size();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n && depth[i] <= listed; ++j) {
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
// Every copy's force summed onto its atom.
std::vector<double> folded(const Domain& d, const std::vector<double>& f) {
  std::vector<double> o(std::size_t(d.n_local) * 3, 0.0);
  for (std::size_t a = 0; a < d.owner.size(); ++a)
    for (int c = 0; c < 3; ++c) o[3 * d.owner[a] + c] += f[3 * a + c];
  return o;
}

// Other ranks in-process, for atom values only (D3's moves): a ghost's owner
// is the atom it is an image of. PET's edge rows are not needed when each
// owned atom's PET neighbourhood is complete -- this domain lists them all --
// so they are passed through untouched (its PET is checked elsewhere).
class AtomExchange : public pet::Exchange {
 public:
  explicit AtomExchange(const Domain& d) : owner_(d.owner), n_local_(d.n_local) {}
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
  void set_live(pet::IView1D) override {}
  void edges(pet::View2D, pet::View2D) override {}
  int forwards = 0;

 private:
  std::vector<int> owner_;
  std::size_t n_local_;
};

}  // namespace

TEST_CASE("D3 in a Calculator adds what the D3 alone gives", "[d3][model]") {
  for (const auto& model : plumbing_models()) {
    Pair c = calculators(model, true);
    if (!c.plain) continue;
    const pet::D3 d3(settings(true));
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results a = c.with->compute(g.system), b = c.plain->compute(g.system), d = d3.compute(g.system);
        require_added(a.energy, b.energy, d.energy, "energy");
        require_added(a.per_atom_energy, b.per_atom_energy, d.per_atom_energy, "per-atom energy");
        require_added(a.forces, b.forces, d.forces, "forces");
        require_added(a.virial, b.virial, d.virial, "virial");
        // A batch, on the device path.
        const std::vector<pet::System> two{g.system, g.system};
        const pet::Results ab = c.with->compute(two), bb = c.plain->compute(two), db = d3.compute(two);
        require_added(ab.energy, bb.energy, db.energy, "batch energy");
        require_added(ab.forces, bb.forces, db.forces, "batch forces");
        require_added(ab.virial, bb.virial, db.virial, "batch virial");
        // Staged on the device.
        const pet::DeviceGeom geom = c.with->stage({g.system});
        const pet::BatchResult r = c.with->compute_device(geom, true);
        const double e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.energy)(0);
        CHECK(std::fabs(e - a.energy[0]) <= 1e-6 * std::fabs(a.energy[0]));
      }
    }
  }
}

TEST_CASE("D3 in response() is its forces' derivative", "[d3][model][hvp]") {
  for (const auto& model : plumbing_models()) {
    Pair c = calculators(model, true);
    if (!c.plain) continue;
    Golden store;
    const Golden* g = periodic_golden(model, store);
    if (!g) continue;
    const pet::System& s = g->system;
    std::vector<double> u(std::size_t(s.n_atoms) * 3), e(9, 0.0);
    for (std::size_t k = 0; k < u.size(); ++k) u[k] = std::sin(1.3 * k + 0.2);
    e[0] = 0.5, e[4] = -0.2, e[1] = e[3] = 0.1;
    const auto a = c.with->response(s, 1, u, e), b = c.plain->response(s, 1, u, e);
    // D3's derivative by a wider central difference, here.
    const pet::D3 d3(settings(true));
    const double h = 1e-3;
    pet::System p = s, m = s;
    for (int i = 0; i < s.n_atoms; ++i)
      for (int x = 0; x < 3; ++x) {
        double sx = 0;
        for (int y = 0; y < 3; ++y) sx += e[3 * x + y] * s.positions[3 * i + y];
        p.positions[3 * i + x] += h * (u[3 * i + x] + sx), m.positions[3 * i + x] -= h * (u[3 * i + x] + sx);
      }
    for (int r = 0; r < 3; ++r)
      for (int x = 0; x < 3; ++x) {
        double sx = 0;
        for (int y = 0; y < 3; ++y) sx += e[3 * x + y] * s.cell[3 * r + y];
        p.cell[3 * r + x] += h * sx, m.cell[3 * r + x] -= h * sx;
      }
    const pet::Results fp = d3.compute(p), fm = d3.compute(m);
    std::vector<double> df(fp.forces.size()), dw(6);
    for (std::size_t k = 0; k < df.size(); ++k) df[k] = (fp.forces[k] - fm.forces[k]) / (2 * h);
    for (int k = 0; k < 6; ++k) dw[k] = (fp.virial[k] - fm.virial[k]) / (2 * h);
    require_added(a.forces, b.forces, df, "dF/dt", 1e-4);
    require_added(a.virial, b.virial, dw, "dW/dt", 1e-4);
    require_added(a.base, b.base, d3.compute(s).forces, "base forces");
  }
}

TEST_CASE("D3 sees an engine's atoms as the periodic structure", "[d3][model][edges]") {
  for (const auto& model : plumbing_models()) {
    Golden store;
    const Golden* g = periodic_golden(model, store);
    if (!g || !find_model(model)) continue;
    // Off its lattice, so the forces are worth comparing.
    pet::System s = g->system;
    for (std::size_t k = 0; k < s.positions.size(); ++k) s.positions[k] += 0.05 * std::sin(2.1 * k + 0.4);

    SECTION("images: a periodic list with shifts, stepped") {
      Pair c = calculators(model, true);
      const pet::D3 d3(settings(true));
      // Every pair within cutoff + skin, with its image.
      std::vector<int> off(s.n_atoms + 1, 0), nbr, sh;
      for (const auto& e : pet::detail::build_raw_edges(s, c.plain->cutoff() + 0.5)) {
        ++off[e.i + 1], nbr.push_back(e.j), sh.insert(sh.end(), {e.sa, e.sb, e.sc});
      }
      for (int i = 0; i < s.n_atoms; ++i) off[i + 1] += off[i];
      pet::EdgeListView v;
      v.n_atoms = s.n_atoms, v.positions = s.positions.data(), v.atomic_numbers = s.atomic_numbers.data();
      v.cell = s.cell.data(), v.offsets = off.data(), v.neighbors = nbr.data(), v.shifts = sh.data();
      const pet::Results a = c.with->compute_edges(v), b = c.plain->compute_edges(v), d = d3.compute(s);
      require_added(a.energy, b.energy, d.energy, "energy");
      require_added(a.forces, b.forces, d.forces, "forces");
      require_added(a.virial, b.virial, d.virial, "virial");
      // Stepped: the grid kept between steps, the atoms moving under it.
      c.with->set_neighbors(v), c.plain->set_neighbors(v);
      pet::System moved = s;
      for (int step = 0; step < 3; ++step) {
        for (std::size_t k = 0; k < moved.positions.size(); ++k) moved.positions[k] += 0.01 * std::sin(0.7 * k + step);
        const pet::Results as = c.with->compute_step(moved.positions.data(), s.cell.data()),
                           bs = c.plain->compute_step(moved.positions.data(), s.cell.data()), ds = d3.compute(moved);
        require_added(as.energy, bs.energy, ds.energy, "step energy");
        require_added(as.forces, bs.forces, ds.forces, "step forces");
      }
    }

    SECTION("owned atoms and a ghost shell, no exchange") {
      Pair c = calculators(model, false);  // ghosts: no cell, so no tail
      const pet::D3 d3(settings(false));
      const double gc = c.with->ghost_cutoff(), rc = c.with->cutoff();
      if (kHostBackend && gc > 12.0) continue;
      const Domain dom = make_domain(s, gc + 0.5, gc - rc, rc + 0.5);
      const pet::EdgeListView v = view(dom);
      const pet::Results a = c.with->compute_edges(v, true, false), b = c.plain->compute_edges(v, true, false);
      const pet::Results d = d3.compute(s);
      require_added(a.energy, b.energy, d.energy, "energy");
      require_added(folded(dom, a.forces), folded(dom, b.forces), d.forces, "forces");
      require_added(a.virial, b.virial, d.virial, "virial");
    }

    SECTION("ghosts as deep as the pair cutoff, with an exchange") {
      Pair c = calculators(model, false);
      const pet::D3 d3(settings(false));
      const double gx = c.with->exchange_ghost_cutoff(), rc = c.with->cutoff();
      // Every owned atom's PET neighbourhood listed, so no PET rows need to cross.
      const Domain dom = make_domain(s, gx + 0.5, 0.0, rc + 0.5);
      AtomExchange xa(dom), xb(dom);
      pet::EdgeListView va = view(dom), vb = view(dom);
      // PET's own exchange is not what is under test: off for the plain one, and
      // the D3 one only moves atom values (its PET edges to ghosts get no partner
      // rows, the same for both calculators).
      va.exchange = &xa, vb.exchange = &xb;
      c.with->set_neighbors(va), c.plain->set_neighbors(vb);
      const pet::Results a = c.with->compute_step(dom.pos.data()), b = c.plain->compute_step(dom.pos.data());
      const pet::Results d = d3.compute(s);
      require_added(a.energy, b.energy, d.energy, "energy");
      require_added(folded(dom, a.forces), folded(dom, b.forces), d.forces, "forces");
      require_added(a.virial, b.virial, d.virial, "virial");
    }
  }
}
