// DFT-D3 (pet/d3.hpp): against s-dftd3 at its own cutoffs, the default
// (tapered) cutoffs against finite differences, and the determinism every
// result in pet-kokkos keeps. The references are tools/make_d3_golden.py's.
#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/d3.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

constexpr double kBohr = 0.529177210903;

struct Case {
  std::string name, functional, damping;
  pet::System sys;
  pet::D3Settings set;
  double energy;
  std::vector<double> forces, virial;
};

std::vector<Case> load_cases() {
  std::ifstream f(std::string(golden_dir()) + "/d3_reference.json");
  REQUIRE(f.good());
  nlohmann::json j;
  f >> j;
  std::vector<Case> out;
  for (const auto& c : j.at("cases")) {
    Case k;
    k.name = c.at("structure").get<std::string>();
    k.functional = c.at("functional").get<std::string>();
    k.damping = c.at("damping").get<std::string>();
    k.sys.atomic_numbers = c.at("numbers").get<std::vector<int>>();
    k.sys.n_atoms = (int) k.sys.atomic_numbers.size();
    for (const auto& p : c.at("positions"))
      for (int d = 0; d < 3; ++d) k.sys.positions.push_back(p[d].get<double>());
    if (c.contains("cell")) {
      int r = 0;
      for (const auto& row : c.at("cell")) {
        for (int d = 0; d < 3; ++d) k.sys.cell[3 * r + d] = row[d].get<double>();
        ++r;
      }
      const auto pbc = c.at("pbc").get<std::vector<bool>>();
      for (int d = 0; d < 3; ++d) k.sys.pbc[d] = pbc[d];
    }
    const auto& p = c.at("params");
    pet::D3Params& q = k.set.params;
    q.damping = pet::d3_damping(k.damping);
    q.s6 = p.at("s6").get<double>(), q.s8 = p.at("s8").get<double>();
    if (q.damping == pet::D3Damping::BJ) {
      q.a1 = p.at("a1").get<double>(), q.a2 = p.at("a2").get<double>() * kBohr;
    } else {
      q.rs6 = p.at("rs6").get<double>(), q.rs8 = p.at("rs8").get<double>();
    }
    q.s9 = c.at("s9").get<double>();
    const auto& cut = c.at("cutoffs_bohr");
    k.set.cutoffs = pet::D3Cutoffs::reference();
    k.set.cutoffs.pair = cut.at("pair").get<double>() * kBohr;
    k.set.cutoffs.cn = cut.at("cn").get<double>() * kBohr;
    k.set.cutoffs.atm = cut.at("atm").get<double>() * kBohr;
    k.energy = c.at("energy").get<double>();
    for (const auto& r : c.at("forces"))
      for (int d = 0; d < 3; ++d) k.forces.push_back(r[d].get<double>());
    k.virial = c.at("virial").get<std::vector<double>>();
    out.push_back(std::move(k));
  }
  return out;
}

pet::System water_box(int n_side, unsigned seed) {
  // n_side^3 waters on a jittered grid in a sheared cell, ~1 g/cm^3.
  pet::System s;
  const double a = 3.104;
  const double L = a * n_side;
  s.cell = {L, 0, 0, 0.2, L, 0, -0.1, 0.15, L};
  s.pbc = {true, true, true};
  unsigned x = seed;
  auto rnd = [&]() {
    x = x * 1664525u + 1013904223u;
    return (x >> 8) / double(1 << 24);
  };
  for (int i = 0; i < n_side; ++i)
    for (int j = 0; j < n_side; ++j)
      for (int k = 0; k < n_side; ++k) {
        const double o[3] = {(i + 0.3 * rnd()) * a, (j + 0.3 * rnd()) * a, (k + 0.3 * rnd()) * a};
        const double t = 6.283 * rnd();
        const double h1[3] = {0.757 * std::cos(t), 0.586, 0.757 * std::sin(t)};
        const double h2[3] = {-0.757 * std::cos(t), 0.586, -0.757 * std::sin(t)};
        s.atomic_numbers.insert(s.atomic_numbers.end(), {8, 1, 1});
        for (int d = 0; d < 3; ++d) s.positions.push_back(o[d]);
        for (int d = 0; d < 3; ++d) s.positions.push_back(o[d] + h1[d]);
        for (int d = 0; d < 3; ++d) s.positions.push_back(o[d] + h2[d]);
      }
  s.n_atoms = (int) s.atomic_numbers.size();
  return s;
}

}  // namespace

TEST_CASE("D3 matches s-dftd3 at its cutoffs", "[d3]") {
  for (const auto& c : load_cases()) {
    DYNAMIC_SECTION(c.name << " " << c.damping << (c.set.params.s9 != 0 ? "+atm" : "")) {
      const pet::D3 d3(c.set);
      const pet::Results r = d3.compute(c.sys, true);
      const int N = c.sys.n_atoms;
      INFO("E " << r.energy[0] << " ref " << c.energy);
      // s-dftd3 and this sum in different orders, and at a hard cutoff a pair
      // a rounding away from it can land on either side: agreement to ~1e-9.
      CHECK(std::fabs(r.energy[0] - c.energy) <= 2e-9 * std::fabs(c.energy) + 1e-12);
      double fmax = 0, fe = 0;
      for (int k = 0; k < 3 * N; ++k) fe = worst(fe, std::fabs(r.forces[k] - c.forces[k])), fmax = std::max(fmax, std::fabs(c.forces[k]));
      INFO("force error " << fe << " of " << fmax);
      CHECK(fe <= 2e-8 * fmax + 1e-11);
      if (c.sys.pbc[0] || c.sys.pbc[1] || c.sys.pbc[2]) {
        double we = 0, wmax = 0;
        for (int k = 0; k < 6; ++k) we = worst(we, std::fabs(r.virial[k] - c.virial[k])), wmax = std::max(wmax, std::fabs(c.virial[k]));
        INFO("virial error " << we << " of " << wmax);
        CHECK(we <= 2e-7 * wmax + 1e-11);
      }
      double sum = 0;
      for (double e : r.per_atom_energy) sum += e;
      CHECK(std::fabs(sum - r.energy[0]) <= 1e-12 * std::fabs(r.energy[0]));
    }
  }
}

TEST_CASE("D3 functional parameters by name", "[d3]") {
  const pet::D3Params p = pet::d3_parameters("PBE");
  CHECK(p.a1 == 0.4289);
  CHECK(std::fabs(p.a2 - 4.4407 * kBohr) < 1e-15);
  CHECK(p.s8 == 0.7875);
  CHECK(p.s9 == 0.0);
  CHECK(pet::d3_parameters("pbe", pet::D3Damping::BJ, true).s9 == 1.0);
  CHECK(pet::d3_parameters("PBEh").a1 == pet::d3_parameters("pbe0").a1);  // an alias
  CHECK(pet::d3_parameters("b3-lyp").a1 == pet::d3_parameters("b3lyp").a1);
  CHECK(pet::d3_parameters("pbe", pet::D3Damping::Zero).rs6 == 1.217);
  CHECK_THROWS(pet::d3_parameters("no-such-functional"));
  CHECK_THROWS(pet::d3_parameters("r2scan", pet::D3Damping::Zero));  // BJ only
  CHECK(pet::d3_functionals().size() > 100);
}

// The default cutoffs taper smoothly: forces and virial are the energy's
// gradient. Central differences, both dampings, with ATM.
TEST_CASE("D3 forces and virial are the energy's gradient", "[d3]") {
  pet::System s = water_box(2, 11);
  for (int atm = 0; atm < 2; ++atm)
    for (auto damping : {pet::D3Damping::BJ, pet::D3Damping::Zero}) {
      DYNAMIC_SECTION((damping == pet::D3Damping::BJ ? "bj" : "zero") << (atm ? "+atm" : "")) {
        pet::D3Settings set = pet::D3Settings::functional("pbe", damping, atm);
        // Short cutoffs, so that many pairs sit inside the tapers.
        set.cutoffs.pair = 7.0, set.cutoffs.pair_taper = 2.5;
        set.cutoffs.cn = 5.0, set.cutoffs.cn_taper = 2.0;
        set.cutoffs.atm = 4.5, set.cutoffs.atm_taper = 1.5;
        const pet::D3 d3(set);
        const pet::Results r = d3.compute(s, true);
        const double h = 1e-4;
        double err = 0, scale = 0;
        for (int i = 0; i < s.n_atoms; i += 5)
          for (int d = 0; d < 3; ++d) {
            pet::System p = s, m = s;
            p.positions[3 * i + d] += h, m.positions[3 * i + d] -= h;
            const double fd = -(d3.compute(p, false).energy[0] - d3.compute(m, false).energy[0]) / (2 * h);
            err = worst(err, std::fabs(fd - r.forces[3 * i + d]));
            scale = std::max(scale, std::fabs(fd));
          }
        INFO("force FD error " << err << " of " << scale);
        CHECK(err < 1e-7 * scale + 1e-10);
        // Virial: W = dE/de for a symmetric strain x -> x (1 + e).
        const int vi[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
        double werr = 0, wscale = 0;
        for (int k = 0; k < 6; ++k) {
          double e[2];
          for (int sg = 0; sg < 2; ++sg) {
            double eps[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            const double t = (sg ? -h : h) * (vi[k][0] == vi[k][1] ? 1.0 : 0.5);
            eps[3 * vi[k][0] + vi[k][1]] += t;
            if (vi[k][0] != vi[k][1]) eps[3 * vi[k][1] + vi[k][0]] += t;
            pet::System q = s;
            auto strain = [&](double* x) {
              double y[3];
              for (int a = 0; a < 3; ++a) y[a] = x[a] + eps[3 * a] * x[0] + eps[3 * a + 1] * x[1] + eps[3 * a + 2] * x[2];
              for (int a = 0; a < 3; ++a) x[a] = y[a];
            };
            for (int i = 0; i < q.n_atoms; ++i) strain(&q.positions[3 * i]);
            for (int r3 = 0; r3 < 3; ++r3) strain(&q.cell[3 * r3]);
            e[sg] = d3.compute(q, false).energy[0];
          }
          const double fd = (e[0] - e[1]) / (2 * h);
          werr = worst(werr, std::fabs(fd - r.virial[k]));
          wscale = std::max(wscale, std::fabs(fd));
        }
        INFO("virial FD error " << werr << " of " << wscale);
        CHECK(werr < 1e-7 * wscale + 1e-10);
      }
    }
}

TEST_CASE("D3 is bit-reproducible and batch-independent", "[d3][determinism]") {
  const pet::System a = water_box(3, 5), b = water_box(2, 9);
  pet::System mol;
  mol.n_atoms = 3, mol.atomic_numbers = {8, 1, 1}, mol.positions = {0, 0, 0, 0.757, 0.586, 0, -0.757, 0.586, 0};
  pet::D3 d3(pet::D3Settings::functional("pbesol", pet::D3Damping::BJ, true));
  const pet::Results ra = d3.compute(a), rb = d3.compute(b), rm = d3.compute(mol);
  const pet::Results r2 = d3.compute(a);
  REQUIRE(ra.energy[0] == r2.energy[0]);
  for (std::size_t k = 0; k < ra.forces.size(); ++k) REQUIRE(ra.forces[k] == r2.forces[k]);
  for (int k = 0; k < 6; ++k) REQUIRE(ra.virial[k] == r2.virial[k]);
  // In a batch, each structure's answer is its own, to the bit.
  const pet::Results all = d3.compute(std::vector<pet::System>{b, mol, a});
  REQUIRE(all.energy.size() == 3);
  CHECK(all.energy[0] == rb.energy[0]);
  CHECK(all.energy[1] == rm.energy[0]);
  CHECK(all.energy[2] == ra.energy[0]);
  const std::size_t ob = 3 * b.n_atoms, om = 3 * mol.n_atoms;
  for (std::size_t k = 0; k < rb.forces.size(); ++k) REQUIRE(all.forces[k] == rb.forces[k]);
  for (std::size_t k = 0; k < rm.forces.size(); ++k) REQUIRE(all.forces[ob + k] == rm.forces[k]);
  for (std::size_t k = 0; k < ra.forces.size(); ++k) REQUIRE(all.forces[ob + om + k] == ra.forces[k]);
  for (int k = 0; k < 6; ++k) REQUIRE(all.virial[12 + k] == ra.virial[k]);
  // A molecule has no virial to speak of but its sum is still W = sum r (x) F.
  // Translation: the energy does not move.
  pet::System t = a;
  for (int i = 0; i < t.n_atoms; ++i) t.positions[3 * i] += 0.37;
  CHECK(std::fabs(d3.compute(t).energy[0] - ra.energy[0]) < 1e-10 * std::fabs(ra.energy[0]));
}

// An engine's view: owned atoms and ghosts, without periodicity. The owned
// atoms' energy and every atom's force must be what the periodic structure
// gives, once the ghosts reach far enough.
TEST_CASE("D3 with ghosts matches the periodic structure", "[d3]") {
  const pet::System s = water_box(2, 3);
  pet::D3Settings set = pet::D3Settings::functional("pbe", pet::D3Damping::BJ, true);
  set.cutoffs.pair = 6.0, set.cutoffs.pair_taper = 1.5, set.cutoffs.cn = 4.0, set.cutoffs.cn_taper = 1.0;
  set.cutoffs.atm = 4.0, set.cutoffs.atm_taper = 1.0;
  const pet::D3 d3(set);
  const pet::Results ref = d3.compute(s);
  // Ghosts: every image within ghost_cutoff (no exchange) of the cell.
  const double reach = d3.ghost_cutoff(false) + 0.5;
  std::vector<double> pos(s.positions);
  std::vector<int> z(s.atomic_numbers), owner;
  for (int i = 0; i < s.n_atoms; ++i) owner.push_back(i);
  const int R = 3;
  for (int a = -R; a <= R; ++a)
    for (int b = -R; b <= R; ++b)
      for (int c = -R; c <= R; ++c) {
        if (!a && !b && !c) continue;
        for (int i = 0; i < s.n_atoms; ++i) {
          double p[3];
          for (int d = 0; d < 3; ++d) p[d] = s.positions[3 * i + d] + a * s.cell[d] + b * s.cell[3 + d] + c * s.cell[6 + d];
          // inside the cell's box grown by reach (the cell is near-cubic)
          bool near = true;
          for (int d = 0; d < 3; ++d) near = near && p[d] > -reach && p[d] < s.cell[4 * d] + reach;
          if (!near) continue;
          pos.insert(pos.end(), p, p + 3), z.push_back(s.atomic_numbers[i]), owner.push_back(i);
        }
      }
  const int N = (int) z.size();
  pet::System g;
  g.n_atoms = N, g.atomic_numbers = z, g.positions = pos;
  pet::DeviceGeom geom = d3.stage({g});
  pet::D3::Atoms at;
  at.pos = geom.pos, at.type = geom.spec, at.sid = geom.sid, at.soff = geom.soff, at.scnt = geom.scnt;
  at.sper = geom.sper, at.scell = geom.scell, at.N = N, at.B = 1, at.n_local = s.n_atoms;
  const pet::D3::Out o = d3.evaluate(at, true);
  auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o.energy);
  auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o.forces);
  auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o.virial);
  CHECK(std::fabs(e(0) - ref.energy[0]) < 1e-10 * std::fabs(ref.energy[0]));
  // Ghost forces summed home, as an engine's reverse communication would.
  std::vector<double> F(3 * s.n_atoms, 0.0);
  for (int a = 0; a < N; ++a)
    for (int d = 0; d < 3; ++d) F[3 * owner[a] + d] += f(a, d);
  double err = 0, scale = 0;
  for (int k = 0; k < 3 * s.n_atoms; ++k) err = worst(err, std::fabs(F[k] - ref.forces[k])), scale = std::max(scale, std::fabs(ref.forces[k]));
  INFO("force error " << err << " of " << scale);
  CHECK(err < 1e-10 * scale + 1e-12);
  for (int k = 0; k < 6; ++k) CHECK(std::fabs(w(0, k) - ref.virial[k]) < 1e-9 * std::fabs(ref.virial[k]) + 1e-11);
}
