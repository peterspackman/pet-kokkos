// The analytic forces and virial against finite differences of the model's own
// energy: the backward must be the forward's derivative, which no golden can
// check. The fp32 energy has a quantization floor, so h is swept: a correct
// backward shows the V (h^2 truncation falling, eps/h noise rising), a wrong one
// a floor.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// The deformation I + E for a Voigt strain, E = [[e0 e3 e4] [e3 e1 e5] [e4 e5 e2]].
// A shear sits in both off-diagonal slots, so dE/de3 = 2 W_xy.
std::array<double, 9> strain_to_def(const std::array<double, 6>& e) {
  return {1.0 + e[0], e[3],       e[4],
          e[3],       1.0 + e[1], e[5],
          e[4],       e[5],       1.0 + e[2]};
}

// 1 for a diagonal, 2 for a shear.
constexpr double voigt_multiplicity(int v) { return v < 3 ? 1.0 : 2.0; }

pet::System strained(const pet::System& s0, int v, double h) {
  std::array<double, 6> e{};
  e[v] = h;
  const auto d = strain_to_def(e);
  pet::System st = s0;
  for (int i = 0; i < s0.n_atoms; ++i) {
    const double* p = &s0.positions[static_cast<std::size_t>(i) * 3];
    for (int r = 0; r < 3; ++r)
      st.positions[static_cast<std::size_t>(i) * 3 + r] =
          d[r * 3 + 0] * p[0] + d[r * 3 + 1] * p[1] + d[r * 3 + 2] * p[2];
  }
  // Lattice vectors (rows) transform like positions.
  for (int a = 0; a < 3; ++a) {
    const double row[3] = {s0.cell[a * 3 + 0], s0.cell[a * 3 + 1], s0.cell[a * 3 + 2]};
    for (int r = 0; r < 3; ++r)
      st.cell[a * 3 + r] = d[r * 3 + 0] * row[0] + d[r * 3 + 1] * row[1] + d[r * 3 + 2] * row[2];
  }
  return st;
}

constexpr double kSteps[] = {1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4};

// Relative or absolute: near equilibrium the forces are themselves the noise
// floor, and the absolute floor is what the fp32 energy can resolve.
constexpr double kRelTol = 1e-3;
constexpr double kForceAbsTol = 3e-3;   // eV/Angstrom
constexpr double kVirialAbsTol = 3e-2;  // eV

}  // namespace

TEST_CASE("analytic forces match -dE/dx by finite difference", "[model][fd]") {
  for (const auto& model : golden_models()) {
    pet::Calculator* calcp = shared_calculator(model);
    if (!calcp) {
      WARN("model '" << model << "' is not installed; skipping its FD check");
      continue;
    }
    pet::Calculator& calc = *calcp;

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results r0 = calc.compute(g.system, true);
        const int N = g.system.n_atoms;

        double max_f = 0.0;
        for (double f : r0.forces) max_f = worst(max_f, std::fabs(f));

        double best = 1e300;
        double best_h = 0.0;
        for (double h : kSteps) {
          double err_h = 0.0;  // worst error at this step
          for (int i = 0; i < N; ++i)
            for (int c = 0; c < 3; ++c) {
              pet::System sp = g.system, sm = g.system;
              sp.positions[static_cast<std::size_t>(i) * 3 + c] += h;
              sm.positions[static_cast<std::size_t>(i) * 3 + c] -= h;
              const double ep = calc.compute(sp, false).energy[0];
              const double em = calc.compute(sm, false).energy[0];
              const double f_num = -(ep - em) / (2 * h);
              const double f_ana = r0.forces[static_cast<std::size_t>(i) * 3 + c];
              err_h = worst(err_h, std::fabs(f_num - f_ana));
            }
          if (err_h < best) {
            best = err_h;
            best_h = h;
          }
          // Stop at the first step that agrees; a real mismatch still tries every step
          // and reports the best.
          if (best / std::max(max_f, 1e-30) < kRelTol || best < kForceAbsTol) break;
        }
        const double rel = best / std::max(max_f, 1e-30);
        INFO("best max|dF| = " << best << " eV/A at h = " << best_h << " (relative " << rel
                               << ", max|F| = " << max_f << ")");
        CHECK((rel < kRelTol || best < kForceAbsTol));
      }
    }
  }
}

TEST_CASE("analytic virial matches dE/dstrain by finite difference", "[model][fd]") {
  for (const auto& model : golden_models()) {
    pet::Calculator* calcp = shared_calculator(model);
    if (!calcp) continue;
    pet::Calculator& calc = *calcp;

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      if (!g.periodic) continue;  // no cell, no virial
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results r0 = calc.compute(g.system, true);
        REQUIRE(r0.virial.size() == 6);

        double max_w = 0.0;
        for (double w : r0.virial) max_w = worst(max_w, std::fabs(w));

        double best = 1e300;
        double best_h = 0.0;
        int worst_v = -1;
        for (double h : kSteps) {
          double err_h = 0.0;  // worst error at this step
          int wv = -1;
          for (int v = 0; v < 6; ++v) {
            const double ep = calc.compute(strained(g.system, v, h), false).energy[0];
            const double em = calc.compute(strained(g.system, v, -h), false).energy[0];
            const double w_num = (ep - em) / (2 * h);
            const double w_ana = r0.virial[v] * voigt_multiplicity(v);
            const double err = std::fabs(w_num - w_ana);
            if (!(err <= err_h)) {  // so that a NaN counts as the worst
              err_h = err;
              wv = v;
            }
          }
          if (err_h < best) {
            best = err_h;
            best_h = h;
            worst_v = wv;
          }
          if (best / std::max(max_w, 1e-30) < kRelTol || best < kVirialAbsTol) break;
        }
        const double rel = best / std::max(max_w, 1e-30);
        INFO("best max|dW| = " << best << " eV at h = " << best_h << " (relative " << rel
                               << ", worst Voigt component " << worst_v << ")");
        CHECK((rel < kRelTol || best < kVirialAbsTol));
      }
    }
  }
}
