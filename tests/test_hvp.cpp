// The exact Hessian-vector product (Calculator::response) against central
// differences of the forces and virial along the same directions: displacements
// of every atom, of one atom, and a strain with a displacement, for each kind
// of model (feedforward and residual featurizers; solver, grid and no adaptive
// cutoff). In the fp64 build this is tight; in fp32 the differences' own error
// sets the tolerance.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

constexpr bool kFp64 = std::is_same_v<pet::Net, double>;

// x(t) = (x + t u)(I + t e)^T and the cell likewise: the direction the product's
// v' = u_j - u_i + e v differentiates along.
pet::System moved(const pet::System& s0, const std::vector<double>& u, const double* e, double t) {
  pet::System s = s0;
  double F[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) F[r * 3 + c] = (r == c) + t * (e ? e[r * 3 + c] : 0.0);
  for (int i = 0; i < s0.n_atoms; ++i) {
    double p[3];
    for (int c = 0; c < 3; ++c) p[c] = s0.positions[i * 3 + c] + t * u[i * 3 + c];
    for (int r = 0; r < 3; ++r) s.positions[i * 3 + r] = F[r * 3] * p[0] + F[r * 3 + 1] * p[1] + F[r * 3 + 2] * p[2];
  }
  if (s0.pbc[0] || s0.pbc[1] || s0.pbc[2])
    for (int a = 0; a < 3; ++a)
      for (int r = 0; r < 3; ++r)
        s.cell[a * 3 + r] =
            F[r * 3] * s0.cell[a * 3] + F[r * 3 + 1] * s0.cell[a * 3 + 1] + F[r * 3 + 2] * s0.cell[a * 3 + 2];
  return s;
}

}  // namespace

TEST_CASE("Hessian-vector products match differences of the forces", "[model][hvp]") {
  // Each model on the goldens' structures of a model that shares its elements.
  const std::pair<std::string, std::string> cases[] = {
      {"pet-mad-xs-v1.6", "pet-mad-xs-v1.6"}, {"pet-mad-s", "pet-mad-xs-v1.6"}, {"pet-mols-s-v1.1", "pet-mols-s-v1.1"},
      {"pet-mad-xs", "pet-mad-xs"}, {"pet-omol-s", "pet-mols-s-v1.1"}};
  for (const auto& [model, structures] : cases) {
    pet::Calculator* calcp = shared_calculator(model);
    if (!calcp) {
      WARN("model '" << model << "' is not installed; skipping its HVP check");
      continue;
    }
    pet::Calculator& calc = *calcp;
    for (const auto& path : golden_paths(structures)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::System& s0 = g.system;
        const int N = s0.n_atoms;
        std::mt19937 rng(7);
        std::normal_distribution<double> nd(0.0, 1.0);
        const int K = 3;
        std::vector<double> disp(std::size_t(K) * N * 3, 0.0), strain(std::size_t(K) * 9, 0.0);
        for (int i = 0; i < N * 3; ++i) disp[i] = nd(rng);       // every atom
        disp[std::size_t(N) * 3 + 1] = 1.0;                      // atom 0, y
        for (int i = 0; i < N * 3; ++i) disp[2 * N * 3 + i] = 0.3 * nd(rng);
        if (g.periodic)                                         // with a strain
          for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c) strain[18 + r * 3 + c] = strain[18 + c * 3 + r] = 0.05 * nd(rng);

        const pet::Calculator::Response hv = calc.response(s0, K, disp, strain);
        const pet::Results r0 = calc.compute(s0, true);
        double fmax = 0.0, ferr = 0.0;
        for (int i = 0; i < N * 3; ++i) {
          fmax = worst(fmax, std::fabs(r0.forces[i]));
          ferr = worst(ferr, std::fabs(hv.base[i] - r0.forces[i]));
        }
        INFO("forces from the product's pass vs compute: " << ferr << " (max " << fmax << ")");
        CHECK(ferr <= (kFp64 ? 1e-9 : 1e-4) * std::max(1.0, fmax));

        // Small: the cosine cutoff's second derivative jumps at both ends of its
        // taper, and a difference straddling an edge there is off by O(h).
        const double h = kFp64 ? 2.5e-4 : 5e-3;
        for (int k = 0; k < K; ++k) {
          const std::vector<double> u(disp.begin() + std::size_t(k) * N * 3, disp.begin() + std::size_t(k + 1) * N * 3);
          const double* e = &strain[std::size_t(k) * 9];
          std::vector<pet::Results> at;
          for (double t : {h, -h, 2 * h, -2 * h}) at.push_back(calc.compute(moved(s0, u, e, t), true));
          auto d4 = [&](auto get) {  // 4-point central difference
            return (8.0 * (get(at[0]) - get(at[1])) - (get(at[2]) - get(at[3]))) / (12.0 * h);
          };
          double scale = 0.0, err = 0.0;
          for (int i = 0; i < N * 3; ++i) {
            const double fd = d4([&](const pet::Results& r) { return r.forces[i]; });
            scale = worst(scale, std::fabs(fd));
            err = worst(err, std::fabs(fd - hv.forces[std::size_t(k) * N * 3 + i]));
          }
          double wscale = 0.0, werr = 0.0;
          if (g.periodic)
            for (int v = 0; v < 6; ++v) {
              const double fd = d4([&](const pet::Results& r) { return r.virial[v]; });
              wscale = worst(wscale, std::fabs(fd));
              werr = worst(werr, std::fabs(fd - hv.virial[std::size_t(k) * 6 + v]));
            }
          INFO("direction " << k << ": max|dF/dt - FD| = " << err << " of max|FD| " << scale
                            << "; virial " << werr << " of " << wscale);
          // In fp32 the reference is what is coarse: its step must stay large against
          // fp32 noise, and in the disordered structures it crosses the cutoffs'
          // kinks. The product's own exactness is the fp64 build's check.
          const double tol = kFp64 ? 1e-6 : 1e-2;
          CHECK(err <= tol * std::max(1.0, scale));
          CHECK(werr <= tol * std::max(1.0, wscale));
        }
      }
    }
  }
}
