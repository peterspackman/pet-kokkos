// The Ozaki GEMM against DGEMM on the same matrices, so a failure (a layout,
// the wrong normalisation axis, too few slices) says which rather than showing
// as a slightly-off energy. Only where the path is live (CUDA, fp64); skipped
// elsewhere.
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "pet/gemm.hpp"
#include "pet/kokkos.hpp"
#include "pet/ozaki.hpp"

namespace {

// Values across `decades` orders of magnitude, so the per-row scaling matters.
void fill_random(const pet::View2D& v, unsigned seed, double decades) {
  auto h = Kokkos::create_mirror_view(v);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_real_distribution<double> e(-decades, decades);
  for (std::size_t i = 0; i < v.extent(0); ++i)
    for (std::size_t j = 0; j < v.extent(1); ++j)
      h(i, j) = (pet::Net)(u(rng) * std::pow(10.0, e(rng)));
  Kokkos::deep_copy(v, h);
}

// Largest relative difference, measured against the reference's own scale.
double max_rel_diff(const pet::View2D& a, const pet::View2D& ref) {
  auto ha = Kokkos::create_mirror_view(a);
  auto hr = Kokkos::create_mirror_view(ref);
  Kokkos::deep_copy(ha, a);
  Kokkos::deep_copy(hr, ref);
  double scale = 0.0;
  for (std::size_t i = 0; i < ref.extent(0); ++i)
    for (std::size_t j = 0; j < ref.extent(1); ++j)
      scale = std::max(scale, std::fabs((double) hr(i, j)));
  if (scale == 0.0) return 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.extent(0); ++i)
    for (std::size_t j = 0; j < a.extent(1); ++j)
      worst = std::max(worst, std::fabs((double) ha(i, j) - (double) hr(i, j)) / scale);
  return worst;
}

struct Case {
  int m, k, n;
  char ta, tb;
};

// The configuration is process-global: restore it on scope exit, so a failed
// CHECK cannot leave it on for later tests.
struct ConfigGuard {
  pet::OzakiConfig saved = pet::ozaki_config();
  ~ConfigGuard() { pet::ozaki_config() = saved; }
  void use(int slices) {
    pet::ozaki_config().mode = pet::GemmMode::Ozaki;
    pet::ozaki_config().slices = slices;
  }
  void native() { pet::ozaki_config().mode = pet::GemmMode::Native; }
};

}  // namespace

TEST_CASE("the Ozaki GEMM reproduces DGEMM as slices increase", "[ozaki]") {
  if (!pet::ozaki_available()) {
    WARN("Ozaki path unavailable in this build (needs CUDA + fp64 precision); skipping");
    return;
  }

  // Network shapes: tall activations against a square weight, both orientations.
  const std::vector<Case> cases = {
      {4096, 256, 768, 'N', 'T'},   // linear: in @ W^T, qkv projection
      {4096, 256, 256, 'N', 'N'},   // linear_bwd: out_adj @ W
      {2048, 1024, 1024, 'N', 'T'}, // a wide d_node layer
  };

  ConfigGuard guard;
  for (const Case& c : cases) {
    DYNAMIC_SECTION("m=" << c.m << " k=" << c.k << " n=" << c.n << " " << c.ta << c.tb) {
      const bool ta = (c.ta == 'T'), tb = (c.tb == 'T');
      pet::View2D A("A", ta ? c.k : c.m, ta ? c.m : c.k);
      pet::View2D B("B", tb ? c.n : c.k, tb ? c.k : c.n);
      pet::View2D C("C", c.m, c.n);
      pet::View2D Cref("Cref", c.m, c.n);
      fill_random(A, 1234, 2.0);
      fill_random(B, 5678, 1.0);

      pet::gemm(c.ta, c.tb, (pet::Net) 1.0, A, B, (pet::Net) 0.0, Cref);

      // Error falls with every slice; a subtly wrong scheme plateaus early.
      double prev = 1e30;
      for (int s : {2, 4, 6, 8}) {
        guard.use(s);
        Kokkos::deep_copy(C, (pet::Net) 0.0);
        pet::gemm_ozaki(c.ta, c.tb, (pet::Net) 1.0, A, B, (pet::Net) 0.0, C, nullptr);
        const double err = max_rel_diff(C, Cref);
        INFO("slices = " << s << ", error " << err << ", previous " << prev);
        CHECK(err < prev);
        prev = err;
      }
      // 56 bits > 53: what remains is the reference's rounding.
      CHECK(prev < 1e-14);
      guard.native();
    }
  }
}

TEST_CASE("a pre-split weight gives the same answer as splitting in place", "[ozaki]") {
  // A weight split once at load (matrix-wide scale) must match splitting per call
  // (per-row scale) to fp64 round-off; not to the bit, as the scales differ.
  if (!pet::ozaki_available()) {
    WARN("Ozaki path unavailable in this build; skipping");
    return;
  }

  const int m = 2048, k = 512, n = 512;
  pet::View2D A("A", m, k), W("W", n, k), C1("C1", m, n), C2("C2", m, n);
  fill_random(A, 11, 2.0);
  fill_random(W, 22, 1.0);

  ConfigGuard guard;
  guard.use(pet::kOzakiMaxSlices);

  // Split in place.
  pet::gemm_ozaki('N', 'T', (pet::Net) 1.0, A, W, (pet::Net) 0.0, C1, nullptr);

  // Split once, as a load-time weight would be, then reuse.
  pet::OzakiSplit wsplit = pet::ozaki_split_weight(W, pet::kOzakiMaxSlices);
  REQUIRE(wsplit.valid());
  REQUIRE(wsplit.uniform);
  pet::gemm_ozaki('N', 'T', (pet::Net) 1.0, A, W, (pet::Net) 0.0, C2, &wsplit);

  const double d = max_rel_diff(C2, C1);
  INFO("pre-split vs in-place: max relative difference " << d);
  CHECK(d < 1e-13);

  pet::View2D Cref("Cref", m, n);
  guard.native();
  pet::gemm('N', 'T', (pet::Net) 1.0, A, W, (pet::Net) 0.0, Cref);
  INFO("pre-split vs DGEMM: " << max_rel_diff(C2, Cref));
  CHECK(max_rel_diff(C2, Cref) < 1e-14);
}

TEST_CASE("beta accumulates rather than overwriting", "[ozaki]") {
  // linear_bwd accumulates with beta = 1; the slice grid is folded into C over
  // several passes, so beta must apply exactly once.
  if (!pet::ozaki_available()) {
    WARN("Ozaki path unavailable in this build; skipping");
    return;
  }
  const int m = 512, k = 256, n = 256;
  pet::View2D A("A", m, k), B("B", k, n), C("C", m, n), Cref("Cref", m, n);
  fill_random(A, 7, 1.0);
  fill_random(B, 8, 1.0);
  fill_random(C, 9, 1.0);
  Kokkos::deep_copy(Cref, C);

  ConfigGuard guard;
  guard.use(pet::kOzakiMaxSlices);
  pet::gemm_ozaki('N', 'N', (pet::Net) 1.0, A, B, (pet::Net) 1.0, C, nullptr);
  guard.native();
  pet::gemm('N', 'N', (pet::Net) 1.0, A, B, (pet::Net) 1.0, Cref);

  const double d = max_rel_diff(C, Cref);
  INFO("beta=1 accumulate: max relative difference " << d);
  CHECK(d < 1e-14);
}

TEST_CASE("Ozaki against DGEMM: throughput", "[ozaki][!benchmark]") {
  // Reported, not asserted: the time against DGEMM at the same shape.
  if (!pet::ozaki_available()) {
    WARN("Ozaki path unavailable in this build; skipping");
    return;
  }
  const int m = 16384, k = 1024, n = 1024;
  pet::View2D A("A", m, k), B("B", n, k), C("C", m, n);
  fill_random(A, 3, 1.0);
  fill_random(B, 4, 1.0);
  ConfigGuard guard;

  auto time_it = [&](auto&& f) {
    f();  // warm
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; ++r) f();
    Kokkos::fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 5;
  };

  const double t_native =
      time_it([&] { pet::gemm('N', 'T', (pet::Net) 1.0, A, B, (pet::Net) 0.0, C); });
  std::printf("    DGEMM                  %8.3f ms\n", t_native * 1e3);

  for (int s : {2, 4, 6, 8}) {
    guard.use(s);
    pet::OzakiSplit ws_s = pet::ozaki_split_weight(B, s);
    const double t = time_it([&] {
      pet::gemm_ozaki('N', 'T', (pet::Net) 1.0, A, B, (pet::Net) 0.0, C, &ws_s);
    });
    std::printf("    Ozaki slices=%d         %8.3f ms   (%.2fx vs DGEMM)\n", s, t * 1e3,
                t_native / t);
  }
  SUCCEED();
}
