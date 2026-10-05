// The reduced-precision GEMM modes against an fp64 product of the same
// matrices: each within its own bound, and each measurably different from
// native, so a mode that silently fell back to fp32 fails too. Only on CUDA with
// an fp32 network; FP8 only from compute capability 8.9.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

#include "pet/gemm.hpp"
#include "pet/kokkos.hpp"

namespace {

struct ModeGuard {
  pet::GemmMode saved = pet::gemm_mode();
  ~ModeGuard() { pet::gemm_mode() = saved; }
};

int compute_capability() {
#if defined(KOKKOS_ENABLE_CUDA)
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major * 10 + minor;
#else
  return 0;
#endif
}

// Activation-like: unit normal entries, plus a few larger ones.
pet::View2D random_matrix(int r, int c, unsigned seed) {
  pet::View2D v("m", r, c);
  auto h = Kokkos::create_mirror_view(v);
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) h(i, j) = pet::Net(n(rng) * ((i + j) % 97 == 0 ? 8.0 : 1.0));
  Kokkos::deep_copy(v, h);
  return v;
}

// max |C - A op(W)| / max |A op(W)|, the product in fp64 on the host.
double error(const pet::View2D& C, const pet::View2D& A, const pet::View2D& W, bool transposed) {
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
  const int m = C.extent(0), n = C.extent(1), k = A.extent(1);
  double worst = 0.0, scale = 0.0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double ref = 0.0;
      for (int p = 0; p < k; ++p) ref += double(ha(i, p)) * double(transposed ? hw(j, p) : hw(p, j));
      worst = std::max(worst, std::fabs(double(hc(i, j)) - ref));
      scale = std::max(scale, std::fabs(ref));
    }
  return worst / scale;
}

// C = A W^T (linear's GEMM) or A W (linear_bwd's), in `mode`.
double run(pet::GemmMode mode, bool transposed) {
  const int m = 300, k = 256, n = 192;
  const pet::View2D A = random_matrix(m, k, 1);
  const pet::View2D W = transposed ? random_matrix(n, k, 2) : random_matrix(k, n, 2);
  pet::View2D C("C", m, n);
  pet::gemm_mode() = mode;
  const char tw = transposed ? 'T' : 'N';
  if (!pet::gemm_lowp('N', tw, 1, A, W, 0, C)) pet::gemm('N', tw, 1, A, W, 0, C);
  return error(C, A, W, transposed);
}

}  // namespace

TEST_CASE("each GEMM mode is as accurate as its precision, and no more", "[gemm]") {
  if (sizeof(pet::Net) != sizeof(float) || compute_capability() < 80) {
    SUCCEED("needs CUDA, compute capability 8.0 and an fp32 network");
    return;
  }
  ModeGuard guard;
  for (const bool transposed : {true, false}) {
    const double native = run(pet::GemmMode::Native, transposed);
    const double tf32 = run(pet::GemmMode::TF32, transposed);
    const double bf16 = run(pet::GemmMode::BF16, transposed);
    const double fp16 = run(pet::GemmMode::FP16, transposed);
    INFO((transposed ? "A W^T" : "A W") << ": native " << native << ", tf32 " << tf32 << ", bf16 " << bf16
                                        << ", fp16 " << fp16);
    CHECK(native < 1e-6);
    CHECK(tf32 > 10 * native);
    CHECK(tf32 < 1e-2);
    CHECK(bf16 > 2 * tf32);  // 8 bits of mantissa against tf32's 11
    CHECK(bf16 < 3e-2);
    CHECK(fp16 > 10 * native);
    CHECK(fp16 < 1e-2);
    if (compute_capability() >= 89) {
      const double fp8 = run(pet::GemmMode::FP8, transposed);
      INFO("fp8 " << fp8);
      CHECK(fp8 > bf16);
      CHECK(fp8 < 0.1);
    }
  }
}

TEST_CASE("BF16 and FP16 run on AMD matrix cores", "[gemm]") {
#if defined(KOKKOS_ENABLE_HIP)
  if (sizeof(pet::Net) != sizeof(float)) return;
  ModeGuard guard;
  for (const bool transposed : {true, false}) {
    const double native = run(pet::GemmMode::Native, transposed);
    const double bf16 = run(pet::GemmMode::BF16, transposed);
    const double fp16 = run(pet::GemmMode::FP16, transposed);
    INFO((transposed ? "A W^T" : "A W") << ": native " << native << ", bf16 " << bf16 << ", fp16 " << fp16);
    CHECK(native < 1e-6);
    CHECK(bf16 > 10 * native);
    CHECK(bf16 < 3e-2);
    CHECK(fp16 > 10 * native);
    CHECK(fp16 < bf16);  // 11 bits of mantissa against 8
  }
#else
  SUCCEED("HIP only");
#endif
}
