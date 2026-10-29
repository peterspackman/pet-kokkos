// The smooth cutoff functions as the device evaluates them, and the constants of
// the adaptive cutoff's solver. The transcendentals are in float: they run over
// every raw edge, Real ones are slow on consumer GPUs, and the factor only
// needs the network's precision. Forward and backward must use these same
// float formulas, or the gradient is of a different function.
#pragma once

#include "pet/kokkos.hpp"

namespace pet::detail {

constexpr double PET_PI = 3.14159265358979323846;  // M_PI is not guaranteed on the device

KOKKOS_INLINE_FUNCTION Real dev_bump_cutoff(Real d, Real rc, Real width) {
  const float x = (float) ((d - (rc - width)) / width);
  if (x <= 0.0f) return 1.0;
  if (x >= 1.0f) return 0.0;
  return Real(0.5) * (Real(1.0) + (Real) Kokkos::tanh(1.0f / Kokkos::tan((float) PET_PI * x)));
}
KOKKOS_INLINE_FUNCTION Real dev_cosine_cutoff(Real d, Real rc, Real width) {
  const float x = Kokkos::fmin(Kokkos::fmax((float) ((d - (rc - width)) / width), 0.0f), 1.0f);
  return Real(0.5) * (Real(1.0) + (Real) Kokkos::cos((float) PET_PI * x));
}
KOKKOS_INLINE_FUNCTION Real dev_cutoff_value(Real d, Real rc, Real width, bool bump) {
  return bump ? dev_bump_cutoff(d, rc, width) : dev_cosine_cutoff(d, rc, width);
}

// d/d(rc) of dev_bump_cutoff, the function the solver root-finds on. With
// s = (d - rc + w)/w, df/drc = (pi / 2w) sech^2(cot(pi s)) / sin^2(pi s). The
// clamp keeps it off 0 * inf at the ends of the taper, as metatrain's does.
KOKKOS_INLINE_FUNCTION Real dev_bump_dcutoff_dr(Real d, Real rc, Real width) {
  const float x = (float) ((d - (rc - width)) / width);
  if (x <= 0.0f || x >= 1.0f) return 0.0;
  const float ps = (float) PET_PI * Kokkos::fmin(Kokkos::fmax(x, 1e-6f), 1.0f - 1e-6f);
  const float si = Kokkos::sin(ps);
  const float tt = Kokkos::tanh(Kokkos::cos(ps) / si);
  return Real(0.5) * (Real(1.0) - (Real) (tt * tt)) * (Real) (((float) PET_PI / (si * si)) / (float) width);
}

// The solver's lower bound on a cutoff (a fraction of the model's), its
// iteration count and its slope floor: metatrain's values, not tuning knobs.
constexpr double PET_SOLVER_MIN_CUTOFF_FACTOR = 1.0 / 16.0;
constexpr int PET_SOLVER_ITERS = 10;
constexpr double PET_SOLVER_DN_FLOOR = 1e-6;

}  // namespace pet::detail
