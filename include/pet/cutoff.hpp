// The smooth cutoff functions as the device evaluates them, and the constants of
// the adaptive cutoff's solver. The transcendentals are in the network's type
// (CutT, float but in the fp64 build): they run over every raw edge, double ones
// are slow on consumer GPUs, and the factor only needs the network's precision.
// Forward and backward must use these same formulas, or the gradient is of a
// different function.
#pragma once

#include "pet/kokkos.hpp"

namespace pet::detail {

using CutT = Net;

constexpr double PET_PI = 3.14159265358979323846;  // M_PI is not guaranteed on the device

KOKKOS_INLINE_FUNCTION double dev_bump_cutoff(double d, double rc, double width) {
  const CutT x = (CutT) ((d - (rc - width)) / width);
  if (x <= CutT(0)) return 1.0;
  if (x >= CutT(1)) return 0.0;
  return 0.5 * (1.0 + (double) Kokkos::tanh(CutT(1) / Kokkos::tan((CutT) PET_PI * x)));
}
KOKKOS_INLINE_FUNCTION double dev_cosine_cutoff(double d, double rc, double width) {
  const CutT x = Kokkos::fmin(Kokkos::fmax((CutT) ((d - (rc - width)) / width), CutT(0)), CutT(1));
  return 0.5 * (1.0 + (double) Kokkos::cos((CutT) PET_PI * x));
}
KOKKOS_INLINE_FUNCTION double dev_cutoff_value(double d, double rc, double width, bool bump) {
  return bump ? dev_bump_cutoff(d, rc, width) : dev_cosine_cutoff(d, rc, width);
}

// d/d(rc) of dev_bump_cutoff, the function the solver root-finds on. With
// s = (d - rc + w)/w, df/drc = (pi / 2w) sech^2(cot(pi s)) / sin^2(pi s). The
// clamp keeps it off 0 * inf at the ends of the taper, as metatrain's does.
KOKKOS_INLINE_FUNCTION double dev_bump_dcutoff_dr(double d, double rc, double width) {
  const CutT x = (CutT) ((d - (rc - width)) / width);
  if (x <= CutT(0) || x >= CutT(1)) return 0.0;
  const CutT ps = (CutT) PET_PI * Kokkos::fmin(Kokkos::fmax(x, CutT(1e-6)), CutT(1) - CutT(1e-6));
  const CutT si = Kokkos::sin(ps);
  const CutT tt = Kokkos::tanh(Kokkos::cos(ps) / si);
  return 0.5 * (1.0 - (double) (tt * tt)) * (double) (((CutT) PET_PI / (si * si)) / (CutT) width);
}

// The solver's lower bound on a cutoff (a fraction of the model's), its
// iteration count and its slope floor: metatrain's values, not tuning knobs.
constexpr double PET_SOLVER_MIN_CUTOFF_FACTOR = 1.0 / 16.0;
constexpr int PET_SOLVER_ITERS = 10;
constexpr double PET_SOLVER_DN_FLOOR = 1e-6;

}  // namespace pet::detail
