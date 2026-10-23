// pet-batch-probe -- does a structure's answer depend on its batch-mates?
//
//   pet-batch-probe MODEL frames.xyz [n_batch]
//
// Every frame is evaluated alone (stage([s]), compute_device) as the reference,
// then inside batches the way a relaxation driver sees it: the whole batch, its
// leading prefixes (a driver packing survivors to the front), and the batch
// reversed (other neighbours, other slot). Reports, per configuration, how many
// structures come back bit-identical to their lone evaluation and the worst
// energy and force difference. A measurement tool, not a test.
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/io.hpp"

namespace {

struct Answer {
  std::vector<double> energy;               // [B]
  std::vector<std::vector<double>> forces;  // [B][n*3]
};

bool g_host = false;  // host neighbour lists, through compute(systems)

Answer eval(const pet::Calculator& calc, const std::vector<pet::System>& systems, int lead = -1) {
  if (g_host) {
    std::vector<pet::System> use(systems.begin(), systems.begin() + (lead > 0 ? lead : systems.size()));
    const pet::Results r = calc.compute(use, true);
    Answer a;
    std::size_t off = 0;
    for (std::size_t b = 0; b < use.size(); ++b) {
      a.energy.push_back(r.energy[b]);
      const std::size_t n = std::size_t(use[b].n_atoms) * 3;
      a.forces.emplace_back(r.forces.begin() + off, r.forces.begin() + off + n);
      off += n;
    }
    return a;
  }
  pet::DeviceGeom g = calc.stage(systems);
  int B = static_cast<int>(systems.size());
  if (lead > 0 && lead < B) {
    int n = 0;
    for (int b = 0; b < lead; ++b) n += systems[b].n_atoms;
    g.B = lead, g.Ntot = n, B = lead;
  }
  const pet::BatchResult r = calc.compute_device(g, true);
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.energy);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r.forces);
  Answer a;
  int off = 0;
  for (int b = 0; b < B; ++b) {
    a.energy.push_back(he(b));
    const int n = systems[b].n_atoms;
    std::vector<double> f(static_cast<std::size_t>(n) * 3);
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) f[i * 3 + c] = hf(off + i, c);
    a.forces.push_back(std::move(f));
    off += n;
  }
  return a;
}

// `idx[b]` is the frame in slot b of `got`.
void report(const char* what, const Answer& got, const std::vector<int>& idx, const std::vector<Answer>& alone,
            const std::vector<pet::System>& frames) {
  int same = 0;
  double de = 0, de_atom = 0, df = 0;
  for (std::size_t b = 0; b < idx.size(); ++b) {
    const Answer& ref = alone[idx[b]];
    bool bit = got.energy[b] == ref.energy[0];
    const double d = std::abs(got.energy[b] - ref.energy[0]);
    de = std::max(de, d);
    de_atom = std::max(de_atom, d / frames[idx[b]].n_atoms);
    for (std::size_t i = 0; i < ref.forces[0].size(); ++i) {
      bit = bit && got.forces[b][i] == ref.forces[0][i];
      df = std::max(df, std::abs(got.forces[b][i] - ref.forces[0][i]));
    }
    same += bit;
  }
  // 1 eV = 96.485 kJ/mol
  std::printf("%-28s B=%-4zu bit-identical %4d/%-4zu  max|dE| %.2e eV (%.2e kJ/mol, %.2e eV/atom)  max|dF| %.2e eV/A\n",
              what, idx.size(), same, idx.size(), de, de * 96.485, de_atom, df);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: pet-batch-probe MODEL frames.xyz [n_batch]\n");
    return 1;
  }
  Kokkos::initialize(argc, argv);
  {
    const std::vector<pet::System> all = pet::read_extxyz(argv[2]);
    const int nb = std::min<int>(argc > 3 ? std::atoi(argv[3]) : 64, static_cast<int>(all.size()));
    const std::vector<pet::System> frames(all.begin(), all.begin() + nb);
    pet::Options opts;
    if (argc > 4 && std::string(argv[4]) == "host") g_host = true, opts.device_neighbors = false;
    pet::Calculator calc(argv[1], opts);

    std::vector<Answer> alone;
    for (const auto& s : frames) alone.push_back(eval(calc, {s}));
    // The lone answer must itself be reproducible, or nothing below means anything.
    {
      std::vector<int> idx;
      Answer again;
      for (int b = 0; b < nb; ++b) {
        const Answer a = eval(calc, {frames[b]});
        again.energy.push_back(a.energy[0]);
        again.forces.push_back(a.forces[0]);
        idx.push_back(b);
      }
      report("alone, repeated", again, idx, alone, frames);
    }
    // klasp's evaluate() sends B == 1 down Calculator::compute(System).
    {
      std::vector<int> idx;
      Answer host;
      for (int b = 0; b < nb; ++b) {
        const pet::Results r = calc.compute(frames[b], true);
        host.energy.push_back(r.energy[0]);
        host.forces.push_back(r.forces);
        idx.push_back(b);
      }
      report("compute(System)", host, idx, alone, frames);
    }

    std::vector<int> idx(nb);
    for (int b = 0; b < nb; ++b) idx[b] = b;
    report("whole batch", eval(calc, frames), idx, alone, frames);
    report("whole batch, repeated", eval(calc, frames), idx, alone, frames);
    for (int lead = nb / 2; lead >= 1; lead /= 2) {
      char what[64];
      std::snprintf(what, sizeof what, "leading %d of the batch", lead);
      report(what, eval(calc, frames, lead), std::vector<int>(idx.begin(), idx.begin() + lead), alone, frames);
    }
    std::vector<pet::System> rev(frames.rbegin(), frames.rend());
    std::vector<int> ridx(idx.rbegin(), idx.rend());
    report("reversed", eval(calc, rev), ridx, alone, frames);
  }
  Kokkos::finalize();
  return 0;
}
