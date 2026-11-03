// PetModel: loading, the host upload, packing, the parts both featurizers share,
// and compute(). The featurizers themselves are in feedforward.cpp and
// residual.cpp; the building blocks in ops.hpp.
#include "ops.hpp"

#include "pet/ozaki.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace pet {

namespace {

// Layer i's last-layer features into ll, as metatrain concatenates them: the
// node head's output in columns [2i*Dh, (2i+1)*Dh), the edge heads' summed
// over each atom's edges with the cutoff factor in the next Dh.
void gather_ll_features(View2D ll, int i, View2D node_h, View2D edge_h, const PackedEdges& pk) {
  const int N = node_h.extent(0), Dh = node_h.extent(1), c0 = 2 * i * Dh;
  auto off = pk.off;
  auto cut = pk.cut;
  Kokkos::parallel_for(
      "ll_features", RangePolicy(0, N * Dh), KOKKOS_LAMBDA(int x) {
        const int n = x / Dh, d = x % Dh;
        double s = 0.0;
        for (int k = off(n); k < off(n + 1); ++k) s += (double) cut(k) * (double) edge_h(k, d);
        ll(n, c0 + d) = node_h(n, d);
        ll(n, c0 + Dh + d) = Net(s);
      });
}

// out(r) = multiplier * |inv * x(r, :)|, inv lower triangular: metatrain's
// LLPR sigma, its triangular solve done as a product with the inverse.
void llpr_sigma(RView2D x, RView2D inv, double multiplier, RView1D out) {
  const int F = inv.extent(0);
  Kokkos::parallel_for(
      "llpr_sigma", TeamPolicy(x.extent(0), Kokkos::AUTO), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank();
        double q = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, F),
            [&](int i, double& acc) {
              double v = 0.0;
              for (int j = 0; j <= i; ++j) v += inv(i, j) * x(r, j);
              acc += v * v;
            },
            q);
        Kokkos::single(Kokkos::PerTeam(t), [&] { out(r) = multiplier * Kokkos::sqrt(q); });
      });
}

}  // namespace

// ---- loading --------------------------------------------------------------------

PetModel::PetModel(const Checkpoint& ckpt)
    : h_(ckpt.hypers),
      energy_scale_(ckpt.energy_scale),
      composition_(ckpt.composition_energies),
      species_to_index_(ckpt.species_to_index),
      nc_force_scale_(ckpt.nc_force_scale),
      nc_stress_scale_(ckpt.nc_stress_scale) {
  load_all(ckpt);

  // The adaptive cutoff's probe grid, on the device.
  const std::vector<double> p = adaptive_probes(h_);
  n_probes_ = (int) p.size();
  probes_ = RView1D("probes", std::max(n_probes_, 1));
  auto hp = Kokkos::create_mirror_view(probes_);
  for (int i = 0; i < n_probes_; ++i) hp(i) = p[i];
  Kokkos::deep_copy(probes_, hp);

  comp_view_ = RView1D("composition", composition_.size());
  auto hc = Kokkos::create_mirror_view(comp_view_);
  for (std::size_t i = 0; i < composition_.size(); ++i) hc(i) = composition_[i];
  Kokkos::deep_copy(comp_view_, hc);

  // The LLPR uncertainty's matrices, in double: its Cholesky factor is badly
  // conditioned (~4e8 for pet-mad-s).
  if (ckpt.llpr_features > 0) {
    if (ckpt.llpr_features != 2 * h_.num_readout_layers * h_.d_head)
      throw std::runtime_error("checkpoint: LLPR features (" + std::to_string(ckpt.llpr_features) +
                               ") are not the model's last-layer features");
    auto load = [&](const std::string& n) {
      const Tensor& t = ckpt.weights.at(n);
      RView2D v(n, t.dim(0), t.dim(1));
      auto hv = Kokkos::create_mirror_view(v);
      for (int a = 0; a < t.dim(0); ++a)
        for (int b = 0; b < t.dim(1); ++b) hv(a, b) = t.data[std::size_t(a) * t.dim(1) + b];
      Kokkos::deep_copy(v, hv);
      return v;
    };
    llpr_inv_ = load("llpr.energy.inv_cholesky");
    if (ckpt.llpr_ensemble > 0) llpr_ensemble_ = load("llpr.energy.ensemble");
    llpr_features_ = ckpt.llpr_features;
    llpr_multiplier_ = ckpt.llpr_multiplier;
  }

  // One force scale for every species, or one each.
  if (has_nc_force()) {
    nc_force_scale_view_ = RView1D("nc_force_scale", composition_.size());
    auto hs = Kokkos::create_mirror_view(nc_force_scale_view_);
    for (std::size_t i = 0; i < composition_.size(); ++i)
      hs(i) = nc_force_scale_[nc_force_scale_.size() == 1 ? 0 : i];
    Kokkos::deep_copy(nc_force_scale_view_, hs);
  }
}

void PetModel::set_uncertainty(bool on) {
  if (on && !has_uncertainty())
    throw std::runtime_error(
        "PetModel: this model has no LLPR uncertainty (or was converted before "
        "tools/convert_pet.py kept it; convert it again)");
  uq_ = on;
}

View2D PetModel::ll_features(int N, const std::string& key) {
  return uq_ ? ws_.n2(key + "llpr_f", N, llpr_features_) : View2D();
}

Uncertainty PetModel::energy_uncertainty(const DeviceEdgeData& dev, const PackedEdges& pk, View2D ll,
                                         RView1D per_atom, const std::string& key) {
  const int N = dev.centres(), NA = dev.n_atoms, NS = dev.n_struct, nl = pk.n_local, F = llpr_features_;
  const IView1D soff = structure_offsets(ws_, key + "uq", dev.struct_id, NA, NS);
  // Each atom's features in double, 0 for atoms outside the energy, and each
  // structure's sum: metatrain's per-atom and per-structure features.
  RView2D fa = ws_.r2(key + "uq_fa", NA, F), fs = ws_.r2(key + "uq_fs", NS, F);
  Kokkos::parallel_for(
      "uq_features", RangePolicy(0, NA * F), KOKKOS_LAMBDA(int x) {
        const int a = x / F, j = x % F;
        fa(a, j) = a < N && a < nl ? (double) ll(a, j) : 0.0;
      });
  sum_by_structure(fa, soff, fs, false);
  Uncertainty u{ws_.r1(key + "uq_e", NS), ws_.r1(key + "uq_ea", NA), {}};
  llpr_sigma(fa, llpr_inv_, llpr_multiplier_, u.per_atom);
  llpr_sigma(fs, llpr_inv_, llpr_multiplier_, u.energy);
  if (llpr_ensemble_.extent(0) == 0) return u;

  // Member k's energy is w_k . f, re-centred so the members average to the
  // model's own energy, as metatrain does (which also restores what the last
  // layer leaves out: composition, biases).
  const int K = llpr_ensemble_.extent(0);
  auto W = llpr_ensemble_;
  RView2D e = ws_.r2(key + "uq_energy", NS, 1);
  sum_by_structure(RView2D(per_atom.data(), NA, 1), soff, e, false);
  u.ensemble = ws_.r2(key + "uq_ens", NS, K);
  auto ens = u.ensemble;
  Kokkos::parallel_for(
      "uq_ensemble", TeamPolicy(NS, Kokkos::AUTO), KOKKOS_LAMBDA(const Team& t) {
        const int b = t.league_rank();
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, K),
            [&](int k, double& acc) {
              double v = 0.0;
              for (int j = 0; j < F; ++j) v += W(k, j) * fs(b, j);
              ens(b, k) = v;
              acc += v;
            },
            sum);
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, K), [&](int k) { ens(b, k) += e(b, 0) - sum / K; });
      });
  return u;
}

void PetModel::set_non_conservative(bool on) {
  if (on && !has_nc_force())
    throw std::runtime_error(
        "PetModel: this model has no non-conservative force head (or was converted before "
        "tools/convert_pet.py kept them; convert it again)");
  nc_ = on;
}

void PetModel::load_all(const Checkpoint& ckpt) {
  const auto& st = ckpt.weights;
  const int G = h_.num_gnn_layers, A = h_.num_attention_layers, R = h_.num_readout_layers;
  const bool ff = h_.featurizer_type == FeaturizerType::FeedForward;
  const bool ln = h_.normalization == Normalization::LayerNorm;
  const bool expanded = h_.expanded_node();
  const bool swiglu = h_.activation == Activation::SwiGLU;

  auto put_mat = [&](const std::string& n, int rows, int cols, const double* d) {
    View2D v(n, rows, cols);
    auto hv = Kokkos::create_mirror_view(v);
    for (int a = 0; a < rows; ++a)
      for (int b = 0; b < cols; ++b) hv(a, b) = d[a * cols + b];
    Kokkos::deep_copy(v, hv);
    mat_.emplace(n, v);
  };
  auto put_vec = [&](const std::string& n, int len, const double* d) {
    View1D v(n, len);
    auto hv = Kokkos::create_mirror_view(v);
    for (int a = 0; a < len; ++a) hv(a) = d[a];
    Kokkos::deep_copy(v, hv);
    vec_.emplace(n, v);
  };
  auto load_mat = [&](const std::string& n) {
    const Tensor& t = st.at(n);
    if (t.ndim() != 2) throw std::runtime_error("expected 2D tensor '" + n + "'");
    put_mat(n, t.dim(0), t.dim(1), t.data.data());
  };
  auto load_vec = [&](const std::string& n) { put_vec(n, st.at(n).numel(), st.at(n).data.data()); };
  auto load_linear = [&](const std::string& p) { load_mat(p + ".weight"), load_vec(p + ".bias"); };
  // A SwiGLU w_in's [v | g] output rows, interleaved to (v_j, g_j): see swiglu.
  auto load_swiglu_in = [&](const std::string& p) {
    const Tensor &W = st.at(p + ".weight"), &b = st.at(p + ".bias");
    const int F = W.dim(0) / 2, K = W.dim(1);
    std::vector<double> w(W.data.size()), bi(b.data.size());
    for (int j = 0; j < 2 * F; ++j) {
      const int src = j % 2 ? F + j / 2 : j / 2;
      std::copy_n(&W.data[std::size_t(src) * K], K, &w[std::size_t(j) * K]);
      bi[j] = b.data[src];
    }
    put_mat(p + ".weight", 2 * F, K, w.data());
    put_vec(p + ".bias", 2 * F, bi.data());
  };
  auto load_mlp_in = [&](const std::string& p) { swiglu ? load_swiglu_in(p) : load_linear(p); };
  auto load_norm = [&](const std::string& p) {  // RMSNorm has no bias
    load_vec(p + ".weight");
    if (ln) load_vec(p + ".bias");
  };

  // Layer L's CompressFold (model.hpp), in double. compress.0's input columns are
  // [edge_emb | species block | input_edge], the species block being the
  // neighbour embedding -- or, at layer 0, input_edge itself (the edge_embedder
  // lookup), with nothing after it.
  auto fold_compress = [&](int L) {
    const std::string g = "gnn_layers." + std::to_string(L), c0 = g + ".compress.0";
    const Tensor &W = st.at(c0 + ".weight"), &b0 = st.at(c0 + ".bias");
    const Tensor &We = st.at(g + ".edge_embedder.weight"), &be = st.at(g + ".edge_embedder.bias");
    const Tensor& T = st.at(L == 0 ? "edge_embedder.weight" : g + ".neighbor_embedder.weight");
    const int D = W.dim(0), K = W.dim(1), ns = T.dim(0);
    auto w = [&](int d, int j) { return W.data[(std::size_t) d * K + j]; };
    std::vector<double> wx(D * 4, 0.0), b(b0.data), tab((std::size_t) ns * D, 0.0), wi;
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < D; ++j) {
        for (int c = 0; c < 4; ++c) wx[d * 4 + c] += w(d, j) * We.data[j * 4 + c];
        b[d] += w(d, j) * be.data[j];
        for (int s = 0; s < ns; ++s) tab[(std::size_t) s * D + d] += T.data[(std::size_t) s * D + j] * w(d, D + j);
      }
    put_mat(c0 + "@x4", D, 4, wx.data());
    put_vec(c0 + "@b", D, b.data());
    put_mat(c0 + "@tab", ns, D, tab.data());
    if (L == 0) return;
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < D; ++j) wi.push_back(w(d, 2 * D + j));
    put_mat(c0 + "@in", D, D, wi.data());
  };

  for (int L = 0; L < G; ++L) {
    const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
    fold_compress(L);
    load_linear(g + ".compress.2");
    for (int a = 0; a < A; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      load_linear(tl + ".attention.input_linear");
      load_linear(tl + ".attention.output_linear");
      load_norm(tl + ".norm_attention");
      load_norm(tl + ".norm_mlp");
      load_mlp_in(tl + ".mlp.w_in");
      load_linear(tl + ".mlp.w_out");
      if (expanded) {  // otherwise metatrain makes these the identity
        load_linear(tl + ".center_contraction");
        load_linear(tl + ".center_expansion");
        load_norm(tl + ".norm_center_features");
        load_mlp_in(tl + ".center_mlp.w_in");
        load_linear(tl + ".center_mlp.w_out");
      }
    }
    if (ff) {
      load_vec("combination_norms." + ls + ".weight");
      load_vec("combination_norms." + ls + ".bias");
      load_linear("combination_mlps." + ls + ".0");
      load_linear("combination_mlps." + ls + ".2");
    }
  }
  load_mat("edge_embedder.weight");
  if (h_.system_conditioning) {
    load_mat("system_conditioning.charge_embedding.weight");
    load_mat("system_conditioning.spin_multiplicity_embedding.weight");
    load_linear("system_conditioning.project.0");
    load_linear("system_conditioning.project.2");
  }
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i);
    load_mat("node_embedders." + si + ".weight");
    load_linear("node_heads.energy." + si + ".0");
    load_linear("node_heads.energy." + si + ".2");
    load_linear("edge_heads.energy." + si + ".0");
    load_linear("edge_heads.energy." + si + ".2");
    load_linear("node_last_layers.energy." + si + ".energy___0");
    load_linear("edge_last_layers.energy." + si + ".energy___0");
  }
  // The non-conservative heads, when the checkpoint has them: the energy's
  // layout, with 3 (force) or 9 (stress) outputs.
  for (const std::string t : {"nc_force", "nc_stress"}) {
    if (t == "nc_force" ? ckpt.nc_force_scale.empty() : ckpt.nc_stress_scale.empty()) continue;
    for (int i = 0; i < R; ++i) {
      const std::string si = std::to_string(i);
      for (const std::string side : {"node", "edge"}) {
        load_linear(side + "_heads." + t + "." + si + ".0");
        load_linear(side + "_heads." + t + "." + si + ".2");
        load_linear(side + "_last_layers." + t + "." + si + "." + t + "___0");
      }
    }
  }

  // The two families this implements; anything else is refused by name.
  if (ff && (h_.transformer_type != TransformerType::PreLN || ln ||
             h_.activation != Activation::SwiGLU || !expanded))
    throw std::runtime_error(
        "PetModel: the feedforward featurizer is only implemented for PreLN + RMSNorm + "
        "SwiGLU with an expanded central token (d_node != d_pet)");
  if (!ff && (h_.transformer_type != TransformerType::PostLN || !ln ||
              h_.activation != Activation::SiLU || expanded))
    throw std::runtime_error(
        "PetModel: the residual featurizer is only implemented for PostLN + LayerNorm + "
        "SiLU with a non-expanded central token (d_node == d_pet)");
  // Its backward has no adaptive-cutoff term: the forces would silently disagree
  // with the energy.
  if (!ff && h_.adaptive())
    throw std::runtime_error(
        "PetModel: the residual featurizer with an adaptive cutoff is not implemented");
  if (A < 1) throw std::runtime_error("PetModel: num_attention_layers must be >= 1");
  if (h_.head_dim > kMaxHeadDim)
    throw std::runtime_error("PetModel: head_dim " + std::to_string(h_.head_dim) +
                             " is above the attention kernels' limit of " +
                             std::to_string(kMaxHeadDim));

  // Ozaki: split every weight once, here. A weight is split with one scale for
  // the whole matrix, so the same slices serve both orientations it is used in;
  // each orientation still gets its own layout.
  if (ozaki_active())
    for (const auto& [name, m] : mat_) {
      wsplit_.emplace(name, ozaki_split_weight(m, ozaki_config().slices, false));
      wsplit_t_.emplace(name, ozaki_split_weight(m, ozaki_config().slices, true));
    }
}

WeightRef PetModel::mat(const std::string& name) const {
  const auto it = mat_.find(name);
  if (it == mat_.end()) throw std::runtime_error("PetModel: missing matrix '" + name + "'");
  const auto s = wsplit_.find(name), t = wsplit_t_.find(name);
  return {it->second, s == wsplit_.end() ? nullptr : &s->second,
          t == wsplit_t_.end() ? nullptr : &t->second};
}

const View1D& PetModel::vec(const std::string& name) const {
  const auto it = vec_.find(name);
  if (it == vec_.end()) throw std::runtime_error("PetModel: missing vector '" + name + "'");
  return it->second;
}

CompressFold PetModel::compress_fold(int L) const {
  const std::string c0 = "gnn_layers." + std::to_string(L) + ".compress.0";
  return {mat(c0 + "@x4").v, vec(c0 + "@b"), mat(c0 + "@tab").v, L ? mat(c0 + "@in") : WeightRef{}};
}

// ---- the host upload --------------------------------------------------------------

DeviceEdgeData PetModel::upload_edge_data(const EdgeData& ed, bool need_reverse) const {
  const int N = ed.n_atoms, M = ed.max_neighbors, S = M + 1, NM = N * M, E = ed.n_raw;
  DeviceEdgeData dev;
  dev.n_atoms = N;
  dev.max_neighbors = M;
  dev.n_raw = E;
  dev.n_edges = (int) std::count_if(ed.mask.begin(), ed.mask.end(), [](char m) { return m != 0; });

  std::vector<double> cf((std::size_t) N * S, 1.0);  // col 0: the central token
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) cf[(std::size_t) n * S + 1 + m] = ed.cutoff_factor[(std::size_t) n * M + m];

  // Per-atom ranges of the raw list and each raw edge's partner, which let the
  // adaptive cutoff's backward gather instead of scattering with atomics. Only
  // built when it will run, and only if the list is grouped by centre and every
  // edge finds its partner; otherwise left empty.
  std::vector<int> off(N + 1, 0), rev;
  bool grouped = E > 0 && need_reverse && h_.adaptive();
  for (int e = 0; e < E && grouped; ++e) {
    if (e > 0 && ed.raw_center[e] < ed.raw_center[e - 1]) grouped = false;
    ++off[ed.raw_center[e] + 1];
  }
  if (grouped) {
    for (int a = 0; a < N; ++a) off[a + 1] += off[a];
    // Both host searches sort each atom's run by neighbour, which puts the edges
    // back to atom i in one contiguous block; checked, not assumed.
    bool sorted = true;
    for (int a = 0; a < N && sorted; ++a)
      for (int f = off[a] + 1; f < off[a + 1] && sorted; ++f) sorted = ed.raw_neigh[f] >= ed.raw_neigh[f - 1];
    // The partner of (i, j, v) is the (j, i, -v) whose vector is closest to -v:
    // that tells periodic images of one pair apart without an exact float compare.
    rev.assign(E, -1);
    for (int e = 0; e < E && grouped; ++e) {
      const int i = ed.raw_center[e], j = ed.raw_neigh[e];
      int lo = off[j], hi = off[j + 1];
      if (sorted) {
        const auto begin = ed.raw_neigh.begin();
        lo = (int) (std::lower_bound(begin + off[j], begin + off[j + 1], i) - begin);
        hi = (int) (std::upper_bound(begin + lo, begin + off[j + 1], i) - begin);
      }
      double best = 1e300;
      for (int f = lo; f < hi; ++f) {
        if (ed.raw_neigh[f] != i) continue;
        double r2 = 0;
        for (int c = 0; c < 3; ++c) r2 += (ed.raw_vec[3 * f + c] + ed.raw_vec[3 * e + c]) * (ed.raw_vec[3 * f + c] + ed.raw_vec[3 * e + c]);
        if (r2 < best) best = r2, rev[e] = f;
      }
      grouped = rev[e] >= 0 && best <= 1e-12;
    }
  }

  const int B = (int) ed.charge.size();
  std::vector<int> spin(B, 1);
  for (int b = 0; b < B && b < (int) ed.spin_multiplicity.size(); ++b) spin[b] = ed.spin_multiplicity[b];

  Upload& up = upload_;
  up.clear();
  const auto o_sp = up.add<int>(ed.species), o_ns = up.add<int>(ed.neigh_species),
             o_rev = up.add<int>(ed.reverse_index), o_ev = up.add<Real>(ed.edge_vec),
             o_d = up.add<Real>(ed.edge_dist), o_mask = up.add<Real>(ed.mask),
             o_pc = up.add<Real>(ed.pair_cutoff), o_cut = up.add<Net>(ed.cutoff_factor),
             o_cf = up.add<Net>(cf), o_rc = up.add<int>(ed.raw_center), o_rj = up.add<int>(ed.raw_neigh),
             o_rd = up.add<Real>(ed.raw_dist), o_rv = up.add<Real>(ed.raw_vec),
             o_q = up.add<int>(ed.charge), o_s = up.add<int>(spin), o_ar = up.add<Real>(ed.adapt_r),
             o_adn = up.add<Real>(ed.adapt_dn), o_off = up.add<int>(off), o_rrev = up.add<int>(rev);
  up.send();

  dev.species = up.get<IView1D>(o_sp, N);
  dev.neigh_species = up.get<IView1D>(o_ns, NM);
  dev.reverse_index = up.get<IView1D>(o_rev, NM);
  dev.edge_vec = up.get<RView2D>(o_ev, NM, 3);
  dev.dist = up.get<RView1D>(o_d, NM);
  dev.mask = up.get<RView1D>(o_mask, NM);
  dev.pair_cutoff = up.get<RView1D>(o_pc, NM);
  dev.cutoff_factor = up.get<View1D>(o_cut, NM);
  dev.cf_seq = up.get<View2D>(o_cf, N, S);
  dev.raw_center = up.get<IView1D>(o_rc, E);
  dev.raw_neigh = up.get<IView1D>(o_rj, E);
  dev.raw_dist = up.get<RView1D>(o_rd, E);
  dev.raw_vec = up.get<RView2D>(o_rv, E, 3);
  if (B > 0) dev.charge = up.get<IView1D>(o_q, B), dev.spin_multiplicity = up.get<IView1D>(o_s, B);
  if ((int) ed.adapt_r.size() == N && N > 0)  // the solver's root and slope
    dev.adapt_r = up.get<RView1D>(o_ar, N), dev.adapt_dn = up.get<RView1D>(o_adn, N);
  if (grouped) dev.raw_off = up.get<IView1D>(o_off, N + 1), dev.raw_reverse = up.get<IView1D>(o_rrev, E);
  return dev;
}

// ---- shared by both featurizers -----------------------------------------------------

// The edges to ghosts PET keeps this time, numbered in the engine's order: each
// one's packed row, and back. The engine learns which they are (set_live), so
// only those rows cross.
static void remote_map(Workspace& ws, const DeviceEdgeData& d, PackedEdges& p) {
  if (!d.exchange) return;
  const int R = p.n_remote;
  auto raw = d.remote_raw, slot = d.raw_slot, slot_edge = p.slot_edge, of = p.remote_of;
  IView1D at = ws.i1("pk:live_at", R + 1);
  Kokkos::parallel_scan(
      "pk_live_scan", RangePolicy(0, R + 1), KOKKOS_LAMBDA(int r, int& c, bool final) {
        if (final) at(r) = c;
        if (r < R) {
          const int s = slot(raw(r));
          c += s >= 0 && slot_edge(s) >= 0;
        }
      });
  // With fixed shapes, a capacity rather than the count, which stays on the
  // device: padding rows are -1, and more live edges than fit raise overflow.
  const int L = d.live_capacity;
  if (L > 0) p.n_live = L;
  else Kokkos::deep_copy(p.n_live, Kokkos::subview(at, R));
  IView1D live = ws.i1("pk:live", p.n_live), packed = p.live_packed = ws.i1("pk:live_packed", p.n_live);
  Kokkos::deep_copy(ExecSpace(), of, -1);
  if (L > 0) {
    Kokkos::deep_copy(ExecSpace(), live, -1), Kokkos::deep_copy(ExecSpace(), packed, -1);
    auto overflow = d.overflow;
    Kokkos::parallel_for(
        "pk_live_capacity", RangePolicy(0, 1), KOKKOS_LAMBDA(int) {
          if (at(R) > L) overflow(0) = 1;
        });
  }
  const int cap = p.n_live;
  Kokkos::parallel_for(
      "pk_remote", RangePolicy(0, R), KOKKOS_LAMBDA(int r) {
        const int s = slot(raw(r)), k = s >= 0 ? slot_edge(s) : -1;
        if (k < 0) return;
        const int l = at(r);
        if (l >= cap) return;  // past a capacity: flagged above
        live(l) = r, packed(l) = k, of(k) = l;
      });
  d.exchange->set_live(live);
}

PackedEdges PetModel::pack_edges(const DeviceEdgeData& d) {
  const int N = d.centres(), M = d.max_neighbors;
  PackedEdges p;
  p.E = d.n_edges;
  p.off = ws_.i1("pk:off", N + 1);
  p.center = ws_.i1("pk:center", p.E), p.species = ws_.i1("pk:species", p.E);
  p.reverse = ws_.i1("pk:reverse", p.E);
  p.vec = ws_.r2("pk:vec", p.E, 3);
  p.dist = ws_.r1("pk:dist", p.E), p.pcut = ws_.r1("pk:pcut", p.E);
  p.cut = ws_.n1("pk:cut", p.E);
  IView1D pos = p.slot_edge = ws_.i1("pk:pos", N * M);
  p.n_local = d.n_local < 0 ? N : d.n_local;
  auto mask = d.mask;
  auto kept = KOKKOS_LAMBDA(int n, int end) {  // atom n's kept slots before `end`
    int c = 0;
    for (int m = 0; m < end; ++m) c += mask(n * M + m) > 0.0;
    return c;
  };
  auto off = p.off, center = p.center, species = p.species, reverse = p.reverse;
  auto vec = p.vec;
  auto dist = p.dist, pcut = p.pcut;
  auto cut = p.cut;
  auto nsp = d.neigh_species, rev = d.reverse_index;
  auto ev = d.edge_vec;
  auto dd = d.dist, dpc = d.pair_cutoff;
  auto dcut = d.cutoff_factor;
  if (d.exchange) {  // mapped once packed (remote_map)
    p.n_remote = d.remote_raw.extent(0);
    p.remote_of = ws_.i1("pk:remote_of", p.E);
  }
  if (d.padded) {  // every slot a row: slot k is edge k
    Kokkos::parallel_for(
        "pk_padded", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
          const int n = k / M;
          if (k % M == 0) off(n) = k;
          if (k == 0) off(N) = N * M;
          pos(k) = k, center(k) = n, species(k) = nsp(k), reverse(k) = rev(k);
          for (int c = 0; c < 3; ++c) vec(k, c) = ev(k, c);
          dist(k) = dd(k), pcut(k) = dpc(k), cut(k) = dcut(k);
        });
    remote_map(ws_, d, p);
    return p;
  }
  Kokkos::parallel_scan(
      "pk_off", RangePolicy(0, N), KOKKOS_LAMBDA(int n, int& upd, bool final) {
        if (final) off(n) = upd;
        upd += kept(n, M);
        if (final && n == N - 1) off(N) = upd;
      });
  if (N == 0) Kokkos::deep_copy(ExecSpace(), off, 0);
  // A capacity: what does not fit is flagged and cut off (the caller redoes the
  // step with more room), and the rows past the kept edges are dead -- no atom's,
  // no partner, zero weight -- so they move nothing.
  const int E = p.E;
  if (d.edge_capacity) {
    auto overflow = d.overflow;
    Kokkos::parallel_for(
        "pk_capacity", RangePolicy(0, N + 1), KOKKOS_LAMBDA(int n) {
          if (n == N && off(N) > E) overflow(0) = 1;
          if (off(n) > E) off(n) = E;
        });
    Kokkos::parallel_for(
        "pk_dead", RangePolicy(0, E), KOKKOS_LAMBDA(int k) {
          if (k < off(N)) return;
          center(k) = 0, species(k) = 0, reverse(k) = -1;
          for (int c = 0; c < 3; ++c) vec(k, c) = 0.0;
          dist(k) = 1.0, pcut(k) = 1.0, cut(k) = Net(0);
        });
  }
  Kokkos::parallel_for(
      "pk_edges", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        const int n = k / M;
        if (mask(k) <= 0.0) return (void) (pos(k) = -1);
        const int e = off(n) + kept(n, k % M);
        if (e >= off(n + 1)) return (void) (pos(k) = -1);  // past a capacity
        pos(k) = e, center(e) = n, species(e) = nsp(k);
        for (int c = 0; c < 3; ++c) vec(e, c) = ev(k, c);
        dist(e) = dd(k), pcut(e) = dpc(k), cut(e) = dcut(k);
      });
  Kokkos::parallel_for(
      "pk_reverse", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        if (pos(k) >= 0) reverse(pos(k)) = rev(k) >= 0 ? pos(rev(k)) : -1;
      });
  remote_map(ws_, d, p);
  return p;
}

// Charge and spin conditioning: per structure, embed the charge and the spin
// multiplicity, project (Linear-SiLU-Linear), and broadcast to its atoms. Added
// to the node features after every GNN layer. It depends on no position, so it
// has no backward. Empty for a model without it.
View2D PetModel::conditioning(const DeviceEdgeData& dev) {
  if (!h_.system_conditioning) return View2D();
  const int N = dev.centres(), NS = dev.n_struct, Dn = h_.d_node, max_q = h_.max_charge;
  IView1D q = dev.charge, sm = dev.spin_multiplicity;
  if ((int) q.extent(0) != NS || (int) sm.extent(0) != NS) {  // metatrain's default: a neutral singlet
    q = ws_.i1("cond_q", NS), sm = ws_.i1("cond_sm", NS);
    Kokkos::deep_copy(ExecSpace(), q, 0);
    Kokkos::deep_copy(ExecSpace(), sm, 1);
  }
  View2D cat = ws_.n2("cond_cat", NS, 2 * Dn);
  const View2D qe = mat("system_conditioning.charge_embedding.weight").v;
  const View2D se = mat("system_conditioning.spin_multiplicity_embedding.weight").v;
  const int nq = qe.extent(0), ns = se.extent(0);
  Kokkos::parallel_for(
      "cond_gather", RangePolicy(0, NS * Dn), KOKKOS_LAMBDA(int i) {
        const int b = i / Dn, d = i % Dn;
        // Clamped: an out-of-range state is the caller's error, not a wild read.
        const int iq = Kokkos::clamp(q(b) + max_q, 0, nq - 1), is = Kokkos::clamp(sm(b) - 1, 0, ns - 1);
        cat(b, d) = qe(iq, d);
        cat(b, Dn + d) = se(is, d);
      });
  View2D h0 = ws_.n2("cond_h0", NS, Dn), per_struct = ws_.n2("cond_ps", NS, Dn);
  linear_silu(h0, View2D(), cat, mat("system_conditioning.project.0.weight"),
              vec("system_conditioning.project.0.bias"));
  linear(per_struct, h0, mat("system_conditioning.project.2.weight"),
         vec("system_conditioning.project.2.bias"));
  View2D out = ws_.n2("cond_atom", N, Dn);
  IView1D sid = dev.struct_id;
  const bool have_sid = (int) sid.extent(0) == N;
  Kokkos::parallel_for(
      "cond_bcast", RangePolicy(0, N * Dn),
      KOKKOS_LAMBDA(int i) { out(i / Dn, i % Dn) = per_struct(have_sid ? sid(i / Dn) : 0, i % Dn); });
  return out;
}

void PetModel::norm(View2D out, View2D in, const std::string& key, View2D add) const {
  const bool ln = h_.normalization == Normalization::LayerNorm;
  norm_fwd(out, in, vec(key + ".weight"), ln ? vec(key + ".bias") : View1D(), add);
}

void PetModel::norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key, bool acc,
                        View2D also) const {
  pet::norm_bwd(in_adj, out_adj, in, vec(key + ".weight"), h_.normalization == Normalization::LayerNorm, acc, also);
}

int PetModel::ffn_pre_width(const std::string& key) const { return mat(key + ".w_in.weight").extent(0); }

void PetModel::feedforward(View2D out, View2D in, const std::string& key, View2D pre, bool save, Net beta) {
  const WeightRef w_in = mat(key + ".w_in.weight"), w_out = mat(key + ".w_out.weight");
  const View1D &b_in = vec(key + ".w_in.bias"), &b_out = vec(key + ".w_out.bias");
  if (h_.activation == Activation::SwiGLU) return swiglu(ws_, out, in, w_in, b_in, w_out, b_out, pre, beta);
  Workspace::Scope scope(ws_);
  View2D h = ws_.tmp(in.extent(0), pre.extent(1));
  linear_silu(h, save ? pre : View2D(), in, w_in, b_in);
  linear(out, h, w_out, b_out, beta);
}

void PetModel::feedforward_bwd(View2D in_adj, View2D out_adj, const std::string& key, View2D pre, Net beta) {
  const WeightRef w_in = mat(key + ".w_in.weight"), w_out = mat(key + ".w_out.weight");
  if (h_.activation == Activation::SwiGLU) return swiglu_bwd(ws_, in_adj, out_adj, pre, w_in, w_out, beta);
  Workspace::Scope scope(ws_);
  View2D h_adj = ws_.tmp(out_adj.extent(0), pre.extent(1));
  linear_silu_bwd(h_adj, out_adj, w_out, pre);
  linear_bwd(in_adj, h_adj, w_in, beta);
}

void PetModel::readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat, View2D out,
                       const PackedEdges& pk, ReadoutSaves* sav, const std::string& key,
                       const std::string& target, View2D ll) {
  const int R = node_feat.size(), N = out.extent(0), C = out.extent(1), E = pk.E, Dh = h_.d_head;
  if (sav)
    for (auto* v : {&sav->nh0, &sav->nh1, &sav->eh0, &sav->eh1, &sav->epred}) v->assign(R, View2D());
  // A buffer the backward reads is saved under a per-layer name, otherwise scratch.
  auto buf = [&](std::vector<View2D>* s, const std::string& name, int i, int rows, int cols) {
    return sav ? ((*s)[i] = ws_.n2(key + "s_" + name + "_" + std::to_string(i), rows, cols))
               : ws_.n2(key + name, rows, cols);
  };
  const std::string w = std::to_string(C);  // the predictions' scratch, by width
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i), nh = "node_heads." + target + "." + si,
                      eh = "edge_heads." + target + "." + si,
                      last = "_last_layers." + target + "." + si + "." + target + "___0";
    View2D nh0 = ws_.n2(key + "nh0", N, Dh), nh1 = ws_.n2(key + "nh1", N, Dh);
    linear_silu(nh0, sav ? buf(&sav->nh0, "nh0", i, N, Dh) : View2D(), node_feat[i], mat(nh + ".0.weight"),
                vec(nh + ".0.bias"));
    linear_silu(nh1, sav ? buf(&sav->nh1, "nh1", i, N, Dh) : View2D(), nh0, mat(nh + ".2.weight"),
                vec(nh + ".2.bias"));
    View2D node_pred = ws_.n2(key + "node_pred" + w, N, C);
    linear(node_pred, nh1, mat("node" + last + ".weight"), vec("node" + last + ".bias"));

    View2D eh0 = ws_.n2(key + "eh0", E, Dh), eh1 = ws_.n2(key + "eh1", E, Dh);
    linear_silu(eh0, sav ? buf(&sav->eh0, "eh0", i, E, Dh) : View2D(), edge_feat[i], mat(eh + ".0.weight"),
                vec(eh + ".0.bias"));
    linear_silu(eh1, sav ? buf(&sav->eh1, "eh1", i, E, Dh) : View2D(), eh0, mat(eh + ".2.weight"),
                vec(eh + ".2.bias"));
    View2D edge_pred = sav ? buf(&sav->epred, "epred", i, E, C) : ws_.n2(key + "edge_pred" + w, E, C);
    linear(edge_pred, eh1, mat("edge" + last + ".weight"), vec("edge" + last + ".bias"));

    readout_accumulate(out, node_pred, edge_pred, pk, i > 0);
    if (ll.extent(0)) gather_ll_features(ll, i, nh1, eh1, pk);
  }
}

DeviceOut PetModel::nc_outputs(const DeviceEdgeData& dev, const PackedEdges& pk,
                               const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
                               RView1D per_atom, const std::string& key) {
  const int N = dev.centres(), NA = dev.n_atoms, NS = dev.n_struct, nl = pk.n_local;  // the model's rows, every atom
  auto species = dev.species;
  const IView1D soff = structure_offsets(ws_, key + "nc", dev.struct_id, NA, NS);

  DeviceOut out{per_atom, ws_.r2(key + "nc_forces", NA, 3), ws_.r2(key + "nc_vir9", NS, 9), {}};
  View2D f = ws_.n2(key + "nc_f", N, 3);
  readout(node_feat, edge_feat, f, pk, nullptr, key + "nc_", "nc_force");
  auto forces = out.forces;
  auto fscale = nc_force_scale_view_;
  Kokkos::parallel_for(
      "nc_forces", RangePolicy(0, NA), KOKKOS_LAMBDA(int a) {
        const bool own = a < N && a < nl;
        for (int c = 0; c < 3; ++c) forces(a, c) = own ? fscale(species(a)) * (double) f(a, c) : 0.0;
      });
  // Each structure's net force taken out, as metatomic does: a predicted force
  // need not sum to zero. Over several ranks a rank sees only its own atoms, so
  // the correction would differ by rank; it is left out there.
  if (!dev.exchange) {
    RView2D net = ws_.r2(key + "nc_net", NS, 3);
    sum_by_structure(forces, soff, net, false);
    auto sid = dev.struct_id;
    Kokkos::parallel_for(
        "nc_net_force", RangePolicy(0, NA), KOKKOS_LAMBDA(int a) {
          if (a >= nl) return;
          const int b = NS == 1 ? 0 : sid(a), end = soff(b + 1);
          const int n = (end < nl ? end : nl) - soff(b);  // the structure's owned atoms
          for (int c = 0; c < 3; ++c) forces(a, c) -= net(b, c) / n;
        });
  }

  auto vir9 = out.vir9;
  if (!has_nc_stress()) {
    Kokkos::deep_copy(ExecSpace(), vir9, 0.0);
    return out;
  }
  // metatrain's stress is the atoms' predictions summed over the volume, so
  // their sum is the virial W = V * stress.
  View2D s = ws_.n2(key + "nc_s", N, 9);
  readout(node_feat, edge_feat, s, pk, nullptr, key + "nc_", "nc_stress");
  RView2D w = ws_.r2(key + "nc_w", NA, 9);
  const double sscale = nc_stress_scale_[0];
  Kokkos::parallel_for(
      "nc_stress", RangePolicy(0, NA), KOKKOS_LAMBDA(int a) {
        const bool own = a < N && a < nl;
        for (int t = 0; t < 9; ++t) w(a, t) = own ? sscale * (double) s(a, t) : 0.0;
      });
  sum_by_structure(w, soff, vir9, false);
  return out;
}

// The adjoint of per_atom_net is 1 per atom, so node_pred's is 1 and edge_pred's
// is the edge's cutoff factor -- whose own adjoint is edge_pred. node_adj and
// edge_adj are written; cutoff_adj is added to when acc_cutoff.
void PetModel::readout_bwd(const ReadoutSaves& sav, int i, const PackedEdges& pk, int N,
                           View2D node_adj, View2D edge_adj, View1D cutoff_adj, bool acc_cutoff) {
  const int E = pk.E, Dh = h_.d_head;
  const bool acc = acc_cutoff;
  const std::string si = std::to_string(i);
  View2D npa = ws_.n2("ro:npa", N, 1), epa = ws_.n2("ro:epa", E, 1);
  auto epred = sav.epred[i];
  auto cut = pk.cut;
  auto center = pk.center;
  const int nl = pk.n_local;  // ghosts shape the energy but are not in it
  Kokkos::parallel_for(
      "ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { npa(n, 0) = n < nl ? Net(1) : Net(0); });
  auto off = pk.off;
  const int nc = pk.off.extent(0) - 1;
  Kokkos::parallel_for(
      "ro_seed_e", RangePolicy(0, E), KOKKOS_LAMBDA(int k) {
        const bool own = k < off(nc) && center(k) < nl;  // a capacity's dead rows: no one's
        epa(k, 0) = own ? cut(k) : Net(0);
        const Net ep = own ? epred(k, 0) : Net(0);
        cutoff_adj(k) = acc ? cutoff_adj(k) + ep : ep;
      });
  View2D nh1_adj = ws_.n2("ro:nh1_adj", N, Dh), nh0_adj = ws_.n2("ro:nh0_adj", N, Dh);
  linear_silu_bwd(nh1_adj, npa, mat("node_last_layers.energy." + si + ".energy___0.weight"), sav.nh1[i]);
  linear_silu_bwd(nh0_adj, nh1_adj, mat("node_heads.energy." + si + ".2.weight"), sav.nh0[i]);
  linear_bwd(node_adj, nh0_adj, mat("node_heads.energy." + si + ".0.weight"), 0);
  View2D eh1_adj = ws_.n2("ro:eh1_adj", E, Dh), eh0_adj = ws_.n2("ro:eh0_adj", E, Dh);
  linear_silu_bwd(eh1_adj, epa, mat("edge_last_layers.energy." + si + ".energy___0.weight"), sav.eh1[i]);
  linear_silu_bwd(eh0_adj, eh1_adj, mat("edge_heads.energy." + si + ".2.weight"), sav.eh0[i]);
  linear_bwd(edge_adj, eh0_adj, mat("edge_heads.energy." + si + ".0.weight"), 0);
}

// ---- evaluation -------------------------------------------------------------------

BatchResult PetModel::energy_forces_batch(const std::vector<System>& systems, bool compute_forces) {
  const int B = systems.size();
  if (B == 0) return {};
  std::vector<EdgeData> parts;
  for (const auto& s : systems) parts.push_back(build_edge_data(s, h_, species_to_index_));
  std::vector<int> sid;
  DeviceEdgeData dev = upload_edge_data(concat_edge_data(parts, sid));
  dev.struct_id = IView1D("struct_id", sid.size());
  Kokkos::deep_copy(dev.struct_id, Kokkos::View<const int*, Kokkos::HostSpace>(sid.data(), sid.size()));
  dev.n_struct = B;
  return energy_forces_batch(dev, compute_forces);
}

BatchResult PetModel::energy_forces_batch(const DeviceEdgeData& dev, bool compute_forces) {
  BatchResult out;
  const int N = dev.n_atoms, B = dev.n_struct;
  out.n_struct = B, out.n_atoms = N, out.struct_id = dev.struct_id;
  if (N == 0 || B == 0) return out;
  compute(dev, nullptr, compute_forces ? &out.forces : nullptr, &out.per_atom,
          compute_forces ? &out.virial : nullptr, &out.edge_grad);
  // Each structure's energy, summed over its atoms in order: the energy itself
  // must not depend on thread arrival order.
  out.energy = ws_.r1("batch_energy", B);
  sum_by_structure(RView2D(out.per_atom.data(), N, 1), structure_offsets(ws_, "batch", dev.struct_id, N, B),
                   RView2D(out.energy.data(), B, 1), false);
  return out;
}

// The rows of a chunk under Wide and Layers (chunk_rows_): an eighth of the
// edges, and at least 64k, where the GEMMs still run at full speed.
// PET_EDGE_CHUNK=rows overrides it, to test the chunking on small structures.
int PetModel::edge_chunk_rows(int E) const {
  if (const char* c = std::getenv("PET_EDGE_CHUNK"))
    if (std::atoi(c) > 0) return std::atoi(c);
  return std::max(65536, (E + 7) / 8);
}

Recompute PetModel::recompute_tier(int N, int S, int E) const {
  // Estimated from the shapes: what one layer saves, how much of that is the two
  // widest activations (qkv and the edge MLP's pre-activation), and the working
  // set every tier needs.
  const std::size_t G = h_.num_gnn_layers, A = h_.num_attention_layers, D = h_.d_pet, Dh = h_.d_head;
  // NS: the token rows (Tokens), an atom's own and one per edge.
  const std::size_t NS = std::size_t(N) + E, NE = E, ew = ffn_pre_width("gnn_layers.0.trans.layers.0.mlp");
  const bool res = h_.featurizer_type == FeaturizerType::Residual;
  // Residual, per block: qkv, the MLP pre-activation, s1, s2 and the attention's
  // merged heads; wide trades qkv and pre for the block's input tokens.
  const std::size_t layer = sizeof(Net) * (res ? A * NS * (6 * D + ew) + NE * D
                                               : A * (NS * 5 * D + NE * (ew + D)) + NE * 5 * D);
  const std::size_t wide = sizeof(Net) * (res ? A * NS * (2 * D + ew) : A * (NS * 3 * D + NE * ew));
  const std::size_t work = sizeof(Net) * (res ? NS * (9 * D + 2 * ew) + NE * (6 * D + 4 * Dh)
                                              : NE * (2 * ew + 6 * D + 4 * Dh));
  // Layers keeps each layer's inputs and one layer's saves at a time. The pool
  // comes out a quarter to a third above these terms (readout saves, packing,
  // the tmp pool's slack; measured on pet-mad-xs and -s), so they are scaled to
  // predict it.
  const std::size_t inputs = sizeof(Net) * G * ((res ? 0 : std::size_t(N) * h_.d_node) + NE * D);
  // Under Wide and Layers the edge MLP's two wide temporaries are a chunk's (and
  // on AMD's matrix cores so are the attention's qkv and its adjoint, which the
  // terms below never counted separately: they share the work term's room).
  const std::size_t chunk = res || !batch_invariant_gemms() ? NE : std::min<std::size_t>(NE, edge_chunk_rows(E));
  const std::size_t work_chunked = work - sizeof(Net) * (NE - chunk) * 2 * ew;
  // The pool came out at 0.86-0.96 of these for Wide and 0.99-1.07 for Layers
  // with the edge MLP and attention chunked (MI250X GCD, pet-mad-s, -xs,
  // pet-omat-s, 8k-166k atoms), so Layers takes a little more.
  auto need = [&](Recompute r) {
    const std::size_t terms = r == Recompute::Never ? work + G * layer
                              : r == Recompute::Wide ? work_chunked + G * (layer - wide)
                                                     : work_chunked + inputs + layer - wide;
    return terms + (r == Recompute::Layers ? terms / 100 * 45 : terms / 3);
  };
  auto fits = [&](Recompute r) { return mem_budget_ == 0 || need(r) <= mem_budget_; };
  const Recompute r = recompute_ != Recompute::Auto                    ? recompute_
                      : fits(Recompute::Never) && tier_floor_ == Recompute::Never ? Recompute::Never
                      : fits(Recompute::Wide) && tier_floor_ != Recompute::Layers ? Recompute::Wide
                                                                                  : Recompute::Layers;
  plan_.tier = r, plan_.estimate = need(r);
  return r;
}

std::string PetModel::memory_report() const {
  constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
  char line[256];
  std::string r;
  const MemoryPlan& p = plan_;
  std::snprintf(line, sizeof line, "  %d atoms, %d neighbours at most, %ld edges; recompute %s", p.atoms,
                p.max_neighbors, p.edges, recompute_name(p.tier));
  r += line;
  if (p.estimate) {
    std::snprintf(line, sizeof line, " (estimated %.2f GiB", p.estimate / GiB);
    r += line;
    if (p.budget) std::snprintf(line, sizeof line, " of a %.2f GiB budget)", p.budget / GiB), r += line;
    else r += ")";
  }
  const auto bufs = ws_.capacity_breakdown();
  std::snprintf(line, sizeof line, "\n  scratch pool %.2f GiB, largest:", ws_.capacity_bytes() / GiB);
  r += line;
  for (std::size_t i = 0; i < bufs.size() && i < 4; ++i)
    std::snprintf(line, sizeof line, " %s %.2f", bufs[i].first.c_str(), bufs[i].second / GiB), r += line;
  std::size_t free_b = 0, total_b = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) free_b = total_b = 0;
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) free_b = total_b = 0;
#endif
  if (total_b) std::snprintf(line, sizeof line, "\n  device %.2f GiB free of %.2f GiB", free_b / GiB, total_b / GiB), r += line;
  return r + "\n";
}

std::size_t PetModel::pass_elements(long centres, int max_neighbors, long edges) const {
  const std::size_t C = std::max(0L, centres), S = std::size_t(std::max(0, max_neighbors)) + 1;
  const std::size_t W = std::max({2 * h_.d_feedforward, 2 * h_.d_pet, h_.d_node, h_.d_head, 3 * h_.num_heads});
  const std::size_t rows = std::max(std::size_t(std::max(0L, edges)), C * (S - 1)) + C;
  return std::max(rows * W, C * std::size_t(h_.num_heads) * S * S);
}

int PetModel::max_index_directions(long centres, int max_neighbors, long edges) const {
  const std::size_t per = std::max<std::size_t>(1, pass_elements(centres, max_neighbors, edges));
  return int(std::min<std::size_t>(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / per)) - 1;
}

std::size_t PetModel::bytes_per_slot() const {
  const std::size_t G = h_.num_gnn_layers, A = h_.num_attention_layers, D = h_.d_pet, Dh = h_.d_head;
  const std::size_t ew = ffn_pre_width("gnn_layers.0.trans.layers.0.mlp");
  return sizeof(Net) * (G * (A * (5 * D + ew + D) + 5 * D) + 2 * ew + 6 * D + 4 * Dh);
}

EnergyResult PetModel::compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                               RView2D* dev_forces, RView1D* dev_per_atom, RView2D* dev_virial,
                               RView2D* dev_edge_grad) {
  const int N = dev.n_atoms, NS = dev.n_struct;
  const bool grad = host_forces || dev_forces;
  peak_max_neighbors_ = std::max(peak_max_neighbors_, dev.max_neighbors);
  peak_edge_slots_ = std::max(peak_edge_slots_, (long) N * std::max(1, dev.max_neighbors));

  // The pass sizes its edge tensors by the kept-edge count, so it needs it on the
  // host before anything a graph records.
  DeviceEdgeData d = dev;
  if (d.n_edges < 0) {
    auto mask = d.mask;
    Kokkos::parallel_reduce(
        "count_edges", RangePolicy(0, d.centres() * d.max_neighbors),
        KOKKOS_LAMBDA(int k, int& c) { c += mask(k) > 0.0; }, d.n_edges);
  }
  auto pass = [&] {
    return h_.featurizer_type == FeaturizerType::Residual ? residual_pass(d, grad) : ff_pass(d, grad);
  };
  // A graph cannot hold the exchange's calls to the engine.
  DeviceOut out;
  // The shapes now; the tier and estimate stay those of the pass that recorded a
  // replayed graph, or are 0 for a featurizer that never recomputes.
  plan_.atoms = N, plan_.max_neighbors = d.max_neighbors, plan_.edges = d.n_edges, plan_.budget = mem_budget_;
  if (max_index_directions(d.centres(), d.max_neighbors, d.n_edges) < 0)
    throw std::runtime_error(
        "pet: " + std::to_string(d.centres()) + " atoms with up to " + std::to_string(d.max_neighbors) +
        " neighbours make tensors of up to " +
        std::to_string(pass_elements(d.centres(), d.max_neighbors, d.n_edges)) +
        " elements, past the 2^31 the kernels index: too many atoms for one device. Split the structure "
        "across more devices (in LAMMPS, more GPUs or ranks).");
  for (;;) {
    try {
      out = graphs_ && !ozaki_active() && !d.exchange ? graph_.run(graph_key(d, grad), pass) : pass();
      break;
    } catch (const std::exception& e) {
      // Kokkos 5 throws a bad_alloc, Kokkos 4 a runtime_error that says so.
      const bool oom = dynamic_cast<const std::bad_alloc*>(&e) || std::strstr(e.what(), "failed to allocate");
      if (!oom) throw;
      // The estimate was wrong for this device: under Auto, give the pool back
      // and run again a tier further down, from now on.
      if (recompute_ == Recompute::Auto && h_.featurizer_type != FeaturizerType::Residual &&
          plan_.tier != Recompute::Layers) {
        tier_floor_ = plan_.tier == Recompute::Never ? Recompute::Wide : Recompute::Layers;
        ws_.release();
        continue;
      }
      throw std::runtime_error(std::string("out of device memory: ") + e.what() + "\n" + memory_report() +
                               "Fewer atoms per device (more ranks or GPUs) would fit.");
    }
  }

  uq_last_ = out.uq;
  EnergyResult res;
  if (dev_per_atom) *dev_per_atom = out.per_atom;
  else {
    auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.per_atom);
    res.per_atom.assign(e.data(), e.data() + N);
    for (double x : res.per_atom) res.total += x;
  }
  if (!grad) return res;

  // The symmetric virial, Voigt order [xx, yy, zz, xy, xz, yz].
  const RView2D w9 = out.vir9;
  if (dev_virial) {
    RView2D w6 = ws_.r2("virial6", NS, 6);
    Kokkos::parallel_for(
        "virial_voigt", RangePolicy(0, NS), KOKKOS_LAMBDA(int b) {
          w6(b, 0) = w9(b, 0), w6(b, 1) = w9(b, 4), w6(b, 2) = w9(b, 8);
          w6(b, 3) = 0.5 * (w9(b, 1) + w9(b, 3));
          w6(b, 4) = 0.5 * (w9(b, 2) + w9(b, 6));
          w6(b, 5) = 0.5 * (w9(b, 5) + w9(b, 7));
        });
    *dev_virial = w6;
  } else {
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w9);
    const double v[6] = {w(0, 0), w(0, 4), w(0, 8), 0.5 * (w(0, 1) + w(0, 3)),
                         0.5 * (w(0, 2) + w(0, 6)), 0.5 * (w(0, 5) + w(0, 7))};
    std::copy(v, v + 6, res.virial);
  }
  if (dev_forces) *dev_forces = out.forces;
  if (dev_edge_grad) *dev_edge_grad = out.edge_grad;
  if (host_forces) {
    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.forces);
    host_forces->assign(f.data(), f.data() + std::size_t(N) * 3);
  }
  return res;
}

// Everything recorded device work depends on besides the model: the shapes,
// where each input lives, and the workspace's allocation generation.
std::vector<std::uintptr_t> PetModel::graph_key(const DeviceEdgeData& d, bool grad) const {
  std::vector<std::uintptr_t> k{(std::uintptr_t) grad,       (std::uintptr_t) nc_,       (std::uintptr_t) uq_,       (std::uintptr_t) d.n_atoms, (std::uintptr_t) d.n_centres,
                                (std::uintptr_t) d.max_neighbors, (std::uintptr_t) d.n_raw,
                                (std::uintptr_t) d.n_struct,  (std::uintptr_t) d.n_edges,
                                (std::uintptr_t) d.n_local,   ws_.generation(),
                                lowp_generation(), numerics_key()};
  auto add = [&k](const auto& v) { k.push_back((std::uintptr_t) v.data()), k.push_back(v.size()); };
  add(d.species), add(d.neigh_species), add(d.reverse_index), add(d.edge_vec), add(d.dist);
  add(d.mask), add(d.pair_cutoff), add(d.cutoff_factor), add(d.cf_seq), add(d.raw_center);
  add(d.raw_neigh), add(d.raw_dist), add(d.raw_vec), add(d.raw_off), add(d.raw_reverse);
  add(d.adapt_eff), add(d.adapt_r), add(d.adapt_dn), add(d.struct_id), add(d.charge);
  add(d.spin_multiplicity), add(d.raw_slot), add(d.orphan_off), add(d.orphan_edge);
  return k;
}

}  // namespace pet
