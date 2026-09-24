// PetModel: loading, the host upload, packing, the parts both featurizers share,
// and compute(). The featurizers themselves are in feedforward.cpp and
// residual.cpp; the building blocks in ops.hpp.
#include "ops.hpp"

#include "pet/ozaki.hpp"

#include <algorithm>
#include <stdexcept>

namespace pet {

// ---- loading --------------------------------------------------------------------

PetModel::PetModel(const Checkpoint& ckpt)
    : h_(ckpt.hypers),
      energy_scale_(ckpt.energy_scale),
      composition_(ckpt.composition_energies),
      species_to_index_(ckpt.species_to_index) {
  load_all(ckpt);

  // The adaptive cutoff's probe grid, spaced by its own taper width. The same
  // grid as neighbors.hpp and device_neighbors.hpp.
  std::vector<double> p;
  for (double r = 0.5; r < h_.cutoff - 1e-12; r += h_.cutoff_width_adaptive / 4.0) p.push_back(r);
  n_probes_ = (int) p.size();
  probes_ = RView1D("probes", std::max(n_probes_, 1));
  auto hp = Kokkos::create_mirror_view(probes_);
  for (int i = 0; i < n_probes_; ++i) hp(i) = p[i];
  Kokkos::deep_copy(probes_, hp);

  comp_view_ = RView1D("composition", composition_.size());
  auto hc = Kokkos::create_mirror_view(comp_view_);
  for (std::size_t i = 0; i < composition_.size(); ++i) hc(i) = composition_[i];
  Kokkos::deep_copy(comp_view_, hc);
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

// Each edge to a ghost: its packed row, and back.
static void remote_map(const DeviceEdgeData& d, PackedEdges& p) {
  if (p.n_remote == 0) return;
  auto raw = d.remote_raw, slot = d.raw_slot, slot_edge = p.slot_edge, of = p.remote_of, packed = p.remote_packed;
  Kokkos::deep_copy(ExecSpace(), of, -1);
  Kokkos::parallel_for(
      "pk_remote", Kokkos::RangePolicy<ExecSpace>(0, p.n_remote), KOKKOS_LAMBDA(int r) {
        const int s = slot(raw(r)), k = s >= 0 ? slot_edge(s) : -1;
        packed(r) = k;
        if (k >= 0) of(k) = r;
      });
}

PackedEdges PetModel::pack_edges(const DeviceEdgeData& d) {
  const int N = d.n_atoms, M = d.max_neighbors;
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
  if (d.exchange && d.remote_raw.extent(0) > 0) {  // mapped once packed (remote_map)
    p.n_remote = d.remote_raw.extent(0);
    p.remote_of = ws_.i1("pk:remote_of", p.E), p.remote_packed = ws_.i1("pk:remote_packed", p.n_remote);
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
    remote_map(d, p);
    return p;
  }
  Kokkos::parallel_scan(
      "pk_off", RangePolicy(0, N), KOKKOS_LAMBDA(int n, int& upd, bool final) {
        if (final) off(n) = upd;
        upd += kept(n, M);
        if (final && n == N - 1) off(N) = upd;
      });
  if (N == 0) Kokkos::deep_copy(ExecSpace(), off, 0);
  Kokkos::parallel_for(
      "pk_edges", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        const int n = k / M;
        if (mask(k) <= 0.0) return (void) (pos(k) = -1);
        const int e = off(n) + kept(n, k % M);
        pos(k) = e, center(e) = n, species(e) = nsp(k);
        for (int c = 0; c < 3; ++c) vec(e, c) = ev(k, c);
        dist(e) = dd(k), pcut(e) = dpc(k), cut(e) = dcut(k);
      });
  Kokkos::parallel_for(
      "pk_reverse", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        if (pos(k) >= 0) reverse(pos(k)) = rev(k) >= 0 ? pos(rev(k)) : -1;
      });
  remote_map(d, p);
  return p;
}

// Charge and spin conditioning: per structure, embed the charge and the spin
// multiplicity, project (Linear-SiLU-Linear), and broadcast to its atoms. Added
// to the node features after every GNN layer. It depends on no position, so it
// has no backward. Empty for a model without it.
View2D PetModel::conditioning(const DeviceEdgeData& dev) {
  if (!h_.system_conditioning) return View2D();
  const int N = dev.n_atoms, NS = dev.n_struct, Dn = h_.d_node, max_q = h_.max_charge;
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

void PetModel::norm(View2D out, View2D in, const std::string& key) const {
  const bool ln = h_.normalization == Normalization::LayerNorm;
  norm_fwd(out, in, vec(key + ".weight"), ln ? vec(key + ".bias") : View1D());
}

void PetModel::norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key, bool acc) const {
  pet::norm_bwd(in_adj, out_adj, in, vec(key + ".weight"), h_.normalization == Normalization::LayerNorm, acc);
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
  linear_bwd(h_adj, out_adj, w_out, 0);
  silu_bwd(h_adj, pre);
  linear_bwd(in_adj, h_adj, w_in, beta);
}

void PetModel::readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
                       View1D per_atom_net, const PackedEdges& pk, ReadoutSaves* sav,
                       const std::string& key) {
  const int R = node_feat.size(), N = per_atom_net.extent(0), E = pk.E, Dh = h_.d_head;
  if (sav)
    for (auto* v : {&sav->nh0, &sav->nh1, &sav->eh0, &sav->eh1, &sav->epred}) v->assign(R, View2D());
  // A buffer the backward reads is saved under a per-layer name, otherwise scratch.
  auto buf = [&](std::vector<View2D>* s, const std::string& name, int i, int rows, int cols) {
    return sav ? ((*s)[i] = ws_.n2(key + "s_" + name + "_" + std::to_string(i), rows, cols))
               : ws_.n2(key + name, rows, cols);
  };
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i), nh = "node_heads.energy." + si, eh = "edge_heads.energy." + si;
    View2D nh0 = ws_.n2(key + "nh0", N, Dh), nh1 = ws_.n2(key + "nh1", N, Dh);
    linear_silu(nh0, sav ? buf(&sav->nh0, "nh0", i, N, Dh) : View2D(), node_feat[i], mat(nh + ".0.weight"),
                vec(nh + ".0.bias"));
    linear_silu(nh1, sav ? buf(&sav->nh1, "nh1", i, N, Dh) : View2D(), nh0, mat(nh + ".2.weight"),
                vec(nh + ".2.bias"));
    View2D node_pred = ws_.n2(key + "node_pred", N, 1);
    linear(node_pred, nh1, mat("node_last_layers.energy." + si + ".energy___0.weight"),
           vec("node_last_layers.energy." + si + ".energy___0.bias"));

    View2D eh0 = ws_.n2(key + "eh0", E, Dh), eh1 = ws_.n2(key + "eh1", E, Dh);
    linear_silu(eh0, sav ? buf(&sav->eh0, "eh0", i, E, Dh) : View2D(), edge_feat[i], mat(eh + ".0.weight"),
                vec(eh + ".0.bias"));
    linear_silu(eh1, sav ? buf(&sav->eh1, "eh1", i, E, Dh) : View2D(), eh0, mat(eh + ".2.weight"),
                vec(eh + ".2.bias"));
    View2D edge_pred = sav ? buf(&sav->epred, "epred", i, E, 1) : ws_.n2(key + "edge_pred", E, 1);
    linear(edge_pred, eh1, mat("edge_last_layers.energy." + si + ".energy___0.weight"),
           vec("edge_last_layers.energy." + si + ".energy___0.bias"));

    readout_accumulate(per_atom_net, node_pred, edge_pred, pk, i > 0);
  }
}

// The adjoint of per_atom_net is 1 per atom, so node_pred's is 1 and edge_pred's
// is the edge's cutoff factor -- whose own adjoint is edge_pred.
void PetModel::readout_bwd(const ReadoutSaves& sav, int i, const PackedEdges& pk, int N,
                           View2D node_adj, View2D edge_adj, View1D cutoff_adj, bool acc) {
  const int E = pk.E, Dh = h_.d_head;
  const Net beta = acc ? 1 : 0;
  const std::string si = std::to_string(i);
  View2D npa = ws_.n2("ro:npa", N, 1), epa = ws_.n2("ro:epa", E, 1);
  auto epred = sav.epred[i];
  auto cut = pk.cut;
  auto center = pk.center;
  const int nl = pk.n_local;  // ghosts shape the energy but are not in it
  Kokkos::parallel_for(
      "ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { npa(n, 0) = n < nl ? Net(1) : Net(0); });
  Kokkos::parallel_for(
      "ro_seed_e", RangePolicy(0, E), KOKKOS_LAMBDA(int k) {
        const bool own = center(k) < nl;
        epa(k, 0) = own ? cut(k) : Net(0);
        const Net ep = own ? epred(k, 0) : Net(0);
        cutoff_adj(k) = acc ? cutoff_adj(k) + ep : ep;
      });
  View2D nh1_adj = ws_.n2("ro:nh1_adj", N, Dh), nh0_adj = ws_.n2("ro:nh0_adj", N, Dh);
  linear_bwd(nh1_adj, npa, mat("node_last_layers.energy." + si + ".energy___0.weight"), beta);
  silu_bwd(nh1_adj, sav.nh1[i]);
  linear_bwd(nh0_adj, nh1_adj, mat("node_heads.energy." + si + ".2.weight"), beta);
  silu_bwd(nh0_adj, sav.nh0[i]);
  linear_bwd(node_adj, nh0_adj, mat("node_heads.energy." + si + ".0.weight"), beta);
  View2D eh1_adj = ws_.n2("ro:eh1_adj", E, Dh), eh0_adj = ws_.n2("ro:eh0_adj", E, Dh);
  linear_bwd(eh1_adj, epa, mat("edge_last_layers.energy." + si + ".energy___0.weight"), beta);
  silu_bwd(eh1_adj, sav.eh1[i]);
  linear_bwd(eh0_adj, eh1_adj, mat("edge_heads.energy." + si + ".2.weight"), beta);
  silu_bwd(eh0_adj, sav.eh0[i]);
  linear_bwd(edge_adj, eh0_adj, mat("edge_heads.energy." + si + ".0.weight"), beta);
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

Recompute PetModel::recompute_tier(int N, int S, int E) const {
  if (recompute_ != Recompute::Auto) return recompute_;
  // Estimated from the shapes: what one layer saves, how much of that is the two
  // widest activations (qkv and the edge MLP's pre-activation), and the working
  // set every tier needs.
  const std::size_t G = h_.num_gnn_layers, A = h_.num_attention_layers, D = h_.d_pet, Dh = h_.d_head;
  const std::size_t NS = std::size_t(N) * S, NE = E, ew = ffn_pre_width("gnn_layers.0.trans.layers.0.mlp");
  const std::size_t layer = sizeof(Net) * (A * (NS * 5 * D + NE * (ew + D)) + NE * 5 * D);
  const std::size_t wide = sizeof(Net) * A * (NS * 3 * D + NE * ew);
  const std::size_t work = sizeof(Net) * NE * (2 * ew + 6 * D + 4 * Dh);
  auto fits = [&](std::size_t b) { return mem_budget_ == 0 || work + b <= mem_budget_; };
  return fits(G * layer) ? Recompute::Never : fits(G * (layer - wide)) ? Recompute::Wide : Recompute::Layers;
}

EnergyResult PetModel::compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                               RView2D* dev_forces, RView1D* dev_per_atom, RView2D* dev_virial,
                               RView2D* dev_edge_grad) {
  const int N = dev.n_atoms, NS = dev.n_struct;
  const bool grad = host_forces || dev_forces;
  last_n_atoms_ = N;
  last_max_neighbors_ = dev.max_neighbors;
  peak_max_neighbors_ = std::max(peak_max_neighbors_, dev.max_neighbors);
  peak_edge_slots_ = std::max(peak_edge_slots_, (long) N * std::max(1, dev.max_neighbors));

  // The pass sizes its edge tensors by the kept-edge count, so it needs it on the
  // host before anything a graph records.
  DeviceEdgeData d = dev;
  if (d.n_edges < 0) {
    auto mask = d.mask;
    Kokkos::parallel_reduce(
        "count_edges", RangePolicy(0, N * d.max_neighbors),
        KOKKOS_LAMBDA(int k, int& c) { c += mask(k) > 0.0; }, d.n_edges);
  }
  auto pass = [&] {
    return h_.featurizer_type == FeaturizerType::Residual ? residual_pass(d, grad) : ff_pass(d, grad);
  };
  // A graph cannot hold the exchange's calls to the engine.
  const DeviceOut out = graphs_ && !ozaki_active() && !d.exchange ? graph_.run(graph_key(d, grad), pass) : pass();

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
  std::vector<std::uintptr_t> k{(std::uintptr_t) grad,       (std::uintptr_t) d.n_atoms,
                                (std::uintptr_t) d.max_neighbors, (std::uintptr_t) d.n_raw,
                                (std::uintptr_t) d.n_struct,  (std::uintptr_t) d.n_edges,
                                (std::uintptr_t) d.n_local,   ws_.generation()};
  auto add = [&k](const auto& v) { k.push_back((std::uintptr_t) v.data()), k.push_back(v.size()); };
  add(d.species), add(d.neigh_species), add(d.reverse_index), add(d.edge_vec), add(d.dist);
  add(d.mask), add(d.pair_cutoff), add(d.cutoff_factor), add(d.cf_seq), add(d.raw_center);
  add(d.raw_neigh), add(d.raw_dist), add(d.raw_vec), add(d.raw_off), add(d.raw_reverse);
  add(d.adapt_eff), add(d.adapt_r), add(d.adapt_dn), add(d.struct_id), add(d.charge);
  add(d.spin_multiplicity), add(d.raw_slot), add(d.orphan_off), add(d.orphan_edge);
  return k;
}

}  // namespace pet
