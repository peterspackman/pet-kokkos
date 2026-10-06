// The residual featurizer (PostLN, LayerNorm, SiLU, non-expanded central token).
//
// Per GNN layer L: the node is re-embedded from species; the edge tokens are
//   et = compress(input_edge, geometry, species);
// a PostLN transformer of A blocks runs over each atom's tokens [node; et]; every
// layer is read out, and the readouts summed; and message passing sets
//   input_edge = 0.5 * (input_edge + out_edge of the reverse edge)
// for the next layer.
#include "ops.hpp"

#include <stdexcept>

namespace pet {

DeviceOut PetModel::residual_pass(const DeviceEdgeData& dev, bool grad) {
  if (dev.exchange)
    throw std::runtime_error("pet: the residual featurizer does not run over several ranks (exchange) yet");
  const int N = dev.n_atoms, S = dev.max_neighbors + 1;
  const int D = h_.d_pet, G = h_.num_gnn_layers, A = h_.num_attention_layers, R = h_.num_readout_layers;
  Workspace::Scope scope(ws_);
  ws_.set_zero(false);

  const PackedEdges pk = pack_edges(dev);
  const int E = pk.E;
  auto off = pk.off, rev = pk.reverse, center = pk.center;
  const View2D cond = conditioning(dev);
  View2D input_edge = ws_.n2("re_input_edge", E, D);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);

  // Saved for the backward, per layer and per block.
  std::vector<View2D> node_feat(G), edge_feat(G), sav_cpre(G);
  std::vector<std::vector<View2D>> sav_qkv(G, std::vector<View2D>(A)), sav_s1 = sav_qkv, sav_s2 = sav_qkv,
                                   sav_pre = sav_qkv;
  auto saved = [&](const std::string& name, int L, int a, int r, int c) {
    return ws_.n2("rs_" + name + "_" + std::to_string(L) + "_" + std::to_string(a), r, c);
  };

  for (int L = 0; L < G; ++L) {
    const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
    View2D node_L = ws_.n2("re_node_L", N, D);
    gather(node_L, mat("node_embedders." + ls + ".weight").v, dev.species);
    View2D cpre = ws_.n2("re_cpre", E, D), et = ws_.n2("re_et", E, D);
    if (grad) sav_cpre[L] = ws_.n2("rs_cpre_" + ls, E, D);
    compress_fwd(cpre, sav_cpre[L], compress_fold(L), input_edge, pk);
    linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

    View2D tokens = ws_.n2("re_tokens", N * S, D);
    Kokkos::parallel_for(
        "re_tok", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
          const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
          tokens(row, d) = row % S == 0 ? node_L(n, d) : k < off(n + 1) ? et(k, d) : Net(0);
        });

    // PostLN: tokens = norm(tokens + attention(tokens)), then norm(t + mlp(t)).
    for (int a = 0; a < A; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string key = "re_attn_" + ls + "_" + std::to_string(a);
      const int pw = ffn_pre_width(tl + ".mlp");
      Workspace::Scope block_scope(ws_);
      View2D qkv = grad ? (sav_qkv[L][a] = saved("qkv", L, a, N * S, 3 * D)) : ws_.n2("re_qkv", N * S, 3 * D);
      View2D s1 = grad ? (sav_s1[L][a] = saved("s1", L, a, N * S, D)) : ws_.n2("re_attn_out", N * S, D);
      View2D s2 = grad ? (sav_s2[L][a] = saved("s2", L, a, N * S, D)) : ws_.n2("re_ff", N * S, D);
      sav_pre[L][a] = grad ? saved("pre", L, a, N * S, pw) : ws_.tmp(N * S, pw);
      linear(qkv, tokens, mat(tl + ".attention.input_linear.weight"), vec(tl + ".attention.input_linear.bias"));
      attention(ws_, key, s1, qkv, dev.cf_seq, mat(tl + ".attention.output_linear.weight"),
                vec(tl + ".attention.output_linear.bias"), N, S, h_.num_heads, h_.head_dim,
                h_.attention_temperature, grad);
      add_inplace(s1, tokens);
      View2D tok_a = ws_.n2("re_tok_a", N * S, D);
      norm(tok_a, s1, tl + ".norm_attention");
      feedforward(s2, tok_a, tl + ".mlp", sav_pre[L][a], grad);
      add_inplace(s2, tok_a);
      norm(tokens, s2, tl + ".norm_mlp");
    }

    View2D node_out = node_feat[L] = ws_.n2("re_nfeat_" + ls, N, D);
    View2D edge_out = edge_feat[L] = ws_.n2("re_efeat_" + ls, E, D);
    Kokkos::parallel_for(
        "re_split", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
          const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
          if (row % S == 0) node_out(n, d) = tokens(row, d);
          else if (k < off(n + 1)) edge_out(k, d) = tokens(row, d);
        });
    if (cond.extent(0) == (std::size_t) N)
      Kokkos::parallel_for(
          "re_cond_add", RangePolicy(0, N * D), KOKKOS_LAMBDA(int i) { node_out(i / D, i % D) += cond(i / D, i % D); });
    if (L + 1 < G)
      Kokkos::parallel_for(
          "re_msg", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
            const int k = i / D, d = i % D;
            input_edge(k, d) = Net(0.5) * (input_edge(k, d) + (rev(k) >= 0 ? edge_out(rev(k), d) : Net(0)));
          });
  }

  View1D net = ws_.n1("re_per_atom_net", N);
  ReadoutSaves rs;
  readout(node_feat, edge_feat, net, pk, grad ? &rs : nullptr, "re_");
  RView1D per_atom = ws_.r1("re_per_atom", N);
  assemble_energy(per_atom, net, dev.species, comp_view_, energy_scale_, pk.n_local);
  if (!grad) return {per_atom, {}, {}, {}};

  // The backward accumulates in many places, so here the pool zeroes what it
  // hands out.
  ws_.set_zero(true);
  View2D x4_adj = ws_.n2("re_x4_adj", E, 4), cf_seq_adj = ws_.n2("re_cf_seq_adj", N, S);
  View1D cutoff_adj = ws_.n1("re_cutoff_adj", E);
  std::vector<View2D> node_adj(G), edge_adj(G);  // of each layer's read-out features
  for (int L = 0; L < G; ++L)
    node_adj[L] = ws_.n2("re_nfa_" + std::to_string(L), N, D), edge_adj[L] = ws_.n2("re_efa_" + std::to_string(L), E, D);
  for (int i = 0; i < R; ++i) readout_bwd(rs, i, pk, N, node_adj[i], edge_adj[i], cutoff_adj, true);

  // ie_adj: the adjoint of the next layer's input_edge (zero past the last).
  View2D ie_adj = ws_.n2("re_ie_adj", E, D);
  for (int L = G - 1; L >= 0; --L) {
    const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
    auto efa = edge_adj[L], nfa = node_adj[L];
    if (L + 1 < G)  // message passing: half of the next input's adjoint, via the reverse edge
      Kokkos::parallel_for(
          "reb_msg", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
            const int k = i / D, d = i % D;
            if (rev(k) >= 0) efa(rev(k), d) += Net(0.5) * ie_adj(k, d);
          });
    View2D tokens_adj = ws_.n2("re_tokens_adj", N * S, D);
    Kokkos::parallel_for(
        "reb_tok", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
          const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
          tokens_adj(row, d) = row % S == 0 ? nfa(n, d) : k < off(n + 1) ? efa(k, d) : Net(0);
        });

    for (int a = A - 1; a >= 0; --a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string key = "re_attn_" + ls + "_" + std::to_string(a);
      // tokens_out = norm(s2), s2 = tok_a + mlp(tok_a)
      View2D s2_adj = ws_.n2("re_s2_adj", N * S, D), tok_a_adj = ws_.n2("re_tok_a_adj", N * S, D);
      norm_bwd(s2_adj, tokens_adj, sav_s2[L][a], tl + ".norm_mlp");
      copy(tok_a_adj, s2_adj);
      feedforward_bwd(tok_a_adj, s2_adj, tl + ".mlp", sav_pre[L][a]);
      // tok_a = norm(s1), s1 = tokens_in + attention(tokens_in): the attention's
      // share accumulates onto the residual's, which is then tokens_in's adjoint.
      View2D s1_adj = ws_.n2("re_s1_adj_" + std::to_string(a % 2), N * S, D);
      norm_bwd(s1_adj, tok_a_adj, sav_s1[L][a], tl + ".norm_attention");
      attention_bwd(ws_, key, s1_adj, cf_seq_adj, s1_adj, sav_qkv[L][a], dev.cf_seq,
                    mat(tl + ".attention.input_linear.weight"), mat(tl + ".attention.output_linear.weight"),
                    N, S, h_.num_heads, h_.head_dim, h_.attention_temperature, 1);
      tokens_adj = s1_adj;
    }

    // tokens = [node_L ; et]: only et depends on the geometry. input_edge feeds
    // compress.0 and, through message passing, carries half of ie_adj. The two
    // buffers alternate by layer, since the pool zeroes what it hands out.
    View2D et_adj = ws_.n2("re_et_adj", E, D), cpre_adj = ws_.n2("re_cpre_adj", E, D);
    View2D ie_next = ws_.n2("re_ie_" + std::to_string(L % 2), E, D);
    const bool has_msg = L + 1 < G;
    Kokkos::parallel_for(
        "reb_et", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
          const int k = i / D, d = i % D, n = center(k);
          et_adj(k, d) = k < off(N) ? tokens_adj(n * S + 1 + k - off(n), d) : Net(0);  // past: a capacity's dead rows
          ie_next(k, d) = has_msg ? Net(0.5) * ie_adj(k, d) : Net(0);
        });
    linear_bwd(cpre_adj, et_adj, mat(g + ".compress.2.weight"));
    compress_bwd(cpre_adj, sav_cpre[L], compress_fold(L), x4_adj, ie_next);
    ie_adj = ie_next;
  }

  DeviceOut out{per_atom, {}, {}, {}};
  forces_and_virial(ws_, dev, pk, h_, probes_, n_probes_, energy_scale_, x4_adj, cutoff_adj, cf_seq_adj,
                    out.forces, out.vir9, out.edge_grad);
  return out;
}

}  // namespace pet
