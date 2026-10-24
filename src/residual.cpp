// The residual featurizer (PostLN, LayerNorm, SiLU, non-expanded central token).
//
// Per GNN layer L: the node is re-embedded from species; the edge tokens are
//   et = compress(input_edge, geometry, species);
// a PostLN transformer of A blocks runs over each atom's tokens [node; et]; every
// layer is read out, and the readouts summed; and message passing sets
//   input_edge = 0.5 * (input_edge + out_edge of the reverse edge)
// for the next layer.
#include "ops.hpp"

namespace pet {

void PetModel::residual_layer(const DeviceEdgeData& dev, const PackedEdges& pk, int L, View2D input_edge,
                              View2D tokens, ResidualSaves* sav, bool save_wide) {
  const int N = dev.centres(), S = dev.max_neighbors + 1, E = pk.E;
  const int D = h_.d_pet, A = h_.num_attention_layers;
  auto off = pk.off;
  const Tokens tk{off, N, dev.edge_capacity};
  const int T = N + E;  // token rows (Tokens): the atoms', then the edges'
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  const bool save = sav != nullptr, wide = save && save_wide;
  if (save) sav->qkv.assign(A, View2D()), sav->pre = sav->tin = sav->s1 = sav->s2 = sav->qkv;
  auto keep = [&](const std::string& name, int a, int r, int c) {
    return ws_.n2("rs_" + name + "_" + sav->tag + "_" + std::to_string(a), r, c);
  };

  // The tokens [node_L ; et] (Tokens), written in place: the species embedding
  // into the atom rows, compress into the edge rows. They become the layer's
  // output there, the caller's node and edge features.
  gather(View2D(tokens.data(), N, D), mat("node_embedders." + ls + ".weight").v, dev.species);
  View2D cpre = ws_.n2("re_cpre", E, D);
  if (save) sav->cpre = ws_.n2("rs_cpre_" + sav->tag, E, D);
  compress_fwd(cpre, save ? sav->cpre : View2D(), compress_fold(L), input_edge, pk);
  linear(View2D(tokens.data() + std::size_t(N) * D, E, D), cpre, mat(g + ".compress.2.weight"),
         vec(g + ".compress.2.bias"));

  // PostLN: tokens = norm(tokens + attention(tokens)), then norm(t + mlp(t)).
  for (int a = 0; a < A; ++a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    const std::string key = "re_attn_" + (save ? sav->tag : ls) + "_" + std::to_string(a);
    const int pw = ffn_pre_width(tl + ".mlp");
    if (save && !wide) copy(sav->tin[a] = keep("tin", a, T, D), tokens);
    View2D s1 = save ? (sav->s1[a] = keep("s1", a, T, D)) : ws_.n2("re_attn_out", T, D);
    View2D s2 = save ? (sav->s2[a] = keep("s2", a, T, D)) : ws_.n2("re_ff", T, D);
    {
      Workspace::Scope scope(ws_);
      View2D qkv = wide ? (sav->qkv[a] = keep("qkv", a, T, 3 * D)) : ws_.tmp(T, 3 * D);
      linear(qkv, tokens, mat(tl + ".attention.input_linear.weight"), vec(tl + ".attention.input_linear.bias"));
      attention(ws_, key, s1, qkv, dev.cf_seq, mat(tl + ".attention.output_linear.weight"),
                vec(tl + ".attention.output_linear.bias"), tk, S, h_.num_heads, h_.head_dim,
                h_.attention_temperature, save);
    }
    View2D tok_a = ws_.n2("re_tok_a", T, D);
    norm(tok_a, s1, tl + ".norm_attention", tokens);  // s1 += tokens first
    {
      Workspace::Scope scope(ws_);
      View2D pre = wide ? (sav->pre[a] = keep("pre", a, T, pw)) : ws_.tmp(T, pw);
      feedforward(s2, tok_a, tl + ".mlp", pre, wide);
    }
    norm(tokens, s2, tl + ".norm_mlp", tok_a);  // s2 += tok_a first
  }
}

DeviceOut PetModel::residual_pass(const DeviceEdgeData& dev, bool grad) {
  const int N = dev.centres(), S = dev.max_neighbors + 1;  // the model's rows
  const int D = h_.d_pet, G = h_.num_gnn_layers, A = h_.num_attention_layers, R = h_.num_readout_layers;
  Workspace::Scope scope(ws_);
  ws_.set_zero(false);

  const PackedEdges pk = pack_edges(dev);
  const int E = pk.E;
  auto off = pk.off, rev = pk.reverse;
  const Tokens tk{off, N, dev.edge_capacity};
  const int T = N + E;
  // Over several ranks, an edge to a ghost passes its message to and from the
  // reverse edge on the ghost's own rank: its row among the live ones (ro), and
  // each live row's packed edge (lp). The exchange is collective: every rank
  // takes part in each one, rows or not.
  const bool remote = dev.exchange != nullptr;
  auto ro = pk.remote_of, lp = pk.live_packed;
  // Rows of `of` for the live edges to ghosts, scaled; the partners' rows back.
  auto swap_rows = [&](View2D of, Net scale) {
    View2D out = ws_.tmp(pk.n_live, D), in = ws_.tmp(pk.n_live, D);
    Kokkos::parallel_for(
        "re_remote_rows", RangePolicy(0, pk.n_live * D), KOKKOS_LAMBDA(int i) {
          const int k = lp(i / D);  // -1: a capacity's padding
          out(i / D, i % D) = k >= 0 ? scale * of(k, i % D) : Net(0);
        });
    dev.exchange->edges(out, in);
    return in;
  };
  const View2D cond = conditioning(dev);
  View2D input_edge = ws_.n2("re_input_edge", E, D);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);

  // What the backward will have: every layer's saves; the same with each block's
  // input tokens in place of the two widest (rebuilt when needed); or only each
  // layer's input_edge, the backward then re-running one layer at a time.
  const Recompute rc = recompute_tier(N, S, E);
  const bool keep_wide = rc == Recompute::Never, ckpt = grad && rc == Recompute::Layers;
  std::vector<ResidualSaves> sav(grad && !ckpt ? G : 0);
  std::vector<View2D> inputs(ckpt ? G : 0);
  // Each layer's tokens (Tokens): its atom rows and edge rows are its node and
  // edge features.
  auto atom_rows = [&](View2D t) { return View2D(t.data(), N, D); };
  auto edge_rows = [&](View2D t) { return View2D(t.data() + std::size_t(N) * D, E, D); };
  std::vector<View2D> node_feat(G), edge_feat(G);

  for (int L = 0; L < G; ++L) {
    const std::string ls = std::to_string(L);
    if (ckpt) copy(inputs[L] = ws_.n2("rck_ie_" + ls, E, D), input_edge);
    if (!sav.empty()) sav[L].tag = ls;
    const View2D tokens = ws_.n2("re_tokens_" + ls, T, D);
    View2D node_out = node_feat[L] = atom_rows(tokens), edge_out = edge_feat[L] = edge_rows(tokens);
    residual_layer(dev, pk, L, input_edge, tokens, sav.empty() ? nullptr : &sav[L], keep_wide);
    if (cond.extent(0) == (std::size_t) N)
      Kokkos::parallel_for(
          "re_cond_add", RangePolicy(0, N * D), KOKKOS_LAMBDA(int i) { node_out(i / D, i % D) += cond(i / D, i % D); });
    if (L + 1 < G) {
      Workspace::Scope msg_scope(ws_);
      const View2D partner = remote ? swap_rows(edge_out, Net(1)) : View2D();
      Kokkos::parallel_for(
          "re_msg", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
            const int k = i / D, d = i % D, r = rev(k);
            const Net back = r >= 0 ? edge_out(r, d) : remote && ro(k) >= 0 ? partner(ro(k), d) : Net(0);
            input_edge(k, d) = Net(0.5) * (input_edge(k, d) + back);
          });
    }
  }

  View1D net = ws_.n1("re_per_atom_net", N);
  ReadoutSaves rs;
  readout(node_feat, edge_feat, net, pk, grad ? &rs : nullptr, "re_");
  RView1D per_atom = ws_.r1("re_per_atom", dev.n_atoms);  // every atom's, 0 past n_local
  assemble_energy(per_atom, net, dev.species, comp_view_, energy_scale_, pk.n_local);
  if (!grad) return {per_atom, {}, {}, {}};

  // As in the forward, every buffer's first writer overwrites it, and the few
  // the backward accumulates into are zeroed here: one memset per evaluation
  // each, not one per buffer handed out.
  View2D x4_adj = ws_.n2("re_x4_adj", E, 4), cf_seq_adj = ws_.n2("re_cf_seq_adj", N, S);
  Kokkos::deep_copy(ExecSpace(), x4_adj, Net(0));
  Kokkos::deep_copy(ExecSpace(), cf_seq_adj, Net(0));
  View1D cutoff_adj = ws_.n1("re_cutoff_adj", E);
  // Of each layer's read-out features: the adjoint of its tokens, by rows.
  std::vector<View2D> tokens_adj_of(G), node_adj(G), edge_adj(G);
  for (int L = 0; L < G; ++L) {
    tokens_adj_of[L] = ws_.n2("re_fa_" + std::to_string(L), T, D);
    node_adj[L] = atom_rows(tokens_adj_of[L]), edge_adj[L] = edge_rows(tokens_adj_of[L]);
    if (L >= R) Kokkos::deep_copy(ExecSpace(), tokens_adj_of[L], Net(0));  // a layer with no read-out of its own
  }
  for (int i = 0; i < R; ++i) readout_bwd(rs, i, pk, N, node_adj[i], edge_adj[i], cutoff_adj, i > 0);

  // A rebuilt activation: lin(x), x = norm(in) when norm_key is given.
  auto rebuild = [&](View2D in, const std::string& norm_key, const std::string& lin) {
    View2D x = in;
    if (!norm_key.empty()) norm(x = ws_.n2("re_tok_a", in.extent(0), in.extent(1)), in, norm_key);  // free by now
    View2D out = ws_.tmp(in.extent(0), mat(lin + ".weight").extent(0));
    if (h_.activation == Activation::SwiGLU && norm_key.size()) swiglu_in(out, {}, x, mat(lin + ".weight"), vec(lin + ".bias"));
    else linear(out, x, mat(lin + ".weight"), vec(lin + ".bias"));
    return out;
  };

  // ie_adj: the adjoint of the next layer's input_edge (zero past the last).
  View2D ie_adj = ws_.n2("re_ie_adj", E, D);
  for (int L = G - 1; L >= 0; --L) {
    const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
    Workspace::Scope layer_scope(ws_);
    ResidualSaves one;
    if (ckpt) {
      one.tag = "r";  // one shared set of saves
      residual_layer(dev, pk, L, inputs[L], ws_.n2("re_tokens_r", T, D), &one, false);
    }
    const ResidualSaves& s = ckpt ? one : sav[L];
    const bool kept_wide = keep_wide && !ckpt;
    auto efa = edge_adj[L];
    if (L + 1 < G) {  // message passing: half of the next input's adjoint, from the reverse edge
      const View2D partner = remote ? swap_rows(ie_adj, Net(0.5)) : View2D();
      Kokkos::parallel_for(
          "reb_msg", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
            const int k = i / D, d = i % D, r = rev(k);
            efa(k, d) += r >= 0 ? Net(0.5) * ie_adj(r, d) : remote && ro(k) >= 0 ? partner(ro(k), d) : Net(0);
          });
    }
    // The read-out's adjoint is the tokens' already (a capacity's dead edge rows
    // carry what they carry: no live row reads them).
    View2D tokens_adj = tokens_adj_of[L];

    for (int a = A - 1; a >= 0; --a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string key = "re_attn_" + s.tag + "_" + std::to_string(a);
      // tokens_out = norm(s2), s2 = tok_a + mlp(tok_a)
      View2D s2_adj = ws_.n2("re_s2_adj", T, D), tok_a_adj = ws_.n2("re_tok_a_adj", T, D);
      norm_bwd(s2_adj, tokens_adj, s.s2[a], tl + ".norm_mlp", false, tok_a_adj);  // and a copy for the residual
      {
        Workspace::Scope scope(ws_);
        feedforward_bwd(tok_a_adj, s2_adj, tl + ".mlp",
                        kept_wide ? s.pre[a] : rebuild(s.s1[a], tl + ".norm_attention", tl + ".mlp.w_in"));
      }
      // tok_a = norm(s1), s1 = tokens_in + attention(tokens_in): the attention's
      // share accumulates onto the residual's, which is then tokens_in's adjoint.
      View2D s1_adj = ws_.n2("re_s1_adj_" + std::to_string(a % 2), T, D);
      norm_bwd(s1_adj, tok_a_adj, s.s1[a], tl + ".norm_attention", false);
      {
        Workspace::Scope scope(ws_);
        attention_bwd(ws_, key, s1_adj, cf_seq_adj, s1_adj,
                      kept_wide ? s.qkv[a] : rebuild(s.tin[a], "", tl + ".attention.input_linear"), dev.cf_seq,
                      mat(tl + ".attention.input_linear.weight"), mat(tl + ".attention.output_linear.weight"),
                      tk, S, h_.num_heads, h_.head_dim, h_.attention_temperature, 1);
      }
      tokens_adj = s1_adj;
    }

    // tokens = [node_L ; et]: only et depends on the geometry. input_edge feeds
    // compress.0 and, through message passing, carries half of ie_adj. The two
    // buffers alternate by layer: this layer's is written while the last's is read.
    View2D et_adj = ws_.n2("re_et_adj", E, D), cpre_adj = ws_.n2("re_cpre_adj", E, D);
    View2D ie_next = ws_.n2("re_ie_" + std::to_string(L % 2), E, D);
    const bool has_msg = L + 1 < G;
    Kokkos::parallel_for(
        "reb_et", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
          const int k = i / D, d = i % D;
          et_adj(k, d) = k < off(N) ? tokens_adj(N + k, d) : Net(0);  // past: a capacity's dead rows
          ie_next(k, d) = has_msg ? Net(0.5) * ie_adj(k, d) : Net(0);
        });
    linear_bwd(cpre_adj, et_adj, mat(g + ".compress.2.weight"), 0);
    compress_bwd(cpre_adj, s.cpre, compress_fold(L), x4_adj, ie_next);
    ie_adj = ie_next;
  }

  DeviceOut out{per_atom, {}, {}, {}};
  forces_and_virial(ws_, dev, pk, h_, probes_, n_probes_, energy_scale_, x4_adj, cutoff_adj, cf_seq_adj,
                    out.forces, out.vir9, out.edge_grad);
  return out;
}

}  // namespace pet
