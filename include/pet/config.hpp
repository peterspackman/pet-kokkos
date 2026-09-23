// The model hyperparameters: the subset of metatrain's PET ModelHypers that
// evaluation needs, read from the <model>.json (checkpoint.hpp).
#pragma once

#include <string>

namespace pet {

enum class Normalization { RMSNorm, LayerNorm };
enum class Activation { SwiGLU, SiLU };
enum class TransformerType { PreLN, PostLN };
enum class FeaturizerType { FeedForward, Residual };
enum class CutoffFunction { Bump, Cosine };
// How the adaptive cutoff is chosen: a Gaussian-weighted average over a probe
// grid, or a Newton-bisection root of n_total(r) = num_neighbors_adaptive. They
// give different answers; a checkpoint that does not say is Grid.
enum class AdaptiveCutoffMethod { Grid, Solver };
// How much of the forward the backward recomputes rather than keeps, cheapest
// first: Never; Wide, the two widest activations (attention qkv, the edge MLP's
// pre-activation), one norm and GEMM each; Layers, each GNN layer re-run from
// its inputs (about one more forward). Auto takes the first that fits memory.
enum class Recompute { Auto, Never, Wide, Layers };

struct Hypers {
  int d_pet = 128;          // edge / transformer width
  int d_head = 128;         // readout hidden width
  int d_node = 512;         // node (central token) width
  int d_feedforward = 256;  // edge MLP inner width
  int num_heads = 8;
  int head_dim = 16;  // d_pet / num_heads
  int num_attention_layers = 1;
  int num_gnn_layers = 2;
  int num_readout_layers = 1;  // feedforward: 1; residual: num_gnn_layers

  Normalization normalization = Normalization::RMSNorm;
  Activation activation = Activation::SwiGLU;
  TransformerType transformer_type = TransformerType::PreLN;
  FeaturizerType featurizer_type = FeaturizerType::FeedForward;
  CutoffFunction cutoff_function = CutoffFunction::Bump;

  double attention_temperature = 1.0;
  double cutoff = 7.5;
  double cutoff_width = 0.5;
  double num_neighbors_adaptive = -1.0;  // <= 0: no adaptive cutoff
  bool adaptive() const { return num_neighbors_adaptive > 0.0; }
  AdaptiveCutoffMethod adaptive_cutoff_method = AdaptiveCutoffMethod::Grid;
  // The taper of the smoothed neighbour count the adaptive cutoff solves on;
  // cutoff_width tapers the edges themselves. Older checkpoints have only that.
  double cutoff_width_adaptive = 0.5;
  // Node features go through their own contract / MLP / expand only when
  // d_node != d_pet; otherwise those are identities with no weights.
  bool expanded_node() const { return d_node != d_pet; }

  bool zbl = false;
  bool long_range_enabled = false;

  // Charge and spin multiplicity, embedded and added to the node features after
  // every GNN layer. A per-structure constant, so it has no gradient.
  bool system_conditioning = false;
  int max_charge = 10;             // table covers charges in [-max_charge, +max_charge]
  int max_spin_multiplicity = 10;  // table covers 2S+1 in [1, max_spin_multiplicity]
};

}  // namespace pet
