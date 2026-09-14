// PET model hyperparameters and small enums.
//
// Mirrors the metatrain PET "ModelHypers" subset that the libtorch-free
// evaluation needs. Populated from the <model>.json emitted by
// tools/convert_pet.py.
#pragma once

#include <string>

namespace pet {

enum class Normalization { RMSNorm, LayerNorm };
enum class Activation { SwiGLU, SiLU };
enum class TransformerType { PreLN, PostLN };
enum class FeaturizerType { FeedForward, Residual };
enum class CutoffFunction { Bump, Cosine };
// How the per-atom adaptive cutoff is chosen. Grid evaluates the smoothed
// neighbour count on a discrete probe grid and takes a Gaussian-weighted
// average of the probes; Solver root-finds n_total(r) = num_neighbors_adaptive
// with Newton-bisection. They do NOT agree, and metatrain switched its default
// from Grid to Solver -- so a checkpoint that does not say which it used is one
// trained before the choice existed, i.e. Grid.
enum class AdaptiveCutoffMethod { Grid, Solver };

struct Hypers {
  int d_pet = 128;            // edge/transformer model dim
  int d_head = 128;           // readout head hidden dim
  int d_node = 512;           // node (central token) feature dim
  int d_feedforward = 256;    // edge FFN inner dim
  int num_heads = 8;
  int head_dim = 16;          // d_pet / num_heads
  int num_attention_layers = 1;
  int num_gnn_layers = 2;
  // Readout layers: feedforward featurizer reads out once (final layer only);
  // residual featurizer reads out from every GNN layer (== num_gnn_layers).
  int num_readout_layers = 1;

  Normalization normalization = Normalization::RMSNorm;
  Activation activation = Activation::SwiGLU;
  TransformerType transformer_type = TransformerType::PreLN;
  FeaturizerType featurizer_type = FeaturizerType::FeedForward;
  CutoffFunction cutoff_function = CutoffFunction::Bump;

  double attention_temperature = 1.0;
  double cutoff = 7.5;
  double cutoff_width = 0.5;
  // num_neighbors_adaptive: <0 means "disabled".
  double num_neighbors_adaptive = -1.0;
  bool adaptive() const { return num_neighbors_adaptive > 0.0; }
  AdaptiveCutoffMethod adaptive_cutoff_method = AdaptiveCutoffMethod::Grid;
  // Taper width for the SMOOTHED NEIGHBOUR COUNT the adaptive scheme minimises
  // over -- a different quantity from cutoff_width, which tapers the edge's own
  // cutoff factor. metatrain passes them separately. A checkpoint predating the
  // split carries only cutoff_width, and the loader mirrors it here, which is
  // the behaviour the shipped goldens are validated against.
  double cutoff_width_adaptive = 0.5;
  // Central-token features are "expanded" (contract d_node->d_pet, run the
  // per-layer center MLP, expand back) only when d_node != d_pet. When equal,
  // metatrain makes center_contraction/expansion/norm_center/center_mlp Identity
  // and the checkpoint carries no center_* weights.
  bool expanded_node() const { return d_node != d_pet; }

  bool zbl = false;
  bool long_range_enabled = false;
};

}  // namespace pet
