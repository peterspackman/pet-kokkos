#!/usr/bin/env python3
"""An analytic FLOP and DRAM-byte model of one PET evaluation, and the time
bounds it implies on a GPU.

    tools/roofline.py MODEL.json --atoms N --edges E --max-neighbors M [options]

The model walks the same sequence of device operations the library runs
(src/feedforward.cpp, src/residual.cpp, src/model.cpp readout, src/attention.cpp,
src/forces.cpp, src/device_neighbors.cpp) and counts, per operation:

  flops  -- the arithmetic the operation needs (2 per multiply-add; attention
            over each atom's real tokens, not the padded S x S the kernels run);
  bytes  -- the DRAM traffic of the operation as it is structured today: every
            input read once and every output written once (a perfect cache
            within one kernel, nothing carried between kernels). Weights are
            counted once per GEMM.

Each operation's bound is max(flops / peak, bytes / bandwidth) (exp counted
against the special-function rate for attention), and the evaluation's bound is
the sum over operations, which is what "% of roofline" of a kernel or a whole
step is judged against. A kernel at its bound moves exactly the bytes counted
here at the full bandwidth; a fusion that removes a DRAM round trip lowers the
bound itself (the `--ideal` notes in docs list the big ones).

Shapes come from MODEL.safetensors' header when it sits next to the json (so
the widths are the checkpoint's own), else from the hypers.

Validated on an RTX 4080 SUPER against Nsight Compute's dram__bytes and FFMA
counters (tools/roofline.md); assumptions are listed there.
"""
import argparse
import json
import math
import os
import struct
import sys

F32, F64, I32 = 4, 8, 4

# Per-device peaks. bw is the HBM/GDDR peak; `eff_bw` the fraction a streaming
# kernel reaches in practice (measured on the 4080, published STREAM-like
# figures elsewhere) -- reported alongside, never used for the bound itself.
GPUS = {
    # MI250X, one GCD: 110 CUs at 1.7 GHz. FP32 matrix (v_mfma_f32_*) 47.9 TF/s,
    # FP32 vector 23.9 TF/s (packed FP32 doubles it, the compilers rarely emit
    # it), HBM2e 1.6 TB/s per GCD. Transcendentals at a quarter of the FMA rate.
    'mi250x': dict(name='MI250X GCD', gemm=47.9e12, attn=47.9e12, vec=23.9e12, sfu=23.9e12 / 4 / 2,
                   bw=1.6e12, eff_bw=0.80),
    # H200 SXM: H100's SMs (132 at 1.98 GHz), FP32 SIMT 67 TF/s (no FP32 tensor
    # path without TF32), HBM3e 4.8 TB/s. MUFU 16/clk/SM -> 4.2 Tops/s.
    'h200': dict(name='H200 SXM', gemm=67e12, attn=67e12, vec=67e12, sfu=132 * 16 * 1.98e9,
                 bw=4.8e12, eff_bw=0.85),
    # RTX 4080 SUPER: 80 SMs x 128 FP32 lanes at the ~2.72 GHz it runs under
    # load here (Nsight's sm__cycles_elapsed.avg.per_second; 2.55 GHz is the
    # rated boost, 52 TF/s), GDDR6X 23 Gb/s x 256 bit = 736 GB/s, 64 MB L2.
    '4080s': dict(name='RTX 4080 SUPER', gemm=80 * 128 * 2 * 2.72e9, attn=80 * 128 * 2 * 2.72e9,
                  vec=80 * 128 * 2 * 2.72e9, sfu=80 * 16 * 2.72e9, bw=736e9, eff_bw=0.88),
}


class Op:
    __slots__ = ('name', 'cat', 'phase', 'flops', 'bytes', 'exps', 'unit')

    def __init__(self, name, cat, phase, flops=0.0, nbytes=0.0, exps=0.0, unit='vec'):
        self.name, self.cat, self.phase = name, cat, phase
        self.flops, self.bytes, self.exps, self.unit = float(flops), float(nbytes), float(exps), unit

    def bound(self, g):
        peak = g[self.unit]
        return max(self.flops / peak, self.bytes / g['bw'], self.exps / g['sfu'])


def shapes_from(path):
    st = os.path.splitext(path)[0] + '.safetensors'
    if not os.path.exists(st):
        return {}
    with open(st, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: v['shape'] for k, v in h.items() if isinstance(v, dict) and 'shape' in v}


class Model:
    def __init__(self, path):
        with open(path) as f:
            j = json.load(f)
        h = j['hypers']
        self.h = h
        sh = shapes_from(path)
        self.sh = sh
        self.D = h['d_pet']
        self.Dn = h.get('d_node', self.D)
        self.Dh = h.get('d_head', self.D)
        self.H = h['num_heads']
        self.hd = h.get('head_dim', self.D // self.H)
        self.G = h['num_gnn_layers']
        self.A = h['num_attention_layers']
        self.ff = h.get('featurizer_type', 'feedforward') == 'feedforward'
        self.R = h.get('num_readout_layers', 1 if self.ff else self.G)
        self.swiglu = h.get('activation', 'SiLU') == 'SwiGLU'
        self.adaptive = h.get('num_neighbors_adaptive', 0) and h['num_neighbors_adaptive'] > 0
        self.solver = h.get('adaptive_cutoff_method', 'grid') == 'solver'
        self.cutoff = h['cutoff']
        self.width_ad = h.get('cutoff_width_adaptive', h.get('cutoff_width', 0.5))
        dff = h.get('d_feedforward', 2 * self.D)

        def rows(name, default):
            s = sh.get(name)
            return s[0] if s else default
        t = 'gnn_layers.0.trans.layers.0.'
        # Pre-activation widths: SwiGLU's w_in has 2F rows (v and g).
        self.ew = rows(t + 'mlp.w_in.weight', 2 * dff if self.swiglu else dff)
        self.cw = rows(t + 'center_mlp.w_in.weight', 4 * self.Dn) if self.Dn != self.D else 0
        self.n_probes = len([r for r in frange(0.5, self.cutoff - 1e-12, self.width_ad / 4.0)])
        self.source = 'safetensors' if sh else 'hypers'


def frange(a, b, s):
    x = a
    while x < b:
        yield x
        x += s


class Counter:
    """Accumulates ops; `backend` decides which epilogues are fused into the GEMM."""

    def __init__(self, backend):
        self.ops = []
        self.fused = backend == 'cuda'  # CUTLASS epilogues (fused_gemm.cpp)
        self.phase = 'fwd'

    def add(self, *a, **k):
        self.ops.append(Op(*a, phase=self.phase, **k))

    # out[M, N] = in[M, K] W^T (+ bias) (+ beta out); extra bytes for an epilogue.
    def gemm(self, name, M, N, K, beta=0, extra=0):
        b = F32 * (M * K + N * K + M * N * (1 + beta)) + extra
        self.add(name, 'gemm', flops=2.0 * M * N * K, nbytes=b, unit='gemm')

    def vec(self, name, cat, nbytes, flops=0, exps=0):
        self.add(name, cat, flops=flops, nbytes=nbytes, exps=exps)

    # silu(in W^T + b), pre saved when `save` (fused on CUDA, a second pass elsewhere).
    def linear_silu(self, name, M, N, K, save):
        if self.fused:
            self.gemm(name, M, N, K, extra=F32 * M * N * save)
        else:
            self.gemm(name, M, N, K)  # into the kept pre-activation, the pass reading it there
            self.vec(name + ':silu', 'elementwise', F32 * M * N * 2, exps=M * N)

    # grad = (out_adj W) * silu'(pre).
    def linear_silu_bwd(self, name, M, N, K):
        if self.fused:
            self.gemm(name, M, N, K, extra=F32 * M * N)
        else:
            self.gemm(name, M, N, K)
            self.vec(name + ':silu_bwd', 'elementwise', F32 * M * N * 3, exps=M * N)

    # pre = in W_in^T (width 2F), h = v * sigmoid(g); the w_out GEMM after.
    def swiglu(self, name, M, F2, K, Nout, beta):
        F = F2 // 2
        if self.fused:
            self.gemm(name + '.w_in', M, F2, K, extra=F32 * M * F)
        else:
            self.gemm(name + '.w_in', M, F2, K)
            self.vec(name + ':swiglu', 'elementwise', F32 * M * (F2 + F), exps=M * F)
        self.gemm(name + '.w_out', M, Nout, F, beta=beta)

    def swiglu_bwd(self, name, M, F2, K, Nout):
        F = F2 // 2
        if self.fused:  # h_adj in registers, the epilogue reads pre and writes pre_adj
            self.gemm(name + '.w_out^T', M, F, Nout, extra=F32 * M * F2 * 2 - F32 * M * F)
        else:
            self.gemm(name + '.w_out^T', M, F, Nout)
            self.vec(name + ':swiglu_bwd', 'elementwise', F32 * M * (F + 2 * F2), exps=M * F)
        self.gemm(name + '.w_in^T', M, K, F2)


def attention_ops(c, m, N, E, S, cnt2, save, fwd):
    """Per (atom, head): QK^T and PV over the atom's tokens (fwd: 4 cnt^2 hd
    flops per head); the backward recomputes QK^T and dO V and forms dQ, dK, dV
    (10 cnt^2 hd). Bytes: qkv and its adjoint, the merged heads, the saved
    max/sum per (atom, head, slot)."""
    D, H, T = m.D, m.H, N + E
    stats = F32 * N * H * S * 2
    if fwd:
        c.add('attention fwd', 'attention', flops=4.0 * cnt2 * D, exps=cnt2 * H,
              nbytes=F32 * (T * 3 * D + T * D + N * S) + stats * save, unit='attn')
    else:
        c.add('attention bwd', 'attention', flops=10.0 * cnt2 * D, exps=cnt2 * H,
              nbytes=F32 * (T * 3 * D + 2 * T * D + N * S + T * 3 * D + N * S * H) + stats)
        c.ops[-1].unit = 'attn'
        c.vec('attn_bwd_cf', 'gather', F32 * (N * S * H + N * S) + F32 * 2 * N * S)


def ff_layer_fwd(c, m, N, E, S, cnt2, L, save, wide):
    D, Dn, T = m.D, m.Dn, N + E
    pre = f'L{L} '
    if L > 0:
        c.gemm(pre + 'compress.0 (input_edge)', E, D, D)
    # compress: (gemm +) bias + species table + wx . (v, |v|), silu; pre saved.
    c.vec(pre + 'compress', 'elementwise', F32 * E * D * ((L > 0) + 1 + save) + (F64 * 4 + I32) * E,
          exps=E * D)
    c.gemm(pre + 'compress.2', E, D, D)
    for a in range(m.A):
        b = f'{pre}b{a} '
        c.gemm(b + 'center_contraction', N, D, Dn)
        # norm_tok: the tokens [input_node; edge_cur], normalised. Block 0's are
        # written in place by the two GEMMs when saving; otherwise gathered.
        c.vec(b + 'norm_tok', 'norm', F32 * T * D * (2 if save and a == 0 else 3 if save else 2))
        c.gemm(b + 'qkv', T, 3 * D, D)
        attention_ops(c, m, N, E, S, cnt2, save, True)
        c.gemm(b + 'attention.output_linear', T, D, D)
        c.gemm(b + 'center_expansion', N, Dn, D)
        # norm_eps: eps = attn(edge rows) + edge_in (kept, and copied to the
        # block's output when saving), normalised.
        c.vec(b + 'norm_eps', 'norm', F32 * E * D * (4 + save))
        c.vec(b + 'norm_nres', 'norm', F32 * N * Dn * (4 + save))
        c.swiglu(b + 'center_mlp', N, m.cw, Dn, Dn, beta=1)
        c.swiglu(b + 'mlp', E, m.ew, D, D, beta=1)
    # norm_cat: [out_edge | out_edge(reverse)] gathered as the norm reads it,
    # input_edge += out_edge on the way; out_edge itself is what is kept.
    c.vec(pre + 'norm_cat', 'norm', F32 * E * D * (2 + 2 + 2) + I32 * E)
    c.linear_silu(pre + 'combination_mlps.0', E, 2 * D, 2 * D, save)
    c.gemm(pre + 'combination_mlps.2', E, D, 2 * D, beta=1)


def ff_layer_bwd(c, m, N, E, S, cnt2, L, kept_wide):
    D, Dn, T = m.D, m.Dn, N + E
    pre = f'L{L} '
    c.linear_silu_bwd(pre + 'combination_mlps.2^T', E, 2 * D, D)
    c.gemm(pre + 'combination_mlps.0^T', E, 2 * D, 2 * D)
    # concat_norm_bwd: out_edge (both halves from it), the norm output's
    # adjoint, input_edge's adjoint in; edge_adj out.
    c.vec(pre + 'concat_norm_bwd', 'norm', F32 * E * D * 5 + I32 * E)
    for a in reversed(range(m.A)):
        b = f'{pre}b{a} '
        if not kept_wide:  # rebuild the edge MLP's pre-activation
            c.vec(b + 'rebuild norm_mlp', 'norm', F32 * E * D * 2)
            c.gemm(b + 'rebuild mlp.w_in', E, m.ew, D)
        c.swiglu_bwd(b + 'mlp', E, m.ew, D, D)
        c.vec(b + 'norm_eps bwd', 'norm', F32 * E * D * 4)
        c.swiglu_bwd(b + 'center_mlp', N, m.cw, Dn, Dn)
        c.vec(b + 'norm_nres bwd', 'norm', F32 * N * Dn * 4)
        c.gemm(b + 'center_expansion^T', N, D, Dn)
        c.gemm(b + 'attention.output_linear^T', T, D, D)
        if not kept_wide:
            c.vec(b + 'rebuild norm_tok', 'norm', F32 * T * D * 2)
            c.gemm(b + 'rebuild qkv', T, 3 * D, D)
        attention_ops(c, m, N, E, S, cnt2, True, False)
        c.gemm(b + 'qkv^T', T, D, 3 * D)
        c.vec(b + 'norm_tok bwd', 'norm', F32 * (T * D * 2 + N * D + 2 * E * D))
        c.gemm(b + 'center_contraction^T', N, Dn, D, beta=1)
    compress_bwd_ops(c, pre, E, D)
    if L > 0:
        c.gemm(pre + 'compress.0^T', E, D, D, beta=1)


def compress_bwd_ops(c, pre, E, D):
    if c.fused:  # silu' in the GEMM's epilogue, the pass only sums the geometry adjoint
        c.gemm(pre + 'compress.2^T', E, D, D, extra=F32 * E * D)
        c.vec(pre + 'compress bwd', 'elementwise', F32 * E * D + F32 * E * 8)
    else:
        c.gemm(pre + 'compress.2^T', E, D, D)
        c.vec(pre + 'compress bwd', 'elementwise', F32 * E * D * 3 + F32 * E * 8, exps=E * D)


def readout_ops(c, m, N, E, rows_node, save, grad):
    Dh = m.Dh
    c.phase = 'fwd'
    for i in range(m.R):
        p = f'readout{i} '
        c.linear_silu(p + 'node_heads.0', N, Dh, rows_node, save)
        c.linear_silu(p + 'node_heads.1', N, Dh, Dh, save)
        c.gemm(p + 'node_last', N, 1, Dh)
        c.linear_silu(p + 'edge_heads.0', E, Dh, m.D, save)
        c.linear_silu(p + 'edge_heads.1', E, Dh, Dh, save)
        c.gemm(p + 'edge_last', E, 1, Dh)
        c.vec(p + 'readout_accumulate', 'gather', F32 * (N + 2 * E) + I32 * N)
    if not grad:
        return
    c.phase = 'bwd'
    for i in range(m.R):
        p = f'readout{i} '
        c.vec(p + 'seeds', 'elementwise', F32 * (N + 3 * E))
        c.linear_silu_bwd(p + 'node_last^T', N, Dh, 1)
        c.linear_silu_bwd(p + 'node_heads.1^T', N, Dh, Dh)
        c.gemm(p + 'node_heads.0^T', N, rows_node, Dh)
        c.linear_silu_bwd(p + 'edge_last^T', E, Dh, 1)
        c.linear_silu_bwd(p + 'edge_heads.1^T', E, Dh, Dh)
        c.gemm(p + 'edge_heads.0^T', E, m.D, Dh)


def geometry_ops(c, m, N, E, M, R, grad):
    """The device neighbour build and edge geometry (device_cell_list.cpp,
    device_neighbors.cpp, model.cpp pack_edges), then the forces' end
    (forces.cpp). Raw edges R: every pair within the model's cutoff."""
    c.phase = 'geom'
    raw = (I32 * 2 + I32 * 3 + F64 * 3 + F64)  # i, j, shift, vec, dist per raw edge
    c.vec('cell list + raw edges', 'geometry', R * raw + N * (F64 * 3 + I32) * 2)
    c.vec('edge topology (count, hash, reverse)', 'geometry', R * (I32 * 3 + 16 + I32 * 4))
    if m.adaptive:
        if m.solver:  # PET_SOLVER_ITERS + 1 sweeps over each atom's raw distances (L1/L2)
            c.vec('adaptive cutoff (solver)', 'geometry', R * F64 + N * F64 * 3, flops=11 * R * 30)
        else:
            c.vec('adaptive cutoff (grid)', 'geometry', R * (F64 + 2 * I32) + N * m.n_probes * F64 * 3)
    c.vec('keep + slots + reverse + cf_seq', 'geometry',
          R * (F64 * 3 + I32 * 3) + N * M * (I32 * 2 + F64 * 6 + F32) + N * (M + 1) * F32)
    c.vec('pack_edges', 'geometry', N * M * (F64 + I32 * 3 + F64 * 5 + F32) + E * (I32 * 3 + F64 * 5 + F32))
    c.vec('embeddings (node, input_edge)', 'gather', F32 * (N * (m.Dn if m.ff else m.D) + E * m.D))
    if not grad:
        return
    c.phase = 'forces'
    c.vec('edge_grad', 'forces', E * (F64 * 3 + F32 * 6 + F64 * 5) + N * (M + 1) * F32)
    c.vec('fold forces + virial', 'forces', E * (F64 * 6 + I32) + N * (F64 * 12))
    if m.adaptive:
        c.vec('adaptive backward (raw edges)', 'forces', R * (F64 * 3 + I32 * 2 + F64 * 4 + F64 * 3 + I32))
        c.vec('raw edge gradients', 'forces', R * (I32 * 2 + F64 * 6))


def evaluate(m, N, E, M, R, cnt2, backend, grad, tier):
    c = Counter(backend)
    S = M + 1
    if not m.ff:
        return evaluate_residual(m, N, E, M, R, cnt2, backend, grad, tier)
    geometry_ops(c, m, N, E, M, R, False)
    c.phase = 'fwd'
    save = grad and tier != 'layers'
    wide = grad and tier == 'never'
    for L in range(m.G):
        if grad and tier == 'layers':
            c.vec(f'L{L} checkpoint copies', 'elementwise', F32 * 2 * (N * m.Dn + E * m.D))
        ff_layer_fwd(c, m, N, E, S, cnt2, L, save, wide)
    readout_ops(c, m, N, E, m.Dn, grad, grad)
    if grad:
        c.phase = 'bwd'
        c.vec('zero x4_adj, cf_seq_adj', 'memset', F32 * (E * 4 + N * S))
        for L in reversed(range(m.G)):
            if tier == 'layers':
                c.phase = 'recompute'
                c.vec(f'L{L} restore inputs', 'elementwise', F32 * 2 * (N * m.Dn + E * m.D))
                ff_layer_fwd(c, m, N, E, S, cnt2, L, True, False)
                c.phase = 'bwd'
            ff_layer_bwd(c, m, N, E, S, cnt2, L, tier == 'never')
        geometry_ops_forces(c, m, N, E, M, R)
    return c.ops


def geometry_ops_forces(c, m, N, E, M, R):
    saved = c.ops
    tmp = Counter('cuda')
    geometry_ops(tmp, m, N, E, M, R, True)
    saved.extend(o for o in tmp.ops if o.phase == 'forces')


def evaluate_residual(m, N, E, M, R, cnt2, backend, grad, tier):
    """The residual featurizer (PostLN, LayerNorm, SiLU): tokens [node; et]
    through A blocks per layer, every layer read out, message passing between."""
    c = Counter(backend)
    S, D, T, pw = M + 1, m.D, N + E, m.ew
    geometry_ops(c, m, N, E, M, R, False)
    c.phase = 'fwd'
    save, wide = grad and tier != 'layers', grad and tier == 'never'

    def layer_fwd(L, save, wide):
        c.vec(f'L{L} node embedding', 'gather', F32 * N * D)
        if L > 0:
            c.gemm(f'L{L} compress.0 (input_edge)', E, D, D)
        c.vec(f'L{L} compress', 'elementwise', F32 * E * D * ((L > 0) + 1 + save) + (F64 * 4 + I32) * E)
        c.gemm(f'L{L} compress.2', E, D, D)
        for a in range(m.A):
            b = f'L{L} b{a} '
            if save and not wide:
                c.vec(b + 'keep tokens', 'elementwise', F32 * T * D * 2)
            c.gemm(b + 'qkv', T, 3 * D, D)
            attention_ops(c, m, N, E, S, cnt2, save, True)
            c.gemm(b + 'attention.output_linear', T, D, D)
            c.vec(b + 'norm_attention (+residual)', 'norm', F32 * T * D * 4)
            c.linear_silu(b + 'mlp.w_in', T, pw, D, wide)
            c.gemm(b + 'mlp.w_out', T, D, pw)
            c.vec(b + 'norm_mlp (+residual)', 'norm', F32 * T * D * 4)
        if L + 1 < m.G:
            c.vec(f'L{L} message', 'gather', F32 * E * D * 3 + I32 * E)

    for L in range(m.G):
        layer_fwd(L, save, wide)
    readout_ops(c, m, N, E, D, grad, grad)
    if not grad:
        return c.ops
    c.phase = 'bwd'
    for L in reversed(range(m.G)):
        if tier == 'layers':
            c.phase = 'recompute'
            layer_fwd(L, True, False)
            c.phase = 'bwd'
        for a in reversed(range(m.A)):
            b = f'L{L} b{a} '
            c.vec(b + 'norm_mlp bwd', 'norm', F32 * T * D * 4)
            if not (tier == 'never'):
                c.gemm(b + 'rebuild mlp.w_in', T, pw, D)
            c.linear_silu_bwd(b + 'mlp.w_out^T', T, pw, D)
            c.gemm(b + 'mlp.w_in^T', T, D, pw, beta=1)
            c.vec(b + 'norm_attention bwd', 'norm', F32 * T * D * 4)
            c.gemm(b + 'attention.output_linear^T', T, D, D)
            if not (tier == 'never'):
                c.gemm(b + 'rebuild qkv', T, 3 * D, D)
            attention_ops(c, m, N, E, S, cnt2, True, False)
            c.gemm(b + 'qkv^T', T, D, 3 * D, beta=1)
        c.vec(f'L{L} ie_next', 'elementwise', F32 * E * D * (2 if L + 1 < m.G else 1))
        compress_bwd_ops(c, f'L{L} ', E, D)
        if L > 0:
            c.gemm(f'L{L} compress.0^T', E, D, D, beta=1)
            c.vec(f'L{L} message bwd', 'gather', F32 * E * D * 3 + I32 * E)
    geometry_ops_forces(c, m, N, E, M, R)
    return c.ops


CATS = ['gemm', 'attention', 'norm', 'elementwise', 'gather', 'geometry', 'forces', 'memset']

# Nsight Compute kernel names -> the categories above (Kokkos kernels carry
# their enclosing function in the functor's type name).
NCU_CATS = [
    ('gemm', r'gemm_kernel|cutlass|cublas|sgemm|gemv|Gemm|ampere_|sm90_|xmma'),
    ('attention', r'Forward<|Pairs<|Queries<|Keys<|attention_impl|tc::|mfma::'),
    ('norm', r'norm_rows|concat_norm_bwd'),
    ('memset', r'ViewFill|init_lock|memset'),
    ('geometry', r'build_raw_edges|edge_topology|edge_geometry|solver_cutoffs|grid_cutoffs|wrap_positions|'
                 r'pack_edges|count_edges|cell_list|UnorderedMap|remote_map'),
    ('forces', r'forces_and_virial|fold_edge|adaptive_|sum_by_structure|structure_offsets|virial'),
    ('gather', r'readout|assemble_energy|ll_features|pet::gather|attention_bwd_impl'),
    ('elementwise', r'compress_|silu|swiglu|ff_layer|residual|copy|cond_|bias'),
]


def load_ncu(path):
    """Per-category [seconds, DRAM bytes, flops] from `ncu --csv --page raw`
    with gpu__time_duration.sum, dram__bytes_{read,write}.sum and the FFMA/FADD/
    FMUL thread-instruction counters."""
    import csv
    import re
    lines = open(path).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.reader(lines[start:]))
    hdr, out = rows[0], {}
    for r in rows[2:]:
        if len(r) != len(hdr):
            continue
        d = dict(zip(hdr, r))

        def f(k):
            try:
                return float(d.get(k, '0').replace(',', ''))
            except ValueError:
                return 0.0
        name = d['Kernel Name']
        cat = next((c for c, pat in NCU_CATS if re.search(pat, name)), 'other')
        t = out.setdefault(cat, [0.0, 0.0, 0.0])
        t[0] += f('gpu__time_duration.sum') * 1e-9
        t[1] += f('dram__bytes_read.sum') + f('dram__bytes_write.sum')
        t[2] += (2 * f('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum') +
                 f('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum') +
                 f('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='model .json (MODEL.safetensors beside it supplies exact widths)')
    ap.add_argument('--atoms', '-N', type=int, required=True)
    ap.add_argument('--edges', '-E', type=int, required=True, help='kept edges (pet-eval --memory prints them)')
    ap.add_argument('--max-neighbors', '-M', type=int, required=True, help='S = M + 1 tokens at most')
    ap.add_argument('--raw-edges', type=int, help='pairs within the model cutoff, before the adaptive cut '
                    '(default 7E; water has 225 per atom within 8 A, 182 within 7.5 A)')
    ap.add_argument('--cnt2', type=float, help='sum over atoms of (tokens)^2 (default: N (E/N + 1)^2, '
                    'i.e. every atom the mean count)')
    ap.add_argument('--energy-only', action='store_true', help='no backward')
    ap.add_argument('--tier', choices=['never', 'wide', 'layers'], default='never',
                    help='recompute tier (pet-eval --memory prints it)')
    ap.add_argument('--gpu', action='append', choices=list(GPUS), help='default: all')
    ap.add_argument('--backend', choices=['cuda', 'hip', 'auto'], default='auto',
                    help='cuda: the CUTLASS fused epilogues; hip: activations as separate passes '
                    '(auto: hip on mi250x, else cuda)')
    ap.add_argument('--ncu', help='an `ncu --csv --page raw` dump of one evaluation to set against the '
                    'bound of the first --gpu (tools/roofline.md has the command)')
    ap.add_argument('--ops', action='store_true', help='list every operation')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)

    m = Model(a.model)
    N, E, M = a.atoms, a.edges, a.max_neighbors
    R = a.raw_edges or 7 * E
    cnt2 = a.cnt2 or N * (E / N + 1) ** 2
    grad = not a.energy_only
    gpus = a.gpu or list(GPUS)
    res = {}
    for g in gpus:
        backend = a.backend if a.backend != 'auto' else ('hip' if g == 'mi250x' else 'cuda')
        res[g] = (backend, evaluate(m, N, E, M, R, cnt2, backend, grad, a.tier))

    if a.json:
        out = {'model': a.model, 'N': N, 'E': E, 'M': M, 'R': R, 'cnt2': cnt2, 'gpus': {}}
        for g, (backend, ops) in res.items():
            G = GPUS[g]
            out['gpus'][g] = {'backend': backend, 'bound_s': sum(o.bound(G) for o in ops), 'ops': [
                dict(name=o.name, cat=o.cat, phase=o.phase, flops=o.flops, bytes=o.bytes, exps=o.exps,
                     bound_s=o.bound(G)) for o in ops]}
        json.dump(out, sys.stdout, indent=1)
        return

    meas = load_ncu(a.ncu) if a.ncu else None
    print(f'{os.path.basename(a.model)}: d_pet {m.D} d_node {m.Dn} d_head {m.Dh} heads {m.H}x{m.hd} '
          f'gnn {m.G} attn {m.A} edge-mlp pre {m.ew} centre-mlp pre {m.cw} ({m.source})')
    print(f'N {N}  E {E} ({E / N:.1f}/atom)  S {M + 1}  raw edges {R}  sum cnt^2 {cnt2:.3g}  '
          f'{"forces" if grad else "energy"}  tier {a.tier}')
    for gi, (g, (backend, ops)) in enumerate(res.items()):
        G = GPUS[g]
        show = meas is not None and gi == 0
        print(f'\n{G["name"]} ({backend} epilogues): {G["gemm"] / 1e12:.1f} TF/s GEMM, '
              f'{G["attn"] / 1e12:.1f} attention, {G["vec"] / 1e12:.1f} vector, {G["bw"] / 1e9:.0f} GB/s')
        hdr = f'  {"category":12s} {"GFLOP":>8s} {"GB":>7s} {"bound ms":>9s} {"compute":>8s}'
        if show:
            hdr += f' {"meas ms":>8s} {"meas GB":>8s} {"bound/meas":>10s}'
        print(hdr)
        tot = [0.0, 0.0, 0.0, 0.0]
        for cat in CATS + ['other']:
            co = [o for o in ops if o.cat == cat]
            mc = meas.get(cat) if show else None
            if not co and not mc:
                continue
            f, b = sum(o.flops for o in co), sum(o.bytes for o in co)
            t = sum(o.bound(G) for o in co)
            # the part of the bound set by arithmetic rather than bytes
            tc = sum(o.bound(G) for o in co if o.bound(G) > o.bytes / G['bw'] * (1 + 1e-9))
            line = f'  {cat:12s} {f / 1e9:8.1f} {b / 1e9:7.2f} {t * 1e3:9.2f} {tc * 1e3:8.2f}'
            if mc:
                line += f' {mc[0] * 1e3:8.2f} {mc[1] / 1e9:8.2f} {t / mc[0] if mc[0] else 0:10.0%}'
                tot[3] += mc[0]
            print(line)
            tot[0] += f
            tot[1] += b
            tot[2] += t
        line = f'  {"total":12s} {tot[0] / 1e9:8.1f} {tot[1] / 1e9:7.2f} {tot[2] * 1e3:9.2f}'
        if show:
            line += f' {"":8s} {tot[3] * 1e3:8.2f} {sum(v[1] for v in meas.values()) / 1e9:8.2f} {tot[2] / tot[3]:10.0%}'
        print(line)
        eff = sum(max(o.flops / G[o.unit], o.bytes / (G['bw'] * G['eff_bw']), o.exps / G['sfu']) for o in ops)
        print(f'  (streaming kernels at {G["eff_bw"]:.0%} of peak bandwidth instead: {eff * 1e3:.2f} ms)')
        if a.ops:
            print(f'    {"op":44s} {"phase":9s} {"cat":11s} {"GFLOP":>8s} {"MB":>9s} {"bound us":>9s} {"AI":>6s}')
            for o in ops:
                ai = o.flops / o.bytes if o.bytes else 0
                print(f'    {o.name[:44]:44s} {o.phase:9s} {o.cat:11s} {o.flops / 1e9:8.2f} {o.bytes / 1e6:9.1f} '
                      f'{o.bound(G) * 1e6:9.1f} {ai:6.1f}')


if __name__ == '__main__':
    main()
