# The roofline of one PET evaluation

`tools/roofline.py` counts the FLOPs and DRAM bytes of every device operation of
one evaluation (forward, and the backward for forces and virial) and turns them
into time bounds per GPU. It is what "% of roofline" means for this code: a
kernel at 100% moves exactly the bytes counted here at the full bandwidth, or
does the counted arithmetic at the peak rate, whichever takes longer.

```
pet-eval MODEL structure.xyz -f --memory      # prints N, M (neighbours at most), E (kept edges), tier
tools/roofline.py build-models/pet-mad-s.json -N 5184 -E 154240 -M 35 --raw-edges 1165840
tools/roofline.py ... --gpu 4080s --ops                         # every operation, its bytes and bound
tools/roofline.py ... --gpu 4080s --ncu prof.csv                # set a profile against the bound
```

The profile for `--ncu` (Nsight Compute works under WSL2; nsys does not):

```
PET_GRAPHS=0 ncu --replay-mode application --clock-control none --cache-control none --csv --page raw \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__cycles_elapsed.avg.per_second \
  pet-eval MODEL structure.xyz -f > prof.csv
```

## The model

The script walks the operations of `ff_pass` / `residual_pass` in order
(feedforward.cpp, residual.cpp, model.cpp readout, attention.cpp, forces.cpp,
device_neighbors.cpp, device_cell_list.cpp). N atoms, E kept edges, T = N + E
token rows, S = M + 1 token slots, D = d_pet, Dn = d_node, widths of the MLPs
from the checkpoint's own tensor shapes (the `.safetensors` header beside the
json). Per feedforward GNN layer and attention block:

| op | GEMM shapes (rows x out x in) | elementwise passes |
|---|---|---|
| compress | E x D x D (layer > 0), E x D x D | compress (bias, species table, geometry, SiLU) |
| centre contraction / expansion | N x D x Dn, N x Dn x D | |
| attention | T x 3D x D (qkv), T x D x D (out) | norm_tok; attention core |
| edge MLP (SwiGLU) | E x 2Fe x D, E x D x Fe | norm_eps |
| centre MLP (SwiGLU) | N x 2Fc x Dn, N x Dn x Fc | norm_nres |
| combination | E x 2D x 2D (SiLU), E x D x 2D | norm_cat |
| readout | N x Dh x Dn, N x Dh x Dh, E x Dh x D, E x Dh x Dh, two 1-wide | readout gather |

The backward needs only input adjoints (no weight gradients), so each GEMM's
backward is one GEMM of the same FLOPs. Forces take the geometry end
(edge_grad, the per-atom folds, the adaptive cutoff's backward over the raw
edges).

Rules:

- **GEMM FLOPs** 2 M N K. **Bytes**: A read once, the weight once, C written once
  (read too when accumulating), plus what the epilogue reads or writes. On CUDA
  (`--backend cuda`) the CUTLASS epilogues carry bias, SiLU/SwiGLU and their
  backward; on HIP (`--backend hip`, the default for mi250x) the MFMA kernel's
  epilogues do the same (fused_gemm_hip.cpp), but for the SwiGLU backward above
  50k rows, whose first GEMM writes h_adj and whose second makes pre_adj as it
  loads it. `--backend unfused` is rocBLAS/hipBLASLt (PET_MFMA_GEMM=0): the
  activations as separate passes, bias in the GEMM.
- **Attention** per (atom, head) over the atom's real tokens cnt (not the padded
  S): forward 4 cnt^2 hd FLOPs (QK^T, PV), backward 10 cnt^2 hd (QK^T and dO V
  recomputed, dQ, dK, dV), one exp per pair; bytes qkv, merged, the adjoints,
  the saved max/sum. sum cnt^2 defaults to N (E/N + 1)^2 (every atom the mean
  count; give `--cnt2` otherwise). The kernels execute padded work (CUDA's
  backward forms S x S tables), which is not counted.
- **Norms, elementwise passes, gathers**: bytes only, as the kernels are
  structured now (each input row read once from DRAM, re-reads within a row
  from L1). These are what a fusion changes; the script tracks the code.
- **Geometry** (cell list, topology hash, adaptive cutoff, packing) and
  **forces**: bytes only. These kernels are latency-bound, so their byte bound
  is far below what they take; it is listed so it is not mistaken for free.
- Bound per op = max(FLOPs / peak, bytes / bandwidth, exps / SFU rate); the
  evaluation's bound is the sum over ops (kernels run one after another).
  L2 reuse between kernels is not credited, launch gaps and host work are not
  charged.

Peaks used (`GPUS` in the script):

| | GEMM / attention | vector | bandwidth | notes |
|---|---|---|---|---|
| MI250X, one GCD | 47.9 TF/s (FP32 MFMA) | 23.9 TF/s | 1.6 TB/s | 8 MB L2 |
| H200 SXM | 67 TF/s (FP32 SIMT) | 67 | 4.8 TB/s | 50 MB L2 |
| RTX 4080 SUPER | 55.7 TF/s | 55.7 | 736 GB/s | 80 SMs x 128 lanes at the 2.72 GHz Nsight measures under load (52.2 at the rated 2.55 GHz boost); GDDR6X 23 Gb/s x 256 bit; 64 MB L2; FP64 1/64 rate |

## Validation (RTX 4080 SUPER, Nsight Compute)

pet-mad-s, water 5184 (every E x D tensor 158 MB, well past the 64 MB L2, so
the DRAM counters see what an MI250X or H200 would), forces, after this
branch's changes:

| category | model GFLOP | measured GFLOP | model GB | measured GB |
|---|---|---|---|---|
| GEMM | 2671 | 2714 | 35.2 | 58.6 (35.3 with the CTA raster fixed, below) |
| attention | 53 (real tokens) | 62 (executed) | 5.97 | 5.91 |
| norms | | | 12.98 | 12.40 |
| elementwise | | | 1.77 | 1.79 |
| gather | | | 0.21 | 0.21 |

Per kernel the byte counts agree to a few percent (e.g. norm_tok 489.8 model /
498.9 MB measured, norm_eps 789.7 / 788.8, attention forward 665.7 / 650.3,
backward 1324.7 / 1342.8, compress 321.4 / 320.6). The GEMMs moved 1.67x the
model's bytes because the CTA grid runs M fastest, so A is streamed once per
128-column tile (qkv read 980 MB of a 163 MB operand); with N fastest they move
35.3 GB against the model's 35.2. Small systems (water 648) do not validate:
there the outputs fit the 64 MB L2 and are never written back within the
kernel, so the DRAM counters read below the model even with
`--cache-control all`.

## Bounds, water 5184 atoms, forces (fp32 default)

N 5184; E and M from `pet-eval --memory`: pet-mad-s E 154240 M 35, pet-omat-s
E 100192 M 25, pet-mad-xs (build-models, the HuggingFace checkpoint) E 124848
M 29; raw edges 1165840 (8 A), 942432 (7.5 A). Bound in ms (GEMM part in
brackets):

| | MI250X GCD | H200 | RTX 4080 SUPER | 4080 measured |
|---|---|---|---|---|
| pet-mad-s | 69.4 (56.1) | 44.4 (40.0) | 86.3 (57.4) | 151.5 (159.9 before this branch) |
| pet-omat-s | 48.6 (39.7) | 31.3 (28.3) | 59.5 (40.1) | 107.5 (111.4) |
| pet-mad-xs | 12.1 (8.3) | 7.0 (5.7) | 22.1 (13.8) | 34.9 (36.8) |

The MI250X column is the MFMA kernel's fused epilogues (`--backend hip`); with
the activations as separate passes (`--backend unfused`, rocBLAS) it is 79.3,
55.2 and 14.7 ms. A loop of nothing but fp32 MFMAs reaches 41.3 TF/s on a GCD
(the practical peak: v_mfma_f32_16x16x4f32 at 1-3 blocks a CU and 4-16
independent chains alike, 40.3 for 32x32x2; 42.7 measured in October), so
pet-mad-s's GEMMs alone take 64.7 ms there, not 56.1: 90% of the 47.9 TF/s
peak is above what the matrix cores sustain.

Against earlier measurements: MI250X pet-mad-s ~150 ms a step in LAMMPS (that
step also carries LAMMPS's own work), H200 native ~94 ms in October (47% of
44 ms).

What each category reaches on the 4080 now (pet-mad-s 5184): GEMM 53% of its
bound (56% with the CTA raster fixed), attention 55%, norms 91%, elementwise
85%, gathers 71%, forces 53%, geometry 7% (latency- and, on this card,
FP64-bound: the cell list's double distances and the solver's double divisions).

## MI250X, measured (one GCD, rocprofv3 kernel trace)

pet-mad-s, water 5184, forces, kernel time per evaluation by category against
the bound (`--backend hip`), perf-integrate and after the perf-mi250x-2 branch
(GEMM after the gemm-redesign branch: 89.3, 93.4 before it in the same job):

| category | bound ms | perf-integrate | now | now / bound |
|---|---|---|---|---|
| GEMM | 56.1 (64.7 at 41.3 TF/s) | 94.1 | 92.9 | 60% (70% of the practical) |
| attention | 3.7 | 14.0 | 9.8 | 38% |
| norms | 8.1 | 17.2 | 10.9 | 74% |
| elementwise | 1.1 | 3.2 | 3.2 | 35% |
| geometry | 0.13 | 3.6 | 3.6 | latency-bound |
| gather, forces | 0.25 | 0.6 | 0.6 | |
| total | 69.4 | 132.8 | 121.3 (121.7 wall) | 57% |

The GEMM kernel alone reaches 32-37 TF/s at the network's shapes with a plain
epilogue (78-90% of the practical peak). In the network each GEMM runs at its
standalone speed with its epilogue: the fused epilogues' extra traffic (a saved
pre-activation, the output accumulated onto, the pre-activation the SiLU
backward reads) and their sigmoids cost 15-30% on those GEMMs, not overlapped
by the matrix cores. The attention kernels move near the minimum of DRAM bytes
(0.94 GB a layer read against 0.82 unique) at a third of the bandwidth:
latency-bound, one 16-token block a wave.

## MI250X: the vendor fp32 GEMMs at the network's shapes

One GCD, ROCm 7.0.1, TF/s (peak 47.9, practical 41.3), the network's GEMMs at
pet-mad-s water 5184 (E = 154240 edge rows, T = 159424 tokens), NT = against
an [N, K] weight (forward), NN = a [K, N] one (the backward's input gradient;
no weight gradients are formed). "ours" is the MFMA kernel before the
gemm-redesign branch, plain epilogue; rocBLAS as `rocblas_gemm_ex` picks it
and the best of all its Tensile solutions for the shape (282-315, each timed);
hipBLASLt the best of its top-32 heuristic algorithms, plain / with its bias /
with bias + Swish epilogue; Composable Kernel the best of its prebuilt fp32
XDL instances (its lds_direct_load ones do not run on gfx90a):

| GEMM | M x N x K | ours | rocBLAS | rocBLAS best | hipBLASLt | + bias | + Swish | CK |
|---|---|---|---|---|---|---|---|---|
| compress NT | E x 256 x 256 | 32.1 | 33.1 | 33.5 | 32.3 | 32.2 | 31.5 | 30.8 |
| qkv NT | T x 768 x 256 | 35.1 | 25.1 | 35.1 | 33.8 | 33.6 | 32.2 | 31.5 |
| mlp.w_in NT | E x 1024 x 256 | 35.0 | 33.8 | 35.1 | 33.2 | 33.2 | 32.2 | 31.5 |
| comb.0 NT | E x 512 x 512 | 36.3 | 34.0 | 36.6 | 34.8 | 34.7 | 34.2 | 32.4 |
| comb.2 NT | E x 256 x 512 | 34.7 | 33.4 | 35.3 | 33.9 | 34.0 | 33.4 | 31.5 |
| cmlp.w_in NT | 5184 x 4096 x 1024 | 36.7 | 37.0 | 37.0 | 34.8 | 34.7 | 34.8 | 31.1 |
| comb.0^T NN | E x 512 x 512 | 36.4 | 34.1 | 38.0 | 36.2 | 35.7 | 35.2 | 33.1 |
| mlp.w_in^T NN | E x 256 x 1024 | 36.9 | 33.9 | 38.7 | 36.7 | 36.1 | 36.0 | 33.2 |
| qkv^T NN | T x 256 x 768 | 35.9 | 33.3 | 37.9 | 35.4 | 35.3 | 35.0 | 32.3 |
| compress^T NN | E x 256 x 256 | 32.9 | 31.0 | 35.8 | 32.5 | 32.4 | 32.0 | 31.3 |
| pet-mad-xs NT | E x 128 x 128 | 18.4 | 25.0 | 27.7 | 23.7 | 23.5 | 23.0 | |
| pet-mad-xs NN | E x 128 x 256 | 25.1 | 26.9 | 31.0 | 27.2 | 27.0 | 26.7 | |
| square NT | 8192^3 | 30.5 | 32.3 | 38.5 | 38.4 | 38.4 | 38.4 | 33.2 |

Over the 15 GEMMs of a layer (both directions): ours 20.9 ms, rocBLAS 22.7,
the best Tensile solution for each shape 20.2, hipBLASLt 21.4 (21.9 with
Swish). None is near peak at these shapes: the vendors' default picks reach
51-77% of 47.9, the best of every solution 70-81%, about ours, which is level
with them NT and 4-9% behind the best NN. Only at width 128 (pet-mad-xs) are
they clearly ahead (25-31 TF/s against 18-25).

What the best Tensile kernels do (names of the solutions timed fastest): the
same v_mfma_f32_16x16x4f32; NT at MT128x128x16 (our tile) or MT64x96x32; NN at
MT256x64x16 with the weight loaded straight into the MFMA's registers (DTVA1),
only the activation through LDS; two k tiles of global loads in flight (PGR2),
LDS operands read several MFMA steps ahead (PLR5-9), one LDS buffer, the loads
and LDS traffic interleaved between the MFMAs by hand in assembly (SIA3), no
direct-to-LDS loads (DTL0) and no split-K. Tried in our compiler-scheduled
kernel, all bit-identical, all slower: two k tiles of register prefetch (-1%,
147 VGPRs, still 3 waves), the weight direct-to-VGPR with the activation
double-buffered (one barrier a k tile; -5 to -25% at 128 x 128 and at
Tensile's 64 x 256), 32-deep k tiles (-5 to -25%), eight-wave 128 x 128
blocks (level), a persistent grid spreading each tile's epilogue over the
next tile's main loop (two sets of accumulators: 2 waves a SIMD and spills,
-20 to -35%), and starting the first blocks staggered (-1 to -3%: the
epilogues do not collide in time). What won: the epilogue's output through
LDS so a store writes whole rows (Composable Kernel's "CShuffle"), +3-17% on
the fused epilogues, and two tile choices where one block spans N.

## Assumptions and limits

- Peaks are vendor numbers; GEMM bounds take the matrix (MI250X) or SIMT (H200,
  4080) FP32 rate, not TF32.
- The uniform-count default for sum cnt^2 underestimates attention by
  (1 + cv^2) for a spread cv of tokens per atom (a few % for water).
- The bytes are of the code's current structure: a fusion that removes a pass
  lowers the bound, so re-run after one. The CUDA epilogues are modelled as
  they are in fused_gemm.cpp, including SwiGLU's pre-activation write even where
  nothing keeps it (energy-only, Wide tier).
- Recompute tiers (`--tier wide|layers`) add the rebuilt GEMMs and the
  re-run forward; pet-mad-s 5184 fits `never` on 16 GB.
- Not validated on MI250X or H200 here (no counters taken there); the residual
  featurizer (pet-mols-s) is modelled but not validated.
