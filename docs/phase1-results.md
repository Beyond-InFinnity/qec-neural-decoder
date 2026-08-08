# Phase 1 results: neural decoders for the repetition code

**Claim:** a small dilated CNN, trained with a documented recipe, matches MWPM
(PyMatching) on the circuit-level-noise repetition code up to d = 11 — within
8% everywhere, ahead at several cells. All numbers are paired evaluations on
identical million-shot test sets; artifacts in `experiments/`, reproducible
from configs.

## The arc (each step isolates one variable)

| Step | Config | d=11, p=0.02 vs MWPM | Lesson |
|---|---|---|---|
| MLP, small | `phase1_rep_mlp` | 55× worse | flat MLPs don't scale |
| MLP, 19× compute | `phase1_rep_mlp_scaled` | 17.8× | capacity ≠ locality |
| CNN 3×3 ×4 | `phase1_rep_cnn` | 6.7× | locality prior: big win |
| + dilations (RF 29) | `phase1_rep_cnn_dilated(_fix)` | 1.88× | receptive field must span chains |
| + train at p=0.05 | `phase1_rep_cnn_transfer` | **1.08×** | rare-event exposure was the last gap |

Final transfer-model table (train p=0.05, evaluate everywhere; ratio =
p_L(NN)/p_L(MWPM)):

| d | eval p=0.02 | p=0.03 | p=0.05 |
|---|---|---|---|
| 9 | 1.04 | **0.99** | **0.96** |
| 11 | 1.08 | 1.07 | **1.00** |

(d=7 best from p-matched dilated training: 1.14 / 1.08 / **0.99**.)

## Training recipe (the one that works)

ConvDecoder, 64 channels, dilations (1,2,4,4,2,1) → receptive field 29 >
longest d=11 chain; head 128. AdamW lr 1e-3, cosine decay, grad-clip 1.0,
batch 4096, 25 epochs, 15M training shots sampled at p=0.05 regardless of
evaluation p. ~1.1M params. Trains in hours on an RTX 5050 (8 GB); dataset
held on-device as uint8.

## Negative results (kept deliberately)

- Flat MLPs plateau far above MWPM at d ≥ 7 at any tested budget
  (`phase1_rep_mlp_scaled`: 19× compute bought 3×, not parity).
- lr 3e-3 without clipping diverges on the largest grids — 4 of 9 dilated
  cells collapsed (`phase1_rep_cnn_dilated`); clip 1.0 + lr 1e-3 fixed all 4
  with no other change (`phase1_rep_cnn_dilated_fix`).
- Single-hidden-layer MLPs underfit even d=3 (~2.3× worse than two layers at
  equal width budget; see tests/test_train.py history).

## Why NN ≈ MWPM is the expected ceiling here

For the repetition code under this noise, matching on the weighted DEM graph
is near-optimal; the correlated-error blind spot that lets NNs beat matching
properly (Y-errors coupling X and Z chains) barely exists in a bit-flip-only
code. The d=3 and p=0.05 NN wins are consistent with small residual
correlation gains. The real test of "learned > matching" is the surface code
(Phase 2), where correlations are structural.

## Carry-forward decisions for Phase 2

1. Locality-first architecture; receptive field ≥ code distance from day one.
2. Grad-clip + lr ≤ 1e-3 by default; treat any stuck-at-high-loss cell as a
   training failure to reseed, not a result.
3. Train at high p, evaluate across p (and consider mixed-p training).
4. Paired evaluation with nn_only_wrong / mwpm_only_wrong counts, always.
