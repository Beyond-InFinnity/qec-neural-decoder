# Phase 3B results: validation on Google's Willow surface-code data

**Dataset:** Zenodo 13273331 (105-qubit processor, Nature 2025). All numbers
are logical error rates on 10k held-out real shots per experiment (20% split
by seeded permutation; fine-tuning saw only the other 80%). Google decoder
numbers come from the dataset's recorded shot-level predictions, subset to
the identical held-out shots.

## Results (13-round memory experiments, X basis)

| decoder | d=3 (q4_5) | d=5 (q4_7) |
|---|---|---|
| Google harmony, RL prior | 0.0782 | 0.0379 |
| Google corr. matching, RL prior | 0.0786 | 0.0425 |
| Google harmony, si1000 prior | 0.0864 | 0.0401 |
| **ours: CNN (DEM pretrain + real fine-tune)** | **0.0800** | **0.0562** |
| Google corr. matching, si1000 prior | 0.0901 | 0.0450 |
| ours: plain MWPM on fitted DEM | 0.1076 | 0.0639 |

## Claims

1. **The decoder beats plain MWPM on real processor data at both distances**
   (−18% at d=3, −12% at d=5) — the plan's target criterion.
2. **At d=3 it is statistically indistinguishable from Google's best
   production pathways**: paired vs harmony-RL, 175 shots only-ours-wrong vs
   157 only-Google's-wrong (McNemar p ≈ 0.32).
3. **Prior quality, not the network, set the d=3 ceiling**: identical
   training on the RL-optimized DEM instead of si1000 improved held-out
   error 11% (0.0896 → 0.0800), fully closing the gap to the si1000-prior
   decoders and matching the RL-prior ones.
4. **At d=5 Google's production decoders remain clearly ahead** (0.038–0.045
   vs our 0.0562). This is the scale boundary of the result at homelab
   compute and model size; stated, not spun.
5. **Sim-to-real transfer works, and real data adds a little**: DEM-pretrained
   models land within 5% of their fine-tuned versions at both distances.

## The d=5 memorization failure (kept)

First d=5 attempt: 8M fixed synthetic shots × 20 epochs → train loss 0.0017
with held-out 0.0748 (worse than MWPM) — the network memorized the finite
sample (healthy d=3 ratio: 0.034 train / 0.094 held-out). Fix: stream fresh
DEM samples every epoch (96M unique shots total; `resample_per_epoch`);
train loss then plateaus at 0.0068 and held-out improves to 0.0562. Rule:
synthetic pretraining sets must be effectively infinite — regenerate, never
recycle, once (dataset size)/(params) gets small.

## Methods notes

XZZX circuits; volume representation built from detector coordinates
(generalized to unit-spaced coords); fitted DEMs and all Google predictions
from the dataset. Training: 48ch ×4 GroupNorm CNN, AdamW lr 1e-3, cosine,
clip 1.0, warmup 500; fine-tune lr 1e-4. Configs `experiments/sycamore_*.json`;
artifacts alongside. d=7 not attempted (VRAM); r=13 only — round-scaling on
real data is future work.
