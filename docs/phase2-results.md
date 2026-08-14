# Phase 2 results: neural decoders for the rotated surface code

**Claims** (all paired, million-shot test sets; artifacts + configs in `experiments/`):

1. A 3D CNN over the (Z/X-channel, time, space) syndrome volume **beats plain
   MWPM everywhere tested**: 0.83–0.88× at d=3, 0.70–0.83× at d=5.
2. Against the strong classical baseline (belief-matching, 20 BP iterations)
   it **wins at d=3 (0.92–0.95×)** and reaches **parity-to-+10% at d=5**
   (1.01× at p=0.004, 1.07×, 1.10×).
3. A bounded 2× scale-up at d=5 **overfits** (train loss 3× lower, all eval
   points worse; 1.11–1.18× vs BM on matched test sets). Naive capacity does
   not close the d=5 gap.

## Model & recipe

`SurfaceConvDecoder`: detector coordinates → (2, r+1, d+1, d+1) volume
(Z-checks channel 0, X-checks channel 1), Conv3d ×4 (48 ch) + GroupNorm +
ReLU, dense head. Train at p=0.008 (near threshold), evaluate at
0.004/0.006/0.008 (train-high-eval-low transfer, per Phase 1). AdamW lr 1e-3,
cosine, grad-clip 1.0, **warmup 500 steps**, batch 2048, 15M shots, 25 epochs.
Checkpoints saved before evaluation, eval chunked at 8192 shots.

| config | d=3 vs MWPM / BM | d=5 vs MWPM / BM |
|---|---|---|
| v1 (48ch ×4) | 0.83–0.88× / 0.92–0.95× | 0.70–0.83× / 1.01–1.10× |
| v2 (64ch ×5, 20M, 30ep) | — | 0.76–0.88× / 1.11–1.18× (overfit) |

Why beating MWPM is possible at all: circuit-level noise produces correlated
X/Z errors (Y components, hook errors) that matching's independent-edge model
discards. Belief-matching recovers much of this classically via BP — hence it,
not plain MWPM, is the bar that matters. The NN clears it at d=3; at d=5 our
best model is statistically tied at low p and ≤10% behind above.

## The training-stability investigation (the freeze)

The d=5 scale-up froze twice at loss = 0.62170 — exactly H(0.3133), the label
base-rate entropy: constant-output collapse. Hypotheses falsified by direct
experiment, in order: seed/lr grid (all train in isolation), Stim corruption
at 20M shots (healthy stats), CUDA gather >2^31 elements (bit-exact vs CPU),
depth-5 without norm (GroupNorm run froze identically), dataset size (freeze
reproduces at 2M), cosine scheduler (freeze reproduces without). Surviving
variable: **the data seed**. Seed-30 sample content deterministically
collapses the fresh network within the first epoch at full lr; seed-100
trains, with identical init and batch order. GroupNorm and grad-clip do not
prevent it. **500-step linear LR warmup rescues the exact pathological case**
(0.649→0.622 frozen becomes 0.252→0.128) and trained the hard seed to the
best d=5 loss trajectory we observed.

Operational lessons the harness now encodes: checkpoint before eval; eval
chunk size is a memory parameter; `pgrep -f` watchers must bracket their
pattern (`[q]ecdec`) or they match themselves; delete stale result artifacts
before relaunch; `apt upgrade` kills live CUDA contexts (see homelab
ORCHESTRATION.md).

## Not claimed

- No d=7 result (VRAM/time; Phase 3 candidate).
- Simulated depolarizing noise only; no real-device data yet (Google's public
  Sycamore memory data is the planned Phase 3 credibility anchor).
- Regularized/augmented scale-up untried at d=5 (would break the bounded-
  iteration rule; noted as future work).

## Carry-forward to Phase 3 (latency frontier)

Checkpoints exist for d=3 and d=5 (v1 = best). Three decoders benchmarkable
on identical shots: NN (GPU, quantizable), PyMatching, belief-matching. The
question: the smallest/fastest network still beating each classical decoder,
in ns/shot, against the ~1 µs cycle budget.
