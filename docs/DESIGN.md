# Design notes

## Data flow

```
stim.Circuit (generated, noise params)
   └─ DetectorErrorModel ──────────────► PyMatching (baseline)
   └─ CompiledDetectorSampler ─► detection events + observable flips (bool arrays)
                                    └─► torch Dataset ─► NN decoder ─► predicted flip
Benchmark: p_L = mean(pred != actual), identical shots fed to both decoders.
```

Key invariant: **NN and MWPM are always evaluated on the same sampled shots**
so comparisons are paired, and error bars can use the paired difference.

## Package sketch

```
qecdec/
  circuits.py     # circuit generation: code family, d, rounds, noise model
  sampling.py     # Stim sampling → numpy/torch; on-the-fly Dataset for training
  baselines.py    # PyMatching / sinter wrappers, latency timing
  models/         # mlp.py, cnn.py, transformer.py — one file per family
  train.py        # config-driven training loop (single entrypoint)
  eval.py         # paired evaluation, bootstrap error bars
  configs.py      # dataclass configs; every run serializes its config
```

## Representation decisions (Phase 2's crux)

Detection events for a d×d rotated surface code with r rounds map naturally to
a (r+1) × (d²−1) tensor, or spatially to per-round images on the (d+1)×(d+1)
plaquette grid (X and Z detectors as 2 channels). CNNs want the spatial
layout; transformers can take per-round tokens. Decide empirically in Phase 2
and record the outcome here.

## Noise models

- Phase 0/1: `after_clifford_depolarization = p` style circuit-level noise via
  Stim's generated circuits (SD6-like). One knob, standard, comparable to
  literature.
- Phase 3B: custom noise insertion (correlated two-qubit bursts, leakage
  proxies via heralded erasure or biased channels). Document any custom
  channel's Stim encoding here when built.

## Benchmarking rules

- Logical error rates always with shot counts sized so the rarer outcome has
  ≥ ~100 events, or report the bound honestly.
- Latency: report median and p99, warm cache, separated into
  batched-throughput (shots/sec) and single-shot latency — they answer
  different questions (offline analysis vs real-time feasibility).
- Never compare NN-favorable settings against MWPM defaults silently; MWPM
  gets its best reasonable configuration (e.g. correlated matching on).

## VRAM budget (8 GB)

Rule of thumb: keep models < 20 M params, use AMP (bf16 on 5050, fp16 on
3070), gradient accumulation before model shrinking. If d = 7 transformer
doesn't fit, chunk rounds or move to the 1080 Ti only if PyTorch still ships
sm_61 wheels (check, don't assume).
