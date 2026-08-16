# Phase 3A results: the accuracy–throughput frontier (d=5 surface code)

**Setting:** rotated surface code, d=5, r=5, circuit-level depolarizing noise,
p=0.008 test sets (100k shots, seed-fixed). NN latencies on RTX 3070 (fp16,
batch 8192); classical decoders on i9-9900 CPU. All decoders evaluated on
identical shots. Figure: `figures/phase3_pareto_d5.png`. Artifacts:
`experiments/ladder_*.latency.json`, `phase3_latency_v1.json`.

## The frontier

| decoder | params | p_L (p=0.008) | µs/shot (batched) |
|---|---|---|---|
| NN c8d2 fp16 | 113k | 0.0525 | **0.40** |
| NN c16d2 fp16 | 229k | 0.0469 | **0.63** |
| NN c16d3 fp16 | 457k | 0.0416 | 0.96 |
| NN c32d3 fp16 | 942k | **0.0401** | 2.33 |
| NN 48d4 (v1) fp16 | 1.52M | 0.0412 | 4.63 |
| MWPM (PyMatching) | — | 0.0491 | 3.78 |
| belief-matching (20 BP iters) | — | 0.0372 | 3437 |

## Claims

1. **A 229k-param fp16 CNN beats MWPM's accuracy at 6× MWPM's throughput,
   inside the 1 µs/shot syndrome budget (batched).** MWPM is strictly
   Pareto-dominated by the c16d2–c32d3 range of the NN family.
2. **The accuracy-matching classical decoder is ~1500–5500× slower than the
   NN family.** Belief-matching keeps a 7% accuracy lead over our best
   network at p=0.008 (ties at p=0.004) but costs 3.4 ms/shot at d=5 —
   3400× the real-time budget, and 26× its own d=3 cost (BP scales badly).
3. **fp16 is free**: 2× throughput at every size, p_L unchanged to 4 decimals.
4. **The size ladder saturates at ~1M params** (c32d3 ≥ c32d4 ≥ v1),
   consistent with Phase 2's overfitting result from the other direction:
   229k → 942k buys 15% accuracy; beyond that, nothing or worse.

## Honest boundary: batched ≠ real-time

Single-shot GPU latency is ~410 µs median regardless of model size —
Python/kernel-launch overhead, not compute. The batched numbers represent a
streaming/windowed decoding regime (decode many syndrome windows in flight),
not a per-shot reactive loop. A real-time claim would need a compiled
persistent-kernel path (TensorRT/CUDA graphs) or FPGA port; the compute-cost
numbers here say that budget exists (0.4–2.3 µs of arithmetic), which is
precisely the actionable finding for decoder-hardware teams.

## Remaining Phase 3 work

- [ ] int8 quantization rung (expected ~2× over fp16; verify p_L survives).
- [ ] CUDA-graph / torch.compile single-shot latency (close the 410 µs gap).
- [ ] Real-device validation on Google's public Sycamore surface-code memory
      data — the credibility anchor for all simulated-noise claims.
- [ ] Multi-seed error bars on the ladder accuracies before any external
      writeup.
