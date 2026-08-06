# Roadmap

Phases are sequential; each has a concrete exit criterion so "done" is
checkable, not vibes. Results that back an exit criterion live in
`experiments/` with a config and a figure in `docs/figures/`.

## Phase 0 — Baselines & harness

Goal: a trustworthy benchmarking harness before any ML.

- [ ] Environment: stim, pymatching, sinter, torch installed; GPU visible.
- [ ] Repetition code: generate circuits (d = 3..11, phenomenological +
      circuit-level noise), decode with PyMatching, plot p_L vs p.
- [ ] Rotated surface code: memory experiment, d = 3/5/7, circuit-level
      depolarizing noise, r = d rounds, MWPM via sinter.
- [ ] Reproduce the standard threshold crossing (~0.5–1% for circuit-level
      depolarizing noise, MWPM). Curves for different d must cross.
- [ ] Latency baseline: PyMatching decode time per shot vs d (batched, single).

**Exit:** threshold plot that matches published MWPM surface-code results;
harness produces (config → metrics artifact) end-to-end.

## Phase 1 — Neural decoder, repetition code

Goal: smallest possible ML decoder that works; establish the training pipeline.

- [ ] Dataset pipeline: Stim detection-event sampling → tensors (memory-mapped
      or generated-on-the-fly; avoid materializing >10 GB).
- [ ] MLP baseline predicting the logical observable flip from the syndrome.
- [ ] Match or beat MWPM logical error rate on rep code, d ≤ 11, at 2–3 values
      of p spanning the threshold region.
- [ ] Training-size study: accuracy vs number of training shots.

**Exit:** NN ≥ MWPM accuracy on repetition code with a documented training
recipe; pipeline reusable for Phase 2.

## Phase 2 — Neural decoder, surface code

Goal: the headline result — a competitive learned decoder under circuit-level noise.

- [ ] Input representation study: raw detection events vs per-round 2D layout
      (d×d detector grid × rounds) — this choice dominates everything after it.
- [ ] Architectures: 2D/3D CNN, then a small transformer over rounds
      (AlphaQubit-style but sized for 8 GB VRAM).
- [ ] Beat MWPM logical error rate at d = 3 and d = 5 (this is achievable —
      MWPM is suboptimal for correlated errors from circuit-level noise;
      compare also against PyMatching with correlated re-weighting).
- [ ] Generalization: train at one p, evaluate across p; train per-d vs shared.
- [ ] d = 7 attempt; document VRAM/throughput limits honestly.

**Exit:** p_L(NN) < p_L(MWPM) at d = 3, 5 under circuit-level depolarizing
noise, with error bars (sinter-style bootstrap), reproducible from configs.

## Phase 3 — Frontier (pick based on Phase 2 findings)

Option A — **Latency & deployment study** (plays to systems strengths):
quantization (int8), torch.compile/TensorRT, batched-streaming decode;
report ns/shot vs the ~1 µs cycle budget; sketch an FPGA feasibility estimate.

Option B — **Noise the matcher can't handle**: inject correlated errors /
leakage-like noise into Stim circuits; show the learned decoder degrades
gracefully where MWPM degrades badly.

Option C — **qLDPC**: BB codes (e.g. [[144,12,12]] gross code) with BP+OSD as
baseline; exploratory — decoders here are genuinely unsettled.

**Exit:** a self-contained writeup in `docs/` suitable for a blog post / arXiv
appendix, with all figures reproducible.

## Non-goals (for now)

- Real hardware data (no access); simulated noise only, stated plainly.
- Decoding at d > 9 or real-time FPGA implementation (estimate, don't build).
- Non-CSS codes, measurement-free QEC, analog decoding.
