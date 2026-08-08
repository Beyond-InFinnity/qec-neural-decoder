# Roadmap

Phases are sequential; each has a concrete exit criterion so "done" is
checkable, not vibes. Results that back an exit criterion live in
`experiments/` with a config and a figure in `docs/figures/`.

## Phase 0 — Baselines & harness

Goal: a trustworthy benchmarking harness before any ML.

- [x] Environment: stim 1.16 / pymatching 2.4 / sinter 1.16 on claude-server
      (CPU-only). Torch+GPU env pends the workstation SSH key
      (scripts/bootstrap_workstation.sh is ready to run).
- [x] Repetition code: circuit-level noise, d = 3..11, MWPM, p_L vs p
      (experiments/phase0_repetition.json → .results.json;
      docs/figures/phase0_repetition_threshold.png). Crossing ≈ 5%.
- [x] Rotated surface code: memory experiment, d = 3/5/7, circuit-level
      depolarizing noise, r = d, MWPM via sinter
      (experiments/phase0_surface.json → .results.json).
- [x] Threshold crossing reproduced at p ≈ 0.007 with clean sub/supra-threshold
      ordering inversion (docs/figures/phase0_surface_threshold.png) —
      consistent with published MWPM circuit-level results.
- [x] Latency baseline (experiments/phase0_latency.results.json), i5-4690K:
      d=7 batched 16 µs/shot, single-shot median 22 µs, p99 49 µs — vs the
      ~1 µs real-time budget. Re-run on the i9-9900 for the number of record.

**Exit:** threshold plot that matches published MWPM surface-code results;
harness produces (config → metrics artifact) end-to-end.

## Phase 1 — Neural decoder, repetition code

Goal: smallest possible ML decoder that works; establish the training pipeline.

- [x] Dataset pipeline: Stim sampling → uint8 on-device tensors (15M-shot
      d=11 set ≈ 1.2 GB VRAM).
- [x] MLP baseline: works at d=3 (beats MWPM 1.05×⁻¹), fails to scale
      (17.8× worse at d=11 even at 19× compute) — documented negative.
- [x] Match or beat MWPM at d ≤ 11: achieved via dilated CNN (RF 29) +
      grad-clip + train-at-p=0.05 transfer. Within 8% everywhere; NN ahead at
      d=9 p≥0.03 and d=7/11 p=0.05. See docs/phase1-results.md.
- [x] Training-budget/generalization studies: small-vs-19× MLP;
      p-matched vs p=0.05-transfer training (transfer strictly better).

**Exit: met (2026-08-08).** Recipe + full arc in docs/phase1-results.md;
pipeline (build_model / cells / eval_ps / paired eval) carries to Phase 2.

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
