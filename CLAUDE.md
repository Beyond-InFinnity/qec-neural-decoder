# CLAUDE.md — qec-neural-decoder

## Project purpose

Build and benchmark neural-network decoders for QEC codes (repetition code →
rotated surface code → stretch goals: correlated noise, qLDPC). Every claim is
benchmarked against an MWPM baseline (PyMatching) on the same Stim-generated
circuits. Two headline metrics: **logical error rate** and **decode latency**.

This is a portfolio/research project for Connor (systems & instrumentation
background, ML engineer). The latency/systems angle is a differentiator — treat
"could this run in a ~1 µs syndrome cycle" as a first-class question, not an
afterthought.

## Current status

Phase 0 complete (2026-08-05): MWPM threshold curves reproduced (surface-code
crossing p≈0.007), latency baseline (i5: d=7 ~22 µs/shot single). Phase 1
complete (2026-08-08): dilated CNN matches MWPM on rep code d≤11 — full arc,
recipe, and negative results in docs/phase1-results.md. Workstation
orchestration live (see ~/Documents/projects/homelab/ORCHESTRATION.md; GPU
runs on cuda:1 = RTX 5050). Next: Phase 2 (surface-code decoder) — carry
forward: locality-first arch, RF ≥ d, grad-clip + lr ≤ 1e-3, train-at-high-p,
paired eval. Still pending: latency re-run on i9-9900; tmux install.

## Hardware context

- Training: RTX 3070 (8 GB) or RTX 5050 (8 GB) in the i9-9900 / 64 GB machine.
- Data generation + MWPM baselines: CPU (Stim is extremely fast on CPU).
- A GTX 1080 Ti (11 GB) exists on a second machine (Z97) — usable for extra
  statevector work but old CUDA arch (sm_61); modern PyTorch may not support it.
  Prefer the 3070/5050.
- 8 GB VRAM is the binding constraint: keep models small, use mixed precision,
  and batch-size accordingly.

## Conventions

- Python ≥3.11, package lives in `src/qecdec/`, installed editable (`pip install -e ".[dev]"`).
- Core deps: stim, pymatching, sinter, torch, numpy. Keep the dependency list lean.
- Every experiment is a config-driven run under `experiments/` producing a
  JSON/CSV artifact with: git SHA, config, seed, metrics. No results in
  notebooks only — anything cited in docs must be reproducible from a config.
- Seed everything. Stim samplers, torch, numpy.
- Physics/decoding logic gets unit tests (`pytest`); e.g. "MWPM on d=3 with
  p=0 gives zero logical errors", "detector counts match circuit structure".
- Plots follow the dataviz skill if available; store under `docs/figures/`.
- Units and terminology: physical error rate `p`, logical error rate per shot
  `p_L`, code distance `d`, rounds `r` (default `r = d`).

## Key technical decisions (append as made)

- Circuit generation via `stim.Circuit.generated(...)` for standard codes;
  custom circuits only when the built-ins can't express the noise model.
- Baseline decoder: PyMatching v2 via `sinter` for threshold-curve collection.
- (pending) Neural architecture choices per phase — record here with rationale.

## Commands

```bash
source .venv/bin/activate
pytest                        # unit tests
pytest -m "not slow"          # skip long sampling tests
```
