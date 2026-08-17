# Phase 3B plan: validation on Google's Willow surface-code data

**Objective:** test whether the decoder family and training recipe developed
on simulated depolarizing noise transfer to real experimental data — the
credibility anchor for every simulation claim in this repo.

**Dataset:** Zenodo 13273331 (Nature 2025, "Quantum error correction below
the surface code threshold"), archive `google_105Q_surface_code_d3_d5_d7.zip`
(5.7 GB): memory experiments on the 105-qubit processor, d = 3/5/7, with
detection events and (per the paper's data release convention) stim circuits
and fitted detector error models. Exact internal format to be confirmed from
the archive README on arrival — the ingestion design below is provisional
until then.

## Design

1. **Ingest** (`qecdec/sycamore.py`): load their stim circuit + fitted DEM +
   detection-event shots per (distance, basis, instance). Our
   `representation.volume_spec` reads detector coordinates from the circuit,
   so the space-time volume mapping should generalize; verify detector
   counts and coordinate conventions explicitly, and add unit tests against
   the archive's documented shapes.
2. **Classical baselines on real data**: PyMatching and belief-matching on
   their fitted DEM, reproducing (or honestly failing to reproduce) the
   paper's reported logical error rates per round. This calibrates our
   pipeline against their published numbers before any NN enters.
3. **NN protocol** (AlphaQubit-style, scaled down):
   a. Pretrain on synthetic shots sampled from their fitted DEM (not our
      depolarizing model) — same cnn3d family, warmup recipe.
   b. Fine-tune on real detection events, holding out a test split by
      experiment instance (never train and test on shots from the same
      instance).
   c. Evaluate all decoders on the same held-out real shots, paired.
4. **Rounds mismatch**: their memory experiments run to r = 250 rounds
   (paper convention: varying r); our architecture assumed r = d. Start with
   the shortest-round subsets matching our volume shapes; if only long-r
   data exists, window it (decode per-window) or extend T in the volume —
   decide from the actual archive contents, document the choice.

## Success criteria (graded, honest)

- Floor: pipeline ingests real data; MWPM numbers on their DEM match the
  paper's within stated uncertainty.
- Target: our NN beats plain MWPM on held-out real shots at d = 3 and d = 5.
- Stretch: NN approaches/beats belief-matching on real data (AlphaQubit-tier
  claim at hobby scale — do not expect this; report whatever happens).

## Constraints

- d = 7 likely exceeds 8 GB VRAM for training — attempt only if d=3/5 go
  well, with the uint8/windowing tricks first.
- All Phase 2/3 rules apply (warmup, checkpoint-before-eval, seeded configs,
  paired evaluation, ledger entries for long jobs).
