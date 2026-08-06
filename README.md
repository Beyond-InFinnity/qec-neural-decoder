# qec-neural-decoder

Neural-network decoders for quantum error correction, benchmarked against
matching decoders under circuit-level noise.

## What this is

Real-time decoding is one of the genuine open bottlenecks in fault-tolerant
quantum computing: syndrome data arrives every ~1 µs and the decoder must keep
up or the backlog grows unboundedly. This project builds ML decoders for the
repetition code and rotated surface code, trained on syndrome data generated
with [Stim](https://github.com/quantumlib/Stim), and benchmarks them against
minimum-weight perfect matching ([PyMatching](https://github.com/oscarhiggott/PyMatching))
on two axes:

1. **Accuracy** — logical error rate vs. physical error rate, per code distance.
2. **Latency** — inference time per shot, batched and single-shot, including
   quantized/compiled variants, measured against the syndrome-cycle budget.

The accuracy/latency framing is deliberate: decoding is as much a streaming
systems problem as an ML problem.

## Roadmap (summary)

| Phase | Goal | Exit criterion |
|-------|------|----------------|
| 0 | Baselines | Reproduce standard threshold curves (rep code + surface code d=3/5/7, MWPM) |
| 1 | First neural decoder | MLP/CNN matches MWPM on repetition code |
| 2 | Surface code | Transformer/CNN decoder competitive with MWPM under circuit-level noise |
| 3 | Frontier | Latency study; correlated/leakage noise or qLDPC codes |

Full detail in [docs/ROADMAP.md](docs/ROADMAP.md); architecture and interfaces
in [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
src/qecdec/      Python package (circuits, sampling, decoders, benchmarks)
tests/           pytest suite
notebooks/       exploratory analysis; results graduate into src/ or docs/
experiments/     run configs + result artifacts (JSON/CSV); large files gitignored
docs/            roadmap, design notes, results writeups
data/            generated syndrome datasets (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU training targets the RTX 3070 / RTX 5050 (8 GB each); Stim sampling and
MWPM baselines run CPU-side (i9-9900, 64 GB RAM).
