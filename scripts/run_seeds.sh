#!/usr/bin/env bash
# Multi-seed retrains for the claim-bearing ladder rungs — run ON the
# workstation, sequential on cuda:1. Seeds 41 and 53 alongside the originals.
set -e
cd ~/projects/qec-neural-decoder
for tag in ladder_c16d2 ladder_c16d3 ladder_c32d3; do
  for seed in 41 53; do
    name="${tag}_s${seed}"
    ./.venv/bin/python - <<PYEOF
import json
c = json.load(open("experiments/${tag}.json"))
c.update(seed=$seed, name="$name")
json.dump(c, open("experiments/${name}.json", "w"), indent=2)
PYEOF
    echo "=== $name ===" >> experiments/seeds_d5.log
    ./.venv/bin/python -m qecdec.train --config experiments/${name}.json \
        --device cuda:1 >> experiments/seeds_d5.log 2>&1
  done
done
echo SEEDS_DONE >> experiments/seeds_d5.log
