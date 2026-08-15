#!/usr/bin/env bash
# Phase 3A model-size ladder at d=5 — run ON the workstation, sequentially on
# cuda:1. Each rung derives its config from the phase2 v1 config (15M shots,
# 25 epochs, eval at 0.004/0.006/0.008) with a smaller model + warmup.
set -e
cd ~/projects/qec-neural-decoder
for m in "8 2 64" "16 2 64" "16 3 128" "32 3 128" "32 4 128"; do
  set -- $m
  ch=$1; dep=$2; head=$3
  tag="ladder_c${ch}d${dep}"
  ./.venv/bin/python - <<PYEOF
import json
c = json.load(open("experiments/phase2_surface_cnn3d.json"))
c.update(cells=[[5, 0.008]], distances=[5], name="$tag", warmup_steps=500,
         model={"arch": "cnn3d", "channels": $ch, "depth": $dep, "head": $head})
json.dump(c, open("experiments/$tag.json", "w"), indent=2)
PYEOF
  echo "=== $tag ===" >> experiments/ladder_d5.log
  ./.venv/bin/python -m qecdec.train --config experiments/$tag.json \
      --device cuda:1 >> experiments/ladder_d5.log 2>&1
done
echo LADDER_DONE >> experiments/ladder_d5.log
