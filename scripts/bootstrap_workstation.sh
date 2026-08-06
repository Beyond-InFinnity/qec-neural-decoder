#!/usr/bin/env bash
# Run FROM the Claude server once `ssh cmfinnerty true` works (key installed).
# Sets up bare repo + working clone + cu128 venv on the workstation, pushes
# current main, and runs the GPU sanity check.
set -euo pipefail

REPO_NAME=qec-neural-decoder
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=cmfinnerty

ssh "$REMOTE" "git init --bare -q ~/repos/${REPO_NAME}.git 2>/dev/null || true; mkdir -p ~/projects"

if ! git -C "$LOCAL_REPO" remote get-url cmfinnerty >/dev/null 2>&1; then
    git -C "$LOCAL_REPO" remote add cmfinnerty "${REMOTE}:repos/${REPO_NAME}.git"
fi
git -C "$LOCAL_REPO" push -q cmfinnerty main

ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
if [ ! -d ~/projects/${REPO_NAME} ]; then
    git clone -q ~/repos/${REPO_NAME}.git ~/projects/${REPO_NAME}
else
    git -C ~/projects/${REPO_NAME} pull -q
fi
cd ~/projects/${REPO_NAME}
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -e ".[dev]"
./.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
EOF
echo "bootstrap complete"
