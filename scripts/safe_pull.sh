#!/usr/bin/env bash
# Run on the workstation. Artifacts get committed on the server after being
# scp'd back, so the workstation's untracked originals collide with the next
# pull. Delete exactly those (untracked here, tracked in incoming), then pull.
set -e
cd "$(dirname "$0")/.."
git fetch -q origin main
git ls-tree -r --name-only origin/main | while read -r f; do
  if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    rm -f "$f"
    echo "removed untracked collision: $f"
  fi
done
git merge -q --ff-only origin/main
git log --oneline -1
