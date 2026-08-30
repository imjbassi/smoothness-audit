#!/bin/bash
set -e
for cfg in configs/*.json; do
  name=$(basename "$cfg" .json)
  if [ -d "results_train/$name" ] && grep -q "finished run successfully" "results_train/$name"/*/logs/log.txt 2>/dev/null; then
    echo "=== skipping $name (already done) ==="
    continue
  fi
  echo "=== $name ==="
  rm -rf "results_train/$name"
  python robomimic/scripts/train.py --config "$cfg"
done
