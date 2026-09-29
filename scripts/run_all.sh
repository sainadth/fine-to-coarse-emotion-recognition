#!/usr/bin/env bash
# Full pipeline: data -> Stage I -> systems M1-M4 -> tables -> error analysis.
# Usage: bash scripts/run_all.sh [extra args passed to every Stage-II run]
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py --dataset all
python scripts/train_stage1.py --config configs/stage1.yaml

for system in M1 M2 M3 M4; do
  echo "=== ${system} ==="
  python scripts/train_stage2.py --system "${system}" "$@"
done

python scripts/build_results_table.py
python scripts/error_analysis.py
