#!/usr/bin/env bash
# Rebuild every table in the report from results/. No GPU, about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_results_table.py --prefix v2_ --seeds 42 1 2   # main table, per-class tables
python scripts/ensemble.py                                          # ensemble table
python scripts/calibrate.py --prefix v2_                            # decision offsets
python scripts/calibrate.py --prefix v2sp_
python scripts/prepare_data.py --dataset meld                       # error analysis reads the MELD CSVs
python scripts/error_analysis.py                                    # contrastive examples
