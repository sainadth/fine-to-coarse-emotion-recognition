#!/usr/bin/env bash
# Train every run behind the report, then rebuild its tables.
#
#   bash scripts/run_all.sh               # about 4 GPU-hours on a Tesla T4
#
# Each seed trains its own Stage-I backbone and the transfer systems of that seed start
# from it. Seeds 42, 1, 2 cover every system; seeds 3-6 add M4, the extension and the
# speaker-aware variant. Extra arguments are passed to every training run as overrides.
# Model weights are deleted after each seed (about 6 GB peak); KEEP_CHECKPOINTS=1 keeps
# them, about 23 GB in total.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"42 1 2 3 4 5 6"}

python scripts/prepare_data.py --dataset all

for seed in $SEEDS; do
  backbone=checkpoints/v2_stage1_s$seed
  python scripts/train_stage1.py --config configs/stage1.yaml --set run_name=v2_stage1_s$seed seed=$seed "$@"

  # full model, extension, and the speaker-aware variant: every seed
  python scripts/train_stage2.py --system M4 --set run_name=v2_M4_s$seed seed=$seed from_stage1=$backbone "$@"
  python scripts/train_stage2.py --config configs/m5_extension_fine_logits.yaml \
      --set run_name=v2_M5_s$seed seed=$seed from_stage1=$backbone "$@"
  python scripts/train_stage2.py --system M4 \
      --set run_name=v3spk_M4_s$seed seed=$seed from_stage1=$backbone speaker_aware=true "$@"

  # the other required systems, and the speaker-prefix recipe: seeds 42, 1, 2 only
  case $seed in 42|1|2)
    python scripts/train_stage2.py --system M1 --set run_name=v2_M1_s$seed seed=$seed "$@"
    python scripts/train_stage2.py --system M3 --set run_name=v2_M3_s$seed seed=$seed "$@"
    python scripts/train_stage2.py --system M2 --set run_name=v2_M2_s$seed seed=$seed from_stage1=$backbone "$@"
    for system in M1 M3; do
      python scripts/train_stage2.py --system $system \
          --set run_name=v2sp_${system}_s$seed seed=$seed speaker_prefix=true "$@"
    done
    for system in M2 M4; do
      python scripts/train_stage2.py --system $system \
          --set run_name=v2sp_${system}_s$seed seed=$seed from_stage1=$backbone speaker_prefix=true "$@"
    done
    python scripts/train_stage2.py --config configs/m5_extension_fine_logits.yaml \
        --set run_name=v2sp_M5_s$seed seed=$seed from_stage1=$backbone speaker_prefix=true "$@"
  ;; esac

  # Weights are ~0.5 GB per run and nothing later reads this seed's; 45 runs would not fit
  # in Kaggle's 20 GB of output. Metrics, predictions and probabilities stay in results/.
  [ -n "${KEEP_CHECKPOINTS:-}" ] || rm -rf checkpoints/*_s"$seed"
done

bash scripts/build_tables.sh
