# Recipe experiments (2026-10-01)

`results.tar.gz` (extract with `tar xzf results.tar.gz -C ../../results`):

- `sweep_*`, `stage1_sweep*` -- 15 M4 recipe variants, seed 42 (summary:
  `docs/tables/sweep.md`). Ran on the pre-fingerprint source, so no dev probabilities.
- `v2_*` -- M1-M5 x seeds 42/1/2 with the baseline recipe, now saving dev/test
  probabilities. Test metrics reproduce `archive/baseline_2026-09-30` exactly.
- `v2sp_*` -- same, with `speaker_prefix=true`.
- `v2_calibration.json`, `v2sp_calibration.json` -- dev-fitted offsets and
  seed ensembles from `scripts/calibrate.py`.

Full Kaggle outputs (logs included) are under `kaggle_output/` (gitignored).
