# Baseline results snapshot (2026-09-30)

Frozen copy of the spec-compliant M1-M4 + extension results (3 seeds each) that the
report draft at commit 4db729f is built on, taken before any recipe changes
(focal loss, dev-tuned offsets, speaker prefix, seed ensembling).

- `results/` -- per-run `metrics.json` and `test_predictions.json` (the live
  `results/` directory is gitignored, so this is the only versioned copy).
  `M5_*` is the fine-grained-logits extension.
- `docs/tables/` -- generated tables and error analysis.
- `docs/report.pdf` / `report.tex` -- ACM two-column build of the report.
- `docs/report_original_layout.tex` -- the single-column article version at 4db729f.

Recipe: RoBERTa-base, plain cross-entropy, M1/M2 5 epochs, M3/M4 6 epochs,
seeds 42/1/2, checkpoint selection on dev macro-F1, test used once per run.
If later changes are found to break the project spec, report these numbers.
