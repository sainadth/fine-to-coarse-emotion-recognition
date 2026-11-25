# Fine-to-Coarse Sequential Emotion Recognition

> **RoBERTa-base → GoEmotions₂₇ → RoBERTa_G + UTE → MELD₇**
> Does fine-grained emotion learning on GoEmotions improve coarse-grained sequential
> emotion recognition on MELD when dialogue context is modelled by an utterance-level
> Transformer?

COSC 6341, Fall 2026 — Computer Science Department, TAMUCC. Project specification:
[`docs/project-spec.pdf`](docs/project-spec.pdf).

---

## 1. What this repository does

Two training stages and four systems.

**Stage I — fine-grained emotion learning.** RoBERTa-base is fine-tuned on GoEmotions
with 27 fine-grained classes (the `neutral` class is excluded). The 27-class head
`W27` is then discarded; the backbone parameters `θ_G` are kept.

**Stage II — sequential coarse emotion learning.** Every utterance of a MELD dialogue is
encoded by the shared backbone, turn-level positional embeddings are added, an
utterance-level Transformer encoder (UTE) contextualises the sequence, and a new 7-class
head predicts the coarse emotion per turn:

```
U_t → RoBERTa_G → h_t → (+ p_t) → UTE → z_t → W7 → MELD emotion
```

`θ_G`, `θ_UTE` and `W7` are all trainable in Stage II — continued fine-tuning, not a
frozen feature extractor.

| System | Backbone | UTE | Config |
|---|---|---|---|
| M₁ | RoBERTa-base | no | `configs/m1_base_noute.yaml` |
| M₂ | RoBERTa_G (GoEmotions) | no | `configs/m2_goemotions_noute.yaml` |
| M₃ | RoBERTa-base | yes | `configs/m3_base_ute.yaml` |
| M₄ | RoBERTa_G (GoEmotions) | yes | `configs/m4_goemotions_ute.yaml` |
| extension | RoBERTa_G + `e_t = [h_t ; p_t^f]` | yes | `configs/m5_extension_fine_logits.yaml` |

M₁ vs M₂ isolates fine-grained transfer, M₁ vs M₃ isolates sequential modelling, and
M₃ vs M₄ asks whether transfer still helps once context is modelled.

## 2. Requirements

* Python 3.10+
* PyTorch 2.1+ (CUDA, Apple MPS, or CPU — the device is picked automatically)
* ~4 GB GPU memory for the dialogue-level systems at the default settings

A free Colab T4 runs the whole pipeline; see [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb).

## 3. Setup

```bash
git clone <this repo> && cd fine-to-coarse-emotion-recognition
python -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install -r requirements.txt
pip install -e .
```

## 4. Get the data

```bash
python scripts/prepare_data.py --dataset all
```

* **GoEmotions** is pulled from the Hugging Face Hub (`google-research-datasets/go_emotions`, `simplified`).
  `neutral` is dropped; because GoEmotions is multi-label and Stage I is a single-label
  cross-entropy classifier, the default policy (`multilabel_policy: single`) keeps only
  examples with exactly one non-neutral label. `first` keeps every example and takes its
  lowest-index label.
* **MELD** is downloaded as the three official CSVs
  (`train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`) into `data/meld/`.
  Expected sizes: 9,989 / 1,109 / 2,610 utterances.

Dialogue boundaries and utterance order are preserved; dialogues longer than
`max_turns` are split into consecutive non-overlapping windows, so no turn is dropped
and no window spans two dialogues. Train / dev / test stay separate, and the test set is
only ever used for the final evaluation — model selection uses dev macro-F1.

## 5. Train

```bash
bash scripts/run_all.sh
```

This trains every run behind the report and then rebuilds its tables (section 6). About
4 GPU-hours on a Tesla T4: 6.5 minutes per Stage-I run, 3.5 to 5 per Stage-II run.

For each of seven seeds (42, 1, 2, 3, 4, 5, 6) it trains a Stage-I backbone and, from it,
M₄, the extension and a speaker-aware M₄. For seeds 42, 1 and 2 it also trains M₁, M₂, M₃
and the speaker-prefix recipe for all five systems. `SEEDS="42 1 2" bash scripts/run_all.sh`
trains a subset. Model weights are deleted after each seed, so the run needs about
6 GB of free disk; `KEEP_CHECKPOINTS=1` keeps them (about 23 GB).

| Run names | Recipe | Seeds |
|---|---|---|
| `v2_stage1_s*` | Stage I | all seven |
| `v2_M1_s*` … `v2_M5_s*` | default recipe (M5 is the extension) | M4, M5: all seven; M1–M3: 42, 1, 2 |
| `v2sp_M*_s*` | `speaker_prefix=true` | 42, 1, 2 |
| `v3spk_M4_s*` | `speaker_aware=true` | all seven |

To train one piece yourself, any config key can be overridden from the command line:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml --set run_name=v2_stage1_s1 seed=1
python scripts/train_stage2.py --system M4 --set run_name=v2_M4_s1 seed=1 from_stage1=checkpoints/v2_stage1_s1
```

A two-minute end-to-end check that touches every code path, on CPU:

```bash
python scripts/train_stage2.py --config configs/smoke.yaml
```

## 5b. Run it on Kaggle's free GPUs

Kaggle is the only free GPU host with a real API, so runs can be launched from a terminal.
The repository source is uploaded as a **private** Kaggle dataset and a **private** notebook
kernel mounts it; nothing is published.

One-time setup: grab an API token at <https://www.kaggle.com/settings> > API >
*Create New Token*. The downloaded `kaggle.json` holds a username and a key; put both in
`.env` at the repository root (`.env` is git-ignored):

```bash
cp .env.example .env    # then fill in KAGGLE_USERNAME and KAGGLE_KEY
pip install kaggle
```

`~/.kaggle/kaggle.json` is still picked up if you prefer keeping the token there.

Internet-enabled kernels need a phone-verified Kaggle account (Settings > Phone Verification) --
without it the kernel cannot reach the Hugging Face Hub or the MELD CSVs.

```bash
python scripts/kaggle_run.py push              # upload/refresh the private source dataset
python scripts/kaggle_run.py submit --wait     # start the GPU run, block, then pull results
python scripts/kaggle_run.py status            # or poll it yourself
python scripts/kaggle_run.py pull              # copy metrics + tables back into the repo
```

Free tier: 30 GPU-hours per week, 12 hours per session -- enough for every run in section 5
in one kernel. `kaggle/run_pipeline.ipynb` lists the seeds and runs it trains; set
`KEEP_CHECKPOINTS` there to keep the Stage-I backbone in the kernel output.

## 6. Tables and report

```bash
bash scripts/build_tables.sh            # every table in the report, from results/
cd docs && tectonic report.tex          # report.pdf
python scripts/inline_report.py         # docs/report_inline.tex, one self-contained source file
```

`build_tables.sh` needs no GPU and takes about a minute. It runs:

| Report item | Command | Output |
|---|---|---|
| Main results, per-class tables | `build_results_table.py --prefix v2_ --seeds 42 1 2` | `docs/tables/table1.{md,tex}`, `per_class.tex`, `per_class_f1.tex` |
| Ensemble table | `ensemble.py` | `docs/tables/ensemble.{md,tex}`, `results/ensemble_selection.json` |
| Decision offsets (negative result) | `calibrate.py --prefix v2_`, then `--prefix v2sp_` | `results/v2_calibration.json`, `results/v2sp_calibration.json` |
| Contrastive examples, per-class deltas | `error_analysis.py` | `docs/tables/error_analysis.json`, `qualitative.tex` |

`--seeds 42 1 2` matters: M₄ and the extension have seven seeds, and without it their cells
would average seven runs next to three for M₁ to M₃. The recipe-sweep table
(`docs/tables/sweep.{md,tex}`) was written once from the sweep kernel
(`python scripts/kaggle_run.py submit --kernel sweep`) and is not rebuilt.

**Checking the reported numbers without training.** `results/` ships with the submission:
the metrics, predictions and probabilities of every run. Rebuild the tables from it and
compare with the committed ones:

```bash
bash scripts/build_tables.sh
git diff --stat docs/tables        # prints nothing: every table is reproduced byte for byte
```

**What to expect.** Main table: M₁ 45.16, M₂ 47.01, M₃ 46.31, M₄ 47.65 test macro-F1 (three
seeds). Single M₄ run over seven seeds: 47.30 ± 0.94. Dev-selected ensemble (M₂ + M₄ +
extension, 17 runs): 49.52 macro-F1, 65.86 accuracy. Speaker-aware against M₄, paired over
seven seeds: −0.50 macro-F1.

MELD is heavily class-imbalanced (47% `neutral`, 2.7% `disgust`), so **macro-F1 is the
headline metric**. `class_weighted_loss: true` enables inverse-frequency weighting; it is
off by default so the four systems stay comparable.

### Post-hoc: ensembling and decision offsets

Both act on saved probabilities, so neither needs a GPU or a retrain. Both choose their
settings on dev and touch test once.

`ensemble.py` averages the class probabilities of runs that differ only in seed, or in
system, and scores every candidate on dev. M4 and M5 have seven seeds (42, 1, 2, 3–6). The
dev-selected ensemble pools M2, M4 and M5 (17 runs) and reaches **49.52 test macro-F1
against 47.30 ± 0.94 for a single M4 run** — a bigger gain than any single architectural
factor in the study, and unlike those it raises accuracy too (65.86). The gain is seed
averaging, not system mixing: seven seeds of M4 alone reach 49.87, and the top three
candidates are within noise of each other. Pooling all five systems is worse.

Speaker-aware modelling (`speaker_aware=true`, runs
`v3spk_M4_s*`) does not help: −0.50 test macro-F1 against M4 over seven paired seeds.

`calibrate.py` fits one additive offset per class on dev to maximise dev macro-F1. It is
kept because the negative result is worth having: the offsets improve dev in all ten
system-by-variant combinations and make **test worse in nine of them**. Seven free
parameters on 1,109 dev utterances (25 of them `fear`) fit the dev sample, not the label
prior. The report presents it as a negative result and uses ensembling instead.

## 7. Outputs

| Artifact | Path |
|---|---|
| Stage-I backbone (`θ_G`) + discarded head | `checkpoints/stage1_goemotions/{backbone,fine_head.pt}` |
| Stage-II checkpoints | `checkpoints/<run_name>/model.pt` |
| Per-run metrics and history | `results/<run_name>/metrics.json` |
| Per-run dev/test predictions and probabilities | `results/<run_name>/{dev,test}_predictions.json` |
| Comparison tables | `docs/tables/table1.md`, `docs/tables/table1.tex` |
| Ensemble table and dev selection | `docs/tables/ensemble.{md,tex}`, `results/ensemble_selection.json` |
| Decision-offset results | `results/v2_calibration.json`, `results/v2sp_calibration.json` |
| Error analysis | `docs/tables/error_analysis.json` |
| Report source | `docs/report.tex`, `docs/report.bib` |
| Report PDF | `docs/report.pdf` (build: `cd docs && tectonic report.tex`) |

## 8. Tests

```bash
pytest -q
```

Covers the GoEmotions label remapping, MELD dialogue grouping / windowing / collation,
UTE padding invariance and order sensitivity, the metric block, and the fact that every
config matches the system it claims to be.

## 9. Repository layout

```
configs/                 one YAML per system (M1-M4, extension, smoke)
data/                    downloaded datasets (git-ignored)
docs/                    project spec, report source, generated tables
notebooks/               Colab runner
results/                 metrics + predictions per run (git-ignored)
scripts/                 prepare_data, train_stage1, train_stage2, tables, error analysis
src/fine2coarse/
  config.py              label vocabularies, paths, YAML helpers
  data/goemotions.py     Stage-I dataset (27 classes, neutral excluded)
  data/meld.py           Stage-II dialogue + utterance views
  models/stage1.py       RoBERTa + 27-class head, backbone export
  models/ute.py          utterance-level Transformer encoder
  models/stage2.py       RoBERTa_G + UTE + 7-class head (all four systems)
  engine.py              training / evaluation loops
  metrics.py             accuracy, macro-F1, weighted-F1, per-class, confusion matrix
tests/
```

## 10. Reproducibility notes

* **Same setup, same numbers.** The reported runs were made on a Kaggle Tesla T4 with
  torch 2.10.0+cu128 and transformers 5.0.0 (section 5b). On that setup training is
  repeatable to the last digit. On 2026-11-25 a fresh `bash scripts/run_all.sh` on Kaggle
  retrained all 52 runs in 243 minutes; every run's test metrics and predictions, and every
  table, came out identical to the ones shipped here.
* **Different setup, close numbers.** On another GPU or other library versions the same
  seeds give close but not identical results; seven seeds of M₄ span 45.6 to 48.6 test
  macro-F1, so a single run can land anywhere in that range.
* **Tables are exact everywhere.** Rebuilding the tables from the shipped `results/`
  (section 6) was tested in a clean `python:3.12-slim` container with torch 2.15.0 and
  transformers 5.24.1, and gave byte-identical tables.
* Every run records its full config, per-epoch history, and test metrics in
  `results/<run_name>/metrics.json`.
* Seeds are set for Python, NumPy and Torch; report the seed with any result.
* Checkpoint selection is by dev macro-F1; the test set is never used for selection.
* Mixed precision is enabled only on CUDA.
* Model weights are not kept (`run_all.sh` deletes them after each seed unless
  `KEEP_CHECKPOINTS=1`); `results/` holds metrics, predictions and probabilities for every run.

## 11. Citations

* Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
* Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
  Conversations*, ACL 2019.
* Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019.
* Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
