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

* **GoEmotions** is pulled from the Hugging Face Hub (`go_emotions`, `simplified`).
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

## 5. Run everything

```bash
bash scripts/run_all.sh
```

That is Stage I, then M₁–M₄, then the result tables and the error analysis. To run the
pieces yourself:

```bash
# Stage I (GoEmotions, 27 classes)
python scripts/train_stage1.py --config configs/stage1.yaml

# Stage II, one system at a time
python scripts/train_stage2.py --system M1
python scripts/train_stage2.py --system M2
python scripts/train_stage2.py --system M3
python scripts/train_stage2.py --system M4

# research extension: fine-grained distribution concatenated into the UTE input
python scripts/train_stage2.py --config configs/m5_extension_fine_logits.yaml
```

Any config key can be overridden from the command line:

```bash
python scripts/train_stage2.py --system M4 --set epochs=8 batch_size=8 seed=1 run_name=M4_seed1
```

A two-minute end-to-end check that touches every code path:

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

Free tier: 30 GPU-hours per week, 12 hours per session -- enough for Stage I plus M1-M4 in one
kernel. Edit `STAGES` in `kaggle/run_pipeline.ipynb` to run a subset, or set `KEEP_CHECKPOINTS`
there to keep the Stage-I backbone in the kernel output.

## 6. Results and analysis

```bash
python scripts/build_results_table.py     # docs/tables/table1.{md,tex}
python scripts/error_analysis.py          # docs/tables/error_analysis.json
```

`build_results_table.py` emits the required comparison table (accuracy, macro-F1,
weighted-F1 per system) plus a per-class F1 table. `error_analysis.py` reports per-class
F1 deltas for the three comparisons, the most frequent confusion pairs, and contrastive
examples — utterances the context-aware system gets right and the context-free system
gets wrong, with their preceding turns.

MELD is heavily class-imbalanced (47% `neutral`, 2.7% `disgust`), so **macro-F1 is the
headline metric**. `class_weighted_loss: true` enables inverse-frequency weighting; it is
off by default so the four systems stay comparable.

## 7. Outputs

| Artifact | Path |
|---|---|
| Stage-I backbone (`θ_G`) + discarded head | `checkpoints/stage1_goemotions/{backbone,fine_head.pt}` |
| Stage-II checkpoints | `checkpoints/<run_name>/model.pt` |
| Per-run metrics and history | `results/<run_name>/metrics.json` |
| Per-run test predictions | `results/<run_name>/test_predictions.json` |
| Comparison tables | `docs/tables/table1.md`, `docs/tables/table1.tex` |
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

* Every run records its full config, per-epoch history, and test metrics in
  `results/<run_name>/metrics.json`.
* Seeds are set for Python, NumPy and Torch (`seed: 42` by default). GPU kernels remain
  non-deterministic, so expect small run-to-run differences; report the seed with results.
* Checkpoint selection is by dev macro-F1; the test set is never used for selection.
* Mixed precision is enabled only on CUDA.

## 11. Citations

* Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
* Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
  Conversations*, ACL 2019.
* Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019.
* Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
