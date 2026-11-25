#!/usr/bin/env python3
"""Seed and system ensembles over Stage-II runs that saved probabilities.

Averaging the class probabilities of runs that differ only in seed removes some
of the variance that a single run carries, and MELD's rare classes (fear and
disgust, 50 and 68 test utterances) are exactly where that variance costs
macro-F1. Averaging across *systems* adds a second kind of diversity, since
M4 and M5 read the same utterance through different inputs.

Every candidate is scored on dev; the winner is evaluated on test once. This is
the same protocol the rest of the project uses, and it is why the dev column
below is the one that chose the reported system.

    python scripts/ensemble.py --prefixes v2_ v2sp_ v3spk_
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.calibration import pooled_predictions, select_by_dev  # noqa: E402
from fine2coarse.config import MELD_LABELS, RESULTS_DIR  # noqa: E402
from fine2coarse.metrics import compute_metrics  # noqa: E402
from fine2coarse.utils import get_logger, read_json, write_json  # noqa: E402

LOGGER = get_logger()
SYSTEMS = ["M1", "M2", "M3", "M4", "M5"]
# Subsets worth trying. "M4+M5" is the pair that carries both the transfer and the
# sequential components; "M2+M4+M5" adds the no-UTE transfer model; "all5" is the
# everything-in baseline that shows diversity alone is not enough.
SUBSETS = {
    "M4": ["M4"],
    "M4+M5": ["M4", "M5"],
    "M2+M4+M5": ["M2", "M4", "M5"],
    "all5": SYSTEMS,
}


def load_runs(prefix: str, system: str, split: str) -> list[dict]:
    """Every seed of one system that saved probabilities for `split`."""
    runs = []
    for directory in sorted(RESULTS_DIR.glob(f"{prefix}{system}_s*")):
        path = directory / f"{split}_predictions.json"
        if not path.exists():
            continue
        payload = read_json(path)
        if payload.get("probs"):
            runs.append(payload)
    return runs


def build_candidates(prefixes: list[str], split: str) -> dict[str, list[dict]]:
    """Named ensembles: each subset of systems, within one prefix and pooled across all."""
    candidates: dict[str, list[dict]] = {}
    groups = [[p] for p in prefixes]
    if len(prefixes) > 1:
        groups.append(prefixes)
    for group in groups:
        tag = group[0].rstrip("_") if len(group) == 1 else "pooled"
        for name, systems in SUBSETS.items():
            per_system = {
                system: [run for prefix in group for run in load_runs(prefix, system, split)]
                for system in systems
            }
            # A subset only counts if every system in it has runs; otherwise a prefix
            # that trained M4 alone would list the same seven runs under four names.
            if all(per_system.values()):
                candidates[f"{tag}:{name}"] = list(itertools.chain.from_iterable(per_system.values()))
    return candidates


def score(runs: list[dict]) -> dict:
    y_true, y_pred = pooled_predictions(runs)
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist(), MELD_LABELS)
    return {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")}


def write_tables(table: dict, best: str, out_dir: Path) -> None:
    """Markdown and LaTeX for the candidate table, dev-ordered, winner marked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(table, key=lambda n: -table[n]["dev"]["macro_f1"])
    pct = lambda row, split, key: 100 * row[split][key]  # noqa: E731

    md = ["| Ensemble | Runs | Dev macro-F1 | Test macro-F1 | Test acc. | Test weighted-F1 |",
          "|---|---|---|---|---|---|"]
    for name in order:
        row = table[name]
        mark = " **(selected)**" if name == best else ""
        md.append(
            f"| {name}{mark} | {row['runs']} | {pct(row,'dev','macro_f1'):.2f} | "
            f"{pct(row,'test','macro_f1'):.2f} | {pct(row,'test','accuracy'):.2f} | "
            f"{pct(row,'test','weighted_f1'):.2f} |"
        )
    (out_dir / "ensemble.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Probability-averaged ensembles of Stage-II runs (\%). The ensemble is "
           r"chosen by dev macro-F1; test is reported once for the chosen row.}",
           r"\label{tab:ensemble}", r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Ensemble & Runs & Dev mF1 & Test mF1 & Test wF1 \\", r"\midrule"]
    for name in order:
        row = table[name]
        label = name.replace("_", r"\_").replace("+", r"$+$")
        if name == best:
            label = r"\textbf{" + label + "}"
        tex.append(
            f"{label} & {row['runs']} & {pct(row,'dev','macro_f1'):.2f} & "
            f"{pct(row,'test','macro_f1'):.2f} & {pct(row,'test','weighted_f1'):.2f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "ensemble.tex").write_text("\n".join(tex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefixes", nargs="+", default=["v2_", "v2sp_", "v3spk_"])
    parser.add_argument("--out", default=str(RESULTS_DIR / "ensemble_selection.json"))
    parser.add_argument("--tables", default="docs/tables", help="directory for ensemble.md / ensemble.tex")
    args = parser.parse_args()

    dev = build_candidates(args.prefixes, "dev")
    test = build_candidates(args.prefixes, "test")
    shared = sorted(set(dev) & set(test))
    if not shared:
        raise SystemExit("no candidate has both dev and test probabilities; run the pipeline first")

    best, dev_scores = select_by_dev({k: dev[k] for k in shared}, num_labels=len(MELD_LABELS))
    table = {
        name: {"runs": len(dev[name]), "dev": score(dev[name]), "test": score(test[name])}
        for name in shared
    }
    payload = {"selected": best, "selected_on": "dev macro-F1", "candidates": table}
    write_json(Path(args.out), payload)
    write_tables(table, best, Path(args.tables))

    order = sorted(shared, key=lambda n: -dev_scores[n])
    LOGGER.info("%-18s %6s %8s %8s", "ensemble", "runs", "dev mF1", "test mF1")
    for name in order:
        row = table[name]
        LOGGER.info(
            "%-18s %6d %8.2f %8.2f%s",
            name, row["runs"], 100 * row["dev"]["macro_f1"], 100 * row["test"]["macro_f1"],
            "   <- selected on dev" if name == best else "",
        )
    LOGGER.info("wrote %s and %s/ensemble.{md,tex}", args.out, args.tables)


if __name__ == "__main__":
    main()
