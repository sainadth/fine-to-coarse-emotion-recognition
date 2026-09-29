#!/usr/bin/env python3
"""Aggregate run metrics into Table 1 (markdown + LaTeX) and a per-class table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import MELD_LABELS, RESULTS_DIR  # noqa: E402
from fine2coarse.utils import get_logger, read_json  # noqa: E402

LOGGER = get_logger()
SYSTEM_ORDER = ["M1", "M2", "M3", "M4"]


def collect(results_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        payload = read_json(metrics_path)
        if payload.get("stage") != "II":
            continue
        runs[payload["system"]] = payload
    return runs


def markdown_table(runs: dict[str, dict]) -> str:
    lines = [
        "| Model | GoEmotions | UTE | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for system in SYSTEM_ORDER:
        payload = runs.get(system)
        if not payload:
            lines.append(f"| {system} | - | - | - | - | - |")
            continue
        test = payload["test"]
        lines.append(
            f"| {system} | {'Yes' if payload['goemotions_transfer'] else 'No'} "
            f"| {'Yes' if payload['ute'] else 'No'} "
            f"| {test['accuracy']:.4f} | {test['macro_f1']:.4f} | {test['weighted_f1']:.4f} |"
        )
    return "\n".join(lines)


def latex_table(runs: dict[str, dict]) -> str:
    rows = []
    for system in SYSTEM_ORDER:
        payload = runs.get(system)
        if not payload:
            rows.append(rf"$M_{system[1]}$ & -- & -- & -- & -- & -- \\")
            continue
        test = payload["test"]
        rows.append(
            rf"$M_{system[1]}$ & {'Yes' if payload['goemotions_transfer'] else 'No'} & "
            rf"{'Yes' if payload['ute'] else 'No'} & {test['accuracy'] * 100:.2f} & "
            rf"{test['macro_f1'] * 100:.2f} & {test['weighted_f1'] * 100:.2f} \\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{MELD test results for the four required systems (\\%).}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Model & GoEmotions & UTE & Accuracy & Macro-F1 & Weighted-F1 \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def per_class_markdown(runs: dict[str, dict]) -> str:
    header = "| Emotion | " + " | ".join(f"{s} F1" for s in SYSTEM_ORDER) + " | Support |"
    separator = "|---" * (len(SYSTEM_ORDER) + 2) + "|"
    lines = [header, separator]
    for label in MELD_LABELS:
        cells = []
        support = "-"
        for system in SYSTEM_ORDER:
            payload = runs.get(system)
            if not payload:
                cells.append("-")
                continue
            entry = payload["test"]["per_class"][label]
            cells.append(f"{entry['f1']:.3f}")
            support = str(entry["support"])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {support} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default="docs/tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(results_dir)
    if not runs:
        raise SystemExit(f"no Stage-II metrics.json found under {results_dir}")
    LOGGER.info("found systems: %s", ", ".join(sorted(runs)))

    main_md = markdown_table(runs)
    per_class_md = per_class_markdown(runs)
    (out_dir / "table1.md").write_text(main_md + "\n\n" + per_class_md + "\n", encoding="utf-8")
    (out_dir / "table1.tex").write_text(latex_table(runs), encoding="utf-8")
    print(main_md)
    print()
    print(per_class_md)
    LOGGER.info("wrote %s and %s", out_dir / "table1.md", out_dir / "table1.tex")


if __name__ == "__main__":
    main()
