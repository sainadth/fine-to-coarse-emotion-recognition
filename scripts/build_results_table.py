#!/usr/bin/env python3
"""Aggregate run metrics into Table 1 (markdown + LaTeX) and a per-class table.

Runs are grouped by system, so repeating the pipeline under several seeds
(`--set seed=... run_name=..._s1`) turns every cell into mean +/- std over
those seeds. A single run per system prints plain numbers.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import MELD_LABELS, RESULTS_DIR  # noqa: E402
from fine2coarse.utils import get_logger, read_json  # noqa: E402

LOGGER = get_logger()
SYSTEM_ORDER = ["M1", "M2", "M3", "M4", "M5"]
SYSTEM_LABEL = {
    "M1": "$M_1$", "M2": "$M_2$", "M3": "$M_3$", "M4": "$M_4$",
    "M5": "ext.",
}
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def run_name_pattern(prefix: str = "") -> re.Pattern:
    """Run directories that belong to one table: `<prefix>M<k>_s<seed>`.

    Sweep runs (`sweep_*`) and other recipes (`v2_M4_s42`) share results/ with the
    baseline, and they are recorded as the same system; without this filter they
    would be averaged into the baseline cells.
    """
    return re.compile(rf"^{re.escape(prefix)}M\d+_s\d+$")


def collect(results_dir: Path, prefix: str = "", seeds: list[int] | None = None) -> dict[str, list[dict]]:
    """Stage-II runs named `<prefix>M<k>_s<seed>`, grouped by system and ordered by seed.

    `seeds` keeps only those seeds, so systems trained on extra seeds stay comparable.
    """
    pattern = run_name_pattern(prefix)
    runs: dict[str, list[dict]] = {}
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        if not pattern.match(metrics_path.parent.name):
            continue
        payload = read_json(metrics_path)
        if payload.get("stage") != "II":
            continue
        payload["seed"] = payload["config"].get("seed")
        if seeds and payload["seed"] not in seeds:
            continue
        runs.setdefault(payload["system"], []).append(payload)
    for group in runs.values():
        group.sort(key=lambda payload: (payload["seed"] is None, payload["seed"]))
    return runs


def summarise(values: list[float]) -> tuple[float, float | None]:
    if len(values) == 1:
        return values[0], None
    return statistics.fmean(values), statistics.stdev(values)


def cell(values: list[float], scale: float = 1.0, digits: int = 4) -> str:
    mean, deviation = summarise([v * scale for v in values])
    if deviation is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {deviation:.{digits}f}"


def present(runs: dict[str, list[dict]]) -> list[str]:
    return [system for system in SYSTEM_ORDER if system in runs] + sorted(set(runs) - set(SYSTEM_ORDER))


def markdown_table(runs: dict[str, list[dict]]) -> str:
    lines = [
        "| Model | GoEmotions | UTE | Seeds | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in present(runs):
        group = runs[system]
        head = group[0]
        cells = [cell([run["test"][metric] for run in group]) for metric in METRICS]
        lines.append(
            f"| {system} | {'Yes' if head['goemotions_transfer'] else 'No'} "
            f"| {'Yes' if head['ute'] else 'No'} | {len(group)} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def latex_table(runs: dict[str, list[dict]]) -> str:
    rows = []
    best = {
        metric: max(present(runs), key=lambda name: statistics.fmean(run["test"][metric] for run in runs[name]))
        for metric in METRICS
    }
    for system in present(runs):
        group = runs[system]
        head = group[0]
        cells = [cell([run["test"][metric] for run in group], scale=100.0, digits=2) for metric in METRICS]
        cells = [c.replace("±", r"$\pm$") for c in cells]
        # Bold the best mean in each metric column so the table can be read at a glance.
        cells = [rf"\textbf{{{c}}}" if system == best[metric] else c for c, metric in zip(cells, METRICS)]
        rows.append(
            f"{SYSTEM_LABEL.get(system, system)} & {'Yes' if head['goemotions_transfer'] else 'No'} & "
            f"{'Yes' if head['ute'] else 'No'} & {len(group)} & " + " & ".join(cells) + r" \\"
        )
    seeds = max(len(group) for group in runs.values())
    caption = (
        "MELD test results for the required systems (\\%)."
        + (f" Mean $\\pm$ standard deviation over {seeds} seeds." if seeds > 1 else "")
    )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{lccrccc}\n\\toprule\n"
        "Model & GoEmotions & UTE & Runs & Accuracy & Macro-F1 & Weighted-F1 \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def per_class_markdown(runs: dict[str, list[dict]]) -> str:
    systems = present(runs)
    header = "| Emotion | " + " | ".join(f"{system} F1" for system in systems) + " | Support |"
    separator = "|---" * (len(systems) + 2) + "|"
    lines = [header, separator]
    for label in MELD_LABELS:
        cells = []
        support = "-"
        for system in systems:
            group = runs[system]
            cells.append(cell([run["test"]["per_class"][label]["f1"] for run in group], digits=3))
            support = str(group[0]["test"]["per_class"][label]["support"])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {support} |")
    return "\n".join(lines)


def per_class_latex(runs: dict[str, list[dict]], systems: list[str] | None = None) -> str:
    """Per-class precision, recall and F1 for the chosen systems.

    The spec asks for all three per class, not F1 alone. Every system at once would
    need fifteen numeric columns, so this reports the baseline and the full model,
    which is the comparison the analysis actually makes.
    """
    systems = systems or [name for name in ("M1", "M4") if name in runs]
    spans = " & ".join(rf"\multicolumn{{3}}{{c}}{{{SYSTEM_LABEL[name]}}}" for name in systems)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(systems)))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-class precision, recall and F1 on the MELD test split (\%), "
        r"mean over three seeds.}",
        r"\label{tab:perclass}",
        r"\begin{tabular}{l" + "rrr" * len(systems) + "r}",
        r"\toprule",
        rf"Emotion & {spans} & Support \\",
        cmid,
        " & " + " & ".join(["P", "R", "F1"] * len(systems)) + r" & \\",
        r"\midrule",
    ]
    for label in MELD_LABELS:
        cells, support = [], "-"
        for name in systems:
            group = runs[name]
            for metric in ("precision", "recall", "f1"):
                values = [100 * run["test"]["per_class"][label][metric] for run in group]
                cells.append(f"{statistics.fmean(values):.1f}")
            support = str(group[0]["test"]["per_class"][label]["support"])
        lines.append(f"{label} & " + " & ".join(cells) + f" & {support} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def per_class_f1_latex(runs: dict[str, list[dict]]) -> str:
    """Per-class F1 for every system, so the per-class claims in the text can be checked."""
    systems = present(runs)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-class F1 on the MELD test split (\%), mean over three seeds. "
        r"Bold marks the best system for each emotion.}",
        r"\label{tab:perclassf1}", r"\small",
        r"\begin{tabular}{l" + "r" * len(systems) + "r}", r"\toprule",
        "Emotion & " + " & ".join(SYSTEM_LABEL[name] for name in systems) + r" & Support \\",
        r"\midrule",
    ]
    for label in MELD_LABELS:
        means = {name: statistics.fmean(100 * run["test"]["per_class"][label]["f1"] for run in runs[name])
                 for name in systems}
        best = max(means, key=means.get)
        cells = [rf"\textbf{{{means[name]:.1f}}}" if name == best else f"{means[name]:.1f}" for name in systems]
        support = runs[systems[0]][0]["test"]["per_class"][label]["support"]
        lines.append(f"{label} & " + " & ".join(cells) + f" & {support} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def deltas_markdown(runs: dict[str, list[dict]]) -> str:
    """The three comparisons the project asks about, in macro-F1 points."""
    comparisons = [
        ("M1", "M2", "fine-grained transfer"),
        ("M1", "M3", "sequential context"),
        ("M3", "M4", "transfer on top of context"),
        ("M4", "M5", "explicit fine-grained distribution"),
    ]
    lines = ["| Comparison | Effect | Macro-F1 delta |", "|---|---|---|"]
    for left, right, description in comparisons:
        if left not in runs or right not in runs:
            continue
        left_mean = statistics.fmean(run["test"]["macro_f1"] for run in runs[left])
        right_mean = statistics.fmean(run["test"]["macro_f1"] for run in runs[right])
        lines.append(f"| {left} → {right} | {description} | {(right_mean - left_mean) * 100:+.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default="docs/tables")
    parser.add_argument("--prefix", default="",
                        help="only runs named <prefix>M<k>_s<seed>, e.g. v2_ for a new recipe")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="only these seeds (M4/M5 have more seeds than M1-M3)")
    args = parser.parse_args()

    runs = collect(Path(args.results_dir), args.prefix, args.seeds)
    if not runs:
        raise SystemExit(f"no Stage-II metrics.json found under {args.results_dir}")
    LOGGER.info("systems: %s", ", ".join(f"{s}×{len(g)}" for s, g in sorted(runs.items())))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_md = markdown_table(runs)
    per_class_md = per_class_markdown(runs)
    deltas_md = deltas_markdown(runs)
    (out_dir / "table1.md").write_text(
        f"{main_md}\n\n{per_class_md}\n\n{deltas_md}\n", encoding="utf-8"
    )
    (out_dir / "table1.tex").write_text(latex_table(runs), encoding="utf-8")
    (out_dir / "per_class.tex").write_text(per_class_latex(runs), encoding="utf-8")
    (out_dir / "per_class_f1.tex").write_text(per_class_f1_latex(runs), encoding="utf-8")
    print(main_md, per_class_md, deltas_md, sep="\n\n")
    LOGGER.info("wrote %s, %s and %s", out_dir / "table1.md", out_dir / "table1.tex",
                out_dir / "per_class.tex")


if __name__ == "__main__":
    main()
