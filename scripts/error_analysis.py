#!/usr/bin/env python3
"""Error analysis over saved MELD test predictions.

Produces, for the systems found under results/:

* per-class F1 deltas for the three comparisons the spec asks about
  (M1 vs M2 = fine-grained transfer, M1 vs M3 = sequential context,
  M3 vs M4 = transfer on top of context);
* the most frequent confusion pairs per system;
* contrastive examples with dialogue context -- utterances that the
  context-aware system gets right and the context-free system gets wrong,
  and vice versa.

Predictions are stored in dataset order (test loaders are never shuffled), so
they are re-aligned with the source utterances here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import MELD_LABELS, RESULTS_DIR  # noqa: E402
from fine2coarse.data.meld import load_meld_dialogues, load_meld_frame  # noqa: E402
from fine2coarse.utils import get_logger, read_json, write_json  # noqa: E402

LOGGER = get_logger()
SYSTEM_ORDER = ["M1", "M2", "M3", "M4"]


def load_runs(results_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        payload = read_json(metrics_path)
        if payload.get("stage") != "II":
            continue
        predictions_path = metrics_path.parent / "test_predictions.json"
        if not predictions_path.exists():
            LOGGER.warning("no predictions for %s", payload["system"])
            continue
        payload["predictions"] = read_json(predictions_path)
        payload["seed"] = payload["config"].get("seed")
        system = payload["system"]
        # With several seeds per system, analyse the lowest-seed run so the
        # contrastive examples come from one consistent set of models.
        if system not in runs or (payload["seed"] or 0) < (runs[system]["seed"] or 0):
            runs[system] = payload
    return runs


def test_records(view: str, max_turns: int) -> list[dict]:
    """Test-set utterances in the exact order the given view's loader yields them."""
    if view == "dialogue":
        records = []
        for dialogue in load_meld_dialogues("test", max_turns=max_turns):
            for position, (text, label) in enumerate(zip(dialogue.utterances, dialogue.labels)):
                records.append(
                    {
                        "dialogue_id": dialogue.dialogue_id,
                        "turn": position,
                        "speaker": dialogue.speakers[position] if dialogue.speakers else "",
                        "text": text,
                        "label": label,
                        "context": dialogue.utterances[max(0, position - 3) : position],
                    }
                )
        return records
    frame = load_meld_frame("test")
    return [
        {
            "dialogue_id": int(row.Dialogue_ID),
            "turn": int(row.Utterance_ID),
            "speaker": str(row.Speaker),
            "text": row.Utterance,
            "label": int(row.label),
            "context": [],
        }
        for row in frame.itertuples()
    ]


class AlignmentError(RuntimeError):
    """Predictions do not line up with the MELD test set for this run's view."""


def align(run: dict) -> list[dict]:
    view = run["config"].get("view", "dialogue")
    records = test_records(view, int(run["config"].get("max_turns", 24)))
    predictions = run["predictions"]
    if len(records) != len(predictions["y_pred"]):
        raise AlignmentError(
            f"{run['system']}: {len(predictions['y_pred'])} predictions but {len(records)} "
            f"test utterances for view={view}; re-run evaluation without --set limit=..."
        )
    for record, true_label, predicted in zip(records, predictions["y_true"], predictions["y_pred"]):
        if record["label"] != true_label:
            raise AlignmentError(
                f"{run['system']}: label mismatch at dialogue {record['dialogue_id']}"
            )
        record["pred"] = predicted
    return records


def per_class_delta(runs: dict[str, dict], left: str, right: str) -> dict[str, float]:
    if left not in runs or right not in runs:
        return {}
    return {
        label: runs[right]["test"]["per_class"][label]["f1"] - runs[left]["test"]["per_class"][label]["f1"]
        for label in MELD_LABELS
    }


def top_confusions(run: dict, top_k: int = 5) -> list[dict]:
    matrix = run["test"]["confusion_matrix"]
    pairs = [
        {"true": MELD_LABELS[i], "pred": MELD_LABELS[j], "count": matrix[i][j]}
        for i in range(len(MELD_LABELS))
        for j in range(len(MELD_LABELS))
        if i != j and matrix[i][j] > 0
    ]
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:top_k]


def contrastive(aligned: dict[str, list[dict]], winner: str, loser: str, limit: int = 15) -> list[dict]:
    if winner not in aligned or loser not in aligned:
        return []
    index = {(r["dialogue_id"], r["turn"], r["text"]): r for r in aligned[loser]}
    examples = []
    for record in aligned[winner]:
        other = index.get((record["dialogue_id"], record["turn"], record["text"]))
        if other is None:
            continue
        if record["pred"] == record["label"] and other["pred"] != other["label"]:
            examples.append(
                {
                    "dialogue_id": record["dialogue_id"],
                    "turn": record["turn"],
                    "speaker": record["speaker"],
                    "context": record["context"],
                    "utterance": record["text"],
                    "gold": MELD_LABELS[record["label"]],
                    f"{winner}_pred": MELD_LABELS[record["pred"]],
                    f"{loser}_pred": MELD_LABELS[other["pred"]],
                }
            )
        if len(examples) >= limit:
            break
    return examples



def _tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def qualitative_latex(examples: dict, per_group: int = 3) -> str:
    """Contrastive cases as a table: last context turn, utterance, gold, both predictions.

    The spec asks for representative predictions in the error analysis; prose alone makes
    the reader reconstruct the contrast. Each group gets its own heading row so the table
    reads as three separate claims, and an example already shown is not repeated.
    """
    groups = [
        ("context_helps_M4_over_M2", r"Context resolves the utterance: $M_4$ correct, $M_2$ wrong", "M4_pred", "M2_pred"),
        ("transfer_helps_M4_over_M3", r"Transfer supplies a rare label: $M_4$ correct, $M_3$ wrong", "M4_pred", "M3_pred"),
        ("context_hurts_M2_over_M4", r"Context misleads: $M_2$ correct, $M_4$ wrong", "M2_pred", "M4_pred"),
    ]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Representative contrastive predictions on the MELD test split. "
        r"\emph{Context} is the preceding turn; \emph{Gold} is the MELD annotation. "
        r"The two prediction columns are the system that gets the turn right and the one that does not.}",
        r"\label{tab:qualitative}", r"\small",
        r"\begin{tabular}{p{0.30\textwidth}p{0.34\textwidth}lll}", r"\toprule",
        r"Context (previous turn) & Utterance & Gold & Correct & Wrong \\",
        r"\midrule",
    ]
    seen: set[tuple] = set()
    for key, title, right, wrong in groups:
        rows = []
        for row in examples.get(key, []):
            ident = (row["dialogue_id"], row["turn"])
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(row)
            if len(rows) == per_group:
                break
        if not rows:
            continue
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\[2pt]")
        for row in rows:
            context = (_tex_escape(_clip(row["context"][-1], 58)) if row.get("context")
                       else r"\textit{dialogue start}")
            utterance = _tex_escape(_clip(f'{row["speaker"]}: {row["utterance"]}', 66))
            lines.append(
                f"{context} & {utterance} & {row['gold']} & {row[right]} & {row[wrong]} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", default="docs/tables/error_analysis.json")
    parser.add_argument("--examples", type=int, default=15)
    args = parser.parse_args()

    runs = load_runs(Path(args.results_dir))
    if not runs:
        raise SystemExit("no Stage-II runs with predictions found")
    LOGGER.info("systems: %s", ", ".join(sorted(runs)))

    aligned: dict[str, list[dict]] = {}
    for system, run in sorted(runs.items()):
        try:
            aligned[system] = align(run)
        except AlignmentError as error:
            LOGGER.warning("skipping contrastive examples for %s: %s", system, error)

    report = {
        "systems": {
            system: {
                "test": {k: run["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                "per_class_f1": {
                    label: run["test"]["per_class"][label]["f1"] for label in MELD_LABELS
                },
                "top_confusions": top_confusions(run),
            }
            for system, run in runs.items()
        },
        "deltas": {
            "M1_to_M2_finegrained_transfer": per_class_delta(runs, "M1", "M2"),
            "M1_to_M3_sequential_context": per_class_delta(runs, "M1", "M3"),
            "M3_to_M4_transfer_with_context": per_class_delta(runs, "M3", "M4"),
        },
        "examples": {
            "context_helps_M4_over_M2": contrastive(aligned, "M4", "M2", args.examples),
            "context_hurts_M2_over_M4": contrastive(aligned, "M2", "M4", args.examples),
            "transfer_helps_M4_over_M3": contrastive(aligned, "M4", "M3", args.examples),
        },
    }
    write_json(args.out, report)
    tex_path = Path(args.out).parent / "qualitative.tex"
    tex_path.write_text(qualitative_latex(report["examples"]), encoding="utf-8")
    LOGGER.info("wrote %s and %s", args.out, tex_path)

    for name, delta in report["deltas"].items():
        if not delta:
            continue
        ranked = sorted(delta.items(), key=lambda item: item[1], reverse=True)
        pretty = ", ".join(f"{label} {value:+.3f}" for label, value in ranked)
        print(f"{name}: {pretty}")


if __name__ == "__main__":
    main()
