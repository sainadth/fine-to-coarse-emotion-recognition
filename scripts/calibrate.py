#!/usr/bin/env python3
"""Dev-fitted decision offsets and seed ensembles for Stage-II runs.

For every run `<prefix>M<k>_s<seed>` that saved `dev_predictions.json` (runs
from before probabilities were saved are skipped), this fits one offset per
class on the dev split to maximise dev macro-F1 and applies it, unchanged, to
test. It also averages the probabilities of each system's seeds (an ensemble)
and fits offsets on the ensemble's dev output the same way.

Nothing is chosen on test: offsets come from dev only, and the ensemble uses
every available seed rather than a selected subset.

    python scripts/calibrate.py --prefix v2_
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.calibration import apply_offsets, average_probabilities, fit_offsets  # noqa: E402
from fine2coarse.config import MELD_LABELS, RESULTS_DIR  # noqa: E402
from fine2coarse.metrics import compute_metrics  # noqa: E402
from fine2coarse.utils import get_logger, read_json, write_json  # noqa: E402

LOGGER = get_logger()


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist(), MELD_LABELS)
    return {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")}


def evaluate(dev: dict, test: dict) -> dict:
    """Plain argmax vs dev-fitted offsets, both reported on dev and test."""
    dev_true, dev_probs = np.asarray(dev["y_true"]), np.asarray(dev["probs"])
    test_true, test_probs = np.asarray(test["y_true"]), np.asarray(test["probs"])
    offsets = fit_offsets(dev_true, dev_probs)
    return {
        "offsets": dict(zip(MELD_LABELS, np.round(offsets, 3).tolist())),
        "argmax": {"dev": score(dev_true, dev_probs.argmax(1)), "test": score(test_true, test_probs.argmax(1))},
        "offset": {"dev": score(dev_true, apply_offsets(dev_probs, offsets)),
                   "test": score(test_true, apply_offsets(test_probs, offsets))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--prefix", default="", help="only runs named <prefix>M<k>_s<seed>")
    parser.add_argument("--out", default=None, help="default: <results-dir>/<prefix>calibration.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pattern = re.compile(rf"^{re.escape(args.prefix)}(M\d+)_s(\d+)$")
    runs: dict[str, list[tuple[str, dict, dict]]] = {}
    skipped = []
    for run_dir in sorted(results_dir.iterdir()):
        match = pattern.match(run_dir.name)
        if not match:
            continue
        if not (run_dir / "dev_predictions.json").exists():
            skipped.append(run_dir.name)
            continue
        dev = read_json(run_dir / "dev_predictions.json")
        test = read_json(run_dir / "test_predictions.json")
        runs.setdefault(match.group(1), []).append((run_dir.name, dev, test))
    if skipped:
        LOGGER.info("no dev probabilities (older runs), skipped: %s", ", ".join(skipped))
    if not runs:
        raise SystemExit("no runs with dev_predictions.json; rerun training with the current code")

    report: dict = {"per_run": {}, "ensemble": {}}
    print(f"{'run':<16}{'test mF1 argmax':>17}{'+offsets':>10}{'dev mF1 argmax':>16}{'+offsets':>10}")
    for system, members in sorted(runs.items()):
        for name, dev, test in members:
            result = evaluate(dev, test)
            report["per_run"][name] = result
            print(f"{name:<16}{result['argmax']['test']['macro_f1']:>17.4f}{result['offset']['test']['macro_f1']:>10.4f}"
                  f"{result['argmax']['dev']['macro_f1']:>16.4f}{result['offset']['dev']['macro_f1']:>10.4f}")
        if len(members) > 1:
            dev_true, dev_mean = average_probabilities([m[1] for m in members])
            test_true, test_mean = average_probabilities([m[2] for m in members])
            result = evaluate({"y_true": dev_true, "probs": dev_mean}, {"y_true": test_true, "probs": test_mean})
            result["members"] = [m[0] for m in members]
            report["ensemble"][system] = result
            print(f"{system + ' ens.':<16}{result['argmax']['test']['macro_f1']:>17.4f}{result['offset']['test']['macro_f1']:>10.4f}"
                  f"{result['argmax']['dev']['macro_f1']:>16.4f}{result['offset']['dev']['macro_f1']:>10.4f}")

    out = Path(args.out) if args.out else results_dir / f"{args.prefix}calibration.json"
    write_json(out, report)
    LOGGER.info("wrote %s", out)


if __name__ == "__main__":
    main()
