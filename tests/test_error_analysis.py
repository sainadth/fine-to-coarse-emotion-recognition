import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "error_analysis.py"
spec = importlib.util.spec_from_file_location("error_analysis", SCRIPT)
error_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(error_analysis)


def _row(dialogue_id, turn, **predictions):
    return {"dialogue_id": dialogue_id, "turn": turn, "speaker": "A", "context": ["50% & more_"],
            "utterance": "x", "gold": "fear", **predictions}


def test_qualitative_table_escapes_and_does_not_repeat_an_example():
    shared = _row(1, 0, M4_pred="fear", M2_pred="joy", M3_pred="anger")
    table = error_analysis.qualitative_latex({
        "context_helps_M4_over_M2": [shared],
        "transfer_helps_M4_over_M3": [shared, _row(2, 3, M4_pred="fear", M3_pred="anger")],
    })
    assert r"50\% \& more\_" in table
    assert table.count("A: x") == 2          # shared row once, the second row once
    assert "Context misleads" not in table   # empty group leaves no heading
