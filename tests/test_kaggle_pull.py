import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_run.py"
spec = importlib.util.spec_from_file_location("kaggle_run", SCRIPT)
kaggle_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kaggle_run)


def _run(root: Path, name: str, metrics: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(metrics)
    (run / "test_predictions.json").write_text("{}")
    return run


def test_new_run_is_copied(tmp_path):
    source = _run(tmp_path / "kaggle", "v2_M4_s42", '{"a": 1}')
    target = tmp_path / "results" / "v2_M4_s42"
    assert kaggle_run.merge_run_dir(source, target) == "new"
    assert (target / "metrics.json").read_text() == '{"a": 1}'


def test_identical_run_is_left_alone(tmp_path):
    source = _run(tmp_path / "kaggle", "M4_s42", '{"a": 1}')
    target = _run(tmp_path / "results", "M4_s42", '{"a": 1}')
    assert kaggle_run.merge_run_dir(source, target) == "identical"


def test_differing_local_run_is_not_clobbered(tmp_path):
    source = _run(tmp_path / "kaggle", "M4_s42", '{"new": true}')
    target = _run(tmp_path / "results", "M4_s42", '{"baseline": true}')
    assert kaggle_run.merge_run_dir(source, target) == "conflict"
    assert (target / "metrics.json").read_text() == '{"baseline": true}'
    assert kaggle_run.merge_run_dir(source, target, force=True) == "overwritten"
    assert (target / "metrics.json").read_text() == '{"new": true}'


def test_every_kernel_has_a_notebook_on_disk():
    root = SCRIPT.parents[1]
    for slug, title, directory, notebook in kaggle_run.KERNELS.values():
        assert (root / directory / notebook).exists(), (slug, directory, notebook)
