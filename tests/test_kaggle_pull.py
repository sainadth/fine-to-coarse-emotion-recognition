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


def test_fingerprint_changes_with_source_and_ignores_caches(tmp_path):
    for name in kaggle_run.SOURCE_DIRS:
        (tmp_path / name).mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    first = kaggle_run.source_fingerprint(tmp_path)
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    (tmp_path / "src" / kaggle_run.SOURCE_ID_FILE).write_text("old\n")
    assert kaggle_run.source_fingerprint(tmp_path) == first
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert kaggle_run.source_fingerprint(tmp_path) != first


def test_both_notebooks_can_be_stamped(tmp_path):
    import shutil

    root = SCRIPT.parents[1]
    for _, _, directory, notebook in kaggle_run.KERNELS.values():
        copy = tmp_path / notebook
        shutil.copy(root / directory / notebook, copy)
        kaggle_run.stamp_notebook(copy, "abc123")
        import json

        cells = json.loads(copy.read_text())["cells"]
        assert any('EXPECTED_SOURCE = "abc123"' in "".join(c["source"]) for c in cells)
