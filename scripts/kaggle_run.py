#!/usr/bin/env python3
"""Run the pipeline on Kaggle's free GPUs from this machine.

The repository source is shipped as a *private* Kaggle dataset (`f2c-src`) and a
private notebook kernel mounts it, trains, and writes metrics back as kernel
output. Nothing is made public.

    python scripts/kaggle_run.py push      # upload/refresh the source dataset
    python scripts/kaggle_run.py submit    # push the kernel and start the run
    python scripts/kaggle_run.py status    # poll the running kernel
    python scripts/kaggle_run.py pull      # download output into results/ and docs/tables/

`push` and `submit` together are the usual start; `submit --wait` blocks until the
kernel finishes and then pulls the output. `--kernel sweep` targets the
hyperparameter-sweep notebook in kaggle/sweep/ instead of the main pipeline.

Every pull keeps the complete kernel output (log, results, tables) under
kaggle_output/<kernel>/<timestamp>/, so each Kaggle session has its own local copy.
Run directories are then copied into results/ only where they do not already
exist; a run that exists locally with different content is left untouched and
reported (pass --force to overwrite). The baseline results therefore cannot be
clobbered by a later session that reuses a run name.

Credentials come from `.env` at the repository root. Kaggle now issues a single
opaque token (it starts with `KGA`), which the CLI reads from KAGGLE_API_TOKEN;
older accounts have a username/key pair in ~/.kaggle/kaggle.json instead. Both
work here -- set KAGGLE_USERNAME plus either KAGGLE_API_TOKEN or KAGGLE_KEY.
Internet-enabled kernels additionally need a phone-verified account.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fine2coarse.utils import get_logger  # noqa: E402

LOGGER = get_logger()

DATASET_SLUG = "f2c-src"
DATASET_TITLE = "Fine-to-Coarse Emotion Recognition source"
# name -> (slug, title, directory, notebook). Kaggle derives the slug from the title.
KERNELS = {
    "pipeline": ("f2c-pipeline", "f2c pipeline", "kaggle", "run_pipeline.ipynb"),
    "sweep": ("f2c-sweep", "f2c sweep", "kaggle/sweep", "sweep.ipynb"),
}
SOURCE_DIRS = ("src", "scripts", "configs")
TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"
ENV_PATH = REPO_ROOT / ".env"
STAGING = REPO_ROOT / ".kaggle_staging"
TERMINAL_STATES = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def load_env(path: Path = ENV_PATH) -> None:
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    executable = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    command = [executable, *args]
    LOGGER.info("$ %s", " ".join(command))
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def username() -> str:
    """Kaggle username, from .env or ~/.kaggle/kaggle.json, after wiring up the token."""
    load_env()
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()

    # Newer Kaggle tokens are a single opaque string starting with "KGA"; the CLI
    # expects those in KAGGLE_API_TOKEN, not in the legacy username/key pair.
    if not token and key.startswith("KG"):
        token = key
        os.environ["KAGGLE_API_TOKEN"] = token
        os.environ.pop("KAGGLE_KEY", None)
    if user and (token or key):
        return user

    if TOKEN_PATH.exists():
        if TOKEN_PATH.stat().st_mode & 0o077:
            TOKEN_PATH.chmod(0o600)
        stored = json.loads(TOKEN_PATH.read_text())
        os.environ.setdefault("KAGGLE_USERNAME", stored["username"])
        os.environ.setdefault("KAGGLE_KEY", stored["key"])
        return stored["username"]

    raise SystemExit(
        "No Kaggle credentials found.\n"
        f"Fill in {ENV_PATH} (copy .env.example):\n"
        "  KAGGLE_USERNAME=...\n  KAGGLE_API_TOKEN=KGA...   # or KAGGLE_KEY for a legacy token\n"
        "Get the token at https://www.kaggle.com/settings > API > Create New Token."
    )


SOURCE_ID_FILE = "SOURCE_ID"  # written into src/, checked by the notebooks
IGNORED = ("__pycache__", ".pyc", ".egg-info")


def source_fingerprint(root: Path = REPO_ROOT) -> str:
    """Short hash of everything shipped to Kaggle (src/, scripts/, configs/).

    Kaggle can mount the previous dataset version for a while after a new one
    reports "ready"; the notebooks compare this against the mounted source and
    stop before training on stale code.
    """
    digest = hashlib.sha256()
    for name in SOURCE_DIRS:
        for path in sorted((root / name).rglob("*")):
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or any(part in relative for part in IGNORED) or path.name == SOURCE_ID_FILE:
                continue
            digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def stamp_notebook(notebook: Path, fingerprint: str) -> None:
    """Set EXPECTED_SOURCE = "<fingerprint>" in the notebook before it is pushed."""
    payload = json.loads(notebook.read_text())
    stamped = False
    for cell in payload["cells"]:
        lines = cell.get("source", [])
        for i, line in enumerate(lines):
            if line.startswith("EXPECTED_SOURCE ="):
                lines[i] = f'EXPECTED_SOURCE = "{fingerprint}"  # set by kaggle_run.py submit\n'
                stamped = True
    if not stamped:
        raise SystemExit(f"{notebook} has no EXPECTED_SOURCE line to stamp")
    notebook.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def stage_source() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for name in SOURCE_DIRS:
        shutil.copytree(
            REPO_ROOT / name,
            STAGING / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", SOURCE_ID_FILE),
        )
    fingerprint = source_fingerprint()
    (STAGING / "src" / SOURCE_ID_FILE).write_text(fingerprint + "\n")
    LOGGER.info("source fingerprint %s", fingerprint)
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": DATASET_TITLE, "id": f"{username()}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
        + "\n"
    )
    size = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file())
    LOGGER.info("staged %s (%.1f KB)", ", ".join(SOURCE_DIRS), size / 1e3)
    return STAGING


def write_kernel_metadata(user: str, kernel: str, enable_gpu: bool = True) -> Path:
    slug, title, directory, notebook = KERNELS[kernel]
    kernel_dir = REPO_ROOT / directory
    kernel_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{user}/{slug}",
        "title": title,
        "code_file": notebook,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    path = kernel_dir / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def cmd_push(args: argparse.Namespace) -> None:
    user = username()
    staged = stage_source()
    existing = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False, capture=True)
    known = existing.returncode == 0 and "ready" in (existing.stdout or "").lower()
    if known and not args.create:
        kaggle("datasets", "version", "-p", str(staged), "-m", args.message, "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(staged), "--dir-mode", "zip")
    LOGGER.info("source dataset ready: %s/%s (private)", user, DATASET_SLUG)


def cmd_submit(args: argparse.Namespace) -> None:
    user = username()
    metadata = write_kernel_metadata(user, args.kernel, enable_gpu=not args.cpu)
    fingerprint = source_fingerprint()
    notebook = metadata.parent / KERNELS[args.kernel][3]
    original = notebook.read_text()
    stamp_notebook(notebook, fingerprint)
    LOGGER.info("notebook expects source %s (run `push` first if the code changed)", fingerprint)
    push = ["kernels", "push", "-p", str(metadata.parent)]
    if not args.cpu:
        # enable_gpu in the metadata is not enough on the current Kaggle images:
        # the accelerator has to be named explicitly or the session comes up CPU-only.
        push += ["--accelerator", args.accelerator]
    try:
        kaggle(*push)
    finally:
        notebook.write_text(original)  # the stamp only needs to exist in the pushed copy
    LOGGER.info("kernel queued: https://www.kaggle.com/code/%s/%s", user, KERNELS[args.kernel][0])
    if args.wait:
        wait_for_completion(user, args.kernel, args.poll)
        cmd_pull(args)


def kernel_status(user: str, kernel: str) -> str:
    result = kaggle("kernels", "status", f"{user}/{KERNELS[kernel][0]}", check=False, capture=True)
    text = (result.stdout or "") + (result.stderr or "")
    for state in ("complete", "running", "queued", "error", "cancelAcknowledged", "cancelRequested"):
        if state.lower() in text.lower():
            return state
    return text.strip() or "unknown"


def wait_for_completion(user: str, kernel: str, poll_seconds: int) -> str:
    started = time.time()
    while True:
        state = kernel_status(user, kernel)
        LOGGER.info("status=%s (%.0f min elapsed)", state, (time.time() - started) / 60)
        if state in TERMINAL_STATES:
            return state
        time.sleep(poll_seconds)


def cmd_status(args: argparse.Namespace) -> None:
    user = username()
    if args.wait:
        wait_for_completion(user, args.kernel, args.poll)
    else:
        print(kernel_status(user, args.kernel))


def _same_file(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def merge_run_dir(source: Path, target: Path, force: bool = False) -> str:
    """Copy one run directory into results/ without clobbering a different local run.

    Returns "new", "identical", "overwritten" or "conflict".
    """
    files = [item for item in source.iterdir() if item.is_file()]
    if target.exists():
        differing = [f.name for f in files if not (target / f.name).exists() or not _same_file(f, target / f.name)]
        if not differing:
            return "identical"
        if not force:
            return "conflict"
        status = "overwritten"
    else:
        status = "new"
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        shutil.copy2(item, target / item.name)
    return status


def cmd_pull(args: argparse.Namespace) -> None:
    user = username()
    slug = KERNELS[args.kernel][0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = Path(args.out) / args.kernel / stamp
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", str(destination))
    files = [f for f in destination.rglob("*") if f.is_file()]
    LOGGER.info("full kernel output saved locally: %s (%d files)", destination, len(files))

    outcome: dict[str, list[str]] = {}
    source_results = destination / "results"
    for metrics in sorted(source_results.glob("*/metrics.json")) if source_results.exists() else []:
        status = merge_run_dir(metrics.parent, REPO_ROOT / "results" / metrics.parent.name, args.force)
        outcome.setdefault(status, []).append(metrics.parent.name)
    tables = destination / "docs" / "tables"
    if tables.exists():
        status = merge_run_dir(tables, REPO_ROOT / "docs" / "tables", args.force)
        outcome.setdefault(status, []).append("docs/tables")
    for status in ("new", "overwritten", "identical", "conflict"):
        if outcome.get(status):
            LOGGER.info("%s: %s", status, ", ".join(outcome[status]))
    if outcome.get("conflict"):
        LOGGER.warning(
            "left %d local run(s) untouched because Kaggle's copy differs; the Kaggle version "
            "is in %s. Re-run with --force to overwrite.", len(outcome["conflict"]), destination,
        )

    log = destination / f"{slug}.log"
    if log.exists():
        LOGGER.info("last lines of the kernel log:")
        print("\n".join(log.read_text(errors="replace").splitlines()[-20:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="upload or refresh the private source dataset")
    push.add_argument("-m", "--message", default="update source")
    push.add_argument("--create", action="store_true", help="force first-time creation")
    push.set_defaults(func=cmd_push)

    submit = sub.add_parser("submit", help="push the kernel and start the run")
    submit.add_argument("--wait", action="store_true", help="block until the kernel finishes, then pull")
    submit.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    submit.add_argument("--cpu", action="store_true", help="run without a GPU (debugging only)")
    submit.add_argument("--accelerator", default="nvidiaTeslaT4",
                        help="Kaggle accelerator name, e.g. nvidiaTeslaT4, nvidiaTeslaP100, nvidiaL4x4")
    submit.add_argument("--out", default="kaggle_output")
    submit.add_argument("--force", action="store_true", help="with --wait: let the pull overwrite differing runs")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="print or follow the kernel status")
    status.add_argument("--wait", action="store_true")
    status.add_argument("--poll", type=int, default=300)
    status.set_defaults(func=cmd_status)

    pull = sub.add_parser("pull", help="download kernel output into the repo")
    pull.add_argument("--out", default="kaggle_output")
    pull.add_argument("--force", action="store_true", help="overwrite local runs that differ from Kaggle's")
    pull.set_defaults(func=cmd_pull)

    for command in (submit, status, pull):
        command.add_argument("--kernel", choices=sorted(KERNELS), default="pipeline")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
