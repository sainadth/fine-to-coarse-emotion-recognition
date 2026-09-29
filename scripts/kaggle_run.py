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
kernel finishes and then pulls the output.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (Kaggle > Settings > API >
Create New Token) and, for internet-enabled kernels, a phone-verified account.
"""

from __future__ import annotations

import argparse
import json
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
KERNEL_SLUG = "fine-to-coarse-emotion-recognition"
KERNEL_TITLE = "Fine-to-Coarse Sequential Emotion Recognition"
SOURCE_DIRS = ("src", "scripts", "configs")
TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"
STAGING = REPO_ROOT / ".kaggle_staging"
KERNEL_DIR = REPO_ROOT / "kaggle"
TERMINAL_STATES = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    executable = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    command = [executable, *args]
    LOGGER.info("$ %s", " ".join(command))
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def username() -> str:
    if not TOKEN_PATH.exists():
        raise SystemExit(
            f"{TOKEN_PATH} not found.\n"
            "Get it at https://www.kaggle.com/settings > API > Create New Token, then:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )
    if TOKEN_PATH.stat().st_mode & 0o077:
        TOKEN_PATH.chmod(0o600)
    return json.loads(TOKEN_PATH.read_text())["username"]


def stage_source() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for name in SOURCE_DIRS:
        shutil.copytree(
            REPO_ROOT / name,
            STAGING / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
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


def write_kernel_metadata(user: str, enable_gpu: bool = True) -> Path:
    KERNEL_DIR.mkdir(exist_ok=True)
    metadata = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "run_pipeline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    path = KERNEL_DIR / "kernel-metadata.json"
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
    write_kernel_metadata(user, enable_gpu=not args.cpu)
    kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    LOGGER.info("kernel queued: https://www.kaggle.com/code/%s/%s", user, KERNEL_SLUG)
    if args.wait:
        wait_for_completion(user, args.poll)
        cmd_pull(args)


def kernel_status(user: str) -> str:
    result = kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}", check=False, capture=True)
    text = (result.stdout or "") + (result.stderr or "")
    for state in ("complete", "running", "queued", "error", "cancelAcknowledged", "cancelRequested"):
        if state.lower() in text.lower():
            return state
    return text.strip() or "unknown"


def wait_for_completion(user: str, poll_seconds: int) -> str:
    started = time.time()
    while True:
        state = kernel_status(user)
        LOGGER.info("status=%s (%.0f min elapsed)", state, (time.time() - started) / 60)
        if state in TERMINAL_STATES:
            return state
        time.sleep(poll_seconds)


def cmd_status(args: argparse.Namespace) -> None:
    user = username()
    if args.wait:
        wait_for_completion(user, args.poll)
    else:
        print(kernel_status(user))


def cmd_pull(args: argparse.Namespace) -> None:
    user = username()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(destination))

    copied = 0
    for metrics in (destination / "results").glob("*/metrics.json") if (destination / "results").exists() else []:
        target = REPO_ROOT / "results" / metrics.parent.name
        target.mkdir(parents=True, exist_ok=True)
        for item in metrics.parent.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
                copied += 1
    tables = destination / "docs" / "tables"
    if tables.exists():
        target = REPO_ROOT / "docs" / "tables"
        target.mkdir(parents=True, exist_ok=True)
        for item in tables.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
                copied += 1
    LOGGER.info("kernel output in %s; copied %d files into results/ and docs/tables/", destination, copied)

    log = destination / f"{KERNEL_SLUG}.log"
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
    submit.add_argument("--out", default="kaggle_output")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="print or follow the kernel status")
    status.add_argument("--wait", action="store_true")
    status.add_argument("--poll", type=int, default=300)
    status.set_defaults(func=cmd_status)

    pull = sub.add_parser("pull", help="download kernel output into the repo")
    pull.add_argument("--out", default="kaggle_output")
    pull.set_defaults(func=cmd_pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
