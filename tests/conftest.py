"""Shared pytest helpers: repo paths, per-test output directories, run_gpu()."""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "out"
TEST_GRAPHS_DIR = REPO_ROOT / "graphs" / "inputs" / "test"
# Override with ECE496_BIN to test a binary built elsewhere.
GPU_BIN = Path(os.environ.get("ECE496_BIN", REPO_ROOT / "bin" / "ece496"))
GPU_TIMEOUT_S = 60


def run_gpu(benchmark, graph_path, source=0, out=None, weight_type="int32",
            scheduler="initial_scheduler", time=False, check=True):
    """Run the GPU program once and return the CompletedProcess.

    A hung run fails after GPU_TIMEOUT_S instead of freezing the test run.
    With check=True a non-zero exit code fails the test. With time=True the
    parsed --time JSON is attached as `.timing`.
    """
    if not GPU_BIN.is_file():
        pytest.fail(f"{GPU_BIN} not found; build it with cmake first")
    cmd = [str(GPU_BIN), benchmark, str(graph_path), str(source), weight_type,
           scheduler]
    if out is not None:
        cmd += ["--out", str(out)]
    if time:
        cmd.append("--time")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=GPU_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail(f"timed out after {GPU_TIMEOUT_S}s: {' '.join(cmd)}")
    if check and proc.returncode != 0:
        pytest.fail(f"exit code {proc.returncode}: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if time and proc.returncode == 0:
        proc.timing = json.loads(proc.stdout.strip().splitlines()[-1])
    return proc


@pytest.fixture(scope="session")
def run_tag():
    """MMDDYY_N, shared by every test in this pytest run.

    N counts up from 0 for each run on the same day, so earlier runs'
    output folders are never overwritten.
    """
    date = datetime.now().strftime("%m%d%y")
    pattern = re.compile(rf"_{date}_(\d+)$")
    runs = [int(m.group(1)) for p in OUT_DIR.rglob(f"*_{date}_*")
            if (m := pattern.search(p.name))]
    return f"{date}_{max(runs, default=-1) + 1}"


@pytest.fixture
def out_dir(request, run_tag):
    """An empty tests/out/<file>/<test name>_<MMDDYY>_<N>/ directory.

    <file> is the test file's name without "test_", e.g. "graphs" for
    test_graphs.py. Files are left in place so failures can be inspected.
    """
    group = Path(request.module.__file__).stem.removeprefix("test_")
    name = re.sub(r"[^\w.-]+", "_", request.node.name).strip("_")
    path = OUT_DIR / group / f"{name}_{run_tag}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
