"""Shared pytest helpers: repo paths, per-test output directories, run_gpu()."""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from graphs import WEIGHT_DTYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "out"
TEST_GRAPHS_DIR = REPO_ROOT / "graphs" / "inputs" / "test"
# Override with ECE496_BIN to test a binary built elsewhere.
GPU_BIN = Path(os.environ.get("ECE496_BIN", REPO_ROOT / "bin" / "ece496"))
GPU_TIMEOUT_S = 180

PR_DAMPING = 0.85
PR_EPSILON = 1e-4


def benchmark_args(benchmark, source=0, destination=None, damping=PR_DAMPING,
                   epsilon=PR_EPSILON):
    """The positional arguments main.cu expects after <scheduler>:
        bfs / sssp   <source>
        a_star       <source> <destination>
        pr           <damping factor> <epsilon>
    Other names get <source>, so main.cu can reject them itself."""
    if benchmark == "a_star":
        if destination is None:
            raise ValueError("a_star needs a destination")
        return [str(source), str(destination)]
    if benchmark == "pr":
        return [repr(float(damping)), repr(float(epsilon))]
    return [str(source)]


def run_gpu(benchmark, graph_path, source=0, out=None, weight_type="int32",
            scheduler="initial_scheduler", time=False, verbose=False,
            check=True, destination=None, damping=PR_DAMPING,
            epsilon=PR_EPSILON):
    """Run the GPU program once and return the CompletedProcess.

    The command is
        bin/ece496 <benchmark> <graph.gr> <weight-type> <scheduler> <args...>
    with <args...> from benchmark_args(). A hung run fails after
    GPU_TIMEOUT_S instead of freezing the test run. With check=True a
    non-zero exit code fails the test. With time=True the parsed --time JSON
    is attached as `.timing`.
    """
    if not GPU_BIN.is_file():
        pytest.fail(f"{GPU_BIN} not found; build it with cmake first")
    cmd = [str(GPU_BIN), benchmark, str(graph_path), weight_type, scheduler]
    cmd += benchmark_args(benchmark, source, destination, damping, epsilon)
    if out is not None:
        cmd += ["--out", str(out)]
    if time:
        cmd.append("--time")
    if verbose:
        cmd.append("--verbose")
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


def out_dtype(benchmark, weight_type="int32"):
    """dtype of the --out file: BFS writes uint32 hop counts, SSSP and A*
    write distances in the weight type, PageRank writes float32 ranks."""
    if benchmark == "bfs":
        return np.dtype("<u4")
    if benchmark == "pr":
        return np.dtype("<f4")
    return WEIGHT_DTYPES[weight_type]


def run_gpu_out(benchmark, graph_path, out_dir, weight_type="int32", **kw):
    """run_gpu with --out, then read the file back as a numpy array."""
    out = out_dir / f"{benchmark}.bin"
    run_gpu(benchmark, graph_path, out=out, weight_type=weight_type, **kw)
    return np.fromfile(out, dtype=out_dtype(benchmark, weight_type))


def assert_same_distances(gpu, expected):
    """Exact match, listing the first vertices that differ."""
    assert gpu.shape == expected.shape, "wrong number of values written"
    wrong = np.flatnonzero(gpu != expected)
    assert wrong.size == 0, (
        f"{wrong.size} vertices differ, first ones: "
        + ", ".join(f"v{v}: gpu={gpu[v]} expected={expected[v]}"
                    for v in wrong[:10]))


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
