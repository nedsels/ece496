"""Shared pytest helpers: repo paths and per-test output directories."""

import re
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "out"
TEST_GRAPHS_DIR = REPO_ROOT / "graphs" / "inputs" / "test"


@pytest.fixture
def out_dir(request):
    """An empty tests/out/<test name>/ directory for this test's files.

    Files are left in place after the test so failures can be inspected.
    """
    name = re.sub(r"[^\w.-]+", "_", request.node.name)
    path = OUT_DIR / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
