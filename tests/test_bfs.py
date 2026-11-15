"""Step 3: BFS on the GPU must match SciPy exactly.

Each test writes its graph and the GPU's distances to
tests/out/bfs/<test name>_<MMDDYY>_<N>/, so a failing case can be rerun or
inspected by hand.
"""

import numpy as np
import pytest

from conftest import TEST_GRAPHS_DIR, run_gpu
from graphs import GRAPH_NAMES, make_graph, random_graph, read_gr, write_gr
from reference import UNREACHED, scipy_bfs

# GPU bugs can be timing dependent, so every case runs several times.
REPEATS = 10


def gpu_bfs(graph_path, source, out_dir, weight_type="int32"):
    """Run BFS on the GPU and read back its distances."""
    out = out_dir / "d.bin"
    run_gpu("bfs", graph_path, source=source, out=out, weight_type=weight_type)
    return np.fromfile(out, dtype=np.uint32)


def assert_same_distances(gpu, expected):
    assert gpu.shape == expected.shape, "wrong number of distances written"
    wrong = np.flatnonzero(gpu != expected)
    assert wrong.size == 0, (
        f"{wrong.size} vertices differ, first ones: "
        + ", ".join(f"v{v}: gpu={gpu[v]} expected={expected[v]}"
                    for v in wrong[:10]))


# Weights up to 100 check that BFS ignores them and counts hops.
@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("name", GRAPH_NAMES)
def test_bfs_matches_scipy(name, repeat, out_dir):
    g = make_graph(name, seed=repeat, max_weight=100)
    write_gr(out_dir / "g.gr", g)
    assert_same_distances(gpu_bfs(out_dir / "g.gr", 0, out_dir),
                          scipy_bfs(g, 0))


@pytest.mark.parametrize("repeat", range(REPEATS))
def test_bfs_tiny_gr(repeat, out_dir):
    path = TEST_GRAPHS_DIR / "tiny.gr"
    assert_same_distances(gpu_bfs(path, 0, out_dir),
                          scipy_bfs(read_gr(path), 0))


@pytest.mark.parametrize("seed", range(5))
def test_bfs_other_sources(seed, out_dir):
    # Directed random graph, so the source changes which vertices are reached.
    g = random_graph(500, 2, seed=seed)
    write_gr(out_dir / "g.gr", g)
    source = int(np.random.default_rng(seed).integers(1, g.shape[0]))
    assert_same_distances(gpu_bfs(out_dir / "g.gr", source, out_dir),
                          scipy_bfs(g, source))


def test_bfs_unreachable_marked(out_dir):
    g = make_graph("disconnected")
    write_gr(out_dir / "g.gr", g)
    gpu = gpu_bfs(out_dir / "g.gr", 0, out_dir)
    np.testing.assert_array_equal(gpu[:5], [0, 1, 2, 3, 4])
    assert np.all(gpu[5:] == UNREACHED)


@pytest.mark.parametrize("weight_type", ["int32", "uint32", "float32"])
def test_bfs_weight_types(weight_type, out_dir):
    g = make_graph("grid", seed=0)
    write_gr(out_dir / "g.gr", g.astype(np.dtype(weight_type)))
    assert_same_distances(gpu_bfs(out_dir / "g.gr", 0, out_dir, weight_type),
                          scipy_bfs(g, 0))


def test_bfs_time_output(out_dir):
    proc = run_gpu("bfs", TEST_GRAPHS_DIR / "tiny.gr", out=out_dir / "d.bin",
                   time=True)
    assert proc.timing["kernel_ms"] >= 0
    assert proc.timing["load_ms"] >= 0


@pytest.mark.parametrize("source", ["-1", "10"])
def test_bfs_rejects_bad_source(source, out_dir):
    # tiny.gr has vertices 0..9.
    proc = run_gpu("bfs", TEST_GRAPHS_DIR / "tiny.gr", source=source,
                   out=out_dir / "d.bin", check=False)
    assert proc.returncode != 0
    assert not (out_dir / "d.bin").exists()
