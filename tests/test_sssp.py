"""SSSP on the GPU must match SciPy's Dijkstra exactly.

Output goes to tests/out/sssp/<test name>_<MMDDYY>_<N>/.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import (TEST_GRAPHS_DIR, assert_same_distances, run_gpu,
                      run_gpu_out)
from graphs import GRAPH_NAMES, make_graph, random_graph, read_gr, write_gr
from reference import scipy_sssp, unreached_distance

REPEATS = 10
SMALL_REPEATS = 3
# a_star.gr is tested in test_a_star.py: its node data doesn't load here.
SMALL_GRAPHS = ["tiny", "pr", "sink_out"]


def gpu_sssp(graph_path, source, out_dir, weight_type="int32"):
    return run_gpu_out("sssp", graph_path, out_dir, weight_type=weight_type,
                       source=source)


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("name", GRAPH_NAMES)
def test_sssp_matches_scipy(name, repeat, out_dir):
    g = make_graph(name, seed=repeat, max_weight=100)
    write_gr(out_dir / "g.gr", g)
    assert_same_distances(gpu_sssp(out_dir / "g.gr", 0, out_dir),
                          scipy_sssp(g, 0))


@pytest.mark.parametrize("repeat", range(SMALL_REPEATS))
@pytest.mark.parametrize("name", SMALL_GRAPHS)
def test_sssp_small_graphs(name, repeat, out_dir):
    path = TEST_GRAPHS_DIR / f"{name}.gr"
    g = read_gr(path)
    for source in range(g.shape[0]):
        assert_same_distances(gpu_sssp(path, source, out_dir),
                              scipy_sssp(g, source))


def test_sssp_tiny_known_distances(out_dir):
    # Worked out by hand from tiny.gr: 0->4 is direct (5), 0->2 goes 0->1->2
    # (12 + 8), and 0->9 goes 0->3->7->9 (11 + 5 + 1).
    gpu = gpu_sssp(TEST_GRAPHS_DIR / "tiny.gr", 0, out_dir)
    assert gpu[[0, 4, 2, 9]].tolist() == [0, 5, 20, 17]


@pytest.mark.parametrize("seed", range(5))
def test_sssp_other_sources(seed, out_dir):
    g = random_graph(500, 2, seed=seed, max_weight=1000)
    write_gr(out_dir / "g.gr", g)
    source = int(np.random.default_rng(seed).integers(1, g.shape[0]))
    assert_same_distances(gpu_sssp(out_dir / "g.gr", source, out_dir),
                          scipy_sssp(g, source))


def test_sssp_weighted_path_beats_fewer_hops(out_dir):
    # 0 -> 2 directly costs 100, 0 -> 1 -> 2 costs 2: BFS and SSSP disagree.
    g = sp.csr_matrix((np.array([100, 1, 1], dtype=np.int32),
                       ([0, 0, 1], [2, 1, 2])), shape=(3, 3))
    write_gr(out_dir / "g.gr", g)
    assert gpu_sssp(out_dir / "g.gr", 0, out_dir).tolist() == [0, 1, 2]


@pytest.mark.parametrize("weight_type", ["int32", "uint32"])
def test_sssp_unreachable_marked(weight_type, out_dir):
    g = make_graph("disconnected", max_weight=10).astype(weight_type)
    write_gr(out_dir / "g.gr", g)
    gpu = gpu_sssp(out_dir / "g.gr", 0, out_dir, weight_type)
    assert_same_distances(gpu, scipy_sssp(g, 0, weight_type))
    assert np.all(gpu[5:] == unreached_distance(weight_type))


@pytest.mark.parametrize("weight_type", ["int32", "uint32"])
def test_sssp_weight_types(weight_type, out_dir):
    g = make_graph("grid", seed=0, max_weight=50).astype(weight_type)
    write_gr(out_dir / "g.gr", g)
    assert_same_distances(gpu_sssp(out_dir / "g.gr", 0, out_dir, weight_type),
                          scipy_sssp(g, 0, weight_type))


def test_sssp_time_output(out_dir):
    proc = run_gpu("sssp", TEST_GRAPHS_DIR / "tiny.gr", out=out_dir / "d.bin",
                   time=True)
    assert proc.timing["kernel_ms"] >= 0


@pytest.mark.parametrize("source", ["-1", "10"])
def test_sssp_rejects_bad_source(source, out_dir):
    proc = run_gpu("sssp", TEST_GRAPHS_DIR / "tiny.gr", source=source,
                   out=out_dir / "d.bin", check=False)
    assert proc.returncode != 0
    assert not (out_dir / "d.bin").exists()
