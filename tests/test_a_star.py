"""A* on the GPU must find the same cost as SciPy's Dijkstra.

The GPU program only writes the cost to the destination (one value in the
weight type), not the path, so that is what is compared. A* needs (x, y)
node coordinates in the .gr file for its straight-line heuristic, so the
generated graphs come from the geometric generators in graphs.py, whose
weights are never shorter than the straight-line distance.

Output goes to tests/out/a_star/<test name>_<MMDDYY>_<N>/.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import (TEST_GRAPHS_DIR, assert_same_distances, run_gpu,
                      run_gpu_out)
from graphs import (GEO_GRAPH_NAMES, make_geo_graph, read_coords, read_gr,
                    write_gr)
from reference import scipy_path_cost, unreached_distance

REPEATS = 10
A_STAR_GR = TEST_GRAPHS_DIR / "a_star.gr"


def gpu_a_star(graph_path, source, destination, out_dir, weight_type="int32"):
    return run_gpu_out("a_star", graph_path, out_dir, weight_type=weight_type,
                       source=source, destination=destination)


def test_a_star_gr_heuristic_is_admissible():
    # Sanity check on the input itself: if an edge were shorter than the
    # straight line between its ends, A* could rightly return a wrong cost.
    g = read_gr(A_STAR_GR).tocoo()
    xy = read_coords(A_STAR_GR)
    assert xy is not None, "a_star.gr has no coordinates"
    straight = np.hypot(*(xy[g.row] - xy[g.col]).T)
    assert np.all(g.data >= straight)


# Paths from graphs/inputs/test/README.md.
@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("source, destination, cost",
                         [(0, 9, 22), (8, 9, 22), (4, 4, 0)])
def test_a_star_gr_known_paths(source, destination, cost, repeat, out_dir):
    gpu = gpu_a_star(A_STAR_GR, source, destination, out_dir)
    assert gpu.tolist() == [cost]


def test_a_star_gr_all_pairs(out_dir):
    g = read_gr(A_STAR_GR)
    n = g.shape[0]
    for source in range(n):
        for destination in range(n):
            assert_same_distances(
                gpu_a_star(A_STAR_GR, source, destination, out_dir),
                scipy_path_cost(g, source, destination))


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("name", GEO_GRAPH_NAMES)
def test_a_star_matches_dijkstra(name, repeat, out_dir):
    g, xy = make_geo_graph(name, seed=repeat)
    write_gr(out_dir / "g.gr", g, coords=xy)
    rng = np.random.default_rng(repeat)
    for source, destination in rng.integers(0, g.shape[0], size=(5, 2)):
        assert_same_distances(
            gpu_a_star(out_dir / "g.gr", source, destination, out_dir),
            scipy_path_cost(g, source, destination))


def test_a_star_far_corner(out_dir):
    # Corner to corner of a grid: the longest search, through every row.
    g, xy = make_geo_graph("geo_grid")
    write_gr(out_dir / "g.gr", g, coords=xy)
    destination = g.shape[0] - 1
    assert_same_distances(gpu_a_star(out_dir / "g.gr", 0, destination,
                                     out_dir),
                          scipy_path_cost(g, 0, destination))


@pytest.mark.parametrize("weight_type", ["int32", "uint32"])
def test_a_star_unreachable_destination(weight_type, out_dir):
    # Two separate grids side by side: nothing in the left one reaches the
    # right one.
    left, xy = make_geo_graph("geo_grid")
    n = left.shape[0]
    g = sp.block_diag([left, left], format="csr").astype(weight_type)
    coords = np.concatenate([xy, xy + [xy[:, 0].max() + 1000, 0]])
    write_gr(out_dir / "g.gr", g, coords=coords)
    gpu = gpu_a_star(out_dir / "g.gr", 0, n, out_dir, weight_type)
    assert gpu.tolist() == [unreached_distance(weight_type)]


def test_a_star_time_output(out_dir):
    proc = run_gpu("a_star", A_STAR_GR, source=0, destination=9,
                   out=out_dir / "d.bin", time=True)
    assert proc.timing["kernel_ms"] >= 0


def test_a_star_prints_cost_without_out(out_dir):
    proc = run_gpu("a_star", A_STAR_GR, source=0, destination=9)
    assert "Distance from 0 to 9 is 22" in proc.stdout


@pytest.mark.parametrize("source, destination",
                         [("-1", "9"), ("10", "9"), ("0", "-1"), ("0", "10")])
def test_a_star_rejects_bad_vertices(source, destination, out_dir):
    proc = run_gpu("a_star", A_STAR_GR, source=source, destination=destination,
                   out=out_dir / "d.bin", check=False)
    assert proc.returncode != 0
    assert not (out_dir / "d.bin").exists()
