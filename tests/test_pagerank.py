"""PageRank-delta on the GPU must land within its error bound of the exact
ranks.

The kernel stops once every leftover residual is below epsilon, so its ranks
are not exact. reference.pagerank_exact() solves for the ranks it converges
to, and pagerank_error_bound() says how far below them the GPU may stop:
each GPU rank must be in [exact - bound, exact], up to float32 rounding.

Output goes to tests/out/pagerank/<test name>_<MMDDYY>_<N>/.
"""

import numpy as np
import pytest

from conftest import PR_DAMPING, PR_EPSILON, TEST_GRAPHS_DIR, run_gpu, run_gpu_out
from graphs import GRAPH_NAMES, make_graph, random_graph, read_gr, write_gr
from reference import pagerank_error_bound, pagerank_exact

REPEATS = 10
# float32 atomicAdd rounding, relative to the rank.
FLOAT_RTOL = 1e-4
# For generated graphs: the bound is n * epsilon / (1 - d), and ranks are
# about 1/n, so epsilon must be tiny for the check to mean anything. With
# 1e-9 on ~300 vertices a single rank 1% too low is caught (1e-7 missed 5%).
TIGHT_EPSILON = 1e-9
# a_star.gr is left out: its node data doesn't load for pr.
SMALL_GRAPHS = ["pr", "sink_out", "tiny"]


def gpu_pagerank(graph_path, out_dir, damping=PR_DAMPING, epsilon=PR_EPSILON):
    return run_gpu_out("pr", graph_path, out_dir, damping=damping,
                       epsilon=epsilon)


def assert_ranks_within_bound(gpu, graph, damping, epsilon):
    exact = pagerank_exact(graph, damping)
    assert gpu.shape == exact.shape, "wrong number of ranks written"
    assert np.all(np.isfinite(gpu)), "non-finite rank"
    slack = FLOAT_RTOL * exact + 1e-7
    bound = pagerank_error_bound(graph.shape[0], damping, epsilon)
    too_high = np.flatnonzero(gpu > exact + slack)
    too_low = np.flatnonzero(gpu < exact - bound - slack)
    bad = np.union1d(too_high, too_low)
    assert bad.size == 0, (
        f"{bad.size} ranks outside [exact - {bound:.3g}, exact], first ones: "
        + ", ".join(f"v{v}: gpu={gpu[v]:.6g} exact={exact[v]:.6g}"
                    for v in bad[:10]))
    # All leftover residual together is also under the bound.
    assert exact.sum() - gpu.sum() <= bound + slack.sum()


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("epsilon", [PR_EPSILON, TIGHT_EPSILON])
@pytest.mark.parametrize("name", SMALL_GRAPHS)
def test_pagerank_small_graphs(name, epsilon, repeat, out_dir):
    path = TEST_GRAPHS_DIR / f"{name}.gr"
    assert_ranks_within_bound(gpu_pagerank(path, out_dir, epsilon=epsilon),
                              read_gr(path), PR_DAMPING, epsilon)


def test_pagerank_pr_gr_readme_values(out_dir):
    # Expected ranks listed in graphs/inputs/test/README.md for d=0.85,
    # epsilon=1e-4. They come from one earlier run, so they are only
    # compared loosely; the bound test above is the real check.
    readme = np.array([0.310829, 0.150852, 0.150792, 0.125495, 0.0614295,
                       0.0894412, 0.0360577, 0.0720129])
    gpu = gpu_pagerank(TEST_GRAPHS_DIR / "pr.gr", out_dir)
    np.testing.assert_allclose(gpu, readme, atol=2e-3)


def test_pagerank_sink_keeps_its_rank(out_dir):
    # sink_out.gr is 0 -> 1 only. Vertex 1 has no out-edges, so it keeps
    # what it gets: r0 = 0.15/2, r1 = 0.15/2 + 0.85 * r0.
    gpu = gpu_pagerank(TEST_GRAPHS_DIR / "sink_out.gr", out_dir)
    np.testing.assert_allclose(gpu, [0.075, 0.13875], rtol=1e-4)


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("name", [n for n in GRAPH_NAMES
                                  if n != "single_vertex"])
def test_pagerank_matches_exact(name, repeat, out_dir):
    g = make_graph(name, seed=repeat)
    write_gr(out_dir / "g.gr", g)
    epsilon = TIGHT_EPSILON
    assert_ranks_within_bound(gpu_pagerank(out_dir / "g.gr", out_dir,
                                           epsilon=epsilon),
                              g, PR_DAMPING, epsilon)


def test_pagerank_single_vertex(out_dir):
    g = make_graph("single_vertex")
    write_gr(out_dir / "g.gr", g)
    gpu = gpu_pagerank(out_dir / "g.gr", out_dir)
    np.testing.assert_allclose(gpu, [1 - PR_DAMPING], rtol=1e-6)


@pytest.mark.parametrize("damping", [0.5, 0.95])
def test_pagerank_other_damping(damping, out_dir):
    g = random_graph(300, 4, seed=1)
    write_gr(out_dir / "g.gr", g)
    epsilon = TIGHT_EPSILON
    assert_ranks_within_bound(gpu_pagerank(out_dir / "g.gr", out_dir,
                                           damping=damping, epsilon=epsilon),
                              g, damping, epsilon)


def test_pagerank_time_output(out_dir):
    proc = run_gpu("pr", TEST_GRAPHS_DIR / "pr.gr", out=out_dir / "r.bin",
                   time=True)
    assert proc.timing["kernel_ms"] >= 0


@pytest.mark.parametrize("damping, epsilon", [(0.85, 0.0), (0.85, -1e-4),
                                              (-0.5, 1e-4)])
def test_pagerank_rejects_bad_parameters(damping, epsilon, out_dir):
    proc = run_gpu("pr", TEST_GRAPHS_DIR / "pr.gr", out=out_dir / "r.bin",
                   damping=damping, epsilon=epsilon, check=False)
    assert proc.returncode != 0
    assert not (out_dir / "r.bin").exists()
