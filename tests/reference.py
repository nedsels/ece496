"""Answer keys: known-correct results computed with SciPy / numpy."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra, shortest_path

from graphs import WEIGHT_DTYPES

# What the GPU program writes for a vertex the source cannot reach.
UNREACHED = np.uint32(0xFFFFFFFF)


def scipy_bfs(graph, source):
    """BFS hop counts from `source` along directed edges, as uint32.

    Edge weights are ignored. Unreachable vertices get UNREACHED, matching
    the distances the GPU program writes with --out.
    """
    dist = shortest_path(graph, unweighted=True, indices=source)
    return np.where(np.isinf(dist), UNREACHED, dist).astype(np.uint32)


def unreached_distance(weight_type):
    """What SSSP / A* write for an unreachable vertex: the largest value of
    the weight type (numeric_limits<weight_type>::max() in sssp.cut and
    a_star.cut). For int32 that is 0x7FFFFFFF, not BFS's 0xFFFFFFFF."""
    return np.iinfo(WEIGHT_DTYPES[weight_type]).max


def scipy_sssp(graph, source, weight_type="int32"):
    """Dijkstra distances from `source` in the GPU's output dtype, with
    unreachable vertices set to unreached_distance(weight_type)."""
    dist = dijkstra(graph, indices=source)
    dtype = WEIGHT_DTYPES[weight_type]
    return np.where(np.isinf(dist), unreached_distance(weight_type),
                    dist).astype(dtype)


def scipy_path_cost(graph, source, destination, weight_type="int32"):
    """Shortest distance source -> destination, which A* must find: a 1-value
    array like the GPU's --out file for a_star."""
    return scipy_sssp(graph, source, weight_type)[[destination]]


def pagerank_exact(graph, damping):
    """The ranks PageRank-delta converges to as epsilon -> 0.

    Solves r = (1 - d)/N + d * W r, where W[v, u] = 1/outdeg(u) for each
    edge u -> v (edge weights and duplicates are ignored, like the kernel's
    getOutDegree). A vertex with no out-edges keeps its rank and passes
    nothing on, as in pagerank.cut, so ranks need not sum to 1.
    """
    n = graph.shape[0]
    g = sp.csr_matrix(graph)
    deg = np.diff(g.indptr).astype(np.float64)
    rows = np.repeat(np.arange(n), np.diff(g.indptr))
    w = sp.csr_matrix((1.0 / deg[rows], (g.indices, rows)), shape=(n, n))
    a = sp.identity(n, format="csc") - damping * w.tocsc()
    return spsolve(a, np.full(n, (1.0 - damping) / n))


def pagerank_error_bound(n, damping, epsilon):
    """How far below the exact ranks the GPU's ranks may be.

    The kernel stops once every vertex's leftover residual is below epsilon.
    Pushing those leftovers through would add at most n * epsilon / (1 - d)
    in total, and residual is only ever moved into ranks, never taken out,
    so each GPU rank is in [exact - bound, exact].
    """
    return n * epsilon / (1.0 - damping)
