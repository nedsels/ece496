"""Answer keys: known-correct results computed with SciPy."""

import numpy as np
from scipy.sparse.csgraph import shortest_path

# What the GPU program writes for a vertex the source cannot reach.
UNREACHED = np.uint32(0xFFFFFFFF)


def scipy_bfs(graph, source):
    """BFS hop counts from `source` along directed edges, as uint32.

    Edge weights are ignored. Unreachable vertices get UNREACHED, matching
    the distances the GPU program writes with --out.
    """
    dist = shortest_path(graph, unweighted=True, indices=source)
    return np.where(np.isinf(dist), UNREACHED, dist).astype(np.uint32)
