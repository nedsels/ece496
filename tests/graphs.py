"""Read/write the .gr graph format and generate test graphs.

The .gr layout mirrors CSRGraph::readFromGR in src/graph/csr_graph.cu.
All values are little-endian:

    u64 version (= 1)
    u64 weight_size (0 = unweighted, else 4)
    u64 nnodes (>= 1)
    u64 nedges
    u64 row_end[nnodes]      end offset of each vertex's edges (no leading 0)
    u32 dst[nedges]          plus one u32 of padding when nedges is odd
    weight[nedges]           only when weight_size != 0
    optional node data, used by A* for its heuristic:
      u64 kind               1 = one i32 per node, 2 = (x, y) i32 pair per node
      i32 values[nnodes] or i32 xy[nnodes][2]

The node data must match what the benchmark reads: A* needs kind 2, and
bfs/sssp/pr (which read kind 1) refuse a file with kind 2.

Graphs are scipy.sparse.csr_matrix objects, where entry (u, v) is the weight
of the directed edge u -> v. Matrices are built straight from their
indptr/indices/data arrays, so duplicate edges and zero weights read from a
file are kept exactly as stored.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

GR_VERSION = 1
HEADER_DTYPE = np.dtype("<u8")
ROW_END_DTYPE = np.dtype("<u8")
DST_DTYPE = np.dtype("<u4")
# CSRGraph::index_type is int32, so counts above this are rejected.
MAX_INDEX = np.iinfo(np.int32).max
# Weight types the GPU program accepts (its <weight-type> argument).
WEIGHT_DTYPES = {
    "int32": np.dtype("<i4"),
    "uint32": np.dtype("<u4"),
    "float32": np.dtype("<f4"),
}


NODE_DATA_XY = 2  # node-data kind for (x, y) coordinates


def write_gr(path, matrix, weighted=True, coords=None):
    """Write a square sparse matrix to `path` in .gr format.

    With weighted=False the file has weight_size 0 and matrix values are
    ignored. Otherwise matrix.data must be int32, uint32 or float32.
    coords, an (nnodes, 2) integer array, is appended as A*'s node data.
    """
    if coords is not None:
        coords = np.asarray(coords)
        if coords.shape != (matrix.shape[0], 2):
            raise ValueError(f"coords must be ({matrix.shape[0]}, 2), "
                             f"got {coords.shape}")
    m = sp.csr_matrix(matrix)
    nnodes, ncols = m.shape
    if nnodes != ncols:
        raise ValueError(f"matrix must be square, got {m.shape}")
    if nnodes < 1:
        raise ValueError("graph must have at least one vertex")
    nedges = int(m.indptr[-1])
    if nnodes > MAX_INDEX or nedges > MAX_INDEX:
        raise ValueError("graph too large for int32 indices")

    if weighted:
        weight_dtype = m.data.dtype.newbyteorder("<")
        if weight_dtype not in WEIGHT_DTYPES.values():
            raise ValueError(
                f"weights must be int32, uint32 or float32, got {m.data.dtype}"
            )
        weight_size = weight_dtype.itemsize
    else:
        weight_size = 0

    dst = np.asarray(m.indices[:nedges], dtype=DST_DTYPE)
    if nedges % 2 == 1:
        dst = np.append(dst, DST_DTYPE.type(0))

    with open(path, "wb") as f:
        f.write(np.array([GR_VERSION, weight_size, nnodes, nedges],
                         dtype=HEADER_DTYPE).tobytes())
        f.write(np.asarray(m.indptr[1:], dtype=ROW_END_DTYPE).tobytes())
        f.write(dst.tobytes())
        if weighted:
            f.write(np.asarray(m.data[:nedges], dtype=weight_dtype).tobytes())
        if coords is not None:
            f.write(np.array([NODE_DATA_XY], dtype=HEADER_DTYPE).tobytes())
            f.write(np.asarray(coords, dtype="<i4").tobytes())


def read_coords(path):
    """The (nnodes, 2) int32 coordinates stored after the edges, or None if
    the file has no node data. Raises ValueError for other node data."""
    raw = Path(path).read_bytes()
    _, weight_size, nnodes, nedges = (
        int(x) for x in np.frombuffer(raw, dtype=HEADER_DTYPE, count=4))
    end = (4 * HEADER_DTYPE.itemsize + nnodes * ROW_END_DTYPE.itemsize
           + (nedges + (nedges & 1)) * DST_DTYPE.itemsize
           + nedges * weight_size)
    if len(raw) == end:
        return None
    kind = int(np.frombuffer(raw, dtype=HEADER_DTYPE, count=1, offset=end)[0])
    if kind != NODE_DATA_XY:
        raise ValueError(f"node data kind {kind}, expected {NODE_DATA_XY}")
    if len(raw) < end + 8 + nnodes * 8:
        raise ValueError("node data is truncated")
    return np.frombuffer(raw, dtype="<i4", count=2 * nnodes,
                         offset=end + 8).reshape(nnodes, 2).copy()


def read_gr(path, weight_type="int32"):
    """Read a .gr file into a csr_matrix.

    `weight_type` says how to interpret 4-byte weights, like the GPU
    program's <weight-type> argument. Unweighted files get uint32 ones.
    Raises ValueError for any file readFromGR would reject.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 4 * HEADER_DTYPE.itemsize:
        raise ValueError("file shorter than the .gr header")
    version, weight_size, nnodes, nedges = (
        int(x) for x in np.frombuffer(raw, dtype=HEADER_DTYPE, count=4))
    weight_dtype = WEIGHT_DTYPES[weight_type]
    if version != GR_VERSION:
        raise ValueError(f"unsupported .gr version {version}")
    if nnodes == 0 or nnodes > MAX_INDEX or nedges > MAX_INDEX:
        raise ValueError(f"bad counts: nnodes={nnodes}, nedges={nedges}")
    if weight_size not in (0, weight_dtype.itemsize):
        raise ValueError(f"weight_size {weight_size} does not match {weight_type}")

    row_offset = 4 * HEADER_DTYPE.itemsize
    dst_offset = row_offset + nnodes * ROW_END_DTYPE.itemsize
    weight_offset = dst_offset + (nedges + (nedges & 1)) * DST_DTYPE.itemsize
    if weight_offset + nedges * weight_size > len(raw):
        raise ValueError("file is truncated")

    row_end = np.frombuffer(raw, dtype=ROW_END_DTYPE, count=nnodes,
                            offset=row_offset)
    indptr = np.concatenate(([0], row_end)).astype(np.int64)
    if np.any(np.diff(indptr) < 0) or indptr[-1] != nedges:
        raise ValueError("row offsets are not monotone or do not end at nedges")

    dst = np.frombuffer(raw, dtype=DST_DTYPE, count=nedges, offset=dst_offset)
    if nedges and int(dst.max()) >= nnodes:
        raise ValueError("edge destination out of range")

    if weight_size:
        data = np.frombuffer(raw, dtype=weight_dtype, count=nedges,
                             offset=weight_offset).copy()
    else:
        data = np.ones(nedges, dtype=np.uint32)

    return sp.csr_matrix((data, dst.astype(np.int32), indptr),
                         shape=(nnodes, nnodes))


# ---------------------------------------------------------------------------
# Generators. Each takes a `seed`, which drives the random weights (and the
# structure, for random_graph), so the same arguments always give the same
# graph. Weights are int32 in [1, max_weight]; the default max_weight=1 gives
# unit weights. Duplicate edges and self-loops are never generated, so SciPy
# and the GPU program see exactly the same edge set.
# ---------------------------------------------------------------------------

def _from_edges(n, src, dst, rng, max_weight):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    pairs = np.unique(np.stack([src[keep], dst[keep]], axis=1), axis=0)
    weights = rng.integers(1, max_weight + 1, size=len(pairs), dtype=np.int32)
    m = sp.csr_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    m.sort_indices()
    return m


def _undirected(src, dst):
    return np.concatenate([src, dst]), np.concatenate([dst, src])


def chain(n, seed=0, max_weight=1):
    """0 - 1 - 2 - ... - (n-1), both directions. n-1 BFS levels from 0."""
    rng = np.random.default_rng(seed)
    a = np.arange(n - 1)
    return _from_edges(n, *_undirected(a, a + 1), rng, max_weight)


def star(n, seed=0, max_weight=1):
    """Vertex 0 joined to every other vertex, both directions."""
    rng = np.random.default_rng(seed)
    leaves = np.arange(1, n)
    return _from_edges(n, *_undirected(np.zeros_like(leaves), leaves), rng,
                       max_weight)


def grid(w, h, seed=0, max_weight=1):
    """w x h 4-neighbour grid, both directions. Vertex id = row * w + col."""
    rng = np.random.default_rng(seed)
    ids = np.arange(w * h).reshape(h, w)
    src = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    dst = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    return _from_edges(w * h, *_undirected(src, dst), rng, max_weight)


def random_graph(n, avg_degree, seed=0, max_weight=1):
    """Directed graph with about n * avg_degree random edges."""
    rng = np.random.default_rng(seed)
    count = int(n * avg_degree)
    src = rng.integers(0, n, size=count)
    dst = rng.integers(0, n, size=count)
    return _from_edges(n, src, dst, rng, max_weight)


def disconnected(seed=0, max_weight=1):
    """Two separate chains (0-4 and 5-9) plus isolated vertex 10.

    From source 0, vertices 5..10 are unreachable.
    """
    rng = np.random.default_rng(seed)
    src = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    return _from_edges(11, *_undirected(src, src + 1), rng, max_weight)


def single_vertex(seed=0, max_weight=1):
    """One vertex, no edges: the smallest graph readFromGR accepts."""
    return _from_edges(1, [], [], np.random.default_rng(seed), max_weight)


def make_graph(name, seed=0, max_weight=1):
    """Small default-sized graph by name, for parametrized tests."""
    if name == "chain":
        return chain(100, seed, max_weight)
    if name == "star":
        return star(1000, seed, max_weight)
    if name == "grid":
        return grid(20, 15, seed, max_weight)
    if name == "random":
        return random_graph(500, 4, seed, max_weight)
    if name == "disconnected":
        return disconnected(seed, max_weight)
    if name == "single_vertex":
        return single_vertex(seed, max_weight)
    raise ValueError(f"unknown graph: {name}")


GRAPH_NAMES = ["chain", "star", "grid", "random", "disconnected",
               "single_vertex"]


# ---------------------------------------------------------------------------
# Geometric graphs for A*: each returns (matrix, coords). Every edge weight is
# at least the Euclidean distance between its endpoints, so A*'s straight-line
# heuristic never overestimates and A* must find the true shortest distance.
# ---------------------------------------------------------------------------

def _geometric(coords, src, dst, rng, max_extra):
    """Edges src -> dst weighted ceil(distance) + a random extra in
    [0, max_extra]. Self-loops and duplicates are dropped."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    pairs = np.unique(np.stack([src[keep], dst[keep]], axis=1), axis=0)
    dist = np.hypot(*(coords[pairs[:, 0]] - coords[pairs[:, 1]]).T)
    weights = (np.ceil(dist).astype(np.int32)
               + rng.integers(0, max_extra + 1, size=len(pairs),
                              dtype=np.int32))
    n = len(coords)
    m = sp.csr_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    m.sort_indices()
    return m


def geo_grid(w, h, seed=0, spacing=10, max_extra=10):
    """w x h 4-neighbour grid, both directions, vertices `spacing` apart."""
    rng = np.random.default_rng(seed)
    ys, xs = np.divmod(np.arange(w * h), w)
    coords = np.stack([xs, ys], axis=1).astype(np.int32) * spacing
    ids = np.arange(w * h).reshape(h, w)
    src = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    dst = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    return _geometric(coords, *_undirected(src, dst), rng, max_extra), coords


def geo_random(n, k, seed=0, size=1000, max_extra=20):
    """n random points in a size x size square, each with directed edges to
    its k nearest neighbours (like a road network: local edges only)."""
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, size, size=(n, 2), dtype=np.int32)
    d2 = ((coords[:, None, :].astype(np.int64) - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.iinfo(np.int64).max)
    nearest = np.argsort(d2, axis=1, kind="stable")[:, :k]
    src = np.repeat(np.arange(n), k)
    return _geometric(coords, src, nearest.ravel(), rng, max_extra), coords


def make_geo_graph(name, seed=0):
    """Small default-sized geometric graph by name: (matrix, coords)."""
    if name == "geo_grid":
        return geo_grid(20, 15, seed)
    if name == "geo_random":
        return geo_random(300, 4, seed)
    raise ValueError(f"unknown geometric graph: {name}")


GEO_GRAPH_NAMES = ["geo_grid", "geo_random"]
