"""Step 2: the .gr helpers and graph generators."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from conftest import TEST_GRAPHS_DIR
from graphs import (GRAPH_NAMES, chain, disconnected, grid, make_graph,
                    random_graph, read_gr, single_vertex, star, write_gr)


def assert_same_arrays(a, b):
    """Exact equality of the stored CSR arrays, not just matrix equality."""
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert a.data.dtype == b.data.dtype
    np.testing.assert_array_equal(a.data, b.data)


# --- Reading a file made outside our Python code ---------------------------

def test_read_tiny_gr_known_contents():
    g = read_gr(TEST_GRAPHS_DIR / "tiny.gr", "int32")
    assert g.shape == (10, 10)
    np.testing.assert_array_equal(g.indptr,
                                  [0, 4, 5, 6, 8, 8, 9, 10, 11, 12, 13])
    np.testing.assert_array_equal(g.indices,
                                  [1, 3, 4, 6, 2, 3, 1, 7, 8, 5, 9, 9, 8])
    np.testing.assert_array_equal(g.data,
                                  [12, 11, 5, 10, 8, 2, 11, 5, 1, 4, 1, 5, 5])


def test_tiny_gr_rewrites_byte_for_byte(out_dir):
    original = (TEST_GRAPHS_DIR / "tiny.gr").read_bytes()
    write_gr(out_dir / "tiny.gr", read_gr(TEST_GRAPHS_DIR / "tiny.gr"))
    assert (out_dir / "tiny.gr").read_bytes() == original


# --- Round trips -----------------------------------------------------------

@pytest.mark.parametrize("name", GRAPH_NAMES)
@pytest.mark.parametrize("seed", range(3))
def test_round_trip_generated(name, seed, out_dir):
    g = make_graph(name, seed=seed, max_weight=100)
    write_gr(out_dir / "g.gr", g)
    assert_same_arrays(read_gr(out_dir / "g.gr", "int32"), g)


@pytest.mark.parametrize("weight_type, data", [
    ("int32", np.array([-5, 0, 7], dtype=np.int32)),
    ("uint32", np.array([0, 1, 2**32 - 1], dtype=np.uint32)),
    ("float32", np.array([0.5, 0.0, 1e30], dtype=np.float32)),
])
def test_round_trip_weight_types(weight_type, data, out_dir):
    # Odd edge count (3) also exercises the destination padding.
    g = sp.csr_matrix((data, np.array([1, 2, 0], dtype=np.int32),
                       np.array([0, 2, 3, 3])), shape=(3, 3))
    write_gr(out_dir / "g.gr", g)
    assert_same_arrays(read_gr(out_dir / "g.gr", weight_type), g)


def test_round_trip_unweighted(out_dir):
    g = grid(4, 3, seed=1, max_weight=50)
    write_gr(out_dir / "g.gr", g, weighted=False)
    back = read_gr(out_dir / "g.gr")
    np.testing.assert_array_equal(back.indptr, g.indptr)
    np.testing.assert_array_equal(back.indices, g.indices)
    np.testing.assert_array_equal(back.data, np.ones(g.nnz, dtype=np.uint32))


def test_round_trip_keeps_duplicates_and_zero_weights(out_dir):
    # SciPy would merge the duplicate 0->1 edges and may drop the zero
    # weight; the .gr file must keep both.
    g = sp.csr_matrix((np.array([3, 4, 0], dtype=np.int32),
                       np.array([1, 1, 0], dtype=np.int32),
                       np.array([0, 2, 3])), shape=(2, 2))
    write_gr(out_dir / "g.gr", g)
    assert_same_arrays(read_gr(out_dir / "g.gr"), g)


# --- File layout -----------------------------------------------------------

@pytest.mark.parametrize("nedges", [0, 1, 2, 3])
def test_file_size_includes_padding(nedges, out_dir):
    # All edges leave the last vertex, so row_end is [0, 0, 0, nedges].
    n = 4
    g = sp.csr_matrix((np.ones(nedges, dtype=np.int32),
                       np.zeros(nedges, dtype=np.int32),
                       np.array([0] * n + [nedges])), shape=(n, n))
    assert g.nnz == nedges
    write_gr(out_dir / "g.gr", g)
    padded = nedges + (nedges & 1)
    expected = 32 + 8 * n + 4 * padded + 4 * nedges
    assert (out_dir / "g.gr").stat().st_size == expected


def test_write_rejects_unsupported_weight_dtype(out_dir):
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int64))
    with pytest.raises(ValueError):
        write_gr(out_dir / "g.gr", g)


def test_write_rejects_empty_graph(out_dir):
    with pytest.raises(ValueError):
        write_gr(out_dir / "g.gr", sp.csr_matrix((0, 0), dtype=np.int32))


def test_read_rejects_mismatched_weight_size(out_dir):
    path = out_dir / "g.gr"
    write_gr(path, chain(3))
    raw = bytearray(path.read_bytes())
    raw[8:16] = (8).to_bytes(8, "little")  # weight_size = 8
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_gr(path)


def test_read_rejects_out_of_range_destination(out_dir):
    path = out_dir / "g.gr"
    write_gr(path, chain(3))
    raw = bytearray(path.read_bytes())
    dst_offset = 32 + 8 * 3
    raw[dst_offset:dst_offset + 4] = (3).to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_gr(path)


# --- Generators ------------------------------------------------------------

@pytest.mark.parametrize("name", GRAPH_NAMES)
def test_generators_are_deterministic(name):
    assert_same_arrays(make_graph(name, seed=7, max_weight=100),
                       make_graph(name, seed=7, max_weight=100))


@pytest.mark.parametrize("name", GRAPH_NAMES)
def test_generators_have_no_self_loops_or_duplicates(name):
    g = make_graph(name, seed=0)
    coo = g.tocoo()
    assert not np.any(coo.row == coo.col)
    pairs = set(zip(coo.row.tolist(), coo.col.tolist()))
    assert len(pairs) == g.nnz
    assert g.has_sorted_indices


@pytest.mark.parametrize("g", [chain(10), star(10), grid(5, 4),
                               disconnected()], ids=["chain", "star", "grid",
                                                     "disconnected"])
def test_undirected_generators_are_symmetric(g):
    assert (g != g.T).nnz == 0


def test_generator_shapes():
    def bfs(g):
        return shortest_path(g, unweighted=True, indices=0)

    np.testing.assert_array_equal(bfs(chain(5)), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(bfs(star(5)), [0, 1, 1, 1, 1])
    assert bfs(grid(4, 3)).max() == (4 - 1) + (3 - 1)
    d = bfs(disconnected())
    np.testing.assert_array_equal(d[:5], [0, 1, 2, 3, 4])
    assert np.all(np.isinf(d[5:]))
    assert single_vertex().shape == (1, 1) and single_vertex().nnz == 0
    assert random_graph(100, 4, seed=0).nnz > 300


def test_max_weight_range():
    g = random_graph(200, 5, seed=3, max_weight=10)
    assert g.data.dtype == np.int32
    assert g.data.min() >= 1 and g.data.max() <= 10
    assert np.all(chain(20).data == 1)
