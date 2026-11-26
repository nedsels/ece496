# Python-based Testing Infrastructure

## Run basic tests

you will need a venv with pytest, scipy, pandas, and numpy

starting the venv (csh on the UG machines):
`source .venv/bin/activate.csh`

running the tests:
`pytest tests`

Build `bin/ece496` before running the GPU tests. On a GPU other than sm_86
(e.g. the RTX 2080 Ti is sm_75), set the architecture:

```csh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j
pytest tests/test_bfs.py -v        # one file
pytest tests -k small_graphs       # every benchmark on graphs/inputs/test/
pytest tests/test_graphs.py        # no GPU needed
```

## What's here

| File | What it does |
|---|---|
| `graphs.py` | Reads and writes `.gr` files, including A*'s (x, y) node data (`write_gr(..., coords=xy)`, `read_coords()`). Generates test graphs (chain, star, grid, random, disconnected, single vertex) and geometric graphs for A* (`geo_grid`, `geo_random`) |
| `reference.py` | Correct answers: `scipy_bfs`, `scipy_sssp`, `scipy_path_cost` (A*), and `pagerank_exact` / `pagerank_error_bound` |
| `conftest.py` | Shared helpers: `run_gpu()`, `run_gpu_out()`, `assert_same_distances()` and the `out_dir` fixture |
| `test_graphs.py` | Checks the `.gr` helpers and the generators |
| `test_bfs.py` | BFS on the GPU against SciPy |
| `test_sssp.py` | SSSP on the GPU against SciPy's Dijkstra |
| `test_a_star.py` | A* path cost on the GPU against Dijkstra |
| `test_pagerank.py` | PageRank on the GPU against the exact ranks, within the error bound |

### `run_gpu()`
Runs `bin/ece496` once, e.g.
`run_gpu("bfs", path, source=0, out=out_dir / "d.bin", time=True)`.
The arguments after `<scheduler>` depend on the benchmark, and `run_gpu`
builds them:

| Benchmark | Command | `run_gpu` arguments |
|---|---|---|
| `bfs`, `sssp` | `... <scheduler> <source>` | `source=` |
| `a_star` | `... <scheduler> <source> <destination>` | `source=`, `destination=` |
| `pr` | `... <scheduler> <damping> <epsilon>` | `damping=` (0.85), `epsilon=` (1e-4) |

`run_gpu_out(benchmark, path, out_dir, ...)` adds `--out` and reads the file
back in the right dtype: uint32 for BFS, the weight type for SSSP and A*
(A* writes one value, the cost to the destination), float32 for PageRank.

- The test fails if the binary is missing, the run takes longer than 180s (GPU_TIMEOUT_S in conftest.py), or
  the exit code isn't 0. The failure message includes the program's output.
- `check=False` skips the exit-code check, for tests that expect an error.
- `verbose=True` passes `--verbose`, so the distances are printed to stdout
  as well as written to `out`.
- `time=True` passes `--time` and puts the parsed JSON on `.timing`, e.g.
  `proc.timing["kernel_ms"]`.
- To test a binary built somewhere else, set `ECE496_BIN=/path/to/ece496`.

### BFS and SSSP tests (`test_bfs.py`, `test_sssp.py`)
Each test runs the GPU program with `--out` and requires an exact match with
SciPy, on the generated graphs and on every source of `tiny.gr`, `pr.gr` and
`sink_out.gr`.
- Other cases: start vertices other than 0, unreachable vertices, the
  `--time` output, and an out-of-range start vertex, which must fail without
  writing a file.
- Unreachable vertices: BFS writes `0xFFFFFFFF`. SSSP and A* write the
  largest value of the weight type, so `0x7FFFFFFF` for int32
  (`reference.unreached_distance()`).

### A* tests (`test_a_star.py`)
A* needs (x, y) coordinates in the `.gr` file, so it runs on `a_star.gr`
(the README's paths, then every source/destination pair) and on generated
geometric graphs, where every edge is at least as long as the straight line
between its ends. The cost must equal Dijkstra's.

### PageRank tests (`test_pagerank.py`)
The kernel stops once every vertex's leftover residual is below epsilon, so
its ranks are a little below the exact ones. `pagerank_exact()` solves for
the exact ranks, and each GPU rank must be in `[exact - n*eps/(1-d), exact]`
(plus float32 rounding). Generated graphs use epsilon = 1e-9 so that bound is
small next to the ranks. Vertices with no out-edges keep their rank and pass
nothing on, as in `pagerank.cut`.

### Graphs in `graphs/inputs/test/`
`a_star.gr` stores (x, y) node data, and the program only loads it as A*'s
node type, so it isn't used for BFS, SSSP or PageRank.

## Output files
Tests that write files get their own folder from the `out_dir` fixture:

```
tests/out/<file>/<test name>_<MMDDYY>_<N>/
```

- `<file>` is the test file without `test_`, e.g. `graphs` or `bfs`.
- `<N>` counts pytest runs on that day, starting at 0. Every test in one run
  shares the same `<N>`, so `ls tests/out/*/*_102626_1` shows one run.
- Each run gets new folders, so old ones are never overwritten. A BFS folder
  holds the graph (`g.gr`) and the GPU's answer (`<benchmark>.bin`), so you
  can rerun a failing case by hand.
- `tests/out/` is gitignored. Clear it with `rm -rf tests/out/*`.
