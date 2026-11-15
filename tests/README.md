# Python-based Testing Infrastructure

## Run basic tests

you will need a venv with pytest, scipy, pandas, and numpy

starting the venv:
`source .venv/bin/activate`

running the tests:
`pytest tests`

Build `bin/ece496` before running the BFS tests. On a GPU other than sm_86
(e.g. the RTX 2080 Ti is sm_75), set the architecture:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j
pytest tests/test_bfs.py -v        # one file
pytest tests -k tiny               # tests whose name contains "tiny"
```

## What's here

| File | What it does |
|---|---|
| `graphs.py` | Reads and writes `.gr` files, and generates test graphs (chain, star, grid, random, disconnected, single vertex) |
| `reference.py` | Correct answers from SciPy. `scipy_bfs(g, source)` gives BFS hop counts, with `0xFFFFFFFF` for unreachable vertices |
| `conftest.py` | Shared helpers: `run_gpu()` and the `out_dir` fixture |
| `test_graphs.py` | Checks the `.gr` helpers and the generators |
| `test_bfs.py` | Checks BFS on the GPU against SciPy |

### `run_gpu()`
Runs `bin/ece496` once, e.g.
`run_gpu("bfs", path, source=0, out=out_dir / "d.bin", time=True)`.
- The test fails if the binary is missing, the run takes longer than 60 s, or
  the exit code isn't 0. The failure message includes the program's output.
- `check=False` skips the exit-code check, for tests that expect an error.
- `time=True` passes `--time` and puts the parsed JSON on `.timing`, e.g.
  `proc.timing["kernel_ms"]`.
- To test a binary built somewhere else, set `ECE496_BIN=/path/to/ece496`.

### BFS tests (`test_bfs.py`)
Each test runs the GPU program with `--out`, reads the distances with
`np.fromfile(..., dtype=np.uint32)` and requires an exact match with SciPy.
- Every generated graph runs 10 times with different seeds, because GPU bugs
  can depend on timing. Edge weights go up to 100, which checks that BFS
  counts hops and ignores weights.
- The checked-in `graphs/inputs/test/tiny.gr` also runs 10 times.
- Other cases: start vertices other than 0, unreachable vertices, all three
  weight types (int32, uint32, float32), the `--time` output, and an
  out-of-range start vertex, which must fail without writing a file.

## Output files
Tests that write files get their own folder from the `out_dir` fixture:

```
tests/out/<file>/<test name>_<MMDDYY>_<N>/
```

- `<file>` is the test file without `test_`, e.g. `graphs` or `bfs`.
- `<N>` counts pytest runs on that day, starting at 0. Every test in one run
  shares the same `<N>`, so `ls tests/out/*/*_102626_1` shows one run.
- Each run gets new folders, so old ones are never overwritten. A BFS folder
  holds the graph (`g.gr`) and the GPU's answer (`d.bin`), so you can rerun a
  failing case by hand.
- `tests/out/` is gitignored. Clear it with `rm -rf tests/out/*`.
