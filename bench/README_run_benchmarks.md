# Running benchmarks

From the repo root:

```bash
# 1. Tell the script where your graphs are (folder holding <name>/<name>.gr)
export GRAPH_DIR="/nfs/ug/groups/capstone_proj_2026F/input_graphs"

# 2. See which graphs it found
bench/run_benchmarks.py --list

# 3. Run everything (builds bin/ece496 first)
bench/run_benchmarks.py

# Or run a subset
bench/run_benchmarks.py -g germany-osm -b bfs -s initial_scheduler --runs 1
```

| Option | Meaning | Default |
|---|---|---|
| `-g` | graphs (names from `--list`) | all that fit in memory |
| `-b` | benchmarks (`bfs`, `sssp`, `pagerank`, ...) | all implemented |
| `-s` | scheduler | `initial_scheduler` |
| `--runs N` | repeats per config | 3 |
| `--no-build` | skip the rebuild | off |

Results go to `bench/results/<scheduler>_<time>/results.csv`, with a log per
run in `logs/`. BFS answers are checked against each graph's `reference.txt`.
See `bench/run_benchmarks.py -h` for every option.

## Running the executable directly

For tools that launch the program themselves (Nsight, gdb, perf, ...):

```bash
cmake -S . -B build && cmake --build build -j

export GRAPH_DIR="/nfs/ug/groups/capstone_proj_2026F/input_graphs"

# bin/ece496 <benchmark> <graph.gr> <source> <weight-type> <scheduler> [--out FILE] [--time]
bin/ece496 bfs "$GRAPH_DIR/germany-osm/germany-osm.gr" 0 int32 initial_scheduler --time --out d.bin
```

Use `--out` on big graphs, otherwise every distance is printed. Each log in
`bench/results/.../logs/` starts with the exact command the script ran, so
you can copy it from there.

Put the tool in front of the same command:

```bash
nsys profile -o bfs_report  bin/ece496 bfs g.gr 0 int32 initial_scheduler --out d.bin   # timeline
ncu -o bfs_kernels          bin/ece496 bfs g.gr 0 int32 initial_scheduler --out d.bin   # per-kernel metrics
compute-sanitizer           bin/ece496 bfs g.gr 0 int32 initial_scheduler --out d.bin   # memory errors
perf record -g              bin/ece496 bfs g.gr 0 int32 initial_scheduler --out d.bin   # host-side CPU profile
cuda-gdb --args             bin/ece496 bfs g.gr 0 int32 initial_scheduler --out d.bin   # debugger (plain gdb works for host code)
```

For debugging, build with debug info (`-G` adds device debug info but makes
kernels much slower, so don't use that build for timing):

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-G" && cmake --build build-debug -j
```

Both builds write to the same `bin/ece496`, so rebuild with `build/` before
timing again.
