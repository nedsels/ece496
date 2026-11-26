# Running benchmarks

From the repo root:

The group's big graphs in `/nfs/ug/groups/capstone_proj_2026F/input_graphs`
are searched automatically on the UG machines, along with `graphs/inputs/`.
For graphs anywhere else, pass `--graph-dir DIR` or set `GRAPH_DIR`
(`setenv GRAPH_DIR /some/dir` in csh).

```csh
# 1. See which graphs it found
bench/run_benchmarks.py --list

# 2. Run everything (builds bin/ece496 first)
bench/run_benchmarks.py

# Or run a subset
bench/run_benchmarks.py -g germany-osm -b bfs -s initial_scheduler --runs 1
```

| Option | Meaning | Default |
|---|---|---|
| `-g` | graphs (names from `--list`) | all that fit in memory |
| `-b` | benchmarks (`bfs`, `sssp`, `a_star`, `pr`) | all implemented |
| `-s` | scheduler | `initial_scheduler` |
| `--runs N` | repeats per config | 3 |
| `--sources N` | first N sources, or pairs for `a_star` (0 for all) | 1 |
| `--source V` | use this source instead; `a_star` also needs `--dest V` | off |
| `--damping D`, `--pr-tol T` | PageRank; the epsilon passed is `T / nodes` | 0.85, 1e-3 |
| `--timeout S` | kill a run after S seconds | 600 |
| `--no-build` | skip the rebuild | off |

Results go to `bench/results/<scheduler>_<time>/results.csv`, with a log per
run in `logs/`. See `bench/run_benchmarks.py -h` for every option.

| Benchmark | One run per | Graph file | Checked against |
|---|---|---|---|
| `bfs`, `sssp` | source in `sources.txt` | `<name>.gr` | `reference.txt` |
| `a_star` | pair in `pairs.txt` | `<name>.agr` | `astar_reference.txt` |
| `pr` | graph | `<name>.gr` | the graph itself: every node's leftover residual, recomputed from the ranks, must be below epsilon (needs numpy) |

`<name>.agr` is a `.gr` with node coordinates after the weights (a `.gr` that
has them works too). Only `a_star` can load it. `check` is `-` when there is
nothing to check against.

On JamesPC (3080 Ti), `sssp` on `usa-road-d` takes about 80 s per run.

## Galois CPU baseline (`run_galois.py`)

`bench/run_galois.py` runs Galois's Lonestar CPU apps (`bfs-cpu`,
`sssp-cpu`) on the same graphs, with the same options (`-g`, `-b`,
`--sources`, `--runs`, `--list`, ...), and writes the same CSV with
`scheduler=galois-cpu` to `bench/results/galois-cpu_<time>/`.

Galois needs Boost, LLVM (with RTTI) and fmt. On the UG machines LLVM and
Boost are installed already, and fmt is in the group folder, so set this
first (put it in `~/.cshrc` to keep it):

```csh
setenv CMAKE_PREFIX_PATH /nfs/ug/groups/capstone_proj_2026F/local
git submodule update --init Galois

python3 bench/run_galois.py --list
python3 bench/run_galois.py -g germany-osm -b bfs sssp --runs 3
```

The first run configures `build-galois/` and builds only `bfs-cpu` and
`sssp-cpu`. Use `--no-build` after that.

| Option | Meaning | Default |
|---|---|---|
| `--threads N` | Galois threads | all cores |
| `--sssp-algo` | `sssp-cpu -algo` (`deltaStep`, `dijkstra`, ...) | Lonestar's auto |
| `--delta N` | `sssp-cpu -delta` (log2 bucket width) | 13 |
| `--bfs-algo` | `bfs-cpu -algo` | Lonestar's |
| `--galois-build` | build directory | `build-galois/` or `$GALOIS_BUILD` |

- `kernel_ms` is Lonestar's `Timer_0`: the algorithm only, in whole
  milliseconds. Graph loading and verification aren't included.
- A run only counts as `ok` if Lonestar prints `Verification successful.`.
  Its reached/max/sum are also checked against `reference.txt` when the
  graph has one, and a failed check blanks the time.
- SSSP needs non-negative integer weights. Graphs without 4-byte weights are
  skipped (`skip sssp on <graph>: ...`), as are `.agr`-only graphs.
- Other machines may share the UG CPU, so check `uptime` before timing.

## Running the executable directly

For tools that launch the program themselves (Nsight, gdb, perf, ...):

```csh
cmake -S . -B build && cmake --build build -j

setenv GRAPH_DIR /nfs/ug/groups/capstone_proj_2026F/input_graphs

# bin/ece496 bfs|sssp <graph.gr>  <weight-type> <scheduler> <source>               [--out FILE] [--time]
# bin/ece496 a_star   <graph.agr> <weight-type> <scheduler> <source> <destination> [--out FILE] [--time]
# bin/ece496 pr       <graph.gr>  <weight-type> <scheduler> <damping> <epsilon>    [--out FILE] [--time]
bin/ece496 bfs "$GRAPH_DIR/germany-osm/germany-osm.gr" int32 initial_scheduler 0 --time --out d.bin
```

Use `--out` on big graphs, otherwise every distance is printed. Each log in
`bench/results/.../logs/` starts with the exact command the script ran, so
you can copy it from there.

Put the tool in front of the same command:

```bash
nsys profile -o bfs_report  bin/ece496 bfs g.gr int32 initial_scheduler 0 --out d.bin   # timeline
ncu -o bfs_kernels          bin/ece496 bfs g.gr int32 initial_scheduler 0 --out d.bin   # per-kernel metrics
compute-sanitizer           bin/ece496 bfs g.gr int32 initial_scheduler 0 --out d.bin   # memory errors
perf record -g              bin/ece496 bfs g.gr int32 initial_scheduler 0 --out d.bin   # host-side CPU profile
cuda-gdb --args             bin/ece496 bfs g.gr int32 initial_scheduler 0 --out d.bin   # debugger (plain gdb works for host code)
```

For debugging, build with debug info (`-G` adds device debug info but makes
kernels much slower, so don't use that build for timing):

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-G" && cmake --build build-debug -j
```

Both builds write to the same `bin/ece496`, so rebuild with `build/` before
timing again.
