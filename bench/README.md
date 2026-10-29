# bench/bench.sh

One script that builds Galois (CPU + LonestarGPU), Gunrock, MBQ, MLMQ and
ADDS at the commits pinned at the top of `bench.sh`, runs every benchmark on every
prepared graph it finds, checks each result, and writes one CSV.

```bash
cd ece496
export GRAPH_DIR=/nfs/ug/groups/capstone_proj_2026F/input_graphs
bench/bench.sh check                 # 1. prerequisites, GPU, which graphs/files it sees
bench/bench.sh build                 # 2. clone + build everything (20-60 min)
RUNS=1 GRAPHS=germany-osm bench/bench.sh run   # 3. smoke test on a small graph
bench/bench.sh run                   # 4. the real thing (hours)
```

Run the long steps inside `tmux` (or `nohup ... &`) so a dropped SSH session
doesn't kill them. `run` resumes where it stopped: finished configurations
are listed in `done.txt` and skipped.

### Running on your own machine (not a UG lab machine)

Copy the prepared graph folders off the UG share first, then set these before
any `bench.sh` command:

```bash
cd ece496
export GRAPH_DIR="/home/james/capstone/input graphs/processed_graphs"   # your local graph folders
export WORK_DIR=/home/james/capstone/bench    # where src/, build/, results/ live; always set it
export LIBS="galois gunrock mbq adds"         # CUDA ≥ 12: no lonestargpu; mlmq hangs
bench/bench.sh check
bench/bench.sh build
bench/bench.sh run
```

If `WORK_DIR` is unset it defaults to `~/bench`, and `run` skips everything
as "not built" when the builds are elsewhere. With CUDA 11.x instead, use
`LIBS="galois lonestargpu mbq adds"` (Gunrock needs CUDA ≥ 12.4).

## Before you start

### Software on the benchmark machine

`bench.sh check` reports what it finds. You need:

| Need | For | Notes |
|---|---|---|
| git, make, internet access | all | Sources are cloned from GitHub; Gunrock's CMake also downloads dependencies. |
| CMake ≥ 3.24 | all | Gunrock's minimum. |
| g++ ≥ 10 (C++17) | all | MBQ was tested with GCC 12. |
| NVIDIA driver + CUDA toolkit ≥ 11.1 | MLMQ, ADDS | Without a GPU these are skipped and the CPU libraries still run. |
| CUDA toolkit **11.x** | LonestarGPU | Its bundled CUB uses texture references, removed in CUDA 12. With CUDA ≥ 12 the script builds Galois's CPU apps only and skips LonestarGPU. |
| CUDA toolkit ≥ 12.4 | Gunrock | Can be a separate, user-installed toolkit used for Gunrock only; see "Gunrock and CUDA 11" below. |
| LLVM development files (with CMake config, built with RTTI) | Galois, MBQ | Ubuntu: `llvm-dev`. MBQ used LLVM 14. Found automatically under `/usr/lib/llvm-*`, or set `LLVM_DIR`. LLVM ≥ 15 packages also need `libzstd-dev libcurl4-openssl-dev libedit-dev`; `check` tests this ("usable from CMake"). |
| Boost ≥ 1.58 (incl. `serialization`, `iostreams`) | Galois, MBQ, MLMQ, ADDS | Ubuntu: `libboost-all-dev`. Otherwise set `BOOST_ROOT`. |
| libfmt ≥ 4 | Galois, MBQ | If missing, `build` compiles fmt 9.1.0 into `WORK_DIR/deps` automatically (no root). |
| clang++ with OpenCilk 2.1 (optional) | PBBS's own MIS baseline | Without it, MBQ's MIS still runs; only the PBBS baseline is skipped. |

On a lab machine without `sudo`, either ask the admins, check
`module avail` for these packages, or install them into a conda env
(`conda install -c conda-forge cmake llvmdev boost fmt`) and point the
`LLVM_DIR`, `BOOST_ROOT` and `fmt_DIR` variables at it.

### Gunrock and CUDA 11

Gunrock needs CUDA ≥ 12.4. If the machine has an older toolkit (e.g. 11.8),
install a newer one in your home directory. This needs no root, but the GPU
**driver** must support it: `nvidia-smi` must show `CUDA Version: 12.4` or
higher (driver ≥ 550). If the driver is older, ask the admins, or skip Gunrock
(`LIBS` without `gunrock`).

```bash
mkdir -p ~/tmp && cd ~/tmp     # ~4.5 GB download, ~8 GB while installing
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.4 \
   --no-man-page --override --tmpdir=$HOME/tmp
rm cuda_12.4.1_550.54.15_linux.run
export GUNROCK_CUDA_HOME=$HOME/cuda-12.4   # used for Gunrock only
bench/bench.sh check                       # "nvcc for Gunrock: 12.4"
```

Keep the system CUDA on `PATH` for everything else: Galois's GPU code ships
an old CUB library that is safest with the toolkit it was written for.

**Host compiler for CUDA 11:** CUDA 11.x accepts at most GCC 11 or Clang 14
as the compiler `nvcc` calls. If the default `g++` is newer, the script
uses `g++-11`/`g++-10`, or `clang++` 14 or older, for the CUDA builds.
`check` shows its choice ("host compiler for nvcc"). Override with
`CUDA_HOST_CXX=/path/to/compiler`.

### Graph folder layout (GRAPH_DIR)

- `GRAPH_DIR` holds **one folder per graph**, as written by
  `graphs/prepare.sh`. Folders without an `info.txt` are ignored.
- The script reads the graph's real name from `info.txt` (`name:`), so a
  folder may be renamed (`hyperlink2012-host-new/` is fine), but **the files
  inside must keep their names** (`hyperlink2012-host.gr`, ...).
- Each folder must have `sources.txt`, `pairs.txt` and `reference.txt`.
  A folder missing one of them is skipped.
- Missing graph files are fine: anything that needs a file the folder
  doesn't have is skipped and listed in `skipped.txt` (for example, with only
  `.gr` + `.sgr` for Hyperlink, Gunrock, PageRank-pull and MBQ MIS are
  skipped). [Which file each run needs](#which-file-each-run-needs) lists them.
- Spaces in paths are fine.
- **NFS:** graphs are read from `GRAPH_DIR` on every run. Load time is not
  part of any reported time, but loading Hyperlink's 17 GB over NFS for every
  run makes the sweep much longer. If the machine has a large local disk,
  copy the graphs there and point `GRAPH_DIR` at the copy.

### Which file each run needs

`file` = the file passed, from that graph's folder; `none` = the library has
no such algorithm; `n/a` = it can't run on that graph. Short names: `.gr` means
`<graph>.gr`, etc.

| Benchmark | Library → program | usa-road-d | germany-osm | livejournal | friendster | hyperlink2012-host |
|---|---|---|---|---|---|---|
| BFS | Galois CPU → `bfs-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| | LonestarGPU → `bfs` | `.gr` | `.gr` | `.gr` | `.gr` ¹ | `.gr` ¹ |
| | Gunrock → `bfs` | `.mtx` | `.mtx` | `.mtx` | n/a ² | `.mtx` ¹ |
| | MBQ → `galois/bfs-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| SSSP | Galois CPU → `sssp-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| | LonestarGPU → `sssp` | `.gr` | `.gr` | `.gr` | `.gr` ¹ | `.gr` ¹ |
| | Gunrock → `sssp` | `.mtx` | `.mtx` | `.mtx` | n/a ² | `.mtx` ¹ |
| | MBQ → `galois/sssp-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| | MLMQ → `SSSP_adaptive/main` (GPU) | `.gr` ⁴ | `.gr` ⁴ | `.gr` ⁴ | n/a ² | `.gr` ¹ ⁴ |
| | ADDS → `ADDS/ads_int/sssp` (GPU) | `.gr` | `.gr` | `.gr` | `.gr` ¹ | `.gr` ¹ |
| PageRank | Galois CPU → `pagerank-pull-cpu` | `.gr` | `.gr` | `.tgr` | `.gr` | `.tgr` |
| | Galois CPU → `pagerank-push-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| | LonestarGPU → `pagerank` | `.gr` | `.gr` | `.gr` | `.gr` ¹ | `.gr` ¹ |
| | Gunrock → `pr` | `.mtx` | `.mtx` | `.mtx` | n/a ² | `.mtx` ¹ |
| | MBQ → `galois/pagerank-push-cpu` | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| PPSP | MBQ → `galois/ppsp-cpu` (others: none) | `.gr` | `.gr` | `.gr` | `.gr` | `.gr` |
| A* | MBQ → `astar/astar` (others: none) | `.bin` | `.bin` | n/a ³ | n/a ³ | n/a ³ |
| MIS | Galois CPU → `maximal-independentset-cpu` | `.gr` | `.gr` | `.sgr` | `.gr` | `.sgr` |
| | LonestarGPU → `maximal-independentset` | `.gr` | `.gr` | `.sgr` | `.gr` ¹ | `.sgr` ¹ |
| | Gunrock: none; `color` is the closest | `.mtx` | `.mtx` | `.sym.mtx` | n/a ² | n/a ² |
| | MBQ → `pbbs/` `MIS` | `.sym.ag` | `.sym.ag` | `.sym.ag` | `.sym.ag` | `.sym.ag` |
| Residual BP | MBQ → `rbp/rbp` (others: none) | no graph file; it generates its own input | | | | |

¹ Needs a GPU with about 40 GB or more; it won't fit on a 12 GB card.
² Over the 32-bit `int` edge limit (2.147B edges) of Gunrock and MLMQ.
³ Needs node coordinates, which only the road graphs have.
⁴ MLMQ always starts at node 0 (`int src = 0;` in `main.cu`). That is the
main source for USA, Germany and LiveJournal, so they need no change. On
Hyperlink node 0 has a single out-edge, so edit that line to the main source
first.

- BFS and SSSP read the start node from `sources.txt`; PPSP and A* read
  (source, target) from `pairs.txt`; `reference.txt` checks every result.
- CPU runs load the whole file into RAM, so the machine needs more RAM than
  the file size: about 30 GB+ for Friendster and 17 GB+ for Hyperlink `.gr`.
- A directed graph needs `.tgr` (PageRank-pull) and `.sgr` (MIS); an
  undirected graph's `.gr` already holds both directions.

### Where builds and results go (WORK_DIR)

Default `~/bench`: `src/` (clones), `build/`, `logs/` (build logs) and
`results/<hostname>/`. Builds take a few GB. Prefer a local disk over an NFS
home with a quota; set `WORK_DIR=/some/local/dir`.

### Memory limits

Before each run the script estimates the graph's size from its `.gr`/`.sgr`
file and skips it if it won't fit: GPU runs need it under 90% of GPU memory,
CPU runs under 80% of RAM. On a 12 GB GPU only the three small graphs run on
the GPU libraries; Friendster needs about 48 GB of RAM or more. Override with
`SKIP_SIZE_CHECK=1`.

## Settings

All are environment variables, e.g. `THREADS="1 8 16" RUNS=10 bench/bench.sh run`.

| Variable | Default | Meaning |
|---|---|---|
| `GRAPH_DIR` | `/nfs/ug/groups/capstone_proj_2026F/input_graphs` | Graph folders |
| `WORK_DIR` | `~/bench` | Sources, builds, results |
| `OUT_DIR` | `$WORK_DIR/results/<hostname>` | Results of this sweep |
| `THREADS` | `1 <physical cores>` | CPU thread counts to sweep |
| `RUNS` | `5` | Timed runs per configuration (after 1 warm-up) |
| `TIMEOUT` | `1800` | Seconds per run before it's killed |
| `LIBS` | `galois lonestargpu gunrock mbq mlmq adds` | Libraries to run |
| `BENCHES` | `bfs sssp pr ppsp astar mis rbp` | Benchmarks to run; add `color` for Gunrock's coloring (its stand-in for MIS; slow, see below) |
| `GRAPHS` | all | Folder names to run, e.g. `"usa-road-d livejournal"` |
| `SOURCES` | `main` | `all`: BFS/SSSP from every source in `sources.txt` that reaches at least half the graph |
| `PAIRS` | `1 11` | Lines of `pairs.txt` for PPSP and A* (1 = a random pair, 11 = a far pair) |
| `MBQ_ALGOS` | `MQBucket OBIM` | MBQ schedulers: any of `MQBucket MQ OBIM PMOD` |
| `ASTAR_VARIANTS` | `orig fixed` | MBQ A* with its published heuristic, and with the haversine fix |
| `PR_TOL` | `1e-3` | PageRank tolerance (Galois CPU, MBQ) |
| `PR_MAX_ITER` | `1000` | LonestarGPU PageRank iteration cap |
| `LSGPU_SSSP_DELTA` | LonestarGPU default (10000) | LonestarGPU SSSP bucket width `-d` |
| `GPU` | `0` | GPU to use |
| `CUDA_ARCH` | detected | e.g. `80` (A100), `86` (RTX 30xx), `90` (H100); `none` = CPU only |
| `JOBS` | `nproc` | Parallel build jobs |
| `GUNROCK_CUDA_HOME` | (nvcc on `PATH`) | CUDA ≥ 12.4 used only to build Gunrock |
| `CUDA_HOST_CXX` | auto | Host compiler for the system `nvcc` (see above) |
| `LLVM_DIR`, `BOOST_ROOT`, `fmt_DIR` | auto | Where to find those libraries if not in standard places |
| `DRY_RUN` | `0` | `1` prints every command without running anything |

`bench.sh build galois mbq` builds just those libraries.

## Output

In `$OUT_DIR`:

- `results.csv`: one row per run:
  `library,algorithm,benchmark,graph,source,target,threads,param,run,time_ms,valid,log`.
  Run 0 is the warm-up, which is also the run that gets checked; runs 1..N are
  timed and not re-checked.
- `summary.csv`: median and minimum of the timed runs per configuration, plus
  the warm-up's check result. Regenerate with `bench/bench.sh summary`.
- `logs/`: the full output of every run.
- `skipped.txt`: everything skipped, with the reason.
- `done.txt`: finished configurations. Delete a line (or the file) to rerun.

`valid` values: `ok` = checked and correct; `-` = this library/benchmark has
no check, or a timed run; `MISMATCH(...)`, `FAIL(rc=N)`, `TIMEOUT` = problem;
`accmax=` = residual BP's largest error versus the serial run.

How each result is checked:

| Library | Check |
|---|---|
| Galois CPU BFS/SSSP, MBQ BFS/SSSP | the program's own verifier ("Verification successful") |
| LonestarGPU BFS/SSSP | its output distances vs `reference.txt` (nodes reached, max, sum) |
| Gunrock BFS/SSSP, color | `--validate` / built-in CPU check ("Number of errors : 0") |
| MLMQ, ADDS | built-in CPU Dijkstra ("mlmq sssp correct!") |
| MBQ PPSP | destination distance vs `reference.txt` |
| MBQ A* | path identical to the Serial run of the same binary. Expect `PATH-DIFFERS` for `astar-orig`: its heuristic overestimates, so the path depends on run order (on germany-osm it always differed; `astar-fixed` always matched) |
| MBQ MIS | PBBS `MISCheck` |
| Galois PR/MIS, LonestarGPU PR/MIS, Gunrock PR | no check (a crash still shows as `FAIL`) |

## Things to know

- **What the times mean differs by library.** Galois CPU: `Timer_0` (the
  algorithm only). LonestarGPU: `Total time` (kernel phase). Gunrock: `GPU
  Elapsed Time`. MBQ: `runtime_ms` (whole milliseconds, so small graphs show
  coarse times). PBBS MIS: each timed round inside one process. ADDS: the mean
  of the 8 samples it runs inside each process. MLMQ: `Elapse time`.
- **PageRank** convergence rules differ between libraries; record `PR_TOL`
  and `PR_MAX_ITER` with the results.
- **Gunrock SSSP on USA** uses `float` distances, exact only below 2^24, and
  USA distances reach 47M, so its check may report small rounding errors there.
- **Parameters:** MBQ's deltas and batch sizes come from MBQ's own run
  scripts (USA, LiveJournal, Twitter → Friendster, Hyperlink). Germany uses
  the USA values with deltas 3 lower, because its weights are metres rather
  than 0.1 m. Galois CPU SSSP uses MBQ's OBIM delta for the same graph.
  Sweep `-delta` if you have time.
- **Hyperlink source 78748048** (4th in `sources.txt`) reaches only 3
  nodes; `SOURCES=all` leaves it out automatically.
- **Gunrock runs once per process** (`-n 1`), because it prints only the last
  run's time; every run reloads the `.mtx`.
- **Residual BP** has no input graph; it is recorded as graph `ising-1000`.
- **Gunrock `color`** (the closest thing Gunrock has to MIS) is opt-in:
  `BENCHES="... color"`. Here it ran for over 500 s on germany-osm without
  finishing.

### Changes the script makes to the libraries

Only in its own clones under `WORK_DIR/src`; your checkouts are never touched.

- **Gunrock:** configured with `-DESSENTIALS_NVIDIA_BACKEND=ON` (this version
  defaults to the AMD backend), and with `thrust/distance.h` and
  `thrust/tuple.h` pre-included, which CUDA 13 no longer pulls in indirectly.
  Its CCCL dependency is pinned to v3.0.3; upstream fetches CCCL's moving
  `main` branch.
  PageRank's tolerance is read from `GUNROCK_PR_TOL` and divided by the node
  count, so it matches Galois/MBQ's per-node `PR_TOL`. The example hard-codes
  1e-6 on ranks that sum to 1, which stopped Germany after about two
  iterations (4 ms instead of 176 ms).
- **MLMQ:** reads the source node from `MLMQ_SRC` (it was hard-coded to 0).
  `GPU_MEMORY` is set to `INT_MAX`: the published `2.0 * 1024^3` overflows
  `int` with current compilers, making every queue `cudaMalloc` fail.
- **ADDS:** `cub::LaneId()` (removed in CUDA 13) is replaced with the same
  PTX register read, and its summary file is written to the run directory
  instead of a hard-coded `/workspace/...` path that crashed it.
- **MLMQ and ADDS:** built with your GPU's arch (their Makefiles hard-code
  `sm_86`); runs get an unlimited stack, since both keep a per-node array on
  the stack.
- **MBQ A\*:** a second copy, `astar-fixed`, with the haversine fix
  (`sin(Δ/2)`); the original is run too (`ASTAR_VARIANTS`).
- **MBQ MIS:** built with g++ (a distro clang may pick a GCC without C++
  headers), without the OpenCilk-only checker dependency; the checker is
  compiled directly in sequential mode.
- **Galois and MBQ's Galois fork:** compiled with `-include cstdint` (GCC 13+
  no longer includes it indirectly). MBQ's fork builds fmt from fmt's
  `master` branch, which no longer exists; it gets a pinned fmt 9.1.0 copy.

### What has been tested

Tested on the prep machine (RTX 3080 Ti, CUDA 13.3, WSL2):
- `check`, and `DRY_RUN=1` on all graphs, against the sources of every
  library.
- The time parsers and validators, on output in each library's format.
- **Gunrock**, built and run for real. BFS and SSSP were checked correct on
  usa-road-d, germany-osm and livejournal, and PageRank ran on all three.
  Road graphs are slow on Gunrock (6-25 s), as expected for its BFS-style
  frontiers on long paths.
- **MLMQ and ADDS**, built and run for real. ADDS gave correct SSSP on USA
  and Germany.
- **MBQ A\*, MIS and residual BP**, built and run for real on germany-osm
  with 1 and 8 threads. MIS passed `MISCheck`, the fixed A\* matched its
  Serial path, and RBP's error versus serial was about 0.0017.
- **Galois CPU and MBQ's Galois fork** (Ubuntu 26.04: GCC 15, LLVM 21, Boost
  1.90), built and run on germany-osm and livejournal with 1 and 8 threads.
  Every BFS, SSSP and PPSP result was checked correct (MQBucket, OBIM, Galois
  deltaStep and SyncTile); PageRank pull/push and MIS completed.
- **LonestarGPU** is not tested: it doesn't build with CUDA 13 (see the
  prerequisites). It should build on a machine with CUDA 11.8.

Two problems on this machine:
- MLMQ hung on every graph, and ADDS on LiveJournal. Both run GPU kernels
  that spin-wait on each other, which may not work on WSL2's GPU driver. They
  are untested on native Linux. A hang costs one `TIMEOUT` per graph; set
  `LIBS` without `mlmq` to avoid it.
- If a build fails, its log is in `$WORK_DIR/logs/build-<lib>.log`; the
  other libraries still build and run.
