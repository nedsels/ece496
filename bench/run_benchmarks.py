#!/usr/bin/env python3
"""Run benchmarks on input graphs against one scheduler.

Builds bin/ece496, runs every (benchmark, graph, source) combination with the
given scheduler (selected at runtime by the binary's <scheduler> argument),
checks the answers, and writes a CSV.

  bfs, sssp   one run per source; checked against reference.txt
  a_star      one run per (source, target) pair on the graph file with node
              coordinates (<name>.agr, or a .gr with coordinates); checked
              against astar_reference.txt
  pr          one run per graph; ranks checked against the graph (needs numpy)

Examples (from the repo root):
  bench/run_benchmarks.py --list                     # graphs found, and why some are skipped
  bench/run_benchmarks.py                            # every benchmark on every graph that fits
  bench/run_benchmarks.py -g germany-osm livejournal -b bfs --runs 5
  GRAPH_DIR="/path/to/processed_graphs" bench/run_benchmarks.py -b bfs sssp
  bench/run_benchmarks.py -g germany-osm -b a_star --sources 5       # first 5 pairs
  bench/run_benchmarks.py -g germany-osm -b a_star --source 4665057 --dest 436940

Graphs are looked up in graphs/inputs/, the group folder
/nfs/ug/groups/capstone_proj_2026F/input_graphs (on the UG machines), and
every --graph-dir (or $GRAPH_DIR, separated by ':'). Both layouts are found:
  <dir>/<name>.gr                 flat file, named <name>
  <dir>/<name>.agr                the same graph with coordinates, for a_star
  <dir>/<name>/<anything>.gr      prepared folder (bench.sh / graph_prep), named
                                  after the folder when it has info.txt or
                                  reference.txt; its sources.txt, pairs.txt,
                                  reference.txt and astar_reference.txt are
                                  used for inputs and checks
"""

import argparse
import csv
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np  # only needed to check PageRank
except ImportError:
    np = None

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BIN = REPO_ROOT / "bin" / "ece496"
# The group's big graphs on the UG machines; only searched when it exists.
GROUP_GRAPH_DIR = Path("/nfs/ug/groups/capstone_proj_2026F/input_graphs")
DEFAULT_GRAPH_DIRS = [REPO_ROOT / "graphs" / "inputs"]
if GROUP_GRAPH_DIR.is_dir():
    DEFAULT_GRAPH_DIRS.append(GROUP_GRAPH_DIR)
DEFAULT_RESULTS_DIR = REPO_ROOT / "bench" / "results"

INT32_MAX = 2**31 - 1  # CSRGraph::index_type is int32_t
# What the binary writes for an unreachable vertex: BFS levels are always
# uint32, the others use the largest value of the weight type.
UNREACHED = {"int32": INT32_MAX, "uint32": 0xFFFFFFFF}

# Benchmarks this script knows about. "inputs" says what one run needs (see
# make_jobs) and "ref_cols" are the reference.txt columns (reached, max, sum)
# that --out distances are checked against. Benchmarks the binary doesn't
# implement yet are reported and skipped, so a source-based one only has to be
# added to main.cu to start running here.
BENCHMARKS = {
    "bfs": {"inputs": "source", "ref_cols": (1, 2, 3)},
    "sssp": {"inputs": "source", "ref_cols": (4, 5, 6)},
    "a_star": {"inputs": "pair"},
    "pr": {"inputs": "pagerank"},
    "cc": {"inputs": "source"},
    "mis": {"inputs": "source"},
}
ALIASES = {"astar": "a_star", "pagerank": "pr"}

CSV_FIELDS = ["scheduler", "benchmark", "graph", "source", "target", "params",
              "run", "status", "kernel_ms", "load_ms", "wall_s", "check",
              "detail"]


@dataclass
class GraphFile:
    path: Path
    nodes: int = 0
    edges: int = 0
    weight_size: int = 0
    has_coords: bool = False  # needed by a_star, rejected by the others
    error: str = ""


@dataclass
class Graph:
    name: str
    plain: GraphFile = None   # without coordinates
    coords: GraphFile = None  # with coordinates
    sources: list = field(default_factory=list)
    pairs: list = field(default_factory=list)      # [(source, target)]
    reference: dict = field(default_factory=dict)  # source -> list of ints
    astar_reference: dict = field(default_factory=dict)  # (s, t) -> distance

    def file_for(self, benchmark):
        pair = BENCHMARKS.get(benchmark, {}).get("inputs") == "pair"
        return self.coords if pair else self.plain


@dataclass
class Job:
    """One configuration to run: its arguments and how to check its output."""
    benchmark: str
    graph: Graph
    cli_args: list        # arguments after <scheduler>
    tag: str
    source: object = ""   # columns in the CSV
    target: object = ""
    params: str = ""
    check: object = None  # out_path -> (check, detail), None for no check


# ---------------------------------------------------------------- graphs

def read_gr_info(path):
    """Header of a Galois .gr v1 file, and whether it has the coordinate
    trailer (u64 kind == 2 after the weights, see CSRGraph::readFromGR)."""
    info = GraphFile(path)
    try:
        with open(path, "rb") as f:
            header = f.read(32)
            if len(header) < 32:
                info.error = "file shorter than the .gr header"
                return info
            version, info.weight_size, info.nodes, info.edges = struct.unpack(
                "<4Q", header)
            if version != 1:
                info.error = f".gr version {version}, expected 1"
                return info
            f.seek(32 + info.nodes * 8
                   + (info.edges + (info.edges & 1)) * 4
                   + info.edges * info.weight_size)
            kind = f.read(8)
            info.has_coords = (len(kind) == 8
                               and struct.unpack("<Q", kind)[0] == 2)
    except OSError as e:
        info.error = f"unreadable: {e}"
    return info


def read_ints_file(path):
    """Rows of ints from a text file, skipping blank and '#' lines.
    "NA" (no reference value) becomes None."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([None if x == "NA" else int(x) for x in line.split()])
    return rows


def load_graph(name, paths, meta_dir=None):
    """A graph from its .gr and/or .agr file."""
    g = Graph(name)
    for path in paths:
        info = read_gr_info(path)
        if info.has_coords or path.suffix == ".agr":
            g.coords = g.coords or info
        else:
            g.plain = g.plain or info
    if meta_dir is not None:
        if (meta_dir / "sources.txt").is_file():
            g.sources = [row[0] for row in
                         read_ints_file(meta_dir / "sources.txt")]
        if (meta_dir / "pairs.txt").is_file():
            g.pairs = [tuple(row[:2]) for row in
                       read_ints_file(meta_dir / "pairs.txt")]
        if (meta_dir / "reference.txt").is_file():
            # The pairs section repeats sources with other columns; keep only
            # the per-source rows, which come first.
            for row in read_ints_file(meta_dir / "reference.txt"):
                if len(row) >= 7:
                    g.reference.setdefault(row[0], row)
        if (meta_dir / "astar_reference.txt").is_file():
            for row in read_ints_file(meta_dir / "astar_reference.txt"):
                g.astar_reference[(row[0], row[1])] = row[2]
            if not g.pairs:
                g.pairs = list(g.astar_reference)
    if not g.sources:
        g.sources = [0]
    return g


def graph_files(d):
    """{stem: [.gr and .agr files with that stem]} in a folder."""
    stems = {}
    for f in sorted(d.glob("*.gr")) + sorted(d.glob("*.agr")):
        stems.setdefault(f.stem, []).append(f)
    return stems


def discover_graphs(dirs):
    graphs = {}
    seen = set()

    def add(name, paths, meta_dir=None):
        paths = [f for f in paths if f.resolve() not in seen]
        if not paths:
            return
        seen.update(f.resolve() for f in paths)
        if name in graphs:
            kept = graphs[name].plain or graphs[name].coords
            print(f"warning: {paths[0]} has the same name as {kept.path}; "
                  "keeping the first", file=sys.stderr)
        else:
            graphs[name] = load_graph(name, paths, meta_dir)

    for d in dirs:
        if not d.is_dir():
            print(f"warning: graph dir {d} not found", file=sys.stderr)
            continue
        for stem, files in graph_files(d).items():
            add(stem, files)
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            stems = graph_files(sub)
            prepared = ((sub / "info.txt").is_file()
                        or (sub / "reference.txt").is_file())
            for stem, files in stems.items():
                if prepared:
                    name = sub.name if len(stems) == 1 else f"{sub.name}/{stem}"
                    add(name, files, meta_dir=sub)
                else:
                    add(stem, files)
    return graphs


def mem_available_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def gpu_free_bytes():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        return int(out.strip().splitlines()[0]) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gib(n):
    return f"{n / 2**30:.1f} GiB"


def skip_reason(g, benchmark, host_free, gpu_free, ignore_size=False):
    """Why this graph can't run this benchmark with the current binary, or "".

    Estimates mirror CSRGraph::readFromGR (reads the whole file, then builds
    int32 CSR arrays) and main.cu (scheduler takes 1/8 of free VRAM after the
    graph and working arrays are on the device).
    """
    f = g.file_for(benchmark)
    if f is None:
        if g.plain is not None:
            return "no file with node coordinates (.agr)"
        return "only has a file with node coordinates"
    if f.error:
        return f.error
    if f.has_coords != (f is g.coords):
        return "no node coordinates in the .agr file"
    if f.nodes > INT32_MAX or f.edges > INT32_MAX:
        return (f"{f.nodes:,} nodes / {f.edges:,} edges exceed the int32 "
                "index_type")
    if f.weight_size not in (0, 4):
        return f"{f.weight_size}-byte weights; binary expects 4"
    if ignore_size:
        return ""
    # row starts, node data, destinations, weights
    csr = ((f.nodes + 1) * 4 + f.nodes * (8 if f.has_coords else 4)
           + f.edges * 4 + f.edges * 4)
    host_need = f.path.stat().st_size + csr
    if host_free is not None and host_need > host_free:
        return f"needs ~{gib(host_need)} host RAM, {gib(host_free)} available"
    # One 4-byte value per node; PageRank keeps ranks and residuals.
    values = f.nodes * (8 if benchmark == "pr" else 4)
    gpu_need = (csr + values) * 8 // 7
    if gpu_free is not None and gpu_need > gpu_free:
        return f"needs ~{gib(gpu_need)} VRAM, {gib(gpu_free)} free"
    return ""


# ---------------------------------------------------------------- build/run

def build(cuda_arch):
    cmd = ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)]
    if cuda_arch:
        cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    cmd = ["cmake", "--build", str(BUILD_DIR), "-j", "--target", "ece496"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def binary_supports(binary, benchmark, scheduler, weight_type):
    """main.cu rejects unknown benchmarks before loading the graph, so a run
    on a missing file tells us whether the benchmark is implemented."""
    proc = subprocess.run(
        [str(binary), benchmark, "/nonexistent.gr", weight_type, scheduler,
         "0", "1"], capture_output=True, text=True, timeout=60)
    if "Unknown scheduler" in proc.stderr:
        raise RuntimeError(f"{binary.name} rejected scheduler {scheduler!r}")
    if "Unsupported weight type" in proc.stderr:
        raise RuntimeError(f"{binary.name} rejected weight type "
                           f"{weight_type!r}")
    return "Unsupported benchmark" not in proc.stderr


def read_u32(path):
    values = array("I")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def summarize_distances(out_path, unreached):
    """(reached, max, sum) of an --out distances file, the same three
    numbers reference.txt and Lonestar's bfs-cpu / sssp-cpu report."""
    dist = read_u32(out_path)
    num_unreached = dist.count(unreached)
    reached = len(dist) - num_unreached
    total = sum(dist) - num_unreached * unreached
    biggest = max((d for d in dist if d != unreached), default=0)
    return reached, biggest, total


def reference_for(benchmark, graph, source):
    """The (reached, max, sum) reference.txt expects, or None if there is
    nothing to check against."""
    ref_cols = BENCHMARKS.get(benchmark, {}).get("ref_cols")
    row = graph.reference.get(source)
    if ref_cols is None or row is None:
        return None
    want = tuple(row[c] for c in ref_cols)
    return None if None in want else want


def compare_summary(got, want):
    """(check, detail) for the CSV."""
    if got == want:
        return "pass", ""
    return "FAIL", (f"reached/max/sum got {got[0]}/{got[1]}/{got[2]}, "
                    f"want {want[0]}/{want[1]}/{want[2]}")


def check_pair_distance(out_path, want, unreached):
    """Compare the one distance A* writes to astar_reference.txt."""
    got = list(read_u32(out_path))
    # graph_prep writes UINT64_MAX when there is no path
    want = unreached if want >= 2**63 else want
    if got == [want]:
        return "pass", ""
    return "FAIL", f"distance got {got}, want {want}"


@lru_cache(maxsize=1)
def read_csr(path):
    """(row_end, dst) of a .gr file as numpy arrays."""
    nodes, edges = np.fromfile(path, "<u8", 4)[2:].astype(int)
    row_end = np.fromfile(path, "<u8", nodes, offset=32).astype(np.int64)
    return row_end, np.memmap(path, "<u4", "r", 32 + 8 * int(nodes),
                              (int(edges),))


def check_ranks(out_path, graph_path, damping, epsilon):
    """Check PageRank from its definition, so no reference is needed.

    The kernel keeps rank = (1-d)/n + d * sum(rank[u] / deg[u] over in-edges)
    - residual for every node, and stops once all residuals are below epsilon.
    So recompute the residuals from the ranks and require 0 <= residual <=
    epsilon, up to float32 rounding in the ranks. As in the kernel, dangling
    nodes don't pass anything on.
    """
    if np is None:
        return "-", "ranks not checked: numpy is not installed"
    row_end, dst = read_csr(graph_path)
    nodes = len(row_end)
    ranks = np.fromfile(out_path, "<f4").astype(np.float64)
    if len(ranks) != nodes:
        return "FAIL", f"{len(ranks)} ranks for {nodes} nodes"
    if not np.isfinite(ranks).all():
        return "FAIL", f"{(~np.isfinite(ranks)).sum()} ranks are not finite"
    degree = np.diff(row_end, prepend=0)
    share = np.divide(ranks, degree, out=np.zeros(nodes), where=degree > 0)
    incoming = np.zeros(nodes)
    # ~16M edges at a time, to bound memory
    first = 0
    while first < nodes:
        start = row_end[first - 1] if first else 0
        last = max(first + 1, int(np.searchsorted(row_end, start + (1 << 24),
                                                  side="right")))
        part = np.repeat(share[first:last], degree[first:last])
        incoming += np.bincount(dst[start:row_end[last - 1]], weights=part,
                                minlength=nodes)
        first = last
    residual = (1 - damping) / nodes + damping * incoming - ranks
    slack = 1e-4 * np.abs(ranks)
    # The kernel compares in float32, so allow epsilon to round too
    bad = (residual > epsilon * (1 + 1e-6) + slack) | (residual < -slack)
    if not bad.any():
        return "pass", ""
    worst = int(np.argmax(np.where(bad, np.abs(residual), -1)))
    return "FAIL", (f"{int(bad.sum())} of {nodes} ranks not converged, worst "
                    f"residual {residual[worst]:.3g} at node {worst} "
                    f"(epsilon {epsilon:.3g})")


def check_in_range(f, *vertices):
    """Raise ValueError if a vertex isn't in graph file f."""
    for v in vertices:
        if not 0 <= v < f.nodes:
            raise ValueError(f"vertex {v} is outside the graph "
                             f"({f.nodes:,} nodes)")


def first_n(items, args):
    """The first --sources items (all of them for 0)."""
    return items[:args.sources] if args.sources > 0 else items


def pick_sources(g, args):
    """--source, or the first --sources of g's sources.txt. Raises
    ValueError if one is outside the graph."""
    sources = ([args.source] if args.source is not None
               else first_n(g.sources, args))
    check_in_range(g.plain, *sources)
    return sources


def job_name(benchmark, g):
    return f"{benchmark}_{g.name.replace('/', '_')}"


def make_jobs(benchmark, g, args):
    """The runs of one benchmark on one graph. Raises ValueError when the
    inputs it needs are missing."""
    inputs = BENCHMARKS.get(benchmark, {}).get("inputs", "source")
    name = job_name(benchmark, g)
    unreached = UNREACHED.get(args.weight_type)
    f = g.file_for(benchmark)

    if inputs == "pagerank":
        epsilon = args.pr_tol / f.nodes
        check = lambda out: check_ranks(out, f.path, args.damping, epsilon)
        return [Job(benchmark, g, [repr(args.damping), repr(epsilon)], name,
                    params=f"damping={args.damping} epsilon={epsilon:.6g}",
                    check=check)]

    if inputs == "pair":
        if args.source is not None or args.dest is not None:
            if args.source is None or args.dest is None:
                raise ValueError("needs both --source and --dest")
            pairs = [(args.source, args.dest)]
        else:
            pairs = first_n(g.pairs, args)
        if not pairs:
            raise ValueError("no pairs.txt; give --source and --dest")
        jobs = []
        for s, t in pairs:
            check_in_range(f, s, t)
            want = g.astar_reference.get((s, t))
            check = None if want is None else (
                lambda out, want=want: check_pair_distance(out, want,
                                                           unreached))
            jobs.append(Job(benchmark, g, [str(s), str(t)], f"{name}_{s}_{t}",
                            source=s, target=t, check=check))
        return jobs

    if benchmark == "bfs":
        unreached = UNREACHED["uint32"]
    jobs = []
    for s in pick_sources(g, args):
        want = reference_for(benchmark, g, s)
        check = None
        if want is not None:
            check = lambda out, want=want: compare_summary(
                summarize_distances(out, unreached), want)
        jobs.append(Job(benchmark, g, [str(s)], f"{name}_{s}", source=s,
                        check=check))
    return jobs


def new_row():
    return {"kernel_ms": "", "load_ms": "", "check": "", "detail": ""}


def run_logged(cmd, timeout, log_path, row):
    """Run cmd with a timeout, write its output to log_path and fill in
    row's wall_s / status / detail. Returns the CompletedProcess when it
    exited 0, else None (status is then "timeout" or "error")."""
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as e:
        row["wall_s"] = f"{time.monotonic() - start:.2f}"
        row["status"] = "timeout"
        row["detail"] = f"killed after {timeout}s"
        log_path.write_text(f"$ {' '.join(cmd)}\n{e.stdout or ''}\n"
                            f"{e.stderr or ''}")
        return None
    row["wall_s"] = f"{time.monotonic() - start:.2f}"
    log_path.write_text(f"$ {' '.join(cmd)}\nexit {proc.returncode}\n"
                        f"--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
    if proc.returncode != 0:
        row["status"] = "error"
        lines = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        row["detail"] = lines[-1] if lines else f"exit {proc.returncode}"
        return None
    row["status"] = "ok"
    return proc


def run_one(binary, job, args, out_path, log_path):
    cmd = [str(binary), job.benchmark,
           str(job.graph.file_for(job.benchmark).path), args.weight_type,
           args.scheduler, *job.cli_args, "--out", str(out_path), "--time"]
    row = new_row()
    proc = run_logged(cmd, args.timeout, log_path, row)
    if proc is None:
        return row
    m = re.search(r'"kernel_ms":\s*([0-9.eE+-]+).*"load_ms":\s*([0-9.eE+-]+)',
                  proc.stdout)
    if m:
        row["kernel_ms"], row["load_ms"] = m.group(1), m.group(2)

    if args.no_check or job.check is None:
        row["check"] = "-"
    elif not out_path.is_file():
        row["check"], row["detail"] = "FAIL", "no --out file written"
    else:
        row["check"], row["detail"] = job.check(out_path)
    return row


# ---------------------------------------------------------------- main

def split_list(values):
    """Accept both `-g a b` and `-g a,b`."""
    return [v for item in values or [] for v in item.split(",") if v]


def common_parser(description, benchmark_names, default_benchmarks):
    """Options shared by run_benchmarks.py and run_galois.py."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-g", "--graphs", nargs="*", metavar="NAME",
                   help="graphs to run (default: every graph that fits)")
    p.add_argument("-b", "--benchmarks", nargs="*", metavar="NAME",
                   help=f"benchmarks to run, from {', '.join(benchmark_names)} "
                        f"(default: {default_benchmarks})")
    p.add_argument("--graph-dir", action="append", type=Path, default=[],
                   help="extra directory to search for graphs (repeatable; "
                        "also read from $GRAPH_DIR)")
    p.add_argument("--sources", type=int, default=1, metavar="N",
                   help="first N sources from each graph's sources.txt, or "
                        "pairs from pairs.txt for a_star (default: 1; 0 for "
                        "all)")
    p.add_argument("--source", type=int, metavar="V",
                   help="use this source vertex on every graph instead")
    p.add_argument("--runs", type=int, default=3,
                   help="repetitions per configuration (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=600,
                   help="seconds before a run is killed (default: %(default)s)")
    p.add_argument("--no-build", action="store_true",
                   help="use the existing binaries instead of rebuilding")
    p.add_argument("--no-check", action="store_true",
                   help="don't check results")
    p.add_argument("--force", action="store_true",
                   help="run graphs even if they look too big")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--list", action="store_true",
                   help="list graphs and whether they fit, then exit")
    return p


def parse_args():
    p = common_parser(__doc__, list(BENCHMARKS),
                      "those the binary implements")
    p.add_argument("-s", "--scheduler", default="initial_scheduler",
                   help="scheduler to run (default: %(default)s)")
    p.add_argument("--dest", type=int, metavar="V",
                   help="a_star target vertex, together with --source")
    p.add_argument("--damping", type=float, default=0.85,
                   help="PageRank damping factor (default: %(default)s)")
    p.add_argument("--pr-tol", type=float, default=1e-3, metavar="T",
                   help="PageRank tolerance; the epsilon passed to the binary "
                        "is T / nodes, since its ranks sum to at most 1 "
                        "(default: %(default)s)")
    p.add_argument("--weight-type", default="int32",
                   choices=["int32", "uint32"])
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN,
                   help="binary to run (default: bin/ece496); implies "
                        "--no-build if not the default")
    p.add_argument("--cuda-arch", help="CMAKE_CUDA_ARCHITECTURES, e.g. 75")
    p.add_argument("--keep-output", action="store_true",
                   help="keep each run's output file")
    return p.parse_args()


def select_graphs(args, print_list):
    """Find the graphs and return the ones -g asks for (all by default).
    For --list, calls print_list(graphs) instead and returns None."""
    graph_dirs = list(DEFAULT_GRAPH_DIRS)
    graph_dirs += [Path(d) for d in os.environ.get("GRAPH_DIR", "").split(":")
                   if d]
    graph_dirs += args.graph_dir

    graphs = discover_graphs(graph_dirs)
    if args.list:
        print_list(graphs)
        return None

    wanted = split_list(args.graphs)
    if wanted:
        unknown = [n for n in wanted if n not in graphs]
        if unknown:
            sys.exit(f"unknown graph(s): {', '.join(unknown)}\n"
                     f"found: {', '.join(graphs) or 'none'} "
                     "(see --list, --graph-dir)")
        selected = [graphs[n] for n in wanted]
    else:
        selected = list(graphs.values())
    if not selected:
        sys.exit("no graphs found (see --list, --graph-dir)")
    return selected


def run_matrix(args, label, jobs, run_fn):
    """Run every job --runs times, write results.csv under
    bench/results/<label>_<time>/ and print a summary. run_fn(job, run_dir,
    tag) returns a CSV row (see new_row) and writes its log to
    run_dir/logs/<tag>.log. Returns the exit code."""
    if not jobs:
        sys.exit("nothing to run")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.results_dir / f"{label}_{stamp}"
    (run_dir / "logs").mkdir(parents=True)
    csv_path = run_dir / "results.csv"
    benchmarks = dict.fromkeys(j.benchmark for j in jobs)
    names = dict.fromkeys(j.graph.name for j in jobs)
    print(f"scheduler {label}; benchmarks {', '.join(benchmarks)}; "
          f"graphs {', '.join(names)}\nresults in {run_dir}")

    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for job in jobs:
            for run in range(args.runs):
                tag = f"{job.tag}_{run}"
                row = run_fn(job, run_dir, tag)
                row.update(scheduler=label, benchmark=job.benchmark,
                           graph=job.graph.name, source=job.source,
                           target=job.target, params=job.params, run=run)
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"  {job.benchmark:8} {job.graph.name:24} "
                      f"{endpoints(row):<24} run={run} {row['status']:7} "
                      f"kernel={row['kernel_ms'] or '-':>10} ms "
                      f"check={row['check'] or '-'} {row['detail']}",
                      flush=True)

    print_summary(rows)
    print(f"\nCSV: {csv_path}")
    bad = [r for r in rows if r["status"] != "ok" or r["check"] == "FAIL"]
    return 1 if bad else 0


def main():
    args = parse_args()
    host_free, gpu_free = mem_available_bytes(), gpu_free_bytes()

    def print_list(graphs):
        for g in graphs.values():
            # bfs and sssp need the same memory, so one line covers both
            for benchmarks, f in ((("bfs", "pr"), g.plain),
                                  (("a_star",), g.coords)):
                if f is None:
                    continue
                ref = "ref" if (g.astar_reference if f is g.coords
                                else g.reference) else "   "
                label = "a_star" if f is g.coords else "bfs sssp pr"
                reasons = {b: skip_reason(g, b, host_free, gpu_free)
                           for b in benchmarks}
                skipped = {b: r for b, r in reasons.items() if r}
                if len(skipped) == len(reasons):
                    status = f"SKIP ({skipped[benchmarks[0]]})"
                elif skipped:  # only PageRank is too big
                    status = f"ok, but SKIP pr ({skipped['pr']})"
                else:
                    status = "ok"
                print(f"{g.name:28} {f.nodes:>13,} nodes {f.edges:>15,} edges "
                      f"{ref}  {label:11}  {status}\n{'':28} {f.path}")

    selected = select_graphs(args, print_list)
    if selected is None:
        return 0

    if args.bin == DEFAULT_BIN and not args.no_build:
        build(args.cuda_arch)
    if not args.bin.is_file():
        sys.exit(f"{args.bin} not found; build it or drop --no-build")

    jobs = []
    for b in split_list(args.benchmarks) or list(BENCHMARKS):
        b = ALIASES.get(b, b)
        try:
            supported = binary_supports(args.bin, b, args.scheduler,
                                        args.weight_type)
        except RuntimeError as e:
            sys.exit(str(e))
        if not supported:
            print(f"skip benchmark {b}: not implemented in {args.bin.name}")
            continue
        for g in selected:
            reason = skip_reason(g, b, host_free, gpu_free,
                                 ignore_size=args.force)
            if not reason:
                try:
                    jobs += make_jobs(b, g, args)
                except ValueError as e:
                    reason = str(e)
            if reason:
                print(f"skip {b} on {g.name}: {reason}")

    def run_fn(job, run_dir, tag):
        out_path = run_dir / f"{tag}.bin"
        row = run_one(args.bin, job, args, out_path,
                      run_dir / "logs" / f"{tag}.log")
        if not args.keep_output:
            out_path.unlink(missing_ok=True)
        return row

    return run_matrix(args, args.scheduler, jobs, run_fn)


def endpoints(row):
    if row["source"] == "":
        return "-"
    return (f"{row['source']}->{row['target']}" if row["target"] != ""
            else str(row["source"]))


def print_summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["benchmark"], r["graph"], endpoints(r)),
                          []).append(r)
    print(f"\n{'benchmark':10}{'graph':26}{'source':>20}{'ok':>6}"
          f"{'median ms':>12}{'min ms':>12}  check")
    for (benchmark, graph, source), rs in groups.items():
        times = [float(r["kernel_ms"]) for r in rs if r["kernel_ms"]]
        ok = sum(r["status"] == "ok" for r in rs)
        checks = {r["check"] for r in rs if r["check"]}
        check = "FAIL" if "FAIL" in checks else ("pass" if "pass" in checks
                                                 else "-")
        med = f"{statistics.median(times):.3f}" if times else "-"
        best = f"{min(times):.3f}" if times else "-"
        print(f"{benchmark:10}{graph:26}{source:>20}{ok:>4}/{len(rs):<1}"
              f"{med:>12}{best:>12}  {check}")


if __name__ == "__main__":
    sys.exit(main())
