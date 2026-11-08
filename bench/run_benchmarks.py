#!/usr/bin/env python3
"""Run benchmarks on input graphs against one scheduler.

Builds bin/ece496, runs every (benchmark, graph, source) combination with the
given scheduler (selected at runtime by the binary's <scheduler> argument),
checks the answers against each graph's reference.txt when there is one, and
writes a CSV.

Examples (from the repo root):
  bench/run_benchmarks.py --list                     # graphs found, and why some are skipped
  bench/run_benchmarks.py                            # every benchmark on every graph that fits
  bench/run_benchmarks.py -g germany-osm livejournal -b bfs --runs 5
  GRAPH_DIR="/path/to/processed_graphs" bench/run_benchmarks.py -b bfs sssp

Graphs are looked up in graphs/inputs/ plus every --graph-dir (or $GRAPH_DIR,
separated by ':'). Both layouts are found:
  <dir>/<name>.gr                 flat file, named <name>
  <dir>/<name>/<anything>.gr      prepared folder (bench.sh / graph_prep), named
                                  after the folder when it has info.txt or
                                  reference.txt; its sources.txt and
                                  reference.txt are used for sources and checks
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BIN = REPO_ROOT / "bin" / "ece496"
DEFAULT_GRAPH_DIRS = [REPO_ROOT / "graphs" / "inputs"]
DEFAULT_RESULTS_DIR = REPO_ROOT / "bench" / "results"

INT32_MAX = 2**31 - 1  # CSRGraph::index_type is int32_t
UNREACHED = 0xFFFFFFFF

# Benchmarks this script knows about. "ref_cols" are the reference.txt columns
# (reached, max, sum) that the --out distances are checked against; None means
# no check. Benchmarks the binary doesn't implement yet are reported and
# skipped, so adding one to main.cu is enough for it to start running here.
BENCHMARKS = {
    "bfs": {"ref_cols": (1, 2, 3)},
    "sssp": {"ref_cols": (4, 5, 6)},
    "pagerank": {"ref_cols": None},
    "cc": {"ref_cols": None},
    "mis": {"ref_cols": None},
}

CSV_FIELDS = ["scheduler", "benchmark", "graph", "source", "run", "status",
              "kernel_ms", "load_ms", "wall_s", "check", "detail"]


@dataclass
class Graph:
    name: str
    path: Path
    nodes: int = 0
    edges: int = 0
    weight_size: int = 0
    sources: list = field(default_factory=list)
    reference: dict = field(default_factory=dict)  # source -> list of ints
    skip_reason: str = ""


# ---------------------------------------------------------------- graphs

def read_gr_header(path):
    """(version, weight_size, nodes, edges) from a Galois .gr v1 header."""
    with open(path, "rb") as f:
        header = f.read(32)
    if len(header) < 32:
        raise ValueError("file shorter than the .gr header")
    return struct.unpack("<4Q", header)


def read_ints_file(path):
    """Rows of ints from a text file, skipping blank and '#' lines.
    "NA" (no reference value) becomes None."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([None if x == "NA" else int(x) for x in line.split()])
    return rows


def load_graph(name, path, meta_dir=None):
    g = Graph(name, path)
    try:
        version, g.weight_size, g.nodes, g.edges = read_gr_header(path)
    except (OSError, ValueError) as e:
        g.skip_reason = f"unreadable header: {e}"
        return g
    if version != 1:
        g.skip_reason = f".gr version {version}, expected 1"
    if meta_dir is not None:
        if (meta_dir / "sources.txt").is_file():
            g.sources = [row[0] for row in
                         read_ints_file(meta_dir / "sources.txt")]
        if (meta_dir / "reference.txt").is_file():
            # The pairs section repeats sources with other columns; keep only
            # the per-source rows, which come first.
            for row in read_ints_file(meta_dir / "reference.txt"):
                if len(row) >= 7:
                    g.reference.setdefault(row[0], row)
    if not g.sources:
        g.sources = [0]
    return g


def discover_graphs(dirs):
    graphs = {}

    def add(g):
        if g.name in graphs:
            print(f"warning: {g.path} has the same name as "
                  f"{graphs[g.name].path}; keeping the first", file=sys.stderr)
        else:
            graphs[g.name] = g

    for d in dirs:
        if not d.is_dir():
            print(f"warning: graph dir {d} not found", file=sys.stderr)
            continue
        for f in sorted(d.glob("*.gr")):
            add(load_graph(f.stem, f))
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            grs = sorted(sub.glob("*.gr"))
            prepared = ((sub / "info.txt").is_file()
                        or (sub / "reference.txt").is_file())
            for f in grs:
                if prepared:
                    name = sub.name if len(grs) == 1 else f"{sub.name}/{f.stem}"
                    add(load_graph(name, f, meta_dir=sub))
                else:
                    add(load_graph(f.stem, f))
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


def check_fits(g, host_free, gpu_free):
    """Set g.skip_reason if the graph can't run with the current binary.

    Estimates mirror CSRGraph::readFromGR (reads the whole file, then builds
    int32 CSR arrays) and main.cu (scheduler takes 1/8 of free VRAM after the
    graph and distances are on the device).
    """
    if g.skip_reason:
        return
    if g.nodes > INT32_MAX or g.edges > INT32_MAX:
        g.skip_reason = (f"{g.nodes:,} nodes / {g.edges:,} edges exceed the "
                         "int32 index_type")
        return
    if g.weight_size not in (0, 4):
        g.skip_reason = f"{g.weight_size}-byte weights; binary expects 4"
        return
    csr = (g.nodes + 1) * 4 + g.edges * 4 + g.edges * 4
    host_need = g.path.stat().st_size + csr
    if host_free is not None and host_need > host_free:
        g.skip_reason = (f"needs ~{gib(host_need)} host RAM, "
                         f"{gib(host_free)} available")
        return
    gpu_need = (csr + g.nodes * 4) * 8 // 7
    if gpu_free is not None and gpu_need > gpu_free:
        g.skip_reason = (f"needs ~{gib(gpu_need)} VRAM, "
                         f"{gib(gpu_free)} free")


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
        [str(binary), benchmark, "/nonexistent.gr", "0", weight_type,
         scheduler], capture_output=True, text=True, timeout=60)
    if "Unknown scheduler" in proc.stderr:
        raise RuntimeError(f"{binary.name} rejected scheduler {scheduler!r}")
    return "Unsupported benchmark" not in proc.stderr


def check_distances(out_path, ref_row, ref_cols):
    """Compare reached / max / sum of the uint32 distances to reference.txt."""
    dist = array("I")
    with open(out_path, "rb") as f:
        dist.frombytes(f.read())
    if sys.byteorder != "little":
        dist.byteswap()
    unreached = dist.count(UNREACHED)
    reached = len(dist) - unreached
    total = sum(dist) - unreached * UNREACHED
    biggest = max((d for d in dist if d != UNREACHED), default=0)
    got = (reached, biggest, total)
    want = tuple(ref_row[c] for c in ref_cols)
    if got == want:
        return "pass", ""
    return "FAIL", (f"reached/max/sum got {got[0]}/{got[1]}/{got[2]}, "
                    f"want {want[0]}/{want[1]}/{want[2]}")


def run_one(binary, benchmark, graph, source, args, out_path, log_path):
    cmd = [str(binary), benchmark, str(graph.path), str(source),
           args.weight_type, args.scheduler, "--out", str(out_path), "--time"]
    row = {"kernel_ms": "", "load_ms": "", "check": "", "detail": ""}
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        row["wall_s"] = f"{time.monotonic() - start:.2f}"
        row["status"] = "timeout"
        row["detail"] = f"killed after {args.timeout}s"
        log_path.write_text(f"$ {' '.join(cmd)}\n{e.stdout or ''}\n"
                            f"{e.stderr or ''}")
        return row
    row["wall_s"] = f"{time.monotonic() - start:.2f}"
    log_path.write_text(f"$ {' '.join(cmd)}\nexit {proc.returncode}\n"
                        f"--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
    if proc.returncode != 0:
        row["status"] = "error"
        lines = proc.stderr.strip().splitlines()
        row["detail"] = lines[-1] if lines else f"exit {proc.returncode}"
        return row
    m = re.search(r'"kernel_ms":\s*([0-9.eE+-]+).*"load_ms":\s*([0-9.eE+-]+)',
                  proc.stdout)
    if m:
        row["kernel_ms"], row["load_ms"] = m.group(1), m.group(2)
    row["status"] = "ok"

    ref_cols = BENCHMARKS.get(benchmark, {}).get("ref_cols")
    ref_row = graph.reference.get(source)
    if (args.no_check or ref_cols is None or ref_row is None
            or any(ref_row[c] is None for c in ref_cols)):
        row["check"] = "-"
    elif not out_path.is_file():
        row["check"], row["detail"] = "FAIL", "no --out file written"
    else:
        row["check"], row["detail"] = check_distances(out_path, ref_row,
                                                      ref_cols)
    return row


# ---------------------------------------------------------------- main

def split_list(values):
    """Accept both `-g a b` and `-g a,b`."""
    return [v for item in values or [] for v in item.split(",") if v]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-g", "--graphs", nargs="*", metavar="NAME",
                   help="graphs to run (default: every graph that fits)")
    p.add_argument("-b", "--benchmarks", nargs="*", metavar="NAME",
                   help="benchmarks to run, e.g. bfs sssp pagerank "
                        f"(default: those of {', '.join(BENCHMARKS)} the "
                        "binary implements)")
    p.add_argument("-s", "--scheduler", default="initial_scheduler",
                   help="scheduler to run (default: %(default)s)")
    p.add_argument("--graph-dir", action="append", type=Path, default=[],
                   help="extra directory to search for graphs (repeatable; "
                        "also read from $GRAPH_DIR)")
    p.add_argument("--sources", type=int, default=1, metavar="N",
                   help="first N sources from each graph's sources.txt "
                        "(default: 1, the main source; 0 for all)")
    p.add_argument("--source", type=int, metavar="V",
                   help="use this source vertex on every graph instead")
    p.add_argument("--runs", type=int, default=3,
                   help="repetitions per configuration (default: %(default)s)")
    p.add_argument("--weight-type", default="int32",
                   choices=["int32", "uint32", "float32"])
    p.add_argument("--timeout", type=float, default=600,
                   help="seconds before a run is killed (default: %(default)s)")
    p.add_argument("--no-build", action="store_true",
                   help="use the existing binary instead of rebuilding")
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN,
                   help="binary to run (default: bin/ece496); implies "
                        "--no-build if not the default")
    p.add_argument("--cuda-arch", help="CMAKE_CUDA_ARCHITECTURES, e.g. 75")
    p.add_argument("--no-check", action="store_true",
                   help="don't check results against reference.txt")
    p.add_argument("--force", action="store_true",
                   help="run graphs even if they look too big")
    p.add_argument("--keep-output", action="store_true",
                   help="keep each run's distances file")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--list", action="store_true",
                   help="list graphs and whether they fit, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    graph_dirs = list(DEFAULT_GRAPH_DIRS)
    graph_dirs += [Path(d) for d in os.environ.get("GRAPH_DIR", "").split(":")
                   if d]
    graph_dirs += args.graph_dir

    graphs = discover_graphs(graph_dirs)
    host_free, gpu_free = mem_available_bytes(), gpu_free_bytes()
    for g in graphs.values():
        check_fits(g, host_free, gpu_free)

    if args.list:
        for g in graphs.values():
            status = f"SKIP ({g.skip_reason})" if g.skip_reason else "ok"
            ref = "ref" if g.reference else "   "
            print(f"{g.name:28} {g.nodes:>13,} nodes {g.edges:>15,} edges "
                  f"{ref}  {status}\n{'':28} {g.path}")
        return 0

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
    runnable = []
    for g in selected:
        if g.skip_reason and not args.force:
            print(f"skip graph {g.name}: {g.skip_reason}")
        else:
            runnable.append(g)
    if not runnable:
        sys.exit("no graphs to run")

    if args.bin == DEFAULT_BIN and not args.no_build:
        build(args.cuda_arch)
    if not args.bin.is_file():
        sys.exit(f"{args.bin} not found; build it or drop --no-build")

    benchmarks = []
    for b in split_list(args.benchmarks) or list(BENCHMARKS):
        try:
            supported = binary_supports(args.bin, b, args.scheduler,
                                        args.weight_type)
        except RuntimeError as e:
            sys.exit(str(e))
        if supported:
            benchmarks.append(b)
        else:
            print(f"skip benchmark {b}: not implemented in {args.bin.name}")
    if not benchmarks:
        sys.exit("no benchmarks to run")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.results_dir / f"{args.scheduler}_{stamp}"
    (run_dir / "logs").mkdir(parents=True)
    csv_path = run_dir / "results.csv"
    print(f"scheduler {args.scheduler}; benchmarks {', '.join(benchmarks)}; "
          f"graphs {', '.join(g.name for g in runnable)}\nresults in {run_dir}")

    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for benchmark in benchmarks:
            for g in runnable:
                if args.source is not None:
                    sources = [args.source]
                elif args.sources > 0:
                    sources = g.sources[:args.sources]
                else:
                    sources = g.sources
                for source in sources:
                    for run in range(args.runs):
                        tag = f"{benchmark}_{g.name.replace('/', '_')}_{source}_{run}"
                        out_path = run_dir / f"{tag}.bin"
                        row = run_one(args.bin, benchmark, g, source, args,
                                      out_path, run_dir / "logs" / f"{tag}.log")
                        if not args.keep_output:
                            out_path.unlink(missing_ok=True)
                        row.update(scheduler=args.scheduler,
                                   benchmark=benchmark, graph=g.name,
                                   source=source, run=run)
                        writer.writerow(row)
                        f.flush()
                        rows.append(row)
                        print(f"  {benchmark:8} {g.name:24} src={source:<10} "
                              f"run={run} {row['status']:7} "
                              f"kernel={row['kernel_ms'] or '-':>10} ms "
                              f"check={row['check'] or '-'} {row['detail']}",
                              flush=True)

    print_summary(rows)
    print(f"\nCSV: {csv_path}")
    bad = [r for r in rows if r["status"] != "ok" or r["check"] == "FAIL"]
    return 1 if bad else 0


def print_summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["benchmark"], r["graph"], r["source"]),
                          []).append(r)
    print(f"\n{'benchmark':10}{'graph':26}{'source':>11}{'ok':>6}"
          f"{'median ms':>12}{'min ms':>12}  check")
    for (benchmark, graph, source), rs in groups.items():
        times = [float(r["kernel_ms"]) for r in rs if r["kernel_ms"]]
        ok = sum(r["status"] == "ok" for r in rs)
        checks = {r["check"] for r in rs if r["check"]}
        check = "FAIL" if "FAIL" in checks else ("pass" if "pass" in checks
                                                 else "-")
        med = f"{statistics.median(times):.3f}" if times else "-"
        best = f"{min(times):.3f}" if times else "-"
        print(f"{benchmark:10}{graph:26}{source:>11}{ok:>4}/{len(rs):<1}"
              f"{med:>12}{best:>12}  {check}")


if __name__ == "__main__":
    sys.exit(main())
