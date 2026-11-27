#!/usr/bin/env python3
"""Run Galois's Lonestar CPU apps on the same graphs as run_benchmarks.py.

This is the Galois CPU baseline (galois-plan.md Step 8). It finds graphs the
same way, runs every (benchmark, graph, source) combination, and writes the
same CSV columns with scheduler=galois-cpu, so the results can be compared
with our GPU runs.

A time is only recorded for a correct answer:
  - Lonestar checks its own answer and exits non-zero if the check fails
    ("Verification successful." must be printed), and
  - the reached / max / sum it prints are compared with the graph's
    reference.txt when there is one.

kernel_ms is Lonestar's Timer_0: the algorithm only, without loading the
graph or verifying. load_ms is left empty.

Examples (from the repo root):
  bench/run_galois.py --list                          # graphs found
  bench/run_galois.py -g germany-osm -b bfs sssp --runs 5
  bench/run_galois.py -b sssp --threads 8 --sssp-algo deltaStep --delta 13

Galois is built into build-galois/ (or $GALOIS_BUILD) with only the targets
needed, unless --no-build is given. It needs Boost, LLVM and fmt; see
bench/README_run_benchmarks.md.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from run_benchmarks import (REPO_ROOT, Job, common_parser, compare_summary,
                            gib, job_name, mem_available_bytes, new_row,
                            pick_sources, reference_for, run_logged,
                            run_matrix, select_graphs, split_list)

GALOIS_SRC = REPO_ROOT / "Galois"
DEFAULT_GALOIS_BUILD = Path(os.environ.get("GALOIS_BUILD",
                                           REPO_ROOT / "build-galois"))
LABEL = "galois-cpu"

# Lonestar apps we can run on a plain .gr from our graph folders.
#   target:  CMake target (also the binary's name)
#   subdir:  where the binary lands under the build dir
#   weights: needs 4-byte integer edge weights in the .gr
# PageRank, CC and MIS need a transposed or symmetric graph (graph-convert
# -gr2tgr / -gr2sgr) and print no reached/max/sum to check, so they aren't
# here yet (galois-plan.md Step 7).
APPS = {
    "bfs": {"target": "bfs-cpu", "subdir": "lonestar/analytics/cpu/bfs",
            "weights": False},
    "sssp": {"target": "sssp-cpu", "subdir": "lonestar/analytics/cpu/sssp",
             "weights": True},
}

# Lonestar prints these through galois::gInfo, i.e. "INFO: # visited ..."
SUMMARY_RES = [re.compile(r"# visited nodes is (\d+)"),
               re.compile(r"Max distance is (\d+)"),
               re.compile(r"Sum of visited distances is (\d+)")]
VERIFIED = "Verification successful."


def app_binary(build_dir, benchmark):
    app = APPS[benchmark]
    return build_dir / app["subdir"] / app["target"]


def build(build_dir, benchmarks):
    if not (GALOIS_SRC / "CMakeLists.txt").is_file():
        sys.exit(f"{GALOIS_SRC} is empty; run "
                 "`git submodule update --init Galois`")
    cmd = ["cmake", "-S", str(GALOIS_SRC), "-B", str(build_dir),
           "-DCMAKE_BUILD_TYPE=Release"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # Only the apps we run: the full Galois build is huge.
    cmd = ["cmake", "--build", str(build_dir), "-j", "--target"]
    cmd += [APPS[b]["target"] for b in benchmarks]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cpu_skip_reason(g, benchmark, host_free, ignore_size=False):
    """Why Lonestar can't run this benchmark on this graph, or "".

    Lonestar runs on the plain .gr (no coordinates). Galois uses 64-bit edge
    offsets, so the GPU binary's int32 limit doesn't apply. Estimate: the
    file is read, then LC_CSR_Graph holds 8-byte edge offsets, 4-byte
    destinations and weights, and 4-byte node data.
    """
    f = g.plain
    if f is None:
        return "only has a file with node coordinates"
    if f.error:
        return f.error
    if APPS[benchmark]["weights"] and f.weight_size != 4:
        return (f"{benchmark} needs 4-byte weights, graph has "
                f"{f.weight_size}")
    if ignore_size:
        return ""
    need = (f.path.stat().st_size + (f.nodes + 1) * 8 + f.edges * 8
            + f.nodes * 4)
    if host_free is not None and need > host_free:
        return f"needs ~{gib(need)} host RAM, {gib(host_free)} available"
    return ""


def parse_summary(stdout):
    """(reached, max, sum) from Lonestar's output, or None."""
    got = []
    for r in SUMMARY_RES:
        m = r.search(stdout)
        if not m:
            return None
        got.append(int(m.group(1)))
    return tuple(got)


def read_timer_ms(stat_path, name="Timer_0"):
    """A timer's total from a Lonestar -statFile. Rows look like
    "STAT, (NULL), Timer_0, TMAX, 123" and timers are in milliseconds."""
    try:
        lines = stat_path.read_text().splitlines()
    except OSError:
        return ""
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[2] == name and parts[3] != "ThreadValues":
            return parts[4]
    return ""


def run_one(binary, job, args, run_dir, tag):
    benchmark, graph, source = job.benchmark, job.graph, job.source
    row = new_row()
    stat_path = run_dir / "logs" / f"{tag}.stats"
    cmd = [str(binary), str(graph.plain.path), f"-startNode={source}",
           f"-t={args.threads}", f"-statFile={stat_path}"]
    if benchmark == "bfs" and args.bfs_algo:
        cmd.append(f"-algo={args.bfs_algo}")
    if benchmark == "sssp":
        if args.sssp_algo:
            cmd.append(f"-algo={args.sssp_algo}")
        if args.delta is not None:
            cmd.append(f"-delta={args.delta}")

    proc = run_logged(cmd, args.timeout, run_dir / "logs" / f"{tag}.log", row)
    if proc is None:
        return row
    if VERIFIED not in proc.stdout:
        row.update(status="error", detail="no 'Verification successful.'")
        return row
    row["kernel_ms"] = read_timer_ms(stat_path)
    if not row["kernel_ms"]:
        row["detail"] = "no Timer_0 in the stat file"

    got = parse_summary(proc.stdout)
    want = None if args.no_check else reference_for(benchmark, graph, source)
    if want is None:
        row["check"] = "-"
    elif got is None:
        row["check"], row["detail"] = "FAIL", "no reached/max/sum printed"
    else:
        row["check"], row["detail"] = compare_summary(got, want)
    if row["check"] == "FAIL":
        row["kernel_ms"] = ""  # never report the time of a wrong answer
    return row


def parse_args():
    p = common_parser(__doc__, list(APPS), "all")
    p.add_argument("--galois-build", type=Path, default=DEFAULT_GALOIS_BUILD,
                   help="Galois build directory (default: build-galois/, "
                        "or $GALOIS_BUILD)")
    p.add_argument("--threads", "-t", type=int, default=os.cpu_count() or 1,
                   help="Galois threads (default: all cores, %(default)s)")
    p.add_argument("--bfs-algo",
                   help="bfs-cpu -algo, e.g. SyncTile, Async (default: "
                        "Lonestar's)")
    p.add_argument("--sssp-algo",
                   help="sssp-cpu -algo, e.g. deltaStep, dijkstra (default: "
                        "Lonestar's AutoAlgo)")
    p.add_argument("--delta", type=int,
                   help="sssp-cpu -delta: log2 of the delta-step bucket "
                        "width (default: Lonestar's, 13)")
    return p.parse_args()


def main():
    args = parse_args()
    host_free = mem_available_bytes()

    def print_list(graphs):
        for g in graphs.values():
            f = g.plain
            if f is None:
                print(f"{g.name:28} SKIP (only has a file with node "
                      "coordinates)")
                continue
            reasons = {b: cpu_skip_reason(g, b, host_free) for b in APPS}
            ok = [b for b, r in reasons.items() if not r]
            status = (f"ok: {' '.join(ok)}" if ok
                      else f"SKIP ({next(iter(reasons.values()))})")
            ref = "ref" if g.reference else "   "
            print(f"{g.name:28} {f.nodes:>13,} nodes {f.edges:>15,} edges "
                  f"{ref}  {status}\n{'':28} {f.path}")

    selected = select_graphs(args, print_list)
    if selected is None:
        return 0

    benchmarks = split_list(args.benchmarks) or list(APPS)
    unknown = [b for b in benchmarks if b not in APPS]
    if unknown:
        sys.exit(f"no Lonestar app for: {', '.join(unknown)} "
                 f"(available: {', '.join(APPS)})")

    if not args.no_build:
        build(args.galois_build, benchmarks)
    binaries = {b: app_binary(args.galois_build, b) for b in benchmarks}
    missing = [str(p) for p in binaries.values() if not p.is_file()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}; build Galois or drop "
                 "--no-build")

    jobs = []
    for b in benchmarks:
        for g in selected:
            reason = cpu_skip_reason(g, b, host_free, ignore_size=args.force)
            if not reason:
                try:
                    jobs += [Job(b, g, [], f"{job_name(b, g)}_{s}", source=s)
                             for s in pick_sources(g, args)]
                except ValueError as e:
                    reason = str(e)
            if reason:
                print(f"skip {b} on {g.name}: {reason}")

    print(f"{args.threads} threads")
    return run_matrix(
        args, LABEL, jobs,
        lambda job, run_dir, tag: run_one(binaries[job.benchmark], job, args,
                                          run_dir, tag))


if __name__ == "__main__":
    sys.exit(main())
