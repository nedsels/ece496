# Summary of `benchmarks.txt` (lab sweep, ug63)

`benchmarks.txt` is the `summary.csv` from the full `bench.sh run` on the lab
machine ug63 (i7-11700 8C/16T, RTX 3070 8 GB, CUDA 11.8; see
[LAB_MACHINE.md](LAB_MACHINE.md)). Same content as
`~/capstone/bench/results/ug63/summary.txt`.

- 168 configurations: 1 warm-up (checked) + 5 timed runs each, median and
  min reported in ms. CPU libraries at 1 and 8 threads; GPU rows say `gpu`.
- Graphs: germany-osm, usa-road-d, livejournal, hyperlink2012-host-new, plus
  `ising-1000` (residual BP's generated input).
- PPSP and A\* use two pairs per graph: line 1 of `pairs.txt` (random) and
  line 11 (far).
- Not in the sweep: **Gunrock** (not built, needs CUDA ≥ 12.4),
  **Friendster** (doesn't fit in RAM), Hyperlink MIS (doesn't fit) and
  Hyperlink Galois PR-pull (no `.tgr` in the `-new` folder), PBBS MIS (no
  OpenCilk).

## Headlines

1. **MQBucket wins SSSP on road graphs and ties elsewhere.** At 8 threads it
   beats Galois deltaStep by 1.36× on Germany (95 vs 129 ms) and 1.09× on USA,
   ties on LiveJournal, and is 4% slower on Hyperlink. Against MBQ's own OBIM
   it is 1.1-1.55× faster on every graph except Hyperlink.
2. **BFS: Galois SyncTile is best or tied everywhere.** MQBucket is 1.7×
   slower than Galois on USA (333 vs 195 ms).
3. **CPU scaling 1 → 8 threads is mostly 4-6×.** Hyperlink is 3.3-4.7× on
   everything (memory-bound). The weakest are Galois PR push-async on roads
   (1.5-2×) and MBQ MIS (2.5× on roads).
4. **GPU:** ADDS SSSP matches or beats the best 8-thread CPU result on roads
   (Germany 95 ms tie, USA 125 vs 219 ms). LonestarGPU MIS is 5-6× faster than
   the best 8-thread CPU MIS on roads and 1.5× on LiveJournal.
5. **PageRank: Galois pull-residual is fastest, 3-4× ahead of MBQ.** MBQ's PageRank
   (MQBucket / OBIMPR) is not competitive and timed out at 1 thread on
   Hyperlink.
6. **A\*: only the fixed heuristic is usable.** `astar-fixed` always matched
   the serial path and scales about 4×. `astar-orig` gave different paths on
   Germany and timed out on USA. Even fixed A\* is not faster than plain
   MQBucket PPSP on the far pairs (Germany 212 vs 108 ms, USA 487 vs 205 ms).
7. **19 configurations failed with `TIMEOUT`,** most of them hangs (below).

## SSSP (ms, median)

| Graph | Galois deltaStep 1t / 8t | MBQ MQBucket 1t / 8t | MBQ OBIM 1t / 8t | ADDS (GPU) |
|---|---|---|---|---|
| germany-osm | 796 / 129 | 549 / **95** | 920 / 147 | **95.2** |
| usa-road-d | 1475 / 238 | 1084 / 219 | 1683 / 275 | **125.0** |
| livejournal | 1447 / 244 | 1063 / **235** | 1561 / 257 | TIMEOUT |
| hyperlink2012-host-new | 28448 / **7030** | 30099 / 7291 | 30643 / 7265 | too big |

MQBucket is also the fastest single-threaded on the three small graphs
(1.3-1.45× over Galois). All rows checked `ok`. LonestarGPU and MLMQ SSSP
timed out on every graph.

## BFS (ms)

| Graph | Galois SyncTile 1t / 8t | MBQ MQBucket 1t / 8t | MBQ OBIM 1t / 8t |
|---|---|---|---|
| germany-osm | 365 / 133 | 572 / **125** | 815 / 130 |
| usa-road-d | 652 / **195** | 1134 / 333 | 1327 / 224 |
| livejournal | 428 / 85 | 411 / **83** | 614 / 105 |
| hyperlink2012-host-new | 12032 / **3545** | 17545 / 3813 | 15391 / 3861 |

All checked `ok`. LonestarGPU BFS timed out on every graph.

## PPSP (MBQ only, ms at 8 threads)

| Graph | Random pair: MQBucket / OBIM | Far pair: MQBucket / OBIM | SSSP MQBucket for comparison |
|---|---|---|---|
| germany-osm | **97** / 137 | **108** / 148 | 95 |
| usa-road-d | **115** / 149 | **205** / 266 | 219 |
| livejournal | **219** / 224 | **245** / 259 | 235 |
| hyperlink2012-host-new | **5493** / 5750 | 7408 / **7312** | 7291 |

MQBucket leads OBIM by about 1.3-1.4× on roads and is level elsewhere. Early
exit at the target saves little: only USA's random pair is clearly faster than
a full SSSP. All `ok`.

## A\* (MBQ, road graphs only, ms)

| Pair | Fixed: Serial / MQBucket 1t / 8t | Original: Serial / MQBucket 1t / 8t |
|---|---|---|
| germany random | 445 / 354 / **89** | 170 / 2932 / 383 (PATH-DIFFERS) |
| germany far | 1179 / 922 / **212** | 4021 / 4998 / 791 (PATH-DIFFERS) |
| usa random | 577 / 495 / **124** | TIMEOUT / TIMEOUT / 1060 (no serial ref) |
| usa far | 2402 / 2044 / **487** | TIMEOUT / TIMEOUT / 61152 (no serial ref) |

- `astar-fixed` (haversine `sin(Δ/2)` fix) matched its serial path every time.
- `astar-orig` overestimates, so its path depends on run order
  (`PATH-DIFFERS`, expected) and its run times swing widely (Germany far 1t:
  median 4998, min 2522). On USA its serial run never finished in 30 min, so
  the 8-thread results have nothing to be checked against.

## PageRank (tolerance 1e-3, ms; no correctness check)

| Graph | Galois pull-residual 1t / 8t | Galois push-async 1t / 8t | MBQ MQBucket 1t / 8t | MBQ OBIMPR 1t / 8t |
|---|---|---|---|---|
| germany-osm | 6693 / **991** | 14016 / 6868 | 20649 / 3315 | 19324 / 3252 |
| usa-road-d | 8215 / **1918** | 22246 / 14593 | 58047 / 11897 | 31428 / 7955 |
| livejournal | 10485 / **1970** | 18839 / 3770 | 38189 / 6832 | 35004 / 6569 |
| hyperlink2012-host-new | n/a (no `.tgr`) | 373752 / **114194** | TIMEOUT / 207514 | TIMEOUT / 199811 |

Some PageRank rows are noisy (median vs min): Galois pull Germany 1t
6693 / 4169, Galois push Germany 8t 6868 / 4891. Rerun those before quoting
them. LonestarGPU PageRank timed out on every graph.

## MIS (ms)

| Graph | Galois prio 1t / 8t | MBQ MQBucket 1t / 8t | LonestarGPU (GPU) |
|---|---|---|---|
| germany-osm | 362 / 54 | 252 / 101 | **9.9** |
| usa-road-d | 839 / 125 | 637 / 255 | **20.4** |
| livejournal | 2677 / 385 | 314 / 81 | **53.0** |

MBQ's MIS is faster single-threaded everywhere but scales only 2.5-3.9×, so
Galois wins at 8 threads on roads and MBQ wins on LiveJournal (4.7×). Only
MBQ's MIS is checked (`MISCheck`, `ok`); Galois and LonestarGPU MIS are
unchecked.

## Residual BP (MBQ, ising-1000)

50374 ms at 1 thread, 8736 ms at 8 threads (5.8×). Largest error versus the
serial result: 0.0016 (1t) and 0.0018 (8t).

## Failures

| Library | What timed out (30 min) | Count |
|---|---|---|
| LonestarGPU | BFS, SSSP, PageRank on all three small graphs (MIS works) | 9 |
| MLMQ | SSSP on all three small graphs | 3 |
| ADDS | SSSP on LiveJournal | 1 |
| MBQ PageRank | Hyperlink at 1 thread (MQBucket and OBIMPR) | 2 |
| MBQ A\* original | USA, Serial and MQBucket 1t, both pairs | 4 |

MLMQ and ADDS-LiveJournal hung the same way under WSL2 on JamesPC, so this is
not a WSL problem. The LonestarGPU and MLMQ hangs alone cost about 6 hours of
the sweep; their `r0` logs are the next thing to look at.

## Reading the numbers

- Times mean different things per library: Galois `Timer_0`, MBQ
  `runtime_ms` (whole ms, so small graphs are coarse), ADDS the mean of 8
  in-process samples, LonestarGPU `Total time`. Compare across libraries with
  that in mind.
- Delta values in the `param` column are MBQ's published settings (Germany:
  USA minus 3). They weren't tuned for this machine.
- `valid = -` means no check exists for that library and benchmark, not that
  it failed.
