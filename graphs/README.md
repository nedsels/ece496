# Input Graphs

Every graph the benchmarks use lives in the shared directory on the UG machines:

```
/nfs/ug/groups/capstone_proj_2026F/input_graphs
```

It holds the raw downloads and, next to them, one converted folder per graph (`<name>/`) that `bench/bench.sh` reads. The converted `<name>.gr` is a Galois binary `.gr` (v1, u32 weights), which `CSRGraph::readFromGR` also loads.

## Graphs already converted

| Name | Raw file(s) | Source format | Weights |
|---|---|---|---|
| `usa-road-d` | `USA-road-d.USA.gr.gz`, `USA-road-d.USA.co.gz` | DIMACS 9 | from the file (0.1 m) |
| `germany-osm` | `road-germany-osm.zip`, `germany.osm.xyz.bz2` | Matrix Market (DIMACS 10) | great-circle metres |
| `livejournal` | `soc-LiveJournal1.txt.gz` | SNAP edge list, directed | random, 1–100 |
| `friendster` | `com-friendster.ungraph.txt.gz` | SNAP edge list, undirected | random, 1–100 |
| `hyperlink2012-host` | `sd-arc.gz` | edge list, directed | random, 1–100 |

The exact command for each is its `case` entry in `graphs/prepare.sh`. You don't need to rerun any of these.

## Adding a new graph

Work from your own copy of the repo in `/nfs/ug/groups/capstone_proj_2026F/<you>_workspace/ece496`.

1. **Build the converters** (they land in `bin/`):
   ```
   cmake --build build --target graph_prep graph_prep_ooc
   ```
2. **Put the raw file in the shared `input_graphs`** if it isn't there already. `.gz`, `.bz2` and `.zip` are read compressed, so don't extract them.
3. **Pick the tool for the input format:**
   - DIMACS 9 text (`p sp N M` / `a u v w`) → `graph_prep --format dimacs`
   - Matrix Market (`.mtx`) → `graph_prep --format mtx`
   - SNAP-style edge list (`u v` per line) → `graph_prep --format snap` (add `--one-based` if ids start at 1)
   - Edge list too large to sort in RAM (billions of edges) → `graph_prep_ooc` (add `--symmetrize` for undirected lists that store each pair once, `--compact-ids` for sparse ids, `--mtx` to also write the Gunrock `.mtx`)
4. **Pick the weights** (`graph_prep --weights`):
   - `input`: the file's weight column (DIMACS)
   - `random`: integer in `[--wmin, --wmax]` (default 1–100), the same for (u,v) and (v,u)
   - `haversine`: great-circle metres; needs `--coords FILE --coords-format dimacs-co|xyz`
5. **Add a `case` entry to `graphs/prepare.sh`** that runs the tool with those options (copy an existing entry), then run it:
   ```
   graphs/prepare.sh <name>
   ```
   Input and output both default to the shared `input_graphs`; set `INPUT_DIR` / `OUTPUT_DIR` to override. The script writes `input_graphs/<name>/` with `<name>.gr`, `.tgr`/`.sgr` (directed graphs only), `.mtx`, `.sym.ag`, `sources.txt`, `pairs.txt`, `reference.txt`, `info.txt` and a `MANIFEST`. Run long conversions inside `tmux` or with `nohup`.

   **Check `input_graphs/<name>/` doesn't already exist first.** If it does, the run replaces the shared copy everyone else is using.
6. **Check the output:**
   - `cd input_graphs/<name> && sha256sum -c MANIFEST`
   - compare the node and edge counts in `info.txt` against the source's published numbers
   - smoke test it in the suite: `RUNS=1 GRAPHS=<name> bench/bench.sh run`
7. **Add the graph to the table above** and commit the `prepare.sh` change.
