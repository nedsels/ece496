#!/usr/bin/env bash
# One-time preparation of the benchmark graphs. See graphs/README.md.
#
# Usage: graphs/prepare.sh <graph>...
#   stage 1 (minutes, in memory):     usa-road-d germany-osm livejournal
#   stage 2 (hours, out of core):     friendster hyperlink2012-host
#
# Reads the raw downloads from $INPUT_DIR and writes one self-contained
# directory per graph to $OUTPUT_DIR (both default to the shared
# /nfs/ug/groups/capstone_proj_2026F/input_graphs). An existing directory for
# the same graph is replaced when the new one is done. Graphs are processed
# one after another. Each directory ends with a MANIFEST of sha256 sums; check
# a copy with: cd <graph> && sha256sum -c MANIFEST
#
# Stage 2 overnight run (about 300 GB free disk and ~18 GB free RAM needed):
#   nohup graphs/prepare.sh friendster hyperlink2012-host > prep_big.log 2>&1 &

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IN=${INPUT_DIR:-/nfs/ug/groups/capstone_proj_2026F/input_graphs}
OUT=${OUTPUT_DIR:-$IN}
PREP="$REPO/bin/graph_prep"
PREP_OOC="$REPO/bin/graph_prep_ooc"

[[ $# -gt 0 ]] || { sed -n '4,6p' "$0" >&2; exit 1; }
for tool in "$PREP" "$PREP_OOC"; do
  [[ -x "$tool" ]] || {
    echo "missing $tool; build with: cmake --build build --target graph_prep graph_prep_ooc" >&2
    exit 1
  }
done

# run <name> <tool> <tool args...>: builds into <name>.tmp, writes the
# MANIFEST, then swaps the directory into place.
run() {
  local name=$1 tool=$2 dir="$OUT/$1" work="$OUT/.work-$1"
  shift 2
  rm -rf "$dir.tmp" "$work"
  mkdir -p "$dir.tmp" "$work"
  echo "== $name: started $(date)"
  df -h "$OUT" | tail -1
  "$tool" --name "$name" --out "$dir.tmp" "$@" 2>&1 | tee "$dir.tmp/prep.log"
  rm -rf "$work"
  echo "== $name: checksums $(date)"
  (cd "$dir.tmp" && sha256sum -- * > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST)
  rm -rf "$dir"
  mv "$dir.tmp" "$dir"
  echo "== $name: done $(date)"
}

for graph in "$@"; do
  case $graph in
    usa-road-d)
      # DIMACS 9 USA distance graph; weights are in 0.1 m.
      run usa-road-d "$PREP" --format dimacs --weights input \
        --coords "$IN/USA-road-d.USA.co.gz" --coords-format dimacs-co \
        --bin-unit-m 0.1 "$IN/USA-road-d.USA.gr.gz" ;;
    germany-osm)
      # DIMACS 10 germany_osm; weights are great-circle metres.
      run germany-osm "$PREP" --format mtx --weights haversine \
        --coords "$IN/germany.osm.xyz.bz2" --coords-format xyz \
        "$IN/road-germany-osm.zip" ;;
    livejournal)
      # SNAP soc-LiveJournal1, directed; random weights in [1, 100].
      run livejournal "$PREP" --format snap --weights random \
        "$IN/soc-LiveJournal1.txt.gz" ;;
    friendster)
      # SNAP com-Friendster: undirected, each pair listed once, sparse ids.
      # No .mtx: 3.6B edges overflow Gunrock's 32-bit edge index.
      run friendster "$PREP_OOC" --work "$OUT/.work-friendster" \
        --symmetrize --compact-ids "$IN/com-friendster.ungraph.txt.gz" ;;
    hyperlink2012-host)
      # Web Data Commons 2012 host graph (sd-arc), directed, ids kept so
      # sd-index.gz still maps them to host names.
      run hyperlink2012-host "$PREP_OOC" --work "$OUT/.work-hyperlink2012-host" \
        --mtx "$IN/sd-arc.gz" ;;
    *)
      echo "unknown graph: $graph" >&2
      exit 1 ;;
  esac
done
