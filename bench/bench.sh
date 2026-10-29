#!/usr/bin/env bash
# Builds every library at a pinned commit and runs every benchmark on the
# prepared graphs. See bench/README.md.
#
# Usage: bench/bench.sh check            prerequisites, GPU, graphs found
#        bench/bench.sh build [lib...]   galois gunrock mbq mlmq (default: all)
#        bench/bench.sh run              all benchmarks on all graphs
#        bench/bench.sh all              check, build, run
#        bench/bench.sh summary          medians from results.csv
#
# Settings are environment variables; defaults below. Example:
#   GRAPH_DIR=/nfs/ug/groups/capstone_proj_2026F/input_graphs bench/bench.sh all

set -uo pipefail   # no -e: one failed run must not stop the sweep

GRAPH_DIR=${GRAPH_DIR:-/nfs/ug/groups/capstone_proj_2026F/input_graphs}
WORK_DIR=${WORK_DIR:-$HOME/bench}          # sources, builds, results
OUT=${OUT_DIR:-$WORK_DIR/results/$(hostname -s)}
SRC=$WORK_DIR/src
BUILD=$WORK_DIR/build

physical_cores() {
  local c
  c=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  (( c > 0 )) && echo "$c" || nproc
}
THREADS=${THREADS:-"1 $(physical_cores)"}  # CPU thread counts to sweep
RUNS=${RUNS:-5}                            # timed runs, after 1 warm-up run
TIMEOUT=${TIMEOUT:-1800}                   # seconds per run
LIBS=${LIBS:-"galois lonestargpu gunrock mbq mlmq adds"}
BENCHES=${BENCHES:-"bfs sssp pr ppsp astar mis rbp"}
GRAPHS=${GRAPHS:-}                         # graph folder names; empty = all
SOURCES=${SOURCES:-main}                   # main | all (BFS and SSSP)
PAIRS=${PAIRS:-"1 11"}                     # lines of pairs.txt for PPSP and A*
MBQ_ALGOS=${MBQ_ALGOS:-"MQBucket OBIM"}    # any of MQBucket MQ OBIM PMOD
ASTAR_VARIANTS=${ASTAR_VARIANTS:-"orig fixed"}  # MBQ heuristic as published / fixed
PR_TOL=${PR_TOL:-1e-3}                     # PageRank tolerance (Galois CPU, MBQ)
PR_MAX_ITER=${PR_MAX_ITER:-1000}           # LonestarGPU PageRank iterations
LSGPU_SSSP_DELTA=${LSGPU_SSSP_DELTA:-}     # LonestarGPU -d; empty = its default
GPU=${GPU:-0}                              # GPU index
CUDA_ARCH=${CUDA_ARCH:-auto}               # e.g. 86; auto = detect; none = no GPU
JOBS=${JOBS:-$(nproc)}                     # parallel build jobs
GUNROCK_CUDA_HOME=${GUNROCK_CUDA_HOME:-}   # CUDA >= 12.4 for Gunrock only; empty = nvcc on PATH
SKIP_SIZE_CHECK=${SKIP_SIZE_CHECK:-0}      # 1 = try graphs that look too big
DRY_RUN=${DRY_RUN:-0}                      # 1 = print run commands only

GALOIS_COMMIT=b67f942
GUNROCK_COMMIT=748f79e
MBQ_COMMIT=25d3583
MLMQ_COMMIT=77d81ab
CCCL_TAG=v3.0.3     # Gunrock fetches CCCL's moving main branch; pin a release
FMT_TAG=9.1.0       # built into WORK_DIR when the system has no libfmt

export CUDA_VISIBLE_DEVICES=$GPU

die() { echo "Error: $*" >&2; exit 1; }
note() { echo "[$(date +%H:%M:%S)] $*"; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1e9}'; }

if [[ $CUDA_ARCH == auto ]]; then
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" 2>/dev/null |
              head -1 | tr -d '. ')
  [[ -n $CUDA_ARCH ]] || CUDA_ARCH=none
fi
GPU_MEM_BYTES=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" \
                    2>/dev/null | head -1 || echo 0) * 1048576 ))
RAM_BYTES=$(( $(awk '/^MemTotal/{print $2}' /proc/meminfo) * 1024 ))

# ---------------------------------------------------------------------------
# check

# Host compiler for the system nvcc. CUDA 11.x accepts at most GCC 11 or
# Clang 14; with a newer default g++, pick a supported one (CUDA_HOST_CXX
# overrides). Prints nothing when the default g++ is fine.
cuda_host_cxx() {
  if [[ -n ${CUDA_HOST_CXX:-} ]]; then echo "$CUDA_HOST_CXX"; return; fi
  local v gmaj c cmaj
  v=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  gmaj=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
  [[ $v == 11 ]] && (( ${gmaj:-0} > 11 )) || return 0
  for c in g++-11 g++-10 g++-9 clang++-14 clang++-13 clang++; do
    command -v "$c" >/dev/null || continue
    case $c in
      g++*) echo "$(command -v "$c")"; return ;;
      clang++*) cmaj=$("$c" -dumpversion | cut -d. -f1)
                (( cmaj <= 14 )) && { echo "$(command -v "$c")"; return; } ;;
    esac
  done
}

gunrock_nvcc() {
  if [[ -n $GUNROCK_CUDA_HOME ]]; then echo "$GUNROCK_CUDA_HOME/bin/nvcc"; else command -v nvcc; fi
}
gunrock_nvcc_version() {  # e.g. "12.4"
  local n; n=$(gunrock_nvcc) || return 0
  [[ -x $n ]] && "$n" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'
}

version_of() { "$@" 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1; }

cmd_check() {
  echo "== tools"
  local t
  for t in git cmake make g++ nvcc clang++; do
    if command -v "$t" >/dev/null; then
      printf '  %-8s %s\n' "$t" "$(version_of "$t" --version)"
    else
      printf '  %-8s MISSING\n' "$t"
    fi
  done
  echo "  need: cmake >= 3.24 (Gunrock), g++ >= 10, nvcc >= 11.1 (LonestarGPU, MLMQ, ADDS)"
  local gv; gv=$(gunrock_nvcc_version)
  echo "  nvcc for Gunrock: ${gv:-none} (${GUNROCK_CUDA_HOME:-PATH}); Gunrock needs >= 12.4, see GUNROCK_CUDA_HOME"
  echo "  clang++ is optional (MBQ MIS; g++ is used otherwise)"
  local hc; hc=$(cuda_host_cxx)
  echo "  host compiler for nvcc: ${hc:-default g++}"
  local llvm; llvm=$(llvm_dir)
  echo "  LLVM cmake config: ${llvm:-NOT FOUND (needed by Galois and MBQ; set LLVM_DIR)}"
  [[ -n $llvm ]] && echo "    usable from CMake: $(llvm_cmake_test "$llvm")"
  local boost
  boost=$(grep -hs 'define BOOST_LIB_VERSION' {${BOOST_ROOT:-/usr}/include,/usr/local/include}/boost/version.hpp | head -1)
  echo "  Boost: ${boost:-NOT FOUND (needed by Galois, MBQ and MLMQ; set BOOST_ROOT)}"
  local fmt
  fmt=$(ls -d /usr/include/fmt /usr/local/include/fmt ${fmt_DIR:-/nonexistent} 2>/dev/null | head -1)
  echo "  fmt: ${fmt:-not found; build will compile fmt $FMT_TAG into $WORK_DIR/deps}"

  echo "== machine"
  echo "  CPU threads to sweep: $THREADS   RAM: $(gb "$RAM_BYTES")"
  if [[ $CUDA_ARCH == none ]]; then
    echo "  GPU: none found; GPU libraries will be skipped"
  else
    echo "  GPU $GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU")," \
         "sm_$CUDA_ARCH, $(gb "$GPU_MEM_BYTES")"
  fi
  echo "  work dir: $WORK_DIR (filesystem: $(df -h --output=avail "$(dirname "$WORK_DIR")" 2>/dev/null | tail -1 | tr -d ' ') free)"
  echo "  WORK_DIR needs ~6 GB (sources + builds); a per-user quota can be much smaller than"
  echo "  the filesystem's free space. Your quotas (if any):"
  if command -v quota >/dev/null; then quota -s 2>&1 | sed 's/^/    /'; else echo "    (no 'quota' command)"; fi

  echo "== graphs in $GRAPH_DIR"
  [[ -d $GRAPH_DIR ]] || die "GRAPH_DIR not found: $GRAPH_DIR"
  local d found=0
  for d in "$GRAPH_DIR"/*/; do
    d=${d%/}
    [[ -f $d/info.txt ]] || continue
    found=1
    local name; name=$(awk -F': ' '$1=="name"{print $2}' "$d/info.txt")
    local files="" ext
    for ext in gr tgr sgr mtx sym.mtx sym.ag bin; do
      [[ -f $d/$name.$ext ]] && files+=" .$ext"
    done
    printf '  %-26s %-11s%s\n' "$(basename "$d")" \
      "$(awk -F': ' '$1=="directed"{print ($2 ~ /^yes/) ? "directed" : "undirected"}' "$d/info.txt")" "$files"
    for ext in sources.txt pairs.txt reference.txt; do
      [[ -f $d/$ext ]] || echo "    MISSING $ext (copy it from the processed graph folder)"
    done
  done
  (( found )) || echo "  no graph folders (folders with info.txt) found"
}

# ---------------------------------------------------------------------------
# build

# fetch <dir> <url> <commit> [submodule paths...]
fetch() {
  local dir=$1 url=$2 commit=$3
  shift 3
  # Partial clones (--filter=blob:none) fetch file contents only for the
  # checked-out commit, which keeps WORK_DIR small.
  if [[ ! -d $dir/.git ]]; then
    git clone -q --filter=blob:none "$url" "$dir" || return 1
  fi
  git -C "$dir" -c advice.detachedHead=false checkout -q "$commit" || return 1
  git -C "$dir" submodule update -q --init --recursive --filter=blob:none -- "$@"
}

# build_step <lib> <description> <command...>: logs to $WORK_DIR/logs/build-<lib>.log
build_step() {
  local lib=$1 what=$2
  shift 2
  local log=$WORK_DIR/logs/build-$lib.log step=$WORK_DIR/logs/.step-$lib.log
  note "$lib: $what"
  "$@" > "$step" 2>&1
  local rc=$?
  { echo "== $what"; cat "$step"; } >> "$log"
  if (( rc != 0 )); then
    grep -E -m 8 'error|Error|ERROR' "$step" | cut -c1-220
    echo "  ... (last lines)"; tail -8 "$step" | cut -c1-220
    echo "$lib: FAILED at '$what'; full log: $log" >&2
  fi
  rm -f "$step"
  return $rc
}

# Galois and MBQ need libfmt. Without a system copy (or fmt_DIR), build
# fmt $FMT_TAG into $WORK_DIR/deps; no root needed.
# fmt source at $FMT_TAG. MBQ's Galois fork also builds fmt itself from
# fmt's "master" branch, which no longer exists; it gets this copy instead.
FMT_SRC=$WORK_DIR/deps/fmt-src
ensure_fmt_src() {
  [[ -f $FMT_SRC/CMakeLists.txt ]] && return 0
  rm -rf "$FMT_SRC"
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$FMT_TAG" \
    https://github.com/fmtlib/fmt "$FMT_SRC"
}

ensure_fmt() {
  [[ -n ${fmt_DIR:-} ]] && return 0
  ls /usr/include/fmt/format.h /usr/local/include/fmt/format.h >/dev/null 2>&1 && return 0
  local d=$WORK_DIR/deps
  fmt_DIR=$d/fmt/lib/cmake/fmt
  [[ -f $fmt_DIR/fmt-config.cmake ]] && return 0
  note "fmt not found; building fmt $FMT_TAG into $d/fmt"
  ensure_fmt_src &&
  cmake -S "$FMT_SRC" -B "$d/fmt-build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$d/fmt" -DCMAKE_INSTALL_LIBDIR=lib -DFMT_TEST=OFF -DFMT_DOC=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$WORK_DIR/logs/build-fmt.log" 2>&1 &&
  cmake --build "$d/fmt-build" -j "$JOBS" --target install >> "$WORK_DIR/logs/build-fmt.log" 2>&1
}

# Configure a tiny project that links LLVMSupport, as Galois does. Newer
# LLVM packages need extra -dev packages (zstd, curl, ...) for this to work.
llvm_cmake_test() {
  local t rc out
  t=$(mktemp -d)
  printf '%s\n' 'cmake_minimum_required(VERSION 3.13)' 'project(t C CXX)' \
    'find_package(LLVM REQUIRED CONFIG)' 'add_executable(t t.cpp)' \
    'target_link_libraries(t LLVMSupport)' > "$t/CMakeLists.txt"
  echo 'int main(){}' > "$t/t.cpp"
  out=$(cmake -S "$t" -B "$t/b" -DLLVM_DIR="$1" 2>&1); rc=$?
  rm -rf "$t"
  if (( rc == 0 )); then echo "yes"; return; fi
  echo "NO: $(grep -oE '[A-Za-z0-9_]+::[A-Za-z0-9_]+' <<< "$out" | sort -u | tr '\n' ' ')"
  echo "      (missing -dev packages; e.g. zstd::* -> libzstd-dev, CURL::* -> libcurl4-openssl-dev)"
}

llvm_dir() {  # LLVM_DIR, else the newest /usr/lib/llvm-N
  if [[ -n ${LLVM_DIR:-} ]]; then echo "$LLVM_DIR"; return; fi
  ls -d /usr/lib/llvm-*/lib/cmake/llvm /usr/lib64/cmake/llvm 2>/dev/null | sort -V | tail -1
}

cmake_args() {
  local a=(-DCMAKE_BUILD_TYPE=Release) llvm
  llvm=$(llvm_dir)
  [[ -n $llvm ]] && a+=(-DLLVM_DIR="$llvm")
  [[ -n ${BOOST_ROOT:-} ]] && a+=(-DBOOST_ROOT="$BOOST_ROOT")
  [[ -n ${fmt_DIR:-} ]] && a+=(-Dfmt_DIR="$fmt_DIR")
  printf '%s\n' "${a[@]}"
}

build_galois() {
  local s=$SRC/galois b=$BUILD/galois args gpu=0 v
  build_step galois "fetch $GALOIS_COMMIT" fetch "$s" \
    https://github.com/IntelligentSoftwareSystems/Galois.git "$GALOIS_COMMIT" \
    external/moderngpu external/cub || return 1
  mapfile -t args < <(cmake_args)
  # GCC >= 13 no longer includes <cstdint> indirectly; this code relies on it.
  args+=(-DCMAKE_CXX_FLAGS="-include cstdint")
  # LonestarGPU needs CUDA 11.x: its bundled CUB uses texture references
  # (removed in CUDA 12) and moderngpu reads device fields removed in CUDA 13.
  if [[ $CUDA_ARCH != none ]]; then
    v=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    if [[ $v == 11 ]]; then
      gpu=1
      args+=(-DGALOIS_CUDA_CAPABILITY="${CUDA_ARCH%?}.${CUDA_ARCH: -1}")
      local hc; hc=$(cuda_host_cxx)
      [[ -n $hc ]] && args+=(-DCMAKE_CUDA_HOST_COMPILER="$hc")
    else
      note "galois: LonestarGPU needs CUDA 11.x (nvcc is ${v:-missing}); building CPU apps only"
    fi
  fi
  build_step galois configure cmake -S "$s" -B "$b" "${args[@]}" || return 1
  build_step galois "compile CPU apps" cmake --build "$b" -j "$JOBS" --target \
    bfs-cpu sssp-cpu pagerank-pull-cpu pagerank-push-cpu maximal-independentset-cpu || return 1
  (( gpu )) || return 0
  build_step galois "compile LonestarGPU apps" cmake --build "$b" -j "$JOBS" --target \
    bfs-gpu sssp-gpu pagerank-gpu maximal-independentset-gpu
}

build_gunrock() {
  [[ $CUDA_ARCH != none ]] || { note "gunrock: no GPU, skipped"; return 0; }
  local s=$SRC/gunrock b=$BUILD/gunrock nvcc v
  nvcc=$(gunrock_nvcc); v=$(gunrock_nvcc_version)
  if [[ -z $v ]] || ! printf '12.4\n%s\n' "$v" | sort -VC; then
    echo "gunrock: needs CUDA >= 12.4, found '${v:-no nvcc}' ($nvcc)." >&2
    echo "  Install a newer toolkit without root and point GUNROCK_CUDA_HOME at it" >&2
    echo "  (see bench/README.md, 'Gunrock and CUDA 11')." >&2
    return 1
  fi
  build_step gunrock "fetch $GUNROCK_COMMIT" fetch "$s" \
    https://github.com/gunrock/gunrock.git "$GUNROCK_COMMIT" || return 1
  build_step gunrock "fetch CCCL $CCCL_TAG" fetch "$SRC/cccl" \
    https://github.com/NVIDIA/cccl.git "$CCCL_TAG" || return 1
  # PageRank: the example hard-codes tol = 1e-6 on ranks that sum to 1, so on
  # a large graph (ranks ~1/n) it stops after a couple of iterations. Read a
  # per-node tolerance in Galois/MBQ units (ranks ~1 each) from
  # $GUNROCK_PR_TOL and divide it by n.
  local prc=$s/examples/algorithms/pr/pr.cu
  git -C "$s" checkout -q -- examples/algorithms/pr/pr.cu
  sed -i -e '/^  weight_t tol = 1e-6;$/d' \
    -e 's|^  size_t n_vertices = G.get_number_of_vertices();$|&\n  weight_t tol = getenv("GUNROCK_PR_TOL") ? atof(getenv("GUNROCK_PR_TOL")) / n_vertices : 1e-6;|' \
    "$prc"
  grep -q 'GUNROCK_PR_TOL' "$prc" && ! grep -q 'weight_t tol = 1e-6;' "$prc" ||
    { echo "gunrock: could not patch the PageRank tolerance" >&2; return 1; }
  # This Gunrock version defaults to its AMD (HIP) backend. It also relies on
  # thrust headers that CUDA 13 no longer includes indirectly.
  build_step gunrock configure cmake -S "$s" -B "$b" -DCMAKE_BUILD_TYPE=Release \
    -DESSENTIALS_NVIDIA_BACKEND=ON -DESSENTIALS_AMD_BACKEND=OFF \
    -DESSENTIALS_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$nvcc" -DCUDAToolkit_ROOT="$(dirname "$(dirname "$nvcc")")" \
    -DFETCHCONTENT_SOURCE_DIR_CCCL="$SRC/cccl" \
    -DCMAKE_CUDA_FLAGS="--pre-include=thrust/distance.h --pre-include=thrust/tuple.h" || return 1
  build_step gunrock "compile bfs sssp pr color" \
    cmake --build "$b" -j "$JOBS" --target bfs sssp pr color
}

build_mbq() {
  local s=$SRC/mbq args cxx ok=0
  build_step mbq "fetch $MBQ_COMMIT" fetch "$s" \
    https://github.com/mcj-group/mbq.git "$MBQ_COMMIT" || return 1
  mapfile -t args < <(cmake_args)

  build_step mbq "fmt $FMT_TAG source" ensure_fmt_src || ok=1
  build_step mbq "galois configure" cmake -S "$s/galois" -B "$BUILD/mbq-galois" "${args[@]}" \
    -DFETCHCONTENT_SOURCE_DIR_FMT="$FMT_SRC" -DFMT_INSTALL=ON -DCMAKE_CXX_FLAGS="-include cstdint" &&
  build_step mbq "galois compile" cmake --build "$BUILD/mbq-galois" -j "$JOBS" \
    --target sssp-cpu ppsp-cpu bfs-cpu pagerank-push-cpu || ok=1

  # A*: as published, and with the haversine fix (sin(d) -> sin(d/2)) in a
  # sibling copy, so ../cps/include still resolves.
  rm -rf "$s/astar-fixed"
  mkdir -p "$s/astar-fixed"
  cp "$s"/astar/{CMakeLists.txt,astar.cpp,util.h} "$s/astar-fixed/"
  sed -i -e 's|std::sin(src->lat - dst->lat)|std::sin((src->lat - dst->lat) / 2)|' \
         -e 's|std::sin(src->lon - dst->lon)|std::sin((src->lon - dst->lon) / 2)|' \
         "$s/astar-fixed/util.h"
  [[ $(grep -c 'dst->l[ao][tn]) / 2)' "$s/astar-fixed/util.h") == 2 ]] ||
    { echo "mbq: could not apply the A* heuristic fix" >&2; ok=1; }
  local v
  for v in astar astar-fixed rbp; do
    build_step mbq "$v configure" cmake -S "$s/$v" -B "$BUILD/mbq-$v" "${args[@]}" &&
    build_step mbq "$v compile" cmake --build "$BUILD/mbq-$v" -j "$JOBS" || ok=1
  done

  # MIS. The MQ version needs no OpenCilk; skip the OpenCilk checker
  # dependency (CHECK=) and build the checker sequentially instead.
  # g++ first: a distro clang can pick a GCC install without C++ headers.
  cxx=${PBBS_CXX:-$(command -v g++ || command -v clang++)}
  local mis=$s/pbbs/maximalIndependentSet
  local pflags=(-DPARLAY_SEQUENTIAL -pthread -mcx16 -O3 -std=c++17 -DNDEBUG -I"$s/pbbs" -I"$s/pbbs/parlaylib/include")
  build_step mbq "MIS (mqMIS, $cxx)" make -C "$mis/mqMIS" MIS CC="$cxx" CHECK= \
    CFLAGS="${pflags[*]}" || ok=1
  # The checker's Makefile insists on OpenCilk clang; compile it directly.
  build_step mbq "MISCheck (sequential)" bash -c "cd '$mis/bench' &&
    '$cxx' ${pflags[*]} -c MISCheck.C -o MISCheck.o && '$cxx' -o MISCheck MISCheck.o -pthread" || ok=1
  if command -v clang++ >/dev/null && clang++ -fopencilk -x c++ -E /dev/null >/dev/null 2>&1; then
    build_step mbq "MIS (PBBS incremental, OpenCilk)" make -C "$mis/incrementalMIS" || ok=1
  else
    note "mbq: no OpenCilk clang; PBBS's own MIS baseline is skipped"
  fi
  return $ok
}

build_mlmq() {
  [[ $CUDA_ARCH != none ]] || { note "mlmq: no GPU, skipped"; return 0; }
  local s=$SRC/mlmq b=$BUILD/mlmq gen="-gencode=arch=compute_$CUDA_ARCH,code=sm_$CUDA_ARCH" hc
  hc=$(cuda_host_cxx)
  [[ -n $hc ]] && gen+=" -ccbin '$hc'"
  build_step mlmq "fetch $MLMQ_COMMIT" fetch "$s" \
    https://github.com/Leo9660/MLMQ.git "$MLMQ_COMMIT" || return 1
  mkdir -p "$b"
  # GPU_MEMORY (2.0 * 2^30 bytes of queue) is passed as an int; 2^31
  # overflows to a negative size with current compilers and every cudaMalloc
  # fails. INT_MAX is the ~2 GB the authors intended.
  git -C "$s" checkout -q -- SSSP_adaptive/sssp.cuh
  sed -i 's|^#define GPU_MEMORY (2.0 \* 1024 \* 1024 \* 1024)|#define GPU_MEMORY (2147483647)|' \
    "$s/SSSP_adaptive/sssp.cuh"
  grep -q '^#define GPU_MEMORY (2147483647)' "$s/SSSP_adaptive/sssp.cuh" ||
    { echo "mlmq: could not patch GPU_MEMORY" >&2; return 1; }
  # MLMQ hard-codes source 0; read it from $MLMQ_SRC instead (default 0).
  sed -e 's|int src = 0;|int src = getenv("MLMQ_SRC") ? atoi(getenv("MLMQ_SRC")) : 0;|' \
      -e '1i #include <cstdlib>' "$s/SSSP_adaptive/main.cu" > "$s/SSSP_adaptive/main_bench.cu"
  grep -q MLMQ_SRC "$s/SSSP_adaptive/main_bench.cu" ||
    { echo "mlmq: could not patch the source node" >&2; return 1; }
  # Same flags as the Makefiles, which hard-code sm_86.
  # Both include Boost headers (header-only use).
  local inc=""
  [[ -n ${BOOST_ROOT:-} ]] && inc="-I'$BOOST_ROOT/include' -I'$BOOST_ROOT'"
  build_step mlmq "MLMQ SSSP_adaptive (sm_$CUDA_ARCH)" bash -c "cd '$s/SSSP_adaptive' &&
    nvcc main_bench.cu csr_graph.cu sssp_run.cu -o '$b/mlmq-sssp' -O3 -m64 $gen \
      -rdc=true -lcuda -lcudart -I../core/include $inc -lcusparse" || return 1
  # ADDS: cub::LaneId() was removed in CUDA 13; use the same PTX register.
  { echo '__device__ __forceinline__ unsigned bench_lane_id() {'
    echo '  unsigned r; asm volatile("mov.u32 %0, %%laneid;" : "=r"(r)); return r; }'
    # It also appends a summary to a hard-coded /workspace/... path and
    # crashes when that can't be opened; write it to the run directory.
    sed -e 's/cub::LaneId()/bench_lane_id()/g' \
        -e 's|/workspace/SSSP/test_bed/result|adds_result.txt|' "$s/ADDS/ads_int/kernel.cu"
  } > "$s/ADDS/ads_int/kernel_bench.cu"
  build_step mlmq "ADDS ads_int (sm_$CUDA_ARCH)" bash -c "cd '$s/ADDS/ads_int' &&
    nvcc kernel_bench.cu support.cu csr_graph.cu -O3 $gen -rdc=true -lcudadevrt \
      -Xptxas -O3 -lcuda -lcudart $inc -o '$b/adds-sssp'"
}

cmd_build() {
  local libs=("$@") lib failed=()
  (( ${#libs[@]} )) || libs=(galois gunrock mbq mlmq)
  mkdir -p "$SRC" "$BUILD" "$WORK_DIR/logs"
  note "building ${libs[*]} (CUDA arch: $CUDA_ARCH, $JOBS jobs)"
  if want galois "${libs[*]}" || want mbq "${libs[*]}"; then
    ensure_fmt || echo "fmt build failed; see $WORK_DIR/logs/build-fmt.log" >&2
  fi
  for lib in "${libs[@]}"; do
    case $lib in
      galois|gunrock|mbq|mlmq) : > "$WORK_DIR/logs/build-$lib.log"
                               "build_$lib" || failed+=("$lib") ;;
      *) die "unknown library: $lib (galois gunrock mbq mlmq)" ;;
    esac
  done
  if (( ${#failed[@]} )); then
    echo "Build failed for: ${failed[*]}. Runs for these will be skipped." >&2
    return 1
  fi
  note "all builds done"
}

# ---------------------------------------------------------------------------
# binaries

B_GALOIS=$BUILD/galois/lonestar/analytics
B_MBQ=$BUILD/mbq-galois/lonestar/analytics/cpu
B_PBBS=$SRC/mbq/pbbs/maximalIndependentSet

gunrock_bin() {
  find "$BUILD/gunrock" -type f -name "$1" -perm -u+x 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# run helpers

want() { [[ " $2 " == *" $1 "* ]]; }

skip() {
  echo "  skip: $*"
  echo "$G_LABEL: $*" >> "$OUT/skipped.txt"
}

# mem_estimate <file>: bytes the graph takes in memory, from its binary form
mem_estimate() {
  local f=$1 base=$G_DIR/$G_NAME
  case $f in
    *.sym.mtx|*.sym.ag|*.sgr) [[ -f $base.sgr ]] && f=$base.sgr || f=$base.gr ;;
    *.mtx|*.tgr) f=$base.gr ;;
  esac
  stat -c %s "$f" 2>/dev/null || echo 0
}

# ok_to_run <lib> <binary> <input file> <cpu|gpu> <label>
ok_to_run() {
  local lib=$1 bin=$2 file=$3 kind=$4 label=$5
  want "$lib" "$LIBS" || return 1
  if [[ $kind == gpu && $CUDA_ARCH == none ]]; then
    skip "$label: no GPU"; return 1
  fi
  if (( ! DRY_RUN )) && [[ ! -x $bin ]]; then
    skip "$label: not built ($bin)"; return 1
  fi
  [[ -f $file ]] || { skip "$label: $(basename "$file") not in this folder"; return 1; }
  (( SKIP_SIZE_CHECK )) && return 0
  local need; need=$(mem_estimate "$file")
  if [[ $kind == gpu ]] && (( need > GPU_MEM_BYTES * 9 / 10 )); then
    skip "$label: graph needs ~$(gb "$need"), GPU has $(gb "$GPU_MEM_BYTES")"; return 1
  fi
  if (( need > RAM_BYTES * 8 / 10 )); then
    skip "$label: graph needs ~$(gb "$need"), RAM is $(gb "$RAM_BYTES")"; return 1
  fi
}

ref_row() {  # ref_row <source>: that source's line of reference.txt
  awk -v s="$1" '$1 == s && NF >= 7 { print; exit }' "$G_DIR/reference.txt"
}

# Time parsers: print milliseconds from a run's log.
p_galois() { awk -F', ' '$3 == "Timer_0" { print $5; exit }' "$1"; }
p_runtime() { awk '$1 == "runtime_ms" { t = $2 } END { if (t != "") print t }' "$1"; }
p_lsgpu() { awk '/^Total time: [0-9]+ ns$/ { printf "%.3f\n", $3 / 1e6; exit }' "$1"; }
p_gunrock() { sed -n 's/^GPU Elapsed Time : \([0-9.eE+-]*\) (ms).*/\1/p' "$1" | head -1; }
p_mlmq() { sed -n 's/^Elapse time: \([0-9.]*\) ms.*/\1/p' "$1" | head -1; }
p_adds() {  # ADDS times 8 samples in one process; report their mean, as ADDS does
  sed -n 's/.*Measured time for sample = \([0-9.]*\)s.*/\1/p' "$1" |
    awk '{ s += $1; n++ } END { if (n) printf "%.3f\n", s / n * 1000 }'
}

# Checkers: run after the warm-up run with <log> <run dir>; print the result.
c_none() { echo "-"; }
c_verified() { grep -q "Verification successful" "$1" && echo ok || echo "no-verify-line"; }
c_gunrock() {
  awk '/^Number of errors :/ { e = $NF } END { print (e == "0") ? "ok" : (e == "" ? "no-check" : "errors=" e) }' "$1"
}
c_mlmq() { grep -q "mlmq sssp correct!" "$1" && echo ok || echo "MISMATCH"; }
c_dist() {  # c_dist <bfs|sssp> <log> <dir>: node-distance file vs reference.txt
  local out=$3/out.txt want got
  [[ -s $out ]] || { echo "no-output"; return; }
  want=$(ref_row "$J_SRC" | awk -v m="$1" '{ print (m == "bfs") ? $2 " " $3 " " $4 : $5 " " $6 " " $7 }')
  [[ -n $want && $want != *NA* ]] || { echo "no-reference"; return; }
  got=$(awk '$2 != "INF" && $2 + 0 < 1073741823 { n++; if ($2 + 0 > m) m = $2 + 0; s += $2 }
             END { printf "%d %d %.0f\n", n, m, s }' "$out")
  [[ $got == "$want" ]] && echo ok || echo "MISMATCH(got $got; want $want)"
}
c_dist_bfs() { c_dist bfs "$@"; }
c_dist_sssp() { c_dist sssp "$@"; }
c_ppsp() {
  local got want
  got=$(sed -n 's/^destination node distance = \([0-9]*\).*/\1/p' "$1" | tail -1)
  want=$(awk -v s="$J_SRC" -v t="$J_DST" 'NF == 4 && $1 == s && $2 == t { print $3; exit }' \
         "$G_DIR/reference.txt")
  [[ -n $want ]] || { echo "no-reference"; return; }
  [[ $got == "$want" ]] && echo ok || echo "MISMATCH(got $got; want $want)"
}
c_astar() {
  [[ -f $J_ASTAR_REF ]] || { echo "no-serial-reference"; return; }
  cmp -s "$2/path.txt" "$J_ASTAR_REF" && echo ok || echo "PATH-DIFFERS"
}
c_mis() {
  [[ -x $B_PBBS/bench/MISCheck ]] || { echo "-"; return; }
  "$B_PBBS/bench/MISCheck" "$J_INPUT" "$2/out.txt" > "$2/check.log" 2>&1 && echo ok ||
    echo "FAIL($(head -c 80 "$2/check.log" | tr ',\n' '; '))"
}
c_rbp() { sed -n 's/^AccuracyMax:\(.*\)/accmax=\1/p' "$1" | head -1; }

# run_job <command...>: one warm-up run (run 0, checked) and $RUNS timed runs.
# Uses J_LIB J_ALGO J_BENCH J_SRC J_DST J_T J_PARAM J_PARSE J_CHECK.
# In the command, "@warm:X" is passed only in the warm-up run and "@timed:X"
# only in the timed runs.
run_job() {
  local key="$J_LIB|$J_ALGO|$J_BENCH|$G_LABEL|$J_SRC|$J_DST|$J_T|$J_PARAM"
  if grep -qxF "$key" "$OUT/done.txt" 2>/dev/null; then
    return 0
  fi
  local tag
  tag=$(printf '%s_%s_%s_%s_s%s_d%s_t%s_%s' "$J_LIB" "$J_ALGO" "$J_BENCH" "$G_LABEL" \
        "$J_SRC" "$J_DST" "$J_T" "$J_PARAM" | tr -c 'A-Za-z0-9_.=-' '_')
  local wd=$OUT/tmp/$tag r a rc t v
  local -a cmd
  mkdir -p "$wd"
  (( DRY_RUN )) || printf '  %-11s %-14s %-6s t=%-4s starting (log: %s.r*.log, timeout %ss)\n' \
    "$J_LIB" "$J_ALGO" "$J_BENCH" "$J_T" "$OUT/logs/$tag" "$TIMEOUT"
  for (( r = 0; r <= RUNS; r++ )); do
    cmd=()
    for a in "$@"; do
      case $a in
        @warm:*) (( r == 0 )) && cmd+=("${a#@warm:}") ;;
        @timed:*) (( r > 0 )) && cmd+=("${a#@timed:}") ;;
        *) cmd+=("$a") ;;
      esac
    done
    if (( DRY_RUN )); then
      echo "  [run $r] ${cmd[*]}"
      (( r >= 1 )) && break
      continue
    fi
    local log=$OUT/logs/$tag.r$r.log
    # MLMQ and ADDS keep an array per node on the stack: lift the 8 MB limit.
    (cd "$wd" && { ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -H -s)"; } &&
     exec timeout "$TIMEOUT" "${cmd[@]}") > "$log" 2>&1
    rc=$?
    t=$("$J_PARSE" "$log")
    v="-"
    if (( rc == 124 )); then v="TIMEOUT"
    elif (( rc != 0 )); then v="FAIL(rc=$rc)"
    elif (( r == 0 )); then v=$("$J_CHECK" "$log" "$wd")
    fi
    echo "$J_LIB,$J_ALGO,$J_BENCH,$G_LABEL,$J_SRC,$J_DST,$J_T,$J_PARAM,$r,${t:-NA},$v,$log" \
      >> "$OUT/results.csv"
    printf '  %-11s %-14s %-6s t=%-4s %-22s run %d: %10s ms  %s\n' "$J_LIB" "$J_ALGO" \
      "$J_BENCH" "$J_T" "$J_PARAM" "$r" "${t:-NA}" "$v"
    (( rc == 0 )) || break   # a failure or timeout will repeat; don't retry
    rm -f "$wd/out.txt"
  done
  rm -rf "$wd"
  (( DRY_RUN )) || echo "$key" >> "$OUT/done.txt"
}

# PBBS MIS times all rounds inside one process ("Parlay time: <s>" per round).
run_pbbs_mis() {
  local key="$J_LIB|$J_ALGO|$J_BENCH|$G_LABEL|-|-|$J_T|$J_PARAM"
  grep -qxF "$key" "$OUT/done.txt" 2>/dev/null && return 0
  local tag wd log rc v r t
  tag=$(printf '%s_%s_mis_%s_t%s' "$J_LIB" "$J_ALGO" "$G_LABEL" "$J_T" | tr -c 'A-Za-z0-9_.=-' '_')
  wd=$OUT/tmp/$tag; log=$OUT/logs/$tag.log
  if (( DRY_RUN )); then echo "  [runs 1-$RUNS] $*"; return 0; fi
  mkdir -p "$wd"
  printf '  %-11s %-14s mis    t=%-4s starting (log: %s, timeout %ss)\n' \
    "$J_LIB" "$J_ALGO" "$J_T" "$log" "$TIMEOUT"
  (cd "$wd" && exec timeout "$TIMEOUT" "$@") > "$log" 2>&1
  rc=$?
  if (( rc != 0 )); then
    v=$([[ $rc == 124 ]] && echo TIMEOUT || echo "FAIL(rc=$rc)")
    echo "$J_LIB,$J_ALGO,mis,$G_LABEL,-,-,$J_T,$J_PARAM,1,NA,$v,$log" >> "$OUT/results.csv"
    printf '  %-11s %-14s mis    t=%-4s %s\n' "$J_LIB" "$J_ALGO" "$J_T" "$v"
  else
    v=$(c_mis "$log" "$wd")
    r=0
    while read -r t; do
      r=$(( r + 1 ))
      echo "$J_LIB,$J_ALGO,mis,$G_LABEL,-,-,$J_T,$J_PARAM,$r,$t,$v,$log" >> "$OUT/results.csv"
      printf '  %-11s %-14s mis    t=%-4s %-22s run %d: %10s ms  %s\n' "$J_LIB" "$J_ALGO" \
        "$J_T" "$J_PARAM" "$r" "$t" "$v"
    done < <(awk -F': ' '/^Parlay time: [0-9.]+$/ { printf "%.3f\n", $2 * 1000 }' "$log")
  fi
  rm -rf "$wd"
  echo "$key" >> "$OUT/done.txt"
}

# MBQ parameters from its run scripts, by graph type:
# usa = USAr, lj = LJ, tw = TW (Friendster), hyper = HYPERH. Germany uses the
# road values with deltas 3 lower, since its weights are metres instead of
# 0.1 m (2^3 ~ 10).
profile() {
  case $1 in
    usa*) echo usa ;; germany*) echo germany ;; livejournal*) echo lj ;;
    friendster*) echo tw ;; hyperlink*) echo hyper ;; *) echo lj ;;
  esac
}
# "<MQ delta> <batch1> <batch2> <stickiness> <prefetch> <OBIM/PMOD delta>"
declare -A MBQ_P=(
  [sssp:usa]="11 64 8 64 1 14"     [sssp:germany]="8 64 8 64 1 11"
  [sssp:lj]="4 128 128 1 1 0"      [sssp:tw]="3 256 128 1 1 0"    [sssp:hyper]="3 256 256 1 1 1"
  [bfs:usa]="0 4 64 16 1 0"        [bfs:germany]="0 4 64 16 1 0"
  [bfs:lj]="0 256 256 1 1 0"       [bfs:tw]="0 256 128 1 1 0"     [bfs:hyper]="0 256 256 1 1 0"
  [ppsp:usa]="12 128 32 64 0 14"   [ppsp:germany]="9 128 32 64 0 11"
  [ppsp:lj]="4 128 128 1 1 0"      [ppsp:tw]="3 256 128 1 1 0"    [ppsp:hyper]="3 256 256 1 1 1"
  [pr:usa]="21 256 256 1 1 21"     [pr:germany]="21 256 256 1 1 21"
  [pr:lj]="20 128 128 1 1 20"      [pr:tw]="20 256 256 1 1 20"    [pr:hyper]="21 64 64 1 1 20"
  [mis:usa]="16 256 256 1 1 0"     [mis:germany]="16 256 256 1 1 0"
  [mis:lj]="11 128 128 1 1 0"      [mis:tw]="21 256 256 1 1 0"    [mis:hyper]="21 256 256 1 1 0"
)

# mbq_galois <bench> <binary> <input> <extra args...>: every MBQ_ALGOS x THREADS
mbq_galois() {
  local bench=$1 bin=$2 input=$3
  shift 3
  local p d b1 b2 st pf od algo t q stick
  read -r d b1 b2 st pf od <<< "${MBQ_P[$bench:$G_PROFILE]}"
  for algo in $MBQ_ALGOS; do
    for t in $THREADS; do
      J_LIB=mbq J_T=$t J_PARSE=p_runtime
      case $bench in
        bfs|sssp) J_CHECK=c_verified ;; ppsp) J_CHECK=c_ppsp ;; *) J_CHECK=c_none ;;
      esac
      case $algo in
        MQBucket|MQ)
          if [[ $bench == pr ]]; then q=$(( 4 * t )); else q=$(( t == 1 ? 4 : 2 * t )); fi
          stick=$(( t == 1 ? 1 : st ))
          J_ALGO=$algo J_PARAM="delta=$d"
          local a=(-threads "$t" -queues "$q" -batch1 "$b1" -batch2 "$b2" -prefetch "$pf" -algo="$algo")
          [[ $algo == MQBucket ]] && a+=(-buckets 64 -delta "$d")
          [[ $algo == MQ && $bench == pr ]] && a+=(-delta "$d")
          [[ $bench != pr ]] && a+=(-stick "$stick")
          [[ $bench == bfs || $bench == sssp ]] && a+=(@timed:-noverify)
          run_job "$bin" "$@" "${a[@]}" "$input" ;;
        OBIM|PMOD)
          local name=$algo
          case $bench in
            sssp|ppsp) name=deltaStep$algo ;; pr) name=${algo}PR ;;
          esac
          J_ALGO=$name J_PARAM="delta=$od"
          local nv=()
          [[ $bench == bfs || $bench == sssp ]] && nv=(@timed:-noverify)
          run_job "$bin" "$@" -t "$t" -delta "$od" -algo="$name" "${nv[@]}" "$input" ;;
        *) die "unknown MBQ_ALGOS entry: $algo" ;;
      esac
    done
  done
}

# ---------------------------------------------------------------------------
# benchmarks for one graph folder

bench_bfs_sssp() {
  local bench=$1 s=$2 gr=$G_DIR/$G_NAME.gr t
  J_SRC=$s J_DST=-
  local galois_bin=$B_GALOIS/cpu/$bench/$bench-cpu
  if ok_to_run galois "$galois_bin" "$gr" cpu "galois $bench"; then
    for t in $THREADS; do
      J_LIB=galois J_T=$t J_PARSE=p_galois J_CHECK=c_verified
      if [[ $bench == bfs ]]; then
        J_ALGO=SyncTile J_PARAM=-
        run_job "$galois_bin" -t "$t" -startNode "$s" @timed:-noverify "$gr"
      else
        local od; od=$(awk '{ print $6 }' <<< "${MBQ_P[sssp:$G_PROFILE]}")
        J_ALGO=deltaStep J_PARAM="delta=$od"
        run_job "$galois_bin" -t "$t" -startNode "$s" -delta "$od" -algo=deltaStep @timed:-noverify "$gr"
      fi
    done
  fi
  local lsgpu_bin=$B_GALOIS/gpu/$bench/$bench-gpu
  if ok_to_run lonestargpu "$lsgpu_bin" "$gr" gpu "lonestargpu $bench"; then
    J_LIB=lonestargpu J_ALGO=default J_T=gpu J_PARSE=p_lsgpu J_CHECK=c_dist_$bench J_PARAM=-
    local extra=()
    if [[ $bench == sssp && -n $LSGPU_SSSP_DELTA ]]; then
      extra=(-d "$LSGPU_SSSP_DELTA"); J_PARAM="d=$LSGPU_SSSP_DELTA"
    fi
    run_job "$lsgpu_bin" -s "$s" "${extra[@]}" @warm:-o @warm:out.txt "$gr"
  fi
  local gbin; gbin=$(gunrock_bin "$bench")
  if ok_to_run gunrock "${gbin:-$BUILD/gunrock/bin/$bench}" "$G_DIR/$G_NAME.mtx" gpu "gunrock $bench"; then
    if (( G_EDGES > 2147483647 )); then
      skip "gunrock $bench: over 2^31 edges"
    else
      J_LIB=gunrock J_ALGO=default J_T=gpu J_PARSE=p_gunrock J_CHECK=c_gunrock J_PARAM=-
      run_job "${gbin:-$BUILD/gunrock/bin/$bench}" -m "$G_DIR/$G_NAME.mtx" -s "$s" -n 1 @warm:--validate
    fi
  fi
  if want mbq "$LIBS" && ok_to_run mbq "$B_MBQ/$bench/$bench-cpu" "$gr" cpu "mbq $bench"; then
    mbq_galois "$bench" "$B_MBQ/$bench/$bench-cpu" "$gr" -startNode "$s"
  fi
  if [[ $bench == sssp ]]; then
    if ok_to_run mlmq "$BUILD/mlmq/mlmq-sssp" "$gr" gpu "mlmq sssp"; then
      if (( G_EDGES > 2147483647 )); then
        skip "mlmq sssp: over 2^31 edges"
      else
        J_LIB=mlmq J_ALGO=adaptive J_T=gpu J_PARSE=p_mlmq J_CHECK=c_mlmq J_PARAM=-
        run_job env MLMQ_SRC="$s" "$BUILD/mlmq/mlmq-sssp" -i "$gr"
      fi
    fi
    if ok_to_run adds "$BUILD/mlmq/adds-sssp" "$gr" gpu "adds sssp"; then
      J_LIB=adds J_ALGO=default J_T=gpu J_PARSE=p_adds J_CHECK=c_mlmq J_PARAM=-
      run_job "$BUILD/mlmq/adds-sssp" -s "$s" "$gr"
    fi
  fi
}

bench_pr() {
  local gr=$G_DIR/$G_NAME.gr tgr t
  J_SRC=- J_DST=-
  [[ $G_DIRECTED == yes ]] && tgr=$G_DIR/$G_NAME.tgr || tgr=$gr
  if ok_to_run galois "$B_GALOIS/cpu/pagerank/pagerank-pull-cpu" "$tgr" cpu "galois pr-pull"; then
    for t in $THREADS; do
      J_LIB=galois J_ALGO=pull-residual J_T=$t J_PARSE=p_galois J_CHECK=c_none J_PARAM="tol=$PR_TOL"
      run_job "$B_GALOIS/cpu/pagerank/pagerank-pull-cpu" -t "$t" -transposedGraph \
        -tolerance="$PR_TOL" @timed:-noverify "$tgr"
    done
  fi
  if ok_to_run galois "$B_GALOIS/cpu/pagerank/pagerank-push-cpu" "$gr" cpu "galois pr-push"; then
    for t in $THREADS; do
      J_LIB=galois J_ALGO=push-async J_T=$t J_PARSE=p_galois J_CHECK=c_none J_PARAM="tol=$PR_TOL"
      run_job "$B_GALOIS/cpu/pagerank/pagerank-push-cpu" -t "$t" -tolerance="$PR_TOL" \
        @timed:-noverify "$gr"
    done
  fi
  if ok_to_run lonestargpu "$B_GALOIS/gpu/pagerank/pagerank-gpu" "$gr" gpu "lonestargpu pr"; then
    J_LIB=lonestargpu J_ALGO=default J_T=gpu J_PARSE=p_lsgpu J_CHECK=c_none J_PARAM="x=$PR_MAX_ITER"
    run_job "$B_GALOIS/gpu/pagerank/pagerank-gpu" -x "$PR_MAX_ITER" "$gr"
  fi
  local gbin; gbin=$(gunrock_bin pr)
  if ok_to_run gunrock "${gbin:-$BUILD/gunrock/bin/pr}" "$G_DIR/$G_NAME.mtx" gpu "gunrock pr"; then
    # Tolerance PR_TOL per node, like Galois and MBQ (see build_gunrock).
    J_LIB=gunrock J_ALGO=default J_T=gpu J_PARSE=p_gunrock J_CHECK=c_none J_PARAM="tol=$PR_TOL/n"
    run_job env GUNROCK_PR_TOL="$PR_TOL" "${gbin:-$BUILD/gunrock/bin/pr}" -m "$G_DIR/$G_NAME.mtx" -n 1
  fi
  if want mbq "$LIBS" && ok_to_run mbq "$B_MBQ/pagerank/pagerank-push-cpu" "$gr" cpu "mbq pr"; then
    mbq_galois pr "$B_MBQ/pagerank/pagerank-push-cpu" "$gr" -tolerance="$PR_TOL"
  fi
}

bench_ppsp() {
  local line=$1 s d
  read -r s d < <(sed -n "${line}p" "$G_DIR/pairs.txt")
  [[ -n ${d:-} ]] || { skip "ppsp: pairs.txt has no line $line"; return; }
  J_SRC=$s J_DST=$d
  if ok_to_run mbq "$B_MBQ/ppsp/ppsp-cpu" "$G_DIR/$G_NAME.gr" cpu "mbq ppsp"; then
    mbq_galois ppsp "$B_MBQ/ppsp/ppsp-cpu" "$G_DIR/$G_NAME.gr" -startNode "$s" -destNode "$d"
  fi
}

bench_astar() {
  local line=$1 s d v bin t bin_dir
  local bin_file=$G_DIR/$G_NAME.bin
  read -r s d < <(sed -n "${line}p" "$G_DIR/pairs.txt")
  [[ -n ${d:-} ]] || { skip "astar: pairs.txt has no line $line"; return; }
  J_SRC=$s J_DST=$d
  for v in $ASTAR_VARIANTS; do
    case $v in orig) bin_dir=mbq-astar ;; fixed) bin_dir=mbq-astar-fixed ;;
      *) die "unknown ASTAR_VARIANTS entry: $v" ;; esac
    bin=$BUILD/$bin_dir/astar
    ok_to_run mbq "$bin" "$bin_file" cpu "mbq astar-$v" || continue
    # The Serial run writes the reference path the parallel runs must match.
    J_ASTAR_REF=$OUT/astar-ref/$G_LABEL.$v.$s.$d.path.txt
    mkdir -p "$OUT/astar-ref"
    if (( ! DRY_RUN )) && [[ ! -f $J_ASTAR_REF ]]; then
      local rd=$OUT/tmp/astar-ref-$$
      mkdir -p "$rd"
      (cd "$rd" && timeout "$TIMEOUT" "$bin" "$bin_file" "$s" "$d" Serial) \
        > "$OUT/logs/astar-ref.$G_LABEL.$v.$s.$d.log" 2>&1 && mv "$rd/path.txt" "$J_ASTAR_REF"
      rm -rf "$rd"
    fi
    J_LIB=mbq J_ALGO=astar-$v-Serial J_T=1 J_PARSE=p_runtime J_CHECK=c_astar J_PARAM=-
    run_job "$bin" "$bin_file" "$s" "$d" Serial
    for t in $THREADS; do
      J_ALGO=astar-$v-MQBucket J_T=$t J_PARAM=delta=11
      run_job "$bin" "$bin_file" "$s" "$d" MQBucket "$t" $(( 4 * t )) 4 4 11 64 0
    done
  done
}

bench_mis() {
  local g s t sym ag=$G_DIR/$G_NAME.sym.ag
  J_SRC=- J_DST=-
  [[ $G_DIRECTED == yes ]] && sym=$G_DIR/$G_NAME.sgr || sym=$G_DIR/$G_NAME.gr
  if ok_to_run galois "$B_GALOIS/cpu/independentset/maximal-independentset-cpu" "$sym" cpu "galois mis"; then
    for t in $THREADS; do
      J_LIB=galois J_ALGO=prio J_T=$t J_PARSE=p_galois J_CHECK=c_none J_PARAM=-
      run_job "$B_GALOIS/cpu/independentset/maximal-independentset-cpu" -t "$t" \
        -symmetricGraph @timed:-noverify "$sym"
    done
  fi
  if ok_to_run lonestargpu "$B_GALOIS/gpu/independentset/maximal-independentset-gpu" "$sym" gpu "lonestargpu mis"; then
    J_LIB=lonestargpu J_ALGO=default J_T=gpu J_PARSE=p_lsgpu J_CHECK=c_none J_PARAM=-
    run_job "$B_GALOIS/gpu/independentset/maximal-independentset-gpu" "$sym"
  fi
  if ok_to_run mbq "$B_PBBS/mqMIS/MIS" "$ag" cpu "mbq mis"; then
    local d b
    read -r d b _ <<< "${MBQ_P[mis:$G_PROFILE]}"
    J_INPUT=$ag
    for g in $MBQ_ALGOS; do
      [[ $g == MQBucket || $g == MQ ]] || continue
      for t in $THREADS; do
        J_LIB=mbq J_ALGO=mis-$g J_T=$t J_PARAM="delta=$d"
        local a=(-o out.txt -r "$RUNS" -type "$g" -threads "$t" -queues $(( 4 * t ))
                 -buckets 64 -batch1 "$b" -batch2 "$b")
        [[ $g == MQBucket ]] && a+=(-delta "$d")
        run_pbbs_mis "$B_PBBS/mqMIS/MIS" "${a[@]}" -prefetch "$ag"
      done
    done
    if [[ -x $B_PBBS/incrementalMIS/MIS ]]; then
      for t in $THREADS; do
        J_LIB=pbbs J_ALGO=mis-incremental J_T=$t J_PARAM=-
        run_pbbs_mis env CILK_NWORKERS="$t" "$B_PBBS/incrementalMIS/MIS" -o out.txt -r "$RUNS" "$ag"
      done
    fi
  fi
}

# Gunrock has no MIS; graph coloring is the closest. Opt-in (BENCHES=color):
# on the prep machine it ran over 500 s on germany-osm.
bench_color() {
  local gbin cmtx
  J_SRC=- J_DST=-
  gbin=$(gunrock_bin color)
  [[ $G_DIRECTED == yes ]] && cmtx=$G_DIR/$G_NAME.sym.mtx || cmtx=$G_DIR/$G_NAME.mtx
  if ok_to_run gunrock "${gbin:-$BUILD/gunrock/bin/color}" "$cmtx" gpu "gunrock color"; then
    J_LIB=gunrock J_ALGO=default J_T=gpu J_PARSE=p_gunrock J_CHECK=c_gunrock J_PARAM=-
    run_job "${gbin:-$BUILD/gunrock/bin/color}" "$cmtx"   # takes only a file name
  fi
}

bench_rbp() {
  local bin=$BUILD/mbq-rbp/rbp t g algo
  want mbq "$LIBS" || return
  if (( ! DRY_RUN )) && [[ ! -x $bin ]]; then skip "rbp: not built"; return; fi
  G_LABEL=ising-1000
  J_SRC=- J_DST=- J_LIB=mbq J_PARSE=p_runtime J_CHECK=c_rbp
  # The serial residual BP run writes output-ising-1000, which the relaxed
  # runs compare against (printed as AccuracyMax).
  local rd=$OUT/rbp
  mkdir -p "$rd"
  if (( ! DRY_RUN )) && [[ ! -f $rd/output-ising-1000 ]]; then
    note "rbp: serial reference"
    (cd "$rd" && timeout "$TIMEOUT" "$bin" residual ising 1000 0 0 0 0 0 0 0 0) \
      > "$OUT/logs/rbp-reference.log" 2>&1 || { skip "rbp: reference run failed"; return; }
  fi
  for g in $MBQ_ALGOS; do
    case $g in MQBucket) algo=bucket ;; MQ) algo=heap ;; *) continue ;; esac
    for t in $THREADS; do
      J_ALGO=rbp-$g J_T=$t J_PARAM=delta=7 J_BENCH=rbp
      run_job bash -c 'ln -sf "$1/output-ising-1000" . && shift && exec "$@"' _ "$rd" \
        "$bin" "$algo" ising 1000 "$t" $(( 4 * t )) 128 128 7 64 1 1
    done
  done
}

run_graph() {
  G_DIR=$1
  G_LABEL=$(basename "$G_DIR")
  G_NAME=$(awk -F': ' '$1=="name"{print $2}' "$G_DIR/info.txt")
  G_DIRECTED=$(awk -F': ' '$1=="directed"{print ($2 ~ /^yes/) ? "yes" : "no"}' "$G_DIR/info.txt")
  G_EDGES=$(awk -F': ' '$1=="edges"{print $2}' "$G_DIR/info.txt")
  G_PROFILE=$(profile "$G_NAME")
  note "graph $G_LABEL ($G_NAME, directed=$G_DIRECTED, $G_EDGES edges)"
  local main s line bench
  local -a srcs
  main=$(head -1 "$G_DIR/sources.txt")
  if [[ $SOURCES == all ]]; then
    # Only sources that reach at least half as many nodes as the main one.
    local reach; reach=$(ref_row "$main" | awk '{ print $2 }')
    mapfile -t srcs < <(while read -r s; do
      ref_row "$s" | awk -v r="$reach" '$2 >= r / 2 { print $1 }'
    done < "$G_DIR/sources.txt")
  else
    srcs=("$main")
  fi
  for bench in $BENCHES; do
    J_BENCH=$bench
    case $bench in
      bfs|sssp) for s in "${srcs[@]}"; do bench_bfs_sssp "$bench" "$s"; done ;;
      pr) bench_pr ;;
      ppsp) for line in $PAIRS; do bench_ppsp "$line"; done ;;
      astar)
        if [[ -f $G_DIR/$G_NAME.bin ]]; then
          for line in $PAIRS; do bench_astar "$line"; done
        elif want mbq "$LIBS"; then
          skip "mbq astar: no $G_NAME.bin (road graphs only)"
        fi ;;
      mis) bench_mis ;;
      color) bench_color ;;
      rbp) ;;   # no input graph; run once below
      *) die "unknown benchmark: $bench" ;;
    esac
  done
}

cmd_run() {
  [[ -d $GRAPH_DIR ]] || die "GRAPH_DIR not found: $GRAPH_DIR"
  mkdir -p "$OUT/logs" "$OUT/tmp"
  [[ -f $OUT/results.csv ]] ||
    echo "library,algorithm,benchmark,graph,source,target,threads,param,run,time_ms,valid,log" \
      > "$OUT/results.csv"
  note "results: $OUT/results.csv (runs already in $OUT/done.txt are skipped)"
  note "threads: $THREADS; runs: 1 warm-up + $RUNS timed; timeout ${TIMEOUT}s"
  # Several folders can hold the same graph (e.g. hyperlink2012-host and the
  # smaller hyperlink2012-host-new). Unless GRAPHS names folders, run only the
  # one with the most graph files.
  local d name n ext
  local -A best=() best_n=()
  for d in "$GRAPH_DIR"/*/; do
    d=${d%/}
    [[ -f $d/info.txt ]] || continue
    name=$(awk -F': ' '$1=="name"{print $2}' "$d/info.txt"); n=0
    for ext in gr tgr sgr mtx sym.mtx sym.ag bin; do [[ -f $d/$name.$ext ]] && n=$((n + 1)); done
    if [[ -z ${best[$name]:-} ]] || (( n > best_n[$name] )); then best[$name]=$d; best_n[$name]=$n; fi
  done
  for d in "$GRAPH_DIR"/*/; do
    d=${d%/}
    [[ -f $d/info.txt ]] || continue
    if [[ -n $GRAPHS ]]; then
      want "$(basename "$d")" "$GRAPHS" || continue
    else
      name=$(awk -F': ' '$1=="name"{print $2}' "$d/info.txt")
      if [[ ${best[$name]} != "$d" ]]; then
        note "skip $(basename "$d"): same graph as $(basename "${best[$name]}"), which has more files"
        continue
      fi
    fi
    for f in sources.txt pairs.txt reference.txt; do
      [[ -f $d/$f ]] || { note "skip $(basename "$d"): no $f"; continue 2; }
    done
    run_graph "$d"
  done
  if want rbp "$BENCHES"; then
    note "residual BP (generated input)"
    bench_rbp
  fi
  rm -rf "$OUT/tmp"
  (( DRY_RUN )) || cmd_summary
}

# Median and min of the timed runs (run >= 1) for each configuration.
cmd_summary() {
  [[ -f $OUT/results.csv ]] || die "no results yet: $OUT/results.csv"
  sort -t, -k1,8 -k10,10g "$OUT/results.csv" | awk -F, '
    $1 == "library" { next }   # the header, wherever sort put it
    { key = $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 }
    $9 == 0 { valid[key] = $11; next }
    !(key in valid) { valid[key] = $11 }   # PBBS MIS has no warm-up row
    $10 != "NA" { n[key]++; t[key, n[key]] = $10 }
    $10 == "NA" { failed[key] = $11 }
    END {
      print "library,algorithm,benchmark,graph,source,target,threads,param,runs,median_ms,min_ms,valid"
      for (k in valid) {
        status = (k in failed) ? failed[k] : valid[k]
        if (!(k in n)) { print k ",0,NA,NA," status; continue }
        m = n[k]; med = (m % 2) ? t[k, (m + 1) / 2] : (t[k, m / 2] + t[k, m / 2 + 1]) / 2
        print k "," m "," med "," t[k, 1] "," status
      }
    }' | (read -r h; echo "$h"; sort) > "$OUT/summary.csv"
  note "summary: $OUT/summary.csv"
  column -s, -t < "$OUT/summary.csv" | cut -c1-200
}

case ${1:-} in
  check) cmd_check ;;
  build) shift; cmd_build "$@" ;;
  run) cmd_run ;;
  all) cmd_check && { cmd_build || true; } && cmd_run ;;
  summary) cmd_summary ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
