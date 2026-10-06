#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build/toolchain-test}"

check_version() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '%s version mismatch: expected %s, found %s\n' "${name}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

check_version cmake "$(cmake --version | awk 'NR == 1 { print $3 }')" 3.25.1
check_version g++ "$(g++ -dumpfullversion -dumpversion)" 12.2.0

nvcc_version="$(nvcc --version)"
if ! grep -q 'release 11\.8, V11\.8\.89' <<<"${nvcc_version}"; then
  printf 'nvcc version mismatch: expected CUDA 11.8.89\n%s\n' "${nvcc_version}" >&2
  exit 1
fi

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"
cmake --build "${build_dir}" --parallel