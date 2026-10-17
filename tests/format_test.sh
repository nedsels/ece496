#!/usr/bin/env bash
set -euo pipefail

unformatted_files=()

while IFS= read -r -d '' file; do
  if [[ "$file" == *.cut ]]; then
    if ! clang-format --style=file --dry-run --Werror \
      --assume-filename="${file%.cut}.cuh" < "$file" >/dev/null; then
      unformatted_files+=("$file")
    fi
  else
    if ! clang-format --style=file --dry-run --Werror "$file"; then
      unformatted_files+=("$file")
    fi
  fi
done < <(
  find src graphs tests -type f \( \
    -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.tpp' -o \
    -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cut' \
  \) -print0
)

if ((${#unformatted_files[@]} > 0)); then
  printf 'Files needing clang-format:\n'
  printf '  %s\n' "${unformatted_files[@]}"
  exit 1
fi

printf 'All checked files are clang-formatted.\n'