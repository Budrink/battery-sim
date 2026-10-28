#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scenario="${1:-$repo_dir/examples/reaction-chain-demo.json}"
if (($#)); then shift; fi

cuda_compiler="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuda_host_compiler="${CUDAHOSTCXX:-$(command -v c++)}"
cmake -S "$repo_dir" -B "$repo_dir/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_compiler" -DCMAKE_CUDA_HOST_COMPILER="$cuda_host_compiler" \
  -DCMAKE_CUDA_ARCHITECTURES="${BATTERY_CUDA_ARCHITECTURES:-native}"
cmake --build "$repo_dir/build" --target battery_sim_viewer -j 4
exec "$repo_dir/build/battery_sim_viewer" "$scenario" "$@"
