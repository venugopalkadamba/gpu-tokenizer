#!/usr/bin/env bash
set -euo pipefail

# Sweep block sizes for baseline and optimized GPU tokenizers and report timings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BLOCK_SIZES=(128 256 512 1024)
CHUNK_TOKENS=2048
INPUT_PATH="data/input/pride_and_prejudice.txt"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "ERROR: Input file '${INPUT_PATH}' not found." >&2
  exit 1
fi

echo "Block size profiling on '${INPUT_PATH}' (chunk_tokens=${CHUNK_TOKENS})"
echo
echo "variant,block_size,cpu_ms,gpu_ms,gpu_tokens_per_s"

for bs in "${BLOCK_SIZES[@]}"; do
  # Build baseline and optimized with this block size
  nvcc -O3 -std=c++17 --expt-extended-lambda -DBLOCK_SIZE="${bs}" \
    gputok.cu -o "gpu_tokenizer_bs${bs}" \
    -Iexternals/cuCollections/include -Iexternals/cccl/include

  nvcc -O3 -std=c++17 --expt-extended-lambda -DBLOCK_SIZE="${bs}" \
    gputok_optimized.cu -o "gpu_tokenizer_optimized_bs${bs}" \
    -Iexternals/cuCollections/include -Iexternals/cccl/include

  # Run baseline
  out_base="$(./gpu_tokenizer_bs${bs} "${CHUNK_TOKENS}" "${INPUT_PATH}")"
  cpu_ms_base="$(grep 'CPU BPE time' <<< "${out_base}" | awk '{print $(NF-1)}')"
  gpu_ms_base="$(grep 'GPU BPE kernel time (' <<< "${out_base}" | awk '{print $(NF-1)}')"
  gpu_tokps_base="$(grep 'GPU kernel throughput:' <<< "${out_base}" | awk '{print $(NF-1)}')"

  echo "baseline,${bs},${cpu_ms_base},${gpu_ms_base},${gpu_tokps_base}"

  # Run optimized
  out_opt="$(./gpu_tokenizer_optimized_bs${bs} "${CHUNK_TOKENS}" "${INPUT_PATH}")"
  cpu_ms_opt="$(grep 'CPU BPE time' <<< "${out_opt}" | awk '{print $(NF-1)}')"
  gpu_ms_opt="$(grep 'GPU BPE kernel time (optimized' <<< "${out_opt}" | awk '{print $(NF-1)}')"
  gpu_tokps_opt="$(grep 'GPU optimized kernel throughput:' <<< "${out_opt}" | awk '{print $(NF-1)}')"

  echo "optimized,${bs},${cpu_ms_opt},${gpu_ms_opt},${gpu_tokps_opt}"
done


