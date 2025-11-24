#!/usr/bin/env bash
set -euo pipefail

# Sweep block sizes and sequence lengths for the BlockBPE GPU tokenizer and report timings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BLOCK_SIZES=(128 256 512 1024)
SEQ_LENS=(128 256 512 1024 2048 4096)
INPUT_PATH="data/input/pride_and_prejudice.txt"

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "ERROR: Input file '${INPUT_PATH}' not found." >&2
  exit 1
fi

echo "BlockBPE block-size profiling on '${INPUT_PATH}'"
echo
echo "block_size,seq_len,cpu_ms,gpu_ms,gpu_tokens_per_s"

for bs in "${BLOCK_SIZES[@]}"; do
  # Build BlockBPE tokenizer with this block size
  nvcc -O3 -std=c++17 --expt-extended-lambda -DBLOCK_SIZE="${bs}" \
    gputok_blockbpe.cu -o "gpu_tokenizer_blockbpe_bs${bs}" \
    -Iexternals/cuCollections/include -Iexternals/cccl/include

  for seq_len in "${SEQ_LENS[@]}"; do
    out="$(./gpu_tokenizer_blockbpe_bs${bs} "${seq_len}" "${INPUT_PATH}")"
    cpu_ms="$(grep 'CPU BPE (greedy Algorithm 1) time' <<< "${out}" | awk '{print $(NF-1)}')"
    gpu_ms="$(grep 'GPU BlockBPE-style greedy kernel time' <<< "${out}" | awk '{print $(NF-1)}')"
    gpu_tokps="$(grep 'GPU greedy kernel throughput:' <<< "${out}" | awk '{print $(NF-1)}')"

    echo "${bs},${seq_len},${cpu_ms},${gpu_ms},${gpu_tokps}"
  done
done



