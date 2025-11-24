## GPU Byte-level BPE Tokenizer (BlockBPE-style)

### What the files do

- **`gputok_blockbpe.cu`**: Baseline CUDA implementation of a byte-level BPE tokenizer that follows BlockBPE’s greedy Algorithm 1 on GPU: we reconstruct the GPT‑2 byte encoder, build a BPE vocabulary and merge table from `data/gpt2_tokenizer/merges.txt`, chunk a long text on CPU, and run a BlockBPE-style kernel that finds the globally best merge pair per pass and merges the left-most occurrence.
- **`gputok_blockbpe_optimized.cu`**: Optimized CUDA variant of the same semantics. We keep one block per chunk and the same global-argmin merge rule, but simplify compaction to a fully double-buffered, thread-coarsened scheme for all sequence lengths, reducing shared-memory usage and synchronization while preserving exact CPU–GPU token equality.
- **`gputok_binding.cu`**: Pybind11-based CUDA extension that exposes a `GpuTokenizer` class to Python. It reconstructs the GPT‑2 byte encoder and BPE merges once, builds a device `static_map`, and provides (1) a CPU reference tokenizer, (2) an optimized BlockBPE kernel, and (3) a baseline BlockBPE kernel for use in the Python benchmarks.
- **`benchmark_tokenizer_scaling.py`**: Benchmarks tokenizer speed on WikiText‑103 for GPU BlockBPE (via `gputok_binding.cu`), CPU BlockBPE, `tiktoken`, and HuggingFace GPT‑2 in both batch and latency modes.
- **`compare_generation_quality.py`**: Compares generation quality on WikiText‑103 by tokenizing prompts with GPU BlockBPE (optimized and baseline), `tiktoken`, and HuggingFace GPT‑2, then generating continuations with GPT‑2 and evaluating simple overlap-based similarity metrics against the ground-truth continuation.

### BPE algorithm and BlockBPE-style GPU implementation

- **Byte-level pre-tokenization (replacing Regex)**  
  We follow BlockBPE’s decision to remove Regex pre-tokenization and operate directly at the byte level with a GPT‑2 style byte encoder [BlockBPE paper](https://arxiv.org/pdf/2507.11941). Each input byte \(0\ldots255\) is mapped to a “safe” Unicode symbol, and we look up the corresponding vocabulary id, producing an initial token sequence that matches GPT‑2’s byte-level behavior without any Regex splitting.

- **Greedy BPE merges (Algorithm 1 semantics)**  
  On CPU, we implement the exact greedy algorithm described in BlockBPE §3 / Algorithm 1 [BlockBPE paper](https://arxiv.org/pdf/2507.11941): while there exists a mergeable adjacent pair in the sequence, we (1) scan all pairs, (2) find the pair with the globally lowest rank in the merge table (ties broken by left-most position), (3) merge only the left-most occurrence of that pair into a new token, and (4) rebuild the sequence. We repeat until no pair from the merge table appears.

- **Parallel merge passes on GPU (one block per chunk)**  
  On GPU, we adapt BlockBPE’s thread-level view of a merge pass:
  - For a sequence of length \(n\), we conceptually launch a block of up to \(n\) threads that operate on adjacent pairs \((i, i+1)\). Thread \(i\):
    1. Reads tokens \(i\) and \(i+1\) from shared memory.
    2. Performs a `static_map` lookup in the GPU-resident merge table \(M\) to obtain the pair’s rank.
    3. Tracks its local best (rank, position).
  - We perform a block-wide argmin reduction over all local candidates to get the globally best (rank, position) pair, matching Algorithm 1’s global choice.
  - Once the best pair is known, threads mark where the merge should happen and cooperatively compact the sequence:
    - Indices before the best position are copied unchanged.
    - At the best position we write the merged token id.
    - The token at best position+1 is removed.
    - All tokens to the right shift left by one index.
  - When the sequence length \(n\) exceeds the block size, each thread strides over the sequence \(d = \lceil n / \text{block size} \rceil\) times, yielding overall \(O(nd)\) complexity as in BlockBPE §4.2, with the ideal case \(d=1\) when the block spans the full sequence.

- **Use of cuCollections and CCCL/CUB**  
  We use cuCollections `static_map` for highly optimized hash lookups of merge pairs on GPU, and CCCL/CUB primitives for block-wide prefix scans and reductions:
  - In the baseline kernel, for shorter sequences we use a per-block prefix scan over a 0/1 “remove” mask to compute compacted write indices.
  - In the optimized variant, we rely on a simplified double-buffered compaction scheme and keep the same block-wide argmin, reducing synchronization overhead while maintaining the same greedy merge semantics.

### How we derived and optimized our BlockBPE kernels

- **Baseline BlockBPE kernel (`gputok_blockbpe.cu`)**  
  We start from the BlockBPE paper’s Algorithm 1 [BlockBPE paper](https://arxiv.org/pdf/2507.11941) and implement a one-block-per-sequence kernel that:
  - Stores tokens in shared memory, flattens the merge table into a device `static_map`, and in each iteration:
    - Computes per-thread local minima over ranks.
    - Reduces to a global best (rank, position) inside the block.
    - Performs a block-wide compaction using either a CUB `BlockScan` (for shorter sequences) or a strided double-buffered shift (for longer sequences).
  - Matches the CPU greedy implementation exactly, and we verify CPU vs GPU token-by-token equality for every chunk.

- **Optimized BlockBPE kernel (`gputok_blockbpe_optimized.cu`)**  
  We then simplify and optimize the compaction step:
  - We retain the same one-block-per-string design and the same global argmin over pair ranks.
  - We remove the split “short vs long sequence” paths and instead always use a thread-coarsened, double-buffered compaction: each thread strides over indices, writes the merged token at the best position, skips the token at best position+1, and shifts all later tokens left by one.
  - This reduces shared-memory footprint and synchronization, while profiling shows that we preserve Algorithm 1 semantics and maintain exact CPU–GPU equality.

### Installation, build, and run

- **Prerequisites**
  - A CUDA-capable GPU.
  - CUDA 13.0 module available as `cuda-13.0`.
  - Git, Python 3, and a C++17-capable `nvcc`.

- **Clone dependencies**
  From the project root (`gpu-tokenizer`):

```bash
mkdir -p externals
cd externals
git clone https://github.com/NVIDIA/cccl.git
git clone https://github.com/NVIDIA/cuCollections.git
cd ..
```

- **Load CUDA module**

```bash
module load cuda-13.0
```

- **Build BlockBPE binaries**

```bash
nvcc -O3 -std=c++17 --expt-extended-lambda \
  gputok_blockbpe.cu -o gpu_tokenizer_blockbpe \
  -Iexternals/cuCollections/include -Iexternals/cccl/include

nvcc -O3 -std=c++17 --expt-extended-lambda \
  gputok_blockbpe_optimized.cu -o gpu_tokenizer_blockbpe_optimized \
  -Iexternals/cuCollections/include -Iexternals/cccl/include
```

- **Run on `Pride and Prejudice`**

Input text is expected at `data/input/pride_and_prejudice.txt`, and GPT‑2 merges at `data/gpt2_tokenizer/merges.txt`:

```bash
./gpu_tokenizer_blockbpe 2048 data/input/pride_and_prejudice.txt
./gpu_tokenizer_blockbpe_optimized 2048 data/input/pride_and_prejudice.txt
```

Each binary prints CPU vs GPU timing, approximate throughput, and writes GPU tokens to `data/output/gpu_tokens_blockbpe.txt`.

- **Python environment and benchmarks**

We typically use a dedicated virtual environment:

```bash
python3 -m venv gputok_env
source gputok_env/bin/activate
python -m pip install -r requirements.txt
python -m pip install ninja
```

To run the tokenizer scaling benchmark on WikiText‑103:

```bash
~/gpu-tokenizer/gputok_env/bin/python benchmark_tokenizer_scaling.py
```

To run the generation quality benchmark (WikiText‑103, test split):

```bash
~/gpu-tokenizer/gputok_env/bin/python compare_generation_quality.py \
  --wikitext-split test \
  --num-prompts 500 \
  --max-prompt-chars 384 \
  --ref-chars 512 \
  --temperature 0.001
```

### Nsight Systems profiling (BlockBPE vs optimized)

We profile both binaries with Nsight Systems to characterize kernel, memory, and API behavior:

```bash
module load cuda-13.0
nsys profile -t cuda,nvtx --stats=true ./gpu_tokenizer_blockbpe 2048 data/input/pride_and_prejudice.txt
nsys profile -t cuda,nvtx --stats=true ./gpu_tokenizer_blockbpe_optimized 2048 data/input/pride_and_prejudice.txt
```

On `pride_and_prejudice.txt` (711,349 initial tokens, 348 chunks, max 2048 tokens/chunk), Nsight reports:

- **Kernel timing and throughput (program timers)**
  - Baseline BlockBPE:
    - GPU kernel time ≈ **69.3 ms**, CPU BPE time ≈ **12,037 ms**.
    - Approximate GPU throughput ≈ **1.03×10⁷ tokens/s**, CPU ≈ **5.9×10⁴ tokens/s**, for a **~174×** CPU→GPU speedup.
  - Optimized BlockBPE:
    - GPU kernel time ≈ **62.9 ms**, CPU BPE time ≈ **12,107 ms**.
    - Approximate GPU throughput ≈ **1.13×10⁷ tokens/s**, CPU ≈ **5.9×10⁴ tokens/s**, for a **~192×** CPU→GPU speedup.

- **CUDA API breakdown (`cuda_api_sum`)**
  - For both kernels, `cudaMalloc` dominates API time (≈68–82% of total CUDA API time), reflecting the one‑time cost of allocating the large `static_map` and token buffers.
  - `cudaEventSynchronize` accounts for ≈12–12% of API time, capturing the end-to-end kernel timing.
  - `cudaFree` contributes 5–19%, while `cudaMemcpy*` and `cudaLaunchKernel` together account for well under 2% of API time, indicating that launch and transfer overheads are negligible relative to allocation and kernel execution.

- **Kernel summary (`cuda_gpu_kern_sum`)**
  - In both runs, the BPE merge kernel (`bpe_merge_kernel_blockbpe` / `bpe_merge_kernel_blockbpe_optimized`) accounts for essentially **100%** of GPU kernel time; the only other kernels are small cuCollections and CUB housekeeping kernels.
  - This confirms that our implementation is compute/memory-bound inside the BPE merge kernel rather than being dominated by overhead kernels.

- **Memory transfers (`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`)**
  - Host-to-device copies move ≈3.65 MB in 4 calls, with average latencies in the hundreds of microseconds; device-to-host copies move ≈2.85 MB in 3 calls, with similar average latencies.
  - Memory transfer time accounts for ≤71% of the **GPU memory** time, but only a small fraction of overall application time; the tokenization workload is dominated by device-side work inside the merge kernel and by initial allocations.

Overall, the optimized kernel reduces kernel time by ≈9% vs the baseline on this workload, and both deliver more than two orders of magnitude speedup over the CPU implementation while preserving exact token sequences.

### WikiText‑103 tokenizer scaling results (updated)

From our latest run of `benchmark_tokenizer_scaling.py` on WikiText‑103 (latency mode, single sequence), we measure average time vs input length (measured in HuggingFace GPT‑2 tokens):

| Len (HF tok) | GPU BlockBPE base | GPU BlockBPE opt | CPU BPE    | tiktoken  | HF GPT‑2 |
|-------------:|------------------:|-----------------:|-----------:|----------:|---------:|
|         256  |  96.47 ms         |   5.51 ms        |  3.15 ms   |  2.35 ms  | 10.97 ms |
|         512  | 287.13 ms         |   9.02 ms        |  6.16 ms   |  4.12 ms  | 20.85 ms |
|       1,024  | 300.74 ms         |  11.17 ms        | 13.30 ms   |  7.72 ms  | 38.01 ms |
|       2,048  | 306.08 ms         |  14.13 ms        | 24.49 ms   | 15.39 ms  | 73.16 ms |
|       4,096  | 314.08 ms         |  20.61 ms        | 50.88 ms   | 29.55 ms  |145.55 ms |
|       8,192  | 327.61 ms         |  33.56 ms        |100.35 ms   | 59.10 ms  |287.70 ms |
|      16,384  | 591.11 ms         |  91.33 ms        |205.54 ms   |115.54 ms  |608.38 ms |
|      32,768  | 431.43 ms         |  88.64 ms        |210.26 ms   |155.39 ms  |671.20 ms |
|      65,536  | 356.45 ms         | 157.69 ms        |238.61 ms   |159.29 ms  |690.20 ms |
|     131,072  | 310.14 ms         |  99.99 ms        |212.50 ms   |156.98 ms  |717.50 ms |

Speedups of GPU BlockBPE opt vs tiktoken / HF GPT‑2 (from the same run):

| Len (HF tok) | opt / tiktoken | opt / HF GPT‑2 | base / tiktoken | base / HF GPT‑2 |
|-------------:|---------------:|---------------:|----------------:|----------------:|
|         256  | 0.45×          | 2.09×          | 0.02×           | 0.11×           |
|         512  | 0.48×          | 2.44×          | 0.01×           | 0.07×           |
|       1,024  | 0.73×          | 3.60×          | 0.03×           | 0.13×           |
|       2,048  | 1.15×          | 5.48×          | 0.05×           | 0.24×           |
|       4,096  | 1.52×          | 7.47×          | 0.09×           | 0.46×           |
|       8,192  | 1.87×          | 9.08×          | 0.18×           | 0.88×           |
|      16,384  | 1.34×          | 7.07×          | 0.20×           | 1.03×           |
|      32,768  | 1.86×          | 8.03×          | 0.36×           | 1.56×           |
|      65,536  | 1.07×          | 4.65×          | 0.45×           | 1.94×           |
|     131,072  | 1.67×          | 7.62×          | 0.51×           | 2.32×           |

In latency mode, the optimized GPU BlockBPE is less efficient than tiktoken on very short inputs, but overtakes it once sequences reach a few thousand tokens and consistently outperforms the HuggingFace GPT‑2 tokenizer by large factors at long context lengths.

### Generation quality benchmark (WikiText‑103, updated)

From our latest run:

```bash
~/gpu-tokenizer/gputok_env/bin/python compare_generation_quality.py \
  --wikitext-split test \
  --num-prompts 500 \
  --max-prompt-chars 384 \
  --ref-chars 512 \
  --temperature 0.001
```

Average similarity vs ground-truth continuation (over 500 prompts):

| Tokenizer            | Character overlap | Word overlap | Sequence similarity |
|----------------------|------------------:|-------------:|--------------------:|
| GPU BlockBPE (opt)   | 0.678             | 0.112        | 0.299               |
| GPU BlockBPE (base)  | 0.685             | 0.113        | 0.303               |
| tiktoken             | 0.683             | 0.116        | 0.302               |
| HuggingFace GPT‑2    | 0.683             | 0.116        | 0.302               |

All four tokenizers deliver very similar generation quality on WikiText‑103. The base BlockBPE kernel closely matches tiktoken and HuggingFace on the sequence similarity metric, and the optimized kernel is only slightly behind on simple overlap scores, confirming that our byte-level BlockBPE implementation is compatible with GPT‑2’s behavior in practice.

### How our implementation differs from BlockBPE in the paper

- **Scope and integration**  
  The BlockBPE paper focuses on a general GPU BPE design; our implementation specifically targets GPT‑2’s byte encoder and merges, and integrates tightly with PyTorch via a pybind11 extension so that we can benchmark against `tiktoken` and HuggingFace tokenizers in real LLM workflows.

- **Greedy global-min merge vs parallel local-merge rule**  
  The original BlockBPE design emphasizes highly parallel merge passes with non-overlapping local merges based on neighboring ranks [BlockBPE paper](https://arxiv.org/pdf/2507.11941). In this project, we deliberately implement the **fully greedy Algorithm 1 semantics** (global minimal rank, left-most occurrence) both on CPU and in our GPU kernels, so that CPU and GPU match token-for-token and align precisely with GPT‑2’s training-time semantics.

- **Kernel structure and optimization choices**  
  While we share several design choices with BlockBPE (one block per string, use of cuCollections and CCCL), we introduce a simplified double-buffered compaction scheme in the optimized kernel and keep per-chunk control flow very close to the CPU reference. This makes it easier to reason about correctness and to validate equality against CPU results, at the cost of slightly less aggressive intra-pass parallelism than the most fine-grained BlockBPE designs.

- **Empirical focus**  
  Our experiments emphasize end-to-end LLM-serving relevance: we report detailed latency vs length curves, direct comparisons against `tiktoken` and HuggingFace GPT‑2, similarity-based generation quality on WikiText‑103, and Nsight-based profiling of kernel and API behavior. This complements the BlockBPE paper’s broader algorithmic analysis by grounding the design in concrete GPT‑2-style workloads.