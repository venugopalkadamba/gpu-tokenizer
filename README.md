## GPU Byte-level BPE Tokenizer (BlockBPE-style)

### What the files do

- **`gputok.cu`**: Baseline CUDA implementation of a byte-level BPE tokenizer using cuCollections `static_map`. It:
  - Reconstructs the GPT-2 byte encoder.
  - Builds a BPE vocabulary and merge table from `data/gpt2_tokenizer/merges.txt`.
  - Tokenizes a large text (default: `data/input/pride_and_prejudice.txt`) into token chunks on CPU.
  - Runs the BPE merge algorithm on CPU and GPU, compares outputs, and reports timing/throughput.
- **`gputok_optimized.cu`**: Optimized CUDA version of the same tokenizer. The algorithm is identical, but the GPU kernel:
  - Caches both the merge **rank** and **new token id** for each adjacent pair in shared memory.
  - Avoids a second `static_map` lookup when actually merging pairs.
  - This reduces global memory traffic and hash lookups, improving GPU kernel throughput.
- **`gputok_binding.cu`**: Pybind11-based CUDA extension used by the Python benchmarks to expose a `GpuTokenizer` class to Python.
- **`benchmark_tokenizer_scaling.py`**: Benchmarks how tokenizer speed scales with input length on WikiText‑103 for:
  - GPU BlockBPE (via `gputok_binding.cu`, if the extension can be built),
  - `tiktoken` GPT‑2 tokenizer,
  - HuggingFace GPT‑2 tokenizer.
- **`compare_generation_quality.py`**: Compares generation quality on WikiText‑103:
  - For each sampled prompt, tokenizes with GPU BlockBPE, `tiktoken`, and HuggingFace GPT‑2,
  - Generates continuations with GPT‑2,
  - Compares each continuation against the ground-truth continuation with simple similarity metrics.

### BPE algorithm (simple explanation, optimized CUDA version)

1. **Byte encoding**: Each input byte (0–255) is mapped to a “safe” Unicode symbol (GPT‑2 byte encoder), giving an initial sequence of token ids (one per byte).
2. **Merge table**: From `merges.txt`, we build a table of adjacent token pairs → (rank, new_token_id). Lower rank = higher priority merge.
3. **Iterative merge passes** (per chunk / per CUDA block):
   - For each position `i`, look up the pair `(token[i], token[i+1])` in a GPU `static_map`.
   - If the pair exists, cache its `rank` and `new_token_id` in shared memory.
   - Decide which positions to merge using a local rule:
     - A pair can merge only if its rank is better than its left neighbor and **not worse** than its right neighbor, ensuring non‑overlapping merges.
   - Sequentially compact the sequence:
     - If `merge_here[i]` is set, write `new_token_id` and skip `i+1`.
     - Otherwise, copy the original token.
   - Repeat until no more merges fire or `max_iters` is reached.
4. **Optimized kernel change**: Compared to `gputok.cu`, the optimized kernel stores both rank and new token id in shared memory per pair, so the actual merge step does **not** re-query the hash map. This yields higher GPU throughput with identical outputs.

### How to build and run the CUDA tokenizers

#### Prerequisites

- CUDA toolkit with `nvcc` available.
- A GPU with sufficient memory.
- cuCollections and CCCL are vendored under `externals/` in this repo.

#### Build

From the project root (`gpu-tokenizer`):

```bash
nvcc -O3 -std=c++17 --expt-extended-lambda \
  gputok.cu -o gpu_tokenizer \
  -Iexternals/cuCollections/include -Iexternals/cccl/include

nvcc -O3 -std=c++17 --expt-extended-lambda \
  gputok_optimized.cu -o gpu_tokenizer_optimized \
  -Iexternals/cuCollections/include -Iexternals/cccl/include
```

#### Run on `Pride and Prejudice`

Input text is expected at `data/input/pride_and_prejudice.txt`, and GPT‑2 merges at `data/gpt2_tokenizer/merges.txt`.

```bash
./gpu_tokenizer 2048 data/input/pride_and_prejudice.txt
./gpu_tokenizer_optimized 2048 data/input/pride_and_prejudice.txt
```

Each binary prints CPU vs GPU timing, throughput, and writes GPU tokens to:

- Baseline: `data/output/gpu_tokens.txt`
- Optimized: `data/output/gpu_tokens_optimized.txt`

### How to run the Python benchmarks

#### 1. Python environment and dependencies

Create virtual environment

```bash
python3 -m venv env_name
```

Activate environment

```bash
source env_name/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```


For GPU BlockBPE via the pybind11 extension, you also need:

- `ninja` (for `torch.utils.cpp_extension.load`),
- CUDA toolkit and a working GPU.

Install `ninja` if needed:

```bash
python -m pip install ninja
```

#### 2. Tokenizer scaling benchmark

Runs WikiText‑103 scaling for tiktoken and HF GPT‑2 (GPU BlockBPE is enabled if the extension builds):

```bash
python benchmark_tokenizer_scaling.py
```

This will:

- Download WikiText‑103,
- Build prompts at various GPT‑2 token lengths,
- Print per‑length throughput and summary tables for batch and per‑prompt (latency) modes.

#### 3. Generation quality benchmark

Runs quality comparison on WikiText‑103 (using subprocess GPU tokenizer if the extension cannot be built):

```bash
python compare_generation_quality.py \
  --wikitext-split test \
  --num-prompts 100 \
  --max-prompt-chars 512 \
  --ref-chars 512
```

This will:

- Sample 100 (prompt, continuation) pairs from WikiText‑103,
- Tokenize prompts with GPU BlockBPE, tiktoken, and HF GPT‑2,
- Generate continuations with GPT‑2,
- Report averaged similarity scores vs. ground truth.

### Key benchmark results (from your runs)

#### A. Single-file GPU tokenizer (`Pride and Prejudice`)

**Input**: `pride_and_prejudice.txt`  
**Total initial tokens**: 711,349 (348 chunks, max 2048 tokens/chunk)  
**Default block size**: `BLOCK_SIZE=256`

| Variant              | CPU BPE time (ms) | GPU kernel time (ms) | CPU tok/s       | GPU tok/s        | CPU→GPU speedup |
|----------------------|-------------------|----------------------|-----------------|------------------|-----------------|
| Baseline (`gputok`)  | 41.50             | 4.07                 | 1.71×10⁷        | 1.75×10⁸         | **10.2×**       |
| Optimized (`gputok_optimized`) | 44.05   | 2.47                 | 1.61×10⁷        | 2.88×10⁸         | **17.9×**       |

**Summary**: With the tuned block size, the optimized kernel delivers roughly **1.6×** higher GPU throughput than the baseline (≈1.75e8 → 2.88e8 tokens/s) while keeping outputs identical to the CPU reference.

#### A.1. Nsight Systems profiling (baseline vs optimized, `BLOCK_SIZE=256`)

Using `nsys profile -t cuda,nvtx --stats=true` on the same workload:

- **Measured kernel time (from the program’s timers)**:
  - Baseline: GPU kernel ≈ **3.9–4.1 ms**, ≈ **1.8×10⁸ tokens/s**.
  - Optimized: GPU kernel ≈ **2.5–2.8 ms**, ≈ **2.6–2.9×10⁸ tokens/s**.
  - Across runs, the optimized kernel is reliably **≈1.4–1.6×** faster than baseline, which is consistent with the throughput improvement reported in the main table.
- **CUDA API breakdown (Nsight `cuda_api_sum`)**:
  - In both variants, **`cudaMalloc` dominates API time** (≈92–96% of total CUDA API time), reflecting the one‑time cost of allocating the large `static_map` and token buffers.
  - `cudaLaunchKernel` contributes only **≈0.3–0.6%** of CUDA API time, and `cudaMemcpy*` another few percent, indicating that launch and transfer overheads are negligible compared to kernel execution and setup.
- **Critical assessment** (is the kernel “healthy”?):
  - Host‑side behavior (allocations, copies, launch counts) is essentially identical between baseline and optimized builds, so the observed speedup must come from the device code itself:
    - The **baseline kernel** does a `static_map` lookup twice per merged pair (once for rank, once for new token id).
    - The **optimized kernel** performs a single lookup per pair, caching both rank and new token id in shared memory and reusing them during compaction.
  - The fact that kernel time scales cleanly with block size (A.2), and that GPU throughput improves without any loss of correctness, suggests the optimized kernel is effectively exploiting on‑chip memory while remaining primarily memory‑lookup bound rather than launch‑bound—a profile that is typical and acceptable for hash‑table–driven GPU algorithms.

#### A.2. Effect of CUDA block size (baseline vs optimized)

Using `profile_block_sizes.sh` to sweep `BLOCK_SIZE ∈ {128, 256, 512, 1024}` on the same workload:

| Variant   | Block size | GPU kernel time (ms) | GPU throughput (tokens/s) |
|-----------|------------|----------------------|---------------------------|
| baseline  | 128        | 3.99974              | 1.78×10⁸                  |
| baseline  | 256        | 3.82157              | 1.86×10⁸                  |
| baseline  | 512        | 5.88902              | 1.21×10⁸                  |
| baseline  | 1024       | 14.8552              | 4.79×10⁷                  |
| optimized | 128        | 2.77811              | 2.56×10⁸                  |
| optimized | 256        | 2.47296              | 2.88×10⁸                  |
| optimized | 512        | 2.48832              | 2.86×10⁸                  |
| optimized | 1024       | 6.20134              | 1.15×10⁸                  |

**Summary**:

- For both kernels, very large blocks (1024 threads) **hurt performance**—kernel time grows and throughput drops, likely due to low SM occupancy and shared-memory pressure per block.
- For the baseline kernel, **256 threads/block** is slightly better than 128 on this GPU; 512+ quickly degrades.
- For the optimized kernel, **256–512 threads/block** are both near-optimal and clearly better than 128; 1024 is again much worse.
- In practice, starting with `BLOCK_SIZE=256` is a good default; on GPUs with more registers/SMs, experimenting with 512 may yield a small additional gain.

#### B. Tokenizer scaling benchmark (WikiText‑103)

With the `gputok_gpu` extension built (using `BLOCK_SIZE=256` under the hood), the scaling benchmark reports GPU BlockBPE, CPU BlockBPE, tiktoken, and HF GPT‑2 throughput.

**Batch throughput (tokens/sec, higher is better):**

| HF tokens | GPU BlockBPE (M) | CPU BlockBPE (M) | tiktoken (M) | HF GPT‑2 (M) | GPU/CPU | GPU/TK | GPU/HF |
|----------:|-----------------:|-----------------:|-------------:|-------------:|--------:|-------:|-------:|
| 256       | 4.26             | 2.10             | 3.64         | 1.70         | 2.03×   | 1.17×  | 2.50×  |
| 512       | 4.74             | 2.38             | 3.63         | 2.59         | 1.99×   | 1.31×  | 1.83×  |
| 1,024     | 5.42             | 2.63             | 3.79         | 3.18         | 2.06×   | 1.43×  | 1.71×  |
| 2,048     | 6.31             | 2.69             | 4.20         | 4.36         | 2.34×   | 1.50×  | 1.45×  |
| 4,096     | 8.40             | 2.91             | 4.51         | 3.68         | 2.89×   | 1.86×  | 2.28×  |
| 8,192     | 7.62             | 2.69             | 4.41         | 3.10         | 2.83×   | 1.73×  | 2.46×  |
| 16,384    | 7.08             | 2.71             | 4.59         | 1.75         | 2.61×   | 1.54×  | 4.05×  |
| 32,768    | 6.22             | 2.97             | 4.50         | 0.83         | 2.09×   | 1.38×  | 7.48×  |

**Latency-mode throughput (per-prompt, tokens/sec):**

| HF tokens | GPU BlockBPE (M) | CPU BlockBPE (M) | tiktoken (M) | HF GPT‑2 (M) | GPU/CPU | GPU/TK | GPU/HF |
|----------:|-----------------:|-----------------:|-------------:|-------------:|--------:|-------:|-------:|
| 256       | 0.47             | 2.08             | 2.55         | 0.75         | 0.22×   | 0.18×  | 0.62×  |
| 512       | 0.80             | 2.27             | 3.39         | 0.83         | 0.36×   | 0.24×  | 0.97×  |
| 1,024     | 1.41             | 2.58             | 3.85         | 0.93         | 0.55×   | 0.37×  | 1.52×  |
| 2,048     | 2.82             | 2.56             | 4.48         | 0.98         | 1.10×   | 0.63×  | 2.88×  |
| 4,096     | 3.94             | 2.56             | 4.22         | 0.92         | 1.54×   | 0.93×  | 4.27×  |
| 8,192     | 6.14             | 2.89             | 3.93         | 0.85         | 2.12×   | 1.56×  | 7.24×  |
| 16,384    | 5.80             | 2.94             | 4.43         | 0.93         | 1.97×   | 1.31×  | 6.24×  |
| 32,768    | 6.46             | 2.47             | 3.83         | 0.88         | 2.61×   | 1.69×  | 7.33×  |

**Summary**: In batch mode, GPU BlockBPE is **2–3× faster than CPU BlockBPE** and consistently faster than both tiktoken and HF GPT‑2, especially at longer lengths. In latency mode, GPU BlockBPE is less efficient for very short prompts but overtakes CPU and HF GPT‑2 once prompts reach a few thousand tokens, and becomes competitive with or faster than tiktoken for long contexts.

#### C. Generation quality benchmark (WikiText‑103, 200 prompts)

Average similarity vs ground‑truth continuation (over 200 prompts):

| Tokenizer     | Char overlap | Word overlap | Seq similarity |
|---------------|-------------:|-------------:|---------------:|
| GPU BlockBPE  | 0.712        | 0.113        | 0.300          |
| tiktoken      | 0.695        | 0.114        | 0.300          |
| HuggingFace   | 0.717        | 0.118        | 0.299          |

**Summary**: All three tokenizers produce **very similar** generation quality. The HF tokenizer is slightly ahead on the simple overlap metrics; GPU BlockBPE matches tiktoken on sequence similarity and stays very close overall, confirming that the custom GPU BPE is compatible with GPT‑2’s behavior in practice.

### How the GPU algorithm differs from naive byte-level BPE

- **Naive byte-level BPE** (typical CPU implementation):
  - Treats the text as one long token sequence and repeatedly scans it left‑to‑right.
  - Each iteration finds mergeable pairs and rebuilds the sequence, largely sequentially.
  - Parallelism is limited and most work happens on scalar CPU state.
- **Current GPU BlockBPE-style algorithm**:
  - Splits long texts into fixed‑size chunks and assigns one CUDA block per chunk, enabling many sequences to be processed in parallel.
  - Within a block, threads:
    - Look up all adjacent pairs in a GPU `static_map`, writing ranks (and, in the optimized kernel, new token ids) into shared memory.
    - Apply a **local non‑overlapping merge rule** based on neighboring ranks, then compact the sequence in-place in shared memory.
  - This design maximizes shared‑memory reuse, minimizes redundant hash lookups (in the optimized kernel), and turns BPE into a highly parallel, GPU‑friendly procedure rather than a naive sequential byte‑level loop.


