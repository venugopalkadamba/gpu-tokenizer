# CUDA BPE Implementation Roadmap

## 📋 Current Status

### ✅ What You Have:

```
Requirements (Data):
├── data/vocab.json        ✓ 50,257 tokens, 2MB
├── data/merges.txt        ✓ 50,000 merge rules, 500KB
└── Test text              ✓ "Hello, world!"

Code:
├── bpe_cpu.py             ✓ Python baseline (working)
├── cuda/bpe_cpu.cu        ✓ C++ version (just created)
├── cuda/bpe_kernels.cu    ✓ GPU kernels (pair counting + merges)
└── cuda/bpe_gpu_demo.cu   ✓ Toy GPU merge demo (build with `make demo`)
```

---

## 🎯 Your Plan (Step-by-Step):

### **Phase 1: CPU Version in C++** (1-2 days)

**Goal:** Understand the algorithm in C++ before GPU

```
Step 1.1: Compile bpe_cpu.cu          ← YOU ARE HERE
Step 1.2: Fix JSON parsing (use library or simple parser)
Step 1.3: Test with "Hello" → verify correctness
Step 1.4: Benchmark CPU performance
```

**Commands:**
```bash
cd cuda
g++ -O3 -std=c++11 -o bpe_cpu bpe_cpu.cu
./bpe_cpu
```

**Why this step?**
- Understand C++ data structures
- Get familiar with CUDA file structure
- Verify logic before parallelizing

---

### **Phase 2: Basic GPU Version** (3-4 days)

**Goal:** Parallelize the two bottlenecks

#### Bottleneck #1: Finding Best Pair
```cpp
// CPU version (SLOW):
Pair find_best_pair(const std::set<Pair>& pairs) {
    for (const Pair& p : pairs) {  // ← Sequential scan
        // Check merge_ranks...
    }
}

// GPU version (FAST):
__global__ void find_best_pair_kernel(
    Pair* pairs,
    int* merge_ranks,
    int num_pairs,
    Pair* best_pair  // Output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks one pair in parallel!
    if (idx < num_pairs) {
        // Parallel reduction to find minimum
    }
}
```

#### Bottleneck #2: Merging Pairs
```cpp
// CPU version (SLOW):
std::vector<Token> merge_pair(const std::vector<Token>& word, const Pair& pair) {
    for (size_t i = 0; i < word.size(); i++) {  // ← Sequential
        // Check and merge...
    }
}

// GPU version (FAST):
__global__ void merge_pair_kernel(
    Token* word_in,
    Token* word_out,
    int* valid_mask,
    Pair pair,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one position in parallel!
    if (idx < length) {
        // Check if position starts the pair to merge
    }
}
```

---

### **Phase 3: Optimization** (3-4 days)

**Goal:** Make it fast!

#### Optimization 1: Batching
```cpp
// Process multiple texts in parallel
__global__ void batch_encode_kernel(
    char** texts,        // Array of text strings
    int* text_lengths,
    int num_texts,
    int** output_tokens  // Array of token arrays
) {
    // Each block processes one text
    int text_idx = blockIdx.x;

    if (text_idx < num_texts) {
        // Tokenize texts[text_idx]
    }
}
```

#### Optimization 2: Shared Memory
```cpp
__global__ void optimized_kernel(...) {
    // Use shared memory for frequently accessed data
    __shared__ int merge_ranks_cache[1024];

    // Load into shared memory (fast!)
    if (threadIdx.x < 1024) {
        merge_ranks_cache[threadIdx.x] = merge_ranks[threadIdx.x];
    }
    __syncthreads();

    // Now use merge_ranks_cache instead of global memory
}
```

#### Optimization 3: Memory Coalescing
```cpp
// BAD: Threads access non-consecutive memory
tokens[threadIdx.x * 100]  // Stride of 100

// GOOD: Threads access consecutive memory
tokens[threadIdx.x]  // Stride of 1
```

---

### **Phase 4: cuCollections (Optional - Advanced)**

**What is cuCollections?**
- NVIDIA library for GPU hash maps, sets, etc.
- Very fast lookups on GPU
- Useful for vocabulary lookup!

**Example Use Case:**
```cpp
#include <cuco/static_map.cuh>

// Create GPU hash map for vocabulary
cuco::static_map<std::string_view, int> gpu_vocab;

// Fast parallel lookup!
__global__ void lookup_tokens_kernel(...) {
    // O(1) lookup instead of O(log n)
    auto result = gpu_vocab.find(token);
}
```

**When to use:**
- After basic GPU version works
- When vocabulary lookup becomes bottleneck
- For 10-20% additional speedup

**Installation:**
```bash
git clone https://github.com/NVIDIA/cuCollections
# Include in your CUDA code
```

---

## 📊 Expected Performance

| Version | Time (1KB text) | Speedup |
|---------|-----------------|---------|
| Python CPU | ~50ms | 1x (baseline) |
| C++ CPU | ~10ms | 5x |
| Basic GPU | ~5ms | 10x |
| Optimized GPU | ~2ms | 25x |
| + Batching (100 texts) | ~20ms total | 250x |
| + cuCollections | ~15ms total | 330x |

---

## 🔧 What You Need (Requirements):

### 1. **Vocabulary (vocab.json)** ✓
```json
{
  "!": 0,
  "\"": 1,
  ...
  "Hello": 15496,
  ...
}
```
**Location:** `data/vocab.json` (already have!)

### 2. **Merge Rules (merges.txt)** ✓
```
#version: 0.2
Ġ t
Ġ a
h e
...
```
**Location:** `data/merges.txt` (already have!)

### 3. **CUDA Compiler (nvcc)** ✓
```bash
module load cuda-12.2  # You already did this!
```

### 4. **JSON Parser (Need to add)**

**Option A: Simple custom parser** (recommended for now)
```cpp
// Parse simple JSON manually
// Good for learning, no dependencies
```

**Option B: Use nlohmann/json library**
```cpp
#include "json.hpp"
// Professional, but adds dependency
```

**Option C: Start without JSON**
```cpp
// Hard-code a small vocabulary for testing
vocab["H"] = 1;
vocab["e"] = 2;
vocab["l"] = 3;
// ...
```

---

## 🎯 Immediate Next Steps:

### Step 1: Compile & Test CPU Version (TODAY)

```bash
cd cuda

# Try compiling
g++ -O3 -std=c++11 -o bpe_cpu bpe_cpu.cu

# If JSON parsing fails, add simple hard-coded vocab first:
vim bpe_cpu.cu
# Add in load_vocab():
#   vocab["H"] = 1;
#   vocab["e"] = 2;
#   vocab["l"] = 3;
#   vocab["l"] = 4;
#   vocab["o"] = 5;

./bpe_cpu
```

### Step 2: Understand Data Flow (TODAY)

```
Input: "Hello"
  ↓
Split to chars: ['H', 'e', 'l', 'l', 'o']
  ↓
Get pairs: {('H','e'), ('e','l'), ('l','l'), ('l','o')}
  ↓
Find best pair: ('l', 'l') has rank 234
  ↓
Merge: ['H', 'e', 'll', 'o']
  ↓
Repeat...
  ↓
Final: ['Hello']
  ↓
Lookup ID: 15496
```

### Step 3: Plan GPU Conversion (TOMORROW)

Read `cuda/bpe_cpu.cu` and identify:
- Which functions to convert to `__global__`?
- What data needs to be on GPU?
- Where are the parallel opportunities?

---

## 💡 Key Insights:

### Why C++ First?
1. **Type safety** - Catch errors before GPU
2. **Debugging** - Easier to debug on CPU
3. **Understanding** - Learn data structures
4. **Verification** - Ensure correctness

### Why cuCollections Later?
1. **Complexity** - Adds another dependency
2. **Diminishing returns** - Maybe only 10-20% speedup
3. **Learning** - Better to master basic CUDA first

### Focus Areas:
```
Week 1: ✓ Python baseline
Week 2: C++ CPU → Basic GPU
Week 3: Optimization (batching, shared memory)
Week 4: Benchmarking, cuCollections (if time), Report
```

---

## 📚 Resources:

### CUDA Learning:
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/
- Your existing code: `cuda/bpe_kernels.cu` (study this!)

### cuCollections:
- GitHub: https://github.com/NVIDIA/cuCollections
- Docs: https://nvidia.github.io/cuCollections/
- When: After basic GPU works!

### BlockBPE Paper:
- PDF: https://arxiv.org/abs/2507.11941
- Key sections: 3 (Algorithm), 4 (Implementation)

---

## ✅ Summary:

**What you have:**
- ✅ Python baseline working
- ✅ Vocabulary and merge files downloaded
- ✅ C++ skeleton code created
- ✅ CUDA environment setup

**What you need:**
- ⏭️ Compile C++ version
- ⏭️ Add JSON parsing (or hard-code small vocab)
- ⏭️ Convert to GPU kernels
- ⏭️ Optimize and benchmark

**Priority:**
1. **NOW:** Get C++ CPU version working
2. **NEXT:** Convert to basic GPU
3. **THEN:** Optimize
4. **MAYBE:** Add cuCollections

---

**Ready to compile the C++ version?** 🚀

```bash
cd cuda
g++ -O3 -std=c++11 -o bpe_cpu bpe_cpu.cu
```
