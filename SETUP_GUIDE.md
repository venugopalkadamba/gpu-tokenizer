# GPU Tokenizer - Complete Setup Guide

## Your Hardware Setup
- **2x NVIDIA GeForce GPUs** (6GB each)
- **CUDA Version**: 11.4
- **Platform**: Linux with module system

---

## Quick Start (First Experiment)

### Step 1: Load CUDA Module
```bash
module load cuda-11.4
nvcc --version  # Verify it works
```

### Step 2: Create Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Dependencies
```bash
# Install PyTorch with CUDA 11.x support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CuPy for CUDA 11.x
pip install cupy-cuda11x

# Install Numba for easy CUDA kernel writing
pip install numba

# Install HuggingFace for baseline comparison
pip install transformers tokenizers datasets

# Install additional utilities
pip install numpy pandas matplotlib tqdm
```

### Step 4: Verify Installation
```bash
python3 test.py
```

---

## 4-Week Project Roadmap

### Week 1: Setup & CUDA Fundamentals (Oct 22-29)
**Goal**: Get comfortable with CUDA programming

**Tasks**:
- [x] Environment validation
- [ ] Install all dependencies
- [ ] Write "Hello World" CUDA kernel (Numba)
- [ ] Implement matrix multiplication in CUDA
  - CPU version
  - Naive GPU version
  - Optimized GPU version (shared memory)
- [ ] Learn CUDA concepts:
  - Thread/block/grid hierarchy
  - Memory types (global, shared, local)
  - Memory coalescing
  - Synchronization

**Deliverable**: Working matrix multiplication benchmark showing GPU speedup

---

### Week 2: BPE Understanding & CPU Baseline (Oct 30 - Nov 5)
**Goal**: Deep understanding of BPE algorithm

**Tasks**:
- [ ] Study BPE algorithm thoroughly
  - Watch: "Byte Pair Encoding Explained" video
  - Read: BlockBPE paper (https://arxiv.org/abs/2507.11941)
- [ ] Download GPT-2 vocabulary files
  - vocab.json (50,257 tokens)
  - merges.txt (50,000 merge rules)
- [ ] Implement CPU BPE tokenizer from scratch
  - Basic BPE encoder
  - Decoder
  - Handle special tokens
- [ ] Create test dataset
  - Small: 1K sentences
  - Medium: 10K sentences
  - Large: 100K sentences
- [ ] Benchmark HuggingFace tokenizer (baseline)
- [ ] Profile CPU implementation (find bottlenecks)

**Deliverable**:
- Working CPU BPE implementation
- Benchmark results vs HuggingFace
- Identified parallelization opportunities

---

### Week 3: GPU Implementation (Nov 6-12)
**Goal**: First working GPU tokenizer

**Key Algorithm Insights from BlockBPE**:
1. **Parallel Merge Detection**: Use GPU to find all mergeable pairs simultaneously
2. **Conflict Resolution**: Handle overlapping merges correctly
3. **Batch Processing**: Process multiple sequences in parallel
4. **Vocabulary Lookup**: Efficient hash table on GPU

**Phase 3A: Basic GPU BPE (Days 1-3)**:
- [ ] Port vocabulary lookup to GPU
- [ ] Implement parallel pair finding kernel
- [ ] Implement merge kernel (naive version)
- [ ] Test on small inputs (100-1000 chars)

**Phase 3B: Optimization (Days 4-7)**:
- [ ] Use shared memory for vocabulary cache
- [ ] Implement memory coalescing for token access
- [ ] Add batching support (process multiple texts)
- [ ] Optimize merge iteration loop
- [ ] Profile with NVIDIA Nsight/nvprof

**Deliverable**:
- Working GPU BPE tokenizer
- 2-3x speedup over CPU (initial target)

---

### Week 4: Optimization & Final Report (Nov 13-19)
**Goal**: Production-ready implementation with comprehensive evaluation

**Tasks**:
- [ ] Advanced optimizations
  - Warp-level primitives
  - Stream compaction for active merges
  - Multi-GPU support (if time permits)
- [ ] Comprehensive benchmarking
  - vs HuggingFace CPU
  - vs HuggingFace Rust (faster)
  - vs tiktoken (OpenAI's tokenizer)
  - Different text sizes (100B, 1KB, 10KB, 100KB, 1MB)
  - Different batch sizes (1, 10, 100, 1000)
- [ ] Create visualizations
  - Speedup graphs
  - Throughput (tokens/sec)
  - Scaling with batch size
- [ ] Write final report
  - Introduction & motivation
  - Algorithm description
  - Implementation details
  - Results & analysis
  - Limitations & future work
- [ ] Prepare presentation

**Target Performance** (based on literature):
- 2-5x speedup vs HuggingFace CPU for single sequences
- 10-50x speedup for large batches
- Higher speedup for longer sequences

**Deliverable**: Complete project with report and code

---

## Critical Concepts to Learn

### CUDA Basics
1. **Thread Hierarchy**:
   - Thread → Warp (32 threads) → Block → Grid
   - `blockIdx`, `threadIdx`, `blockDim`, `gridDim`

2. **Memory Hierarchy** (speed & size):
   - Registers (fastest, per-thread, ~64KB)
   - Shared memory (fast, per-block, ~48KB)
   - L1/L2 cache (automatic)
   - Global memory (slow, 6GB)
   - Constant memory (read-only, cached)

3. **Memory Access Patterns**:
   - Coalesced access: threads access consecutive addresses
   - Bank conflicts in shared memory
   - Alignment requirements

4. **Optimization Techniques**:
   - Use shared memory for frequently accessed data
   - Minimize global memory accesses
   - Avoid warp divergence (if/else with different threads)
   - Maximize occupancy

### BPE-Specific Challenges
1. **Variable-length sequences**: Need dynamic scheduling
2. **Iterative merges**: Multiple kernel launches or loops
3. **Dependency chain**: Later merges depend on earlier ones
4. **Vocabulary size**: 50K+ tokens, need efficient lookup

### Key Papers to Read
1. **BlockBPE** (YOUR PRIMARY REFERENCE): https://arxiv.org/abs/2507.11941
   - Read sections 3 (Algorithm) and 4 (Implementation) carefully
   - Note their use of "blocks" for parallel processing

2. **NVIDIA Subword Tokenizer**: https://developer.nvidia.com/blog/gpu-subword-tokenizer/
   - Focus on WordPiece (similar to BPE)
   - Learn vocabulary lookup optimization

---

## Development Tools

### Debugging
```bash
# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Check for CUDA errors in Python
import torch
torch.cuda.synchronize()
```

### Profiling
```bash
# PyTorch profiler
import torch.profiler

# Numba profile
from numba import cuda
cuda.profile_start()
# ... your code ...
cuda.profile_stop()

# NVIDIA Nsight Compute (detailed kernel analysis)
ncu --set full -o profile python your_script.py

# NVIDIA Nsight Systems (timeline view)
nsys profile -o timeline python your_script.py
```

### Testing
```python
# Always validate against HuggingFace
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Hello, world!"

# Reference output
expected = tokenizer.encode(text)

# Your output
actual = your_gpu_tokenizer.encode(text)

assert expected == actual, "Tokenization mismatch!"
```

---

## Recommended Learning Path

### Days 1-2: CUDA Basics
- [ ] CUDA Programming Guide (read chapters 2-3)
- [ ] Tutorial: Matrix multiplication with CUDA
- [ ] Practice: Vector addition, reduction kernels

### Days 3-4: BPE Algorithm
- [ ] Watch BPE explanation videos
- [ ] Implement BPE in pure Python (100 lines)
- [ ] Test with GPT-2 vocabulary

### Days 5-7: Study BlockBPE Paper
- [ ] Read paper 3 times (skim → detailed → notes)
- [ ] Understand parallel merge algorithm
- [ ] Sketch pseudocode

### Week 2+: Implementation
Follow the weekly roadmap above

---

## Success Metrics

### Minimum Viable Project (Grade: B)
- Working GPU BPE tokenizer
- Correct output matching HuggingFace
- 1.5-2x speedup over CPU
- Basic report with methodology and results

### Strong Project (Grade: A-)
- 3-5x speedup over CPU
- Batching support
- Comprehensive benchmarks
- Good code quality with comments
- Detailed report with analysis

### Excellent Project (Grade: A/A+)
- 5-10x+ speedup
- Multiple optimizations (shared memory, coalescing, etc.)
- Comparison with multiple baselines
- Scaling analysis (batch size, sequence length)
- Publication-quality report with insights
- Clean, documented, reproducible code

---

## Common Pitfalls to Avoid

1. **Don't implement everything in pure CUDA C++** - Start with Numba/CuPy
2. **Don't optimize prematurely** - Get correctness first, then speed
3. **Don't skip the CPU baseline** - You need something to beat!
4. **Don't ignore memory bottlenecks** - Profile early and often
5. **Don't hardcode vocabulary** - Use GPT-2's actual vocab files
6. **Don't forget to batch** - Single-sequence performance is less impressive
7. **Don't skip validation** - Always check against HuggingFace output

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate
module load cuda-11.4

# Run tests
python test.py

# Run your tokenizer
python bpe_cpu.py  # Week 2
python bpe_gpu.py  # Week 3-4

# Check GPU usage
nvidia-smi

# Profile code
python -m cProfile -o profile.stats your_script.py
```

---

## Getting Help

- **CUDA Documentation**: https://docs.nvidia.com/cuda/
- **Numba CUDA**: https://numba.readthedocs.io/en/stable/cuda/
- **PyTorch CUDA**: https://pytorch.org/docs/stable/notes/cuda.html
- **BlockBPE Code**: Check if authors released code on GitHub

---

**Remember**: Focus on correctness first, then performance. You have 4 weeks - use them wisely!

Good luck!