#!/usr/bin/env python3
"""
Python Wrapper for CUDA BPE Kernels

This loads your compiled CUDA code and calls it from Python
"""

import ctypes
import numpy as np
from pathlib import Path
import time

class BPETokenizerCUDA:
    """
    GPU-accelerated BPE tokenizer using your CUDA kernels
    """

    def __init__(self, cuda_lib_path="cuda/libbpe_cuda.so"):
        """Load CUDA library"""
        lib_path = Path(cuda_lib_path)

        if not lib_path.exists():
            raise FileNotFoundError(
                f"CUDA library not found: {lib_path}\n"
                f"Compile it first:\n"
                f"  cd cuda\n"
                f"  make\n"
            )

        # Load shared library
        self.lib = ctypes.CDLL(str(lib_path))

        # Define function signatures
        self._setup_function_signatures()

        print(f"✓ Loaded CUDA library: {lib_path}")

    def _setup_function_signatures(self):
        """Define C function signatures for ctypes"""

        # void cuda_hello()
        self.lib.cuda_hello.argtypes = []
        self.lib.cuda_hello.restype = None

        # void cuda_count_pairs(const int*, int, int*, int)
        self.lib.cuda_count_pairs.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,                   # length
            ctypes.POINTER(ctypes.c_int),  # pair_counts
            ctypes.c_int                    # vocab_size
        ]
        self.lib.cuda_count_pairs.restype = None

    def hello(self):
        """Test function - prints GPU info"""
        self.lib.cuda_hello()

    def count_pairs_gpu(self, tokens: np.ndarray, vocab_size: int) -> np.ndarray:
        """
        Count adjacent pairs using GPU

        Args:
            tokens: numpy array of token IDs (int32)
            vocab_size: size of vocabulary

        Returns:
            2D numpy array of pair counts (vocab_size x vocab_size)
        """
        # Ensure correct dtype
        tokens = np.asarray(tokens, dtype=np.int32)
        length = len(tokens)

        # Allocate output array
        pair_counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        # Get pointers
        tokens_ptr = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        counts_ptr = pair_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call CUDA function
        self.lib.cuda_count_pairs(
            tokens_ptr,
            length,
            counts_ptr,
            vocab_size
        )

        return pair_counts

    def benchmark_pair_counting(self, text_size=10000, vocab_size=1000, num_runs=100):
        """
        Benchmark GPU pair counting vs CPU

        This demonstrates the speedup!
        """
        print(f"\n{'='*70}")
        print(f" Pair Counting Benchmark: GPU vs CPU")
        print(f"{'='*70}")
        print(f"Text size: {text_size} tokens")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Runs: {num_runs}")

        # Generate random tokens
        tokens = np.random.randint(0, vocab_size, size=text_size, dtype=np.int32)

        # CPU version
        print(f"\nRunning CPU version...")
        cpu_times = []
        for _ in range(num_runs):
            start = time.perf_counter()

            # Count pairs on CPU
            pair_counts_cpu = np.zeros((vocab_size, vocab_size), dtype=np.int32)
            for i in range(len(tokens) - 1):
                pair_counts_cpu[tokens[i], tokens[i+1]] += 1

            elapsed = time.perf_counter() - start
            cpu_times.append(elapsed)

        cpu_avg = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)

        # GPU version
        print(f"Running GPU version...")
        gpu_times = []
        for _ in range(num_runs):
            start = time.perf_counter()

            # Count pairs on GPU
            pair_counts_gpu = self.count_pairs_gpu(tokens, vocab_size)

            elapsed = time.perf_counter() - start
            gpu_times.append(elapsed)

        gpu_avg = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)

        # Verify correctness
        pair_counts_cpu = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        for i in range(len(tokens) - 1):
            pair_counts_cpu[tokens[i], tokens[i+1]] += 1

        pair_counts_gpu = self.count_pairs_gpu(tokens, vocab_size)

        if np.array_equal(pair_counts_cpu, pair_counts_gpu):
            correctness = "✓ CORRECT"
        else:
            correctness = "✗ MISMATCH"
            diff = np.sum(np.abs(pair_counts_cpu - pair_counts_gpu))
            correctness += f" (diff: {diff})"

        # Results
        speedup = cpu_avg / gpu_avg

        print(f"\n{'─'*70}")
        print(f" Results")
        print(f"{'─'*70}")
        print(f"CPU Time:   {cpu_avg*1000:.3f} ± {cpu_std*1000:.3f} ms")
        print(f"GPU Time:   {gpu_avg*1000:.3f} ± {gpu_std*1000:.3f} ms")
        print(f"Speedup:    {speedup:.2f}x")
        print(f"Correctness: {correctness}")
        print(f"{'─'*70}")

        if speedup > 1.0:
            print(f"\n🚀 GPU is {speedup:.2f}x FASTER!")
        else:
            print(f"\n🐌 GPU is {1/speedup:.2f}x SLOWER")
            print(f"   Note: GPU has overhead. Try larger text_size (100K+)")

        return {
            'cpu_time': cpu_avg,
            'gpu_time': gpu_avg,
            'speedup': speedup,
            'correct': np.array_equal(pair_counts_cpu, pair_counts_gpu)
        }


def main():
    """Demo the CUDA tokenizer"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║            BPE CUDA Tokenizer - Python Wrapper                   ║
    ║            Calling YOUR CUDA C++ Code!                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Load CUDA library
        tokenizer = BPETokenizerCUDA()

        # Test 1: GPU info
        print("\n" + "="*70)
        print(" Test 1: GPU Information")
        print("="*70)
        tokenizer.hello()

        # Test 2: Simple pair counting
        print("\n" + "="*70)
        print(" Test 2: Simple Pair Counting")
        print("="*70)

        tokens = np.array([1, 2, 3, 2, 3], dtype=np.int32)
        print(f"Input tokens: {tokens}")

        pair_counts = tokenizer.count_pairs_gpu(tokens, vocab_size=10)

        print(f"\nPair counts:")
        for i in range(10):
            for j in range(10):
                if pair_counts[i, j] > 0:
                    print(f"  ({i}, {j}): {pair_counts[i, j]}")

        # Test 3: Benchmark
        print("\n" + "="*70)
        print(" Test 3: Performance Benchmark")
        print("="*70)

        # Small size (GPU overhead visible)
        print("\n--- Small text (1K tokens) ---")
        tokenizer.benchmark_pair_counting(text_size=1000, vocab_size=100, num_runs=50)

        # Medium size
        print("\n--- Medium text (10K tokens) ---")
        tokenizer.benchmark_pair_counting(text_size=10000, vocab_size=1000, num_runs=50)

        # Large size (GPU should win!)
        print("\n--- Large text (100K tokens) ---")
        tokenizer.benchmark_pair_counting(text_size=100000, vocab_size=1000, num_runs=20)

        print("\n" + "="*70)
        print(" ✓ All tests complete!")
        print("="*70)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo compile CUDA code:")
        print("  1. Load CUDA: module load cuda-11.4")
        print("  2. cd cuda")
        print("  3. make")
        print("  4. cd ..")
        print("  5. python bpe_cuda_wrapper.py")


if __name__ == "__main__":
    main()
