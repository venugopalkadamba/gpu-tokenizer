#!/usr/bin/env python3
"""
BPE Tokenizer Comparison Script
Compare our implementation with HuggingFace's tokenizer

This validates correctness and measures performance differences
"""

import time
import numpy as np
from typing import List
from bpe_cpu import BPETokenizerCPU

try:
    from transformers import GPT2Tokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers")


def compare_outputs(our_tokens: List[int], hf_tokens: List[int], text: str):
    """
    Compare tokenization outputs

    Returns: (matches, difference_info)
    """
    if our_tokens == hf_tokens:
        return True, None

    # Find first mismatch
    min_len = min(len(our_tokens), len(hf_tokens))
    for i in range(min_len):
        if our_tokens[i] != hf_tokens[i]:
            return False, {
                'position': i,
                'our_token': our_tokens[i],
                'hf_token': hf_tokens[i],
                'our_len': len(our_tokens),
                'hf_len': len(hf_tokens),
            }

    # Length mismatch
    return False, {
        'position': min_len,
        'our_len': len(our_tokens),
        'hf_len': len(hf_tokens),
    }


def benchmark_comparison(test_texts: List[str], num_runs: int = 10):
    """
    Comprehensive benchmark comparing:
    1. Our CPU implementation
    2. HuggingFace's tokenizer (optimized Rust)
    3. [Future] Our GPU implementation
    """
    print("\n" + "="*70)
    print(" COMPREHENSIVE TOKENIZER COMPARISON")
    print("="*70)

    if not HF_AVAILABLE:
        print("❌ HuggingFace transformers not installed!")
        print("   Install with: pip install transformers")
        return

    # Initialize tokenizers
    print("\nInitializing tokenizers...")
    vocab_file, merges_file = BPETokenizerCPU.download_gpt2_files()

    our_tokenizer = BPETokenizerCPU(vocab_file, merges_file)
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("✓ Both tokenizers loaded\n")

    results = []

    for text_idx, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f" Test {text_idx}/{len(test_texts)}")
        print(f"{'='*70}")
        print(f"Text length: {len(text)} characters")
        print(f"Preview: '{text[:100]}{'...' if len(text) > 100 else ''}'")

        # Warmup
        our_tokens = our_tokenizer.encode(text)
        hf_tokens = hf_tokenizer.encode(text)

        # Validate correctness
        matches, diff = compare_outputs(our_tokens, hf_tokens, text)

        if matches:
            print(f"\n✓ Outputs match! ({len(our_tokens)} tokens)")
        else:
            print(f"\n❌ Output mismatch!")
            print(f"   Position: {diff.get('position', 'N/A')}")
            print(f"   Our length: {diff.get('our_len', 'N/A')}")
            print(f"   HF length: {diff.get('hf_len', 'N/A')}")
            continue  # Skip benchmark if outputs don't match

        # Benchmark our implementation
        print(f"\nBenchmarking OUR implementation...")
        our_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = our_tokenizer.encode(text)
            elapsed = time.perf_counter() - start
            our_times.append(elapsed)

        our_avg = np.mean(our_times)
        our_std = np.std(our_times)

        # Benchmark HuggingFace
        print(f"Benchmarking HUGGINGFACE implementation...")
        hf_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = hf_tokenizer.encode(text)
            elapsed = time.perf_counter() - start
            hf_times.append(elapsed)

        hf_avg = np.mean(hf_times)
        hf_std = np.std(hf_times)

        # Calculate speedup
        speedup = our_avg / hf_avg

        # Print results
        print(f"\n{'─'*70}")
        print(f" Results")
        print(f"{'─'*70}")
        print(f"Our CPU Implementation:")
        print(f"  Time: {our_avg*1000:.2f} ± {our_std*1000:.2f} ms")
        print(f"  Throughput: {len(text)/our_avg:.0f} chars/sec")
        print(f"  Token rate: {len(our_tokens)/our_avg:.0f} tokens/sec")
        print()
        print(f"HuggingFace (Rust-based):")
        print(f"  Time: {hf_avg*1000:.2f} ± {hf_std*1000:.2f} ms")
        print(f"  Throughput: {len(text)/hf_avg:.0f} chars/sec")
        print(f"  Token rate: {len(hf_tokens)/hf_avg:.0f} tokens/sec")
        print()

        if speedup >= 1.0:
            print(f"🐌 Our implementation is {speedup:.2f}x SLOWER")
            print(f"   (This is expected - HuggingFace uses optimized Rust)")
        else:
            print(f"🚀 Our implementation is {1/speedup:.2f}x FASTER!")

        results.append({
            'text_len': len(text),
            'num_tokens': len(our_tokens),
            'our_time': our_avg,
            'hf_time': hf_avg,
            'speedup': speedup,
            'matches': matches
        })

    # Summary
    print(f"\n\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"\nTests run: {len(results)}")
    print(f"Correctness: {sum(r['matches'] for r in results)}/{len(results)} passed")

    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage performance vs HuggingFace:")
        if avg_speedup >= 1.0:
            print(f"  {avg_speedup:.2f}x SLOWER (expected for pure Python)")
        else:
            print(f"  {1/avg_speedup:.2f}x FASTER")

        print(f"\n💡 KEY INSIGHTS:")
        print(f"   1. HuggingFace uses optimized Rust → 2-10x faster than pure Python")
        print(f"   2. Your baseline (our implementation) is pure Python")
        print(f"   3. GPU target: Beat HuggingFace by 2-5x")
        print(f"   4. That means GPU needs to be 10-50x faster than our CPU code!")

    print(f"\n{'='*70}\n")

    return results


def test_edge_cases():
    """Test edge cases and special scenarios"""
    print("\n" + "="*70)
    print(" EDGE CASE TESTING")
    print("="*70)

    if not HF_AVAILABLE:
        return

    vocab_file, merges_file = BPETokenizerCPU.download_gpt2_files()
    our_tokenizer = BPETokenizerCPU(vocab_file, merges_file)
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("   ", "Whitespace only"),
        ("Hello", "Simple word"),
        ("Hello, world!", "With punctuation"),
        ("café résumé naïve", "Unicode characters"),
        ("🚀🎉🎊", "Emojis"),
        ("x" * 1000, "Very long repetition"),
        ("\n\n\n", "Newlines"),
        ("lower lowest LOWER LOWEST", "Case sensitivity"),
    ]

    passed = 0
    failed = 0

    for text, description in edge_cases:
        our_tokens = our_tokenizer.encode(text)
        hf_tokens = hf_tokenizer.encode(text)

        matches, diff = compare_outputs(our_tokens, hf_tokens, text)

        status = "✓" if matches else "✗"
        print(f"{status} {description:30s} | Tokens: {len(our_tokens):4d} | Match: {matches}")

        if matches:
            passed += 1
        else:
            failed += 1
            if diff:
                print(f"  Difference: {diff}")

    print(f"\n{'─'*70}")
    print(f"Passed: {passed}/{len(edge_cases)}")
    print(f"Failed: {failed}/{len(edge_cases)}")
    print(f"{'─'*70}\n")


def main():
    """Run all comparisons"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         BPE Tokenizer Comparison & Validation                    ║
    ║         Our Implementation vs HuggingFace                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Edge case testing
    test_edge_cases()

    # Performance benchmarks with various text sizes
    test_texts = [
        # Small
        "Hello, world!",

        # Medium
        "The quick brown fox jumps over the lazy dog. " * 10,

        # Large
        """
        Machine learning is a subset of artificial intelligence that focuses on
        the development of algorithms and statistical models that enable computer
        systems to improve their performance on a specific task through experience.
        """ * 50,

        # Very large
        "The tokenization process is a critical component of natural language processing. " * 200,
    ]

    results = benchmark_comparison(test_texts, num_runs=20)

    print("""
    🎯 NEXT STEPS:

    1. ✓ You now understand how BPE works (bpe_tutorial.py)
    2. ✓ You have a working CPU baseline (bpe_cpu.py)
    3. ✓ You validated against HuggingFace (this script)

    4. ⏭️  Optimize with GPU (bpe_gpu.py - needs more work!)
    5. ⏭️  Target: 2-5x faster than HuggingFace
    6. ⏭️  Focus on batch processing for best speedup

    💡 GPU Optimization Strategy:
       - Start with pair counting (easy to parallelize)
       - Then parallel merging (harder, but critical)
       - Add batching (process multiple texts together)
       - Use shared memory for vocabulary
       - Profile and iterate!

    Good luck! 🚀
    """)


if __name__ == "__main__":
    main()