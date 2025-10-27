#!/usr/bin/env python3
"""
BPE (Byte Pair Encoding) Interactive Tutorial
Learn how BPE tokenization works step-by-step!

What is BPE?
- BPE is a compression algorithm adapted for tokenization
- It starts with individual characters and iteratively merges the most frequent pairs
- GPT-2, GPT-3, and many LLMs use BPE for tokenization
"""

from collections import Counter
from typing import List, Tuple, Dict
import pprint

def visualize_text(text: str, tokens: List[str], step: int = 0):
    """Pretty print tokens with separators"""
    print(f"\n{'='*70}")
    print(f"Step {step}: {len(tokens)} tokens")
    print(f"{'='*70}")
    print("Tokens: [" + " | ".join(tokens) + "]")
    print(f"Reconstructed: '{' '.join(tokens)}'")
    print(f"{'='*70}")

def get_pair_counts(tokens: List[str]) -> Counter:
    """
    Count all adjacent pairs in the token sequence

    Example: ["l", "o", "w", "e", "r"]
    → {("l", "o"): 1, ("o", "w"): 1, ("w", "e"): 1, ("e", "r"): 1}
    """
    pairs = Counter()
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs[pair] += 1
    return pairs

def merge_pair(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
    """
    Merge all occurrences of a pair in the token list

    Example: merge_pair(["l", "o", "w"], ("l", "o"))
    → ["lo", "w"]
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        # Check if current position matches the pair to merge
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            # Merge the pair
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2  # Skip the next token since we merged it
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def train_bpe(text: str, num_merges: int = 10, verbose: bool = True) -> List[Tuple[str, str]]:
    """
    Train BPE by finding the most frequent pairs and merging them

    Args:
        text: Training text
        num_merges: Number of merge operations to perform
        verbose: Print step-by-step explanation

    Returns:
        List of merge rules (pair tuples) in order
    """
    print("\n" + "="*70)
    print(" BPE TRAINING - Learning Merge Rules")
    print("="*70)
    print(f"\nOriginal text: '{text}'")
    print(f"Target: Learn {num_merges} merge rules")

    # Start with character-level tokens
    tokens = list(text)

    if verbose:
        visualize_text(text, tokens, step=0)

    merge_rules = []

    for step in range(1, num_merges + 1):
        # Count all pairs
        pairs = get_pair_counts(tokens)

        if not pairs:
            print(f"\nNo more pairs to merge! Stopping at {step-1} merges.")
            break

        # Find the most frequent pair
        most_frequent_pair = pairs.most_common(1)[0][0]
        count = pairs[most_frequent_pair]

        print(f"\n--- Merge {step} ---")
        print(f"Most frequent pair: {most_frequent_pair} (appears {count} times)")

        if verbose and len(pairs) <= 10:
            print(f"All pairs: {dict(pairs.most_common())}")

        # Merge this pair everywhere
        tokens = merge_pair(tokens, most_frequent_pair)
        merge_rules.append(most_frequent_pair)

        if verbose:
            visualize_text(text, tokens, step)

    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nLearned {len(merge_rules)} merge rules:")
    for i, (a, b) in enumerate(merge_rules, 1):
        print(f"  {i}. '{a}' + '{b}' → '{a}{b}'")

    return merge_rules

def apply_bpe(text: str, merge_rules: List[Tuple[str, str]], verbose: bool = True) -> List[str]:
    """
    Apply learned BPE merge rules to tokenize text

    Args:
        text: Text to tokenize
        merge_rules: Ordered list of merge rules from training

    Returns:
        List of tokens after all merges
    """
    print("\n" + "="*70)
    print(" BPE ENCODING - Applying Merge Rules")
    print("="*70)
    print(f"\nText to encode: '{text}'")

    # Start with characters
    tokens = list(text)

    if verbose:
        print(f"\nInitial (character-level): {tokens}")

    # Apply each merge rule in order
    for i, pair in enumerate(merge_rules, 1):
        old_len = len(tokens)
        tokens = merge_pair(tokens, pair)
        new_len = len(tokens)

        if verbose and old_len != new_len:
            print(f"\nAfter merge {i} {pair}:")
            print(f"  {tokens}")
            print(f"  ({old_len} → {new_len} tokens, reduced by {old_len - new_len})")

    print(f"\n{'='*70}")
    print(f"Final tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"{'='*70}")

    return tokens

# ============================================================================
# EXAMPLE 1: Simple word "lower"
# ============================================================================

def example1_simple():
    """Example 1: Tokenize a simple word"""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "EXAMPLE 1: Simple Word" + " "*26 + "║")
    print("╚" + "="*68 + "╝")

    text = "lower"

    # Train BPE
    merge_rules = train_bpe(text, num_merges=3, verbose=True)

    # Apply to same text
    tokens = apply_bpe(text, merge_rules, verbose=True)

    print("\n💡 KEY INSIGHT:")
    print("  BPE learns to merge frequently occurring character pairs.")
    print("  In 'lower': 'l'+'o' appears once, 'o'+'w' appears once, etc.")
    print("  The order matters - we can only merge adjacent tokens!")

# ============================================================================
# EXAMPLE 2: Repeated patterns
# ============================================================================

def example2_patterns():
    """Example 2: Text with repeated patterns"""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "EXAMPLE 2: Repeated Patterns" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    text = "aaabdaaabac"

    print("\n🎯 GOAL: See how BPE handles repeated patterns")
    print("   Notice how 'aa' appears 4 times - it should merge first!\n")

    # Train BPE
    merge_rules = train_bpe(text, num_merges=5, verbose=True)

    # Test on new text with same patterns
    test_text = "aaabdaaaba"
    print(f"\n\n{'='*70}")
    print(" TESTING ON NEW TEXT")
    print(f"{'='*70}")
    tokens = apply_bpe(test_text, merge_rules, verbose=True)

# ============================================================================
# EXAMPLE 3: Real-world scenario - multiple words
# ============================================================================

def example3_realistic():
    """Example 3: More realistic multi-word text"""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*16 + "EXAMPLE 3: Multi-word Text" + " "*26 + "║")
    print("╚" + "="*68 + "╝")

    # Use spaces to separate words (this is how GPT-2 does it)
    text = "low low low lower lower newest newest"

    print("\n🎯 GOAL: See how BPE learns subword units")
    print("   'low' appears 5 times - watch it get merged into a single token!\n")

    # Train with more merges
    merge_rules = train_bpe(text, num_merges=15, verbose=True)

    # Test on related text
    test_texts = [
        "low",
        "lower",
        "lowest",
        "new",
        "newer",
    ]

    print(f"\n\n{'='*70}")
    print(" TESTING ON NEW WORDS")
    print(f"{'='*70}")

    for test_text in test_texts:
        tokens = apply_bpe(test_text, merge_rules, verbose=False)
        print(f"\n'{test_text}' → {tokens}")

# ============================================================================
# EXAMPLE 4: Understanding the bottleneck (why we need GPU)
# ============================================================================

def example4_bottleneck():
    """Example 4: Understand computational complexity"""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*12 + "EXAMPLE 4: Why BPE Needs GPU Acceleration" + " "*15 + "║")
    print("╚" + "="*68 + "╝")

    text = "a" * 100 + "b" * 100  # 200 characters

    print(f"\nText: {text[:50]}... ({len(text)} characters)")
    print("\n🔍 COMPUTATIONAL ANALYSIS:")
    print(f"   - Initial tokens: {len(text)}")
    print(f"   - For EACH merge step:")
    print(f"     1. Count ALL pairs: O(n) - scan entire sequence")
    print(f"     2. Find most frequent: O(n) - count unique pairs")
    print(f"     3. Merge all occurrences: O(n) - scan and merge")
    print(f"   - Total for k merges: O(k * n)")
    print(f"\n   For GPT-2 vocabulary (50K merges, 1M+ text):")
    print(f"     → 50,000 * 1,000,000 = 50 BILLION operations!")
    print(f"     → This is why we need GPU parallelization!")

    import time

    print("\n⏱️  Timing CPU implementation...")
    start = time.time()
    merge_rules = train_bpe(text, num_merges=50, verbose=False)
    elapsed = time.time() - start

    print(f"\n   CPU time for 200 chars, 50 merges: {elapsed:.4f} seconds")
    print(f"   Estimated for 1M chars: {elapsed * 5000:.2f} seconds = {elapsed * 5000 / 60:.1f} minutes")
    print(f"\n   🎯 GPU Goal: 10-100x speedup → seconds instead of minutes!")

# ============================================================================
# MAIN: Run all examples
# ============================================================================

def main():
    """Run all BPE examples"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              BPE (Byte Pair Encoding) Tutorial                   ║
    ║              Understanding Tokenization Step-by-Step             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    This tutorial will teach you:
    1. How BPE finds and merges frequent character pairs
    2. Why merge order matters (iterative process)
    3. How to apply learned rules to new text
    4. Why we need GPU acceleration

    """)

    input("Press Enter to start Example 1...")
    example1_simple()

    input("\n\nPress Enter for Example 2...")
    example2_patterns()

    input("\n\nPress Enter for Example 3...")
    example3_realistic()

    input("\n\nPress Enter for Example 4...")
    example4_bottleneck()

    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*26 + "TUTORIAL COMPLETE!" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    print("\n📚 NEXT STEPS:")
    print("   1. Run 'python bpe_cpu.py' - Full CPU implementation")
    print("   2. Run 'python bpe_gpu.py' - GPU-accelerated version")
    print("   3. Run 'python bpe_benchmark.py' - Compare performance")
    print("\n")

if __name__ == "__main__":
    # You can run all examples or just one
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example1_simple()
        elif example_num == "2":
            example2_patterns()
        elif example_num == "3":
            example3_realistic()
        elif example_num == "4":
            example4_bottleneck()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python bpe_tutorial.py [1|2|3|4]")
    else:
        main()