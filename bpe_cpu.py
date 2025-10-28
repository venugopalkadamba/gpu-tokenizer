#!/usr/bin/env python3
"""
CPU-based BPE Tokenizer Implementation
Works with real GPT-2 vocabulary for realistic benchmarking

This is the BASELINE implementation that we'll optimize with GPU!
"""

import json
import time
import regex as re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import urllib.request


class BPETokenizerCPU:
    """
    CPU-based BPE Tokenizer following GPT-2's approach

    Key components:
    1. vocab: Maps token strings to token IDs
    2. merges: List of merge rules (pair → merged token)
    3. encoder: The actual encoding logic
    """

    def __init__(self, vocab_file: Optional[str] = None, merges_file: Optional[str] = None):
        """Initialize tokenizer with vocabulary and merge rules"""
        self.vocab = {}
        self.merges = []
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        if vocab_file and merges_file:
            self.load_vocab(vocab_file, merges_file)

    @staticmethod
    def _bytes_to_unicode():
        """
        GPT-2 uses a clever trick to handle all bytes as unicode characters
        This avoids issues with special bytes that aren't valid UTF-8
        """
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def load_vocab(self, vocab_file: str, merges_file: str):
        """Load GPT-2 vocabulary and merge rules"""
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        print(f"Loading merges from {merges_file}...")
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges_data = f.read().split('\n')[1:-1]  # Skip header and empty last line

        self.merges = [tuple(merge.split()) for merge in merges_data]

        # Create inverse vocabulary for decoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Create merge priority dictionary for fast lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        print(f"✓ Loaded {len(self.vocab)} tokens and {len(self.merges)} merge rules")

    @staticmethod
    def download_gpt2_files(output_dir: str = "data"):
        """Download GPT-2 vocabulary files from HuggingFace"""
        Path(output_dir).mkdir(exist_ok=True)

        vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
        merges_url = "https://huggingface.co/openai-community/gpt2/raw/main/merges.txt"

        vocab_file = f"{output_dir}/vocab.json"
        merges_file = f"{output_dir}/merges.txt"

        if not Path(vocab_file).exists():
            print(f"Downloading {vocab_url}...")
            urllib.request.urlretrieve(vocab_url, vocab_file)
            print(f"✓ Saved to {vocab_file}")

        if not Path(merges_file).exists():
            print(f"Downloading {merges_url}...")
            urllib.request.urlretrieve(merges_url, merges_file)
            print(f"✓ Saved to {merges_file}")

        return vocab_file, merges_file

    def get_pairs(self, word: Tuple[str, ...]) -> set:
        """
        Get all adjacent pairs in a word

        Example: ('l', 'o', 'w') → {('l', 'o'), ('o', 'w')}
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> str:
        """
        Apply BPE merges to a token

        This is THE CORE ALGORITHM that we'll parallelize on GPU!

        Steps:
        1. Start with character-level representation
        2. Find all adjacent pairs
        3. Find the highest-priority pair (earliest in merge list)
        4. Merge that pair
        5. Repeat until no more merges possible
        """
        if token in self.vocab:
            return token

        # Convert to tuple of characters
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        # Keep merging until no valid pairs remain
        while True:
            # Find the pair with highest priority (lowest rank = earliest in merge list)
            # THIS IS THE BOTTLENECK: O(n) scan for each iteration
            bigram = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))

            if bigram not in self.merge_ranks:
                break  # No more valid merges

            first, second = bigram
            new_word = []
            i = 0

            # Merge all occurrences of the bigram
            # THIS IS ANOTHER BOTTLENECK: O(n) scan
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        return ' '.join(word)

    def encode(self, text: str, verbose: bool = False) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            verbose: Print timing information

        Returns:
            List of token IDs
        """
        if verbose:
            print(f"\nEncoding text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            start = time.time()

        # GPT-2 pattern to split text into tokens before BPE
        # This handles spaces, punctuation, etc.
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        tokens = []
        for match in re.finditer(pat, text):
            token = match.group()

            # Convert to bytes then to unicode chars
            token_bytes = token.encode('utf-8')
            token_unicode = ''.join(self.byte_encoder[b] for b in token_bytes)

            # Apply BPE
            bpe_tokens = self.bpe(token_unicode).split(' ')

            # Convert to IDs
            tokens.extend([self.vocab[bpe_token] for bpe_token in bpe_tokens])

        if verbose:
            elapsed = time.time() - start
            print(f"✓ Encoded to {len(tokens)} tokens in {elapsed*1000:.2f}ms")
            print(f"  Throughput: {len(text)/elapsed:.0f} chars/sec")

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        text = ''.join([self.inverse_vocab[token] for token in tokens])
        text_bytes = bytearray([self.byte_decoder[c] for c in text])
        return text_bytes.decode('utf-8', errors='replace')

    def benchmark(self, text: str, num_runs: int = 10):
        """Benchmark encoding performance"""
        print(f"\n{'='*70}")
        print(f" CPU BPE Benchmark")
        print(f"{'='*70}")
        print(f"Text length: {len(text)} characters")
        print(f"Number of runs: {num_runs}")

        # Warmup
        _ = self.encode(text)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            tokens = self.encode(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nResults:")
        print(f"  Tokens produced: {len(tokens)}")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Min time: {min_time*1000:.2f} ms")
        print(f"  Max time: {max_time*1000:.2f} ms")
        print(f"  Throughput: {len(text)/avg_time:.0f} chars/sec")
        print(f"  Token rate: {len(tokens)/avg_time:.0f} tokens/sec")
        print(f"{'='*70}\n")

        return {
            'avg_time': avg_time,
            'throughput_chars': len(text) / avg_time,
            'throughput_tokens': len(tokens) / avg_time,
            'num_tokens': len(tokens)
        }


def main():
    """Demo the CPU tokenizer"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                CPU BPE Tokenizer                                 ║
    ║                Baseline Implementation                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Download GPT-2 files if needed
    vocab_file, merges_file = BPETokenizerCPU.download_gpt2_files()

    # Initialize tokenizer
    tokenizer = BPETokenizerCPU(vocab_file, merges_file)

    # Test examples
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-2 uses Byte Pair Encoding for tokenization.",
        "lower lowest",
    ]

    print("\n" + "="*70)
    print(" Testing Tokenization")
    print("="*70)

    for text in test_texts:
        tokens = tokenizer.encode(text, verbose=True)
        decoded = tokenizer.decode(tokens)
        print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  Decoded: '{decoded}'")
        assert text == decoded, "Decode mismatch!"
        print()

    # Benchmark with longer text
    long_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development
    of algorithms and statistical models that enable computer systems to improve their
    performance on a specific task through experience. The fundamental premise of machine
    learning is to build systems that can learn from and make decisions based on data.
    """ * 10

    tokenizer.benchmark(long_text.strip(), num_runs=100)

    print("✓ CPU tokenizer working correctly!")
    print("\n📊 These numbers are your BASELINE to beat with GPU!")


if __name__ == "__main__":
    main()