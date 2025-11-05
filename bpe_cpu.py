#!/usr/bin/env python3
"""
CPU-based BPE Tokenizer Implementation
Works with real GPT-2 vocabulary for realistic benchmarking

This is the BASELINE implementation that we'll optimize with GPU!
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import time
import regex as re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import urllib.request


# ==============================================================================
# MAIN TOKENIZER CLASS
# ==============================================================================
class BPETokenizerCPU:
    """
    CPU-based BPE Tokenizer following GPT-2's approach

    Key components:
    1. vocab: Maps token strings to token IDs
    2. merges: List of merge rules (pair → merged token)
    3. encoder: The actual encoding logic
    """

    # ==========================================================================
    # SECTION 1: INITIALIZATION
    # ==========================================================================

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

    # ==========================================================================
    # SECTION 2: LOADING VOCABULARY & MERGE RULES
    # ==========================================================================

    def load_vocab(self, vocab_file: str, merges_file: str):
        """Load GPT-2 vocabulary and merge rules"""
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        print(f"Loading merges from {merges_file}...")
        with open(merges_file, 'r', encoding='utf-8') as f:
            # Skip header line; keep every non-empty rule exactly as stored
            merges_data = [line for line in f.read().splitlines()[1:] if line]

        self.merges = [tuple(merge.split()) for merge in merges_data]

        # Create inverse vocabulary for decoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Create merge priority dictionary for fast lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        print(f"✓ Loaded {len(self.vocab)} tokens and {len(self.merges)} merge rules")

    # ==========================================================================
    # SECTION 3: DOWNLOADING GPT-2 FILES
    # ==========================================================================

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

    # ==========================================================================
    # SECTION 4: CORE BPE ALGORITHM *** THIS IS THE BOTTLENECK! ***
    # ==========================================================================

    def get_pairs(self, word: Tuple[str, ...]) -> set:
        """
        Get all adjacent pairs in a word

        Example: ('l', 'o', 'w') → {('l', 'o'), ('o', 'w')}
        """
        if len(word) < 2:
            return set()

        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> str:
        """
        Apply BPE merges to a token

        *** THIS IS THE CORE ALGORITHM THAT WE'LL PARALLELIZE ON GPU! ***

        Steps:
        1. Start with character-level representation
        2. Find all adjacent pairs
        3. Find the highest-priority pair (earliest in merge list)
        4. Merge that pair
        5. Repeat until no more merges possible
        """
        # ---- Quick check: token already in vocabulary? ----
        if token in self.vocab:
            return token

        # ---- STEP 1: Convert to character-level tuple ----
        # Example: "Hello" → ('H', 'e', 'l', 'l', 'o')
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        # ---- STEP 2: ITERATIVE MERGING LOOP (THE BOTTLENECK!) ----
        while True:
            # --- BOTTLENECK #1: Find best pair to merge ---
            # Scans ALL pairs to find which one has lowest rank (highest priority)
            # 🎯 GPU OPPORTUNITY: Can parallelize this scan!
            bigram = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))

            if bigram not in self.merge_ranks:
                break  # No more valid merges

            first, second = bigram
            new_word = []
            i = 0

            # --- BOTTLENECK #2: Merge all occurrences of the pair ---
            # Scans entire word to find and merge the bigram
            # 🎯 GPU OPPORTUNITY: Can parallelize this too!
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

            # Check if done merging
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        return ' '.join(word)

    # ==========================================================================
    # SECTION 5: ENCODING TEXT TO TOKEN IDs (THE MAIN ENTRY POINT!)
    # ==========================================================================

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

        # ---- STEP A: Split text into chunks using regex pattern ----
        # Pattern splits on: contractions ('s, 't), letters, numbers, punctuation, spaces
        # Example: "Hello, world!" → ["Hello", ",", " world", "!"]
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        tokens = []

        # ---- STEP B: Process each chunk ----
        for match in re.finditer(pat, text):
            token = match.group()

            # --- STEP B1: Convert to byte representation ---
            # Handles special characters safely using byte_encoder mapping
            token_bytes = token.encode('utf-8')
            token_unicode = ''.join(self.byte_encoder[b] for b in token_bytes)

            # --- STEP B2: Apply BPE merging (THE SLOW PART!) ---
            # This calls bpe() method which does iterative character merging
            bpe_tokens = self.bpe(token_unicode).split(' ')

            # --- STEP B3: Look up token IDs in vocabulary ---
            # Convert token strings to their numeric IDs
            # Example: "Hello" → 15496
            tokens.extend([self.vocab[bpe_token] for bpe_token in bpe_tokens])

        if verbose:
            elapsed = time.time() - start
            print(f"✓ Encoded to {len(tokens)} tokens in {elapsed*1000:.2f}ms")
            print(f"  Throughput: {len(text)/elapsed:.0f} chars/sec")

        return tokens

    # ==========================================================================
    # SECTION 6: DECODING TOKEN IDs BACK TO TEXT
    # ==========================================================================

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        text = ''.join([self.inverse_vocab[token] for token in tokens])
        text_bytes = bytearray([self.byte_decoder[c] for c in text])
        return text_bytes.decode('utf-8', errors='replace')

    # ==========================================================================
    # SECTION 7: BENCHMARKING (Measure CPU Performance)
    # ==========================================================================

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


# ==============================================================================
# MAIN DEMO FUNCTION
# ==============================================================================

def main():
    """Demo the CPU tokenizer"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                CPU BPE Tokenizer                                 ║
    ║                Baseline Implementation                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # --------------------------------------------------------------------------
    # PART 1: Setup - Download GPT-2 vocabulary files
    # --------------------------------------------------------------------------
    vocab_file, merges_file = BPETokenizerCPU.download_gpt2_files()

    # --------------------------------------------------------------------------
    # PART 2: Initialize tokenizer with vocab
    # --------------------------------------------------------------------------
    tokenizer = BPETokenizerCPU(vocab_file, merges_file)

    # --------------------------------------------------------------------------
    # PART 3: Test with example texts
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # PART 4: Benchmark with longer text
    # --------------------------------------------------------------------------
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
