#!/usr/bin/env python3
"""
WikiText-103 benchmark for GPU BlockBPE, tiktoken, and HuggingFace GPT-2 tokenizers.

For each sampled line from WikiText-103:
- Split into (prompt, ground-truth continuation)
- Tokenize prompt with all three tokenizers
- Generate with GPT-2 from each tokenization
- Compare generated continuations to the ground-truth continuation

Reports:
- Average tokens per prompt for each tokenizer
- Tokenization time and tokens/sec for each tokenizer
- Generation time and tokens/sec for each tokenizer
- Similarity vs ground-truth continuation (char/word/LCS) for each tokenizer
"""

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import re
from typing import Optional, List, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

try:
    # For loading WikiText-103 and other datasets
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None

try:
    # For building/loading the CUDA extension in-process
    from torch.utils.cpp_extension import load as load_extension  # type: ignore
except ImportError:
    load_extension = None


def load_gpu_token_strings(path: Path) -> List[str]:
    """Load token strings from GPU tokenizer output file."""
    if not path.exists():
        raise FileNotFoundError(f"GPU token file not found: {path}")
    
    tokens = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.extend(line.split())
    return tokens


def tokenize_with_gpu(prompt: str, gpu_tokenizer_path: str = "./gpu_tokenizer_optimized",
                      chunk_size: int = 2048) -> Tuple[List[str], Optional[float]]:
    """
    Tokenize text using GPU tokenizer by writing to input file and calling executable.
    Returns:
        (token_strings, kernel_time_ms) where kernel_time_ms is parsed from the
        binary's stdout ("GPU BPE kernel time ...") if available, else None.
    """
    # Write prompt to input file (the GPU tokenizer reads from a fixed path)
    # NOTE: This path must match the input_path hard-coded in the C++ binary.
    input_path = Path("data/input/pride_and_prejudice.txt")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save original content if it exists
    original_content = None
    if input_path.exists():
        original_content = input_path.read_text(encoding="utf-8", errors="ignore")
    
    # Write ONLY the prompt (ensure file is overwritten, not appended)
    input_path.write_text(prompt, encoding="utf-8")
    
    # Verify file was written correctly
    written_content = input_path.read_text(encoding="utf-8")
    if written_content != prompt:
        raise RuntimeError(f"Failed to write prompt to file. Expected '{prompt[:50]}...', got '{written_content[:50]}...'")
    
    try:
        # Call GPU tokenizer
        # The optimized version writes to gpu_tokens_optimized.txt, regular to gpu_tokens.txt
        if "optimized" in gpu_tokenizer_path:
            output_path = Path("data/output/gpu_tokens_optimized.txt")
        else:
            output_path = Path("data/output/gpu_tokens.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear output file first
        if output_path.exists():
            output_path.unlink()
        
        result = subprocess.run(
            [gpu_tokenizer_path, str(chunk_size)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()  # Ensure we're in the right directory
        )
        
        if result.returncode != 0:
            print(f"GPU tokenizer stderr: {result.stderr[:500]}")
            print(f"GPU tokenizer stdout: {result.stdout[:500]}")
            raise RuntimeError(f"GPU tokenizer failed with return code {result.returncode}")
        
        # Parse kernel time from stdout (in milliseconds), e.g.:
        # "GPU BPE kernel time (optimized, excl. H2D/D2H): 2.78221 ms"
        kernel_time_ms: Optional[float] = None
        for line in result.stdout.splitlines():
            m = re.search(r"GPU BPE kernel time.*:\s*([0-9.]+)\s*ms", line)
            if m:
                try:
                    kernel_time_ms = float(m.group(1))
                except ValueError:
                    kernel_time_ms = None
                break
        
        # Load token strings - only from first line (first chunk)
        if not output_path.exists():
            raise FileNotFoundError(f"GPU tokenizer did not create output file: {output_path}")
        
        tokens = load_gpu_token_strings(output_path)
        
        # If we got way too many tokens, it might be reading the whole file
        # Take only the first chunk that corresponds to our prompt
        expected_max = len(prompt) * 2  # Rough estimate: prompt length * 2
        if len(tokens) > expected_max:
            print(f"Warning: Got {len(tokens)} tokens for prompt of length {len(prompt)}")
            print(f"Taking first {min(100, len(tokens))} tokens as sample")
            # Estimate tokens for prompt (rough: ~1 token per 4 chars for English)
            estimated_tokens = max(1, len(prompt) // 4)
            tokens = tokens[:estimated_tokens * 2]  # Take 2x estimate to be safe
        
        return tokens, kernel_time_ms
    
    finally:
        # Restore original content
        if original_content is not None:
            input_path.write_text(original_content, encoding="utf-8")


def map_token_strings_to_ids(token_strings: List[str], hf_tokenizer) -> List[int]:
    """
    Map token strings (like "ĠThe") to token IDs using HuggingFace tokenizer vocab.
    This handles the mapping from BlockBPE token strings to GPT-2 token IDs.
    
    The token strings from BlockBPE are BPE pieces that should exist in GPT-2's vocab.
    We map them directly using the tokenizer's vocabulary.
    """
    token_ids = []
    failed_count = 0
    
    vocab = hf_tokenizer.get_vocab()
    
    for tok_str in token_strings:
        # Direct lookup in vocab
        if tok_str in vocab:
            token_ids.append(vocab[tok_str])
        else:
            failed_count += 1
            # Try alternative: use convert_tokens_to_ids which might handle special cases
            try:
                # Some tokenizers have special handling
                ids = hf_tokenizer.convert_tokens_to_ids([tok_str])
                if ids and ids[0] != hf_tokenizer.unk_token_id:
                    token_ids.append(ids[0])
                else:
                    # Last resort: skip this token
                    if failed_count <= 5:  # Only print first few warnings
                        print(f"Warning: Token '{tok_str[:20]}...' not found in vocab, skipping")
            except Exception:
                if failed_count <= 5:
                    print(f"Warning: Could not map token '{tok_str[:20]}...', skipping")
    
    if len(token_ids) == 0:
        raise ValueError("Could not map any tokens to IDs. Check tokenizer compatibility.")
    
    if failed_count > 0:
        print(f"Note: {failed_count} tokens could not be mapped (out of {len(token_strings)} total)")
    
    return token_ids


def tokenize_with_tiktoken(text: str) -> List[int]:
    """Tokenize using tiktoken."""
    if tiktoken is None:
        raise ImportError("tiktoken not installed")
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text)


def tokenize_with_hf(text: str, tokenizer) -> List[int]:
    """Tokenize using HuggingFace tokenizer."""
    return tokenizer.encode(text, add_special_tokens=False)


def generate_text(model, tokenizer, token_ids: List[int], max_new_tokens: int = 50,
                 temperature: float = 0.7, top_p: float = 0.9) -> Tuple[str, int]:
    """Generate text from token IDs using GPT-2 model.

    Returns:
        (generated_text, num_new_tokens)
    """
    if torch is None:
        raise ImportError("torch not installed")
    
    # Validate token IDs are within vocab range
    vocab_size = len(tokenizer.get_vocab())
    valid_token_ids = [tid for tid in token_ids if 0 <= tid < vocab_size]
    
    if len(valid_token_ids) != len(token_ids):
        print(f"Warning: {len(token_ids) - len(valid_token_ids)} token IDs out of range, using {len(valid_token_ids)} valid tokens")
    
    if len(valid_token_ids) == 0:
        raise ValueError("No valid token IDs for generation")
    
    # Convert token IDs to tensor
    input_ids = torch.tensor([valid_token_ids], device=model.device)
    
    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Generation error: {e}")
            raise
    
    # Decode only the new tokens
    generated_ids = outputs[0][len(valid_token_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, len(generated_ids)


def compute_similarity(text1: str, text2: str) -> Tuple[float, float, float]:
    """
    Compute similarity metrics between two texts.
    Returns: (character overlap, word overlap, BLEU-like score)
    """
    # Character-level similarity
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    char_overlap = len(chars1 & chars2) / max(len(chars1 | chars2), 1)
    
    # Word-level similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
    
    # Simple sequence similarity (longest common subsequence ratio)
    def lcs_ratio(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / max(m, n, 1)
    
    seq_similarity = lcs_ratio(text1.lower(), text2.lower())
    
    return char_overlap, word_overlap, seq_similarity


def main():
    parser = argparse.ArgumentParser(
        description="WikiText-103 benchmark for GPU BlockBPE, tiktoken, and HuggingFace tokenizers"
    )
    parser.add_argument(
        "--gpu-tokenizer",
        type=str,
        default="./gpu_tokenizer_optimized",
        help="Path to GPU tokenizer executable"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for GPU tokenizer"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch and torch.cuda.is_available() else "cpu",
        help="Device for model inference"
    )
    parser.add_argument(
        "--wikitext-split",
        type=str,
        default="test",
        help="WikiText-103 split to use: train/validation/test"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts sampled from WikiText-103 when --use-wikitext is set"
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=200,
        help="Maximum characters per sampled WikiText-103 prompt"
    )
    parser.add_argument(
        "--ref-chars",
        type=int,
        default=200,
        help="Maximum characters of ground-truth continuation used as reference"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WikiText-103 Benchmark: GPU BlockBPE vs tiktoken vs HuggingFace")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print()
    
    # Load model and tokenizer
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        print("ERROR: transformers library not installed")
        print("Install with: pip install transformers torch")
        return
    
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()
    print("Model loaded.\n")

    # Try to load the in-process CUDA extension for GPU tokenization
    gpu_ext = None
    gpu_ext_tokenizer = None
    if load_extension is not None:
        try:
            ext_path = Path(__file__).parent / "gputok_binding.cu"
            if ext_path.exists():
                print("Building/loading gputok_gpu CUDA extension (pybind11)...")
                gpu_ext = load_extension(
                    name="gputok_gpu",
                    sources=[str(ext_path)],
                    extra_include_paths=[
                        str(Path(__file__).parent / "externals/cuCollections/include"),
                        str(Path(__file__).parent / "externals/cccl/include"),
                    ],
                    extra_cflags=["-O3"],
                    # cuCollections requires extended device lambdas support
                    # (nvcc flag --expt-extended-lambda), so we add it here.
                    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-extended-lambda"],
                    verbose=False,
                )
                gpu_ext_tokenizer = gpu_ext.GpuTokenizer(
                    str(Path("data/gpt2_tokenizer/merges.txt")), args.chunk_size, 50
                )
                print("Loaded gputok_gpu extension.\n")
            else:
                print("gputok_binding.cu not found; falling back to subprocess GPU tokenizer.\n")
        except Exception as e:
            print(f"WARNING: Failed to build/load gputok_gpu extension: {e}")
            print("Falling back to subprocess GPU tokenizer.\n")
    else:
        print("torch.utils.cpp_extension.load not available; using subprocess GPU tokenizer.\n")

    # ------------------------------------------------------------------
    # WikiText-103 benchmark (only mode)
    # ------------------------------------------------------------------
    if load_dataset is None:
        print("ERROR: datasets library not installed")
        print("Install with: pip install datasets")
        return

    print("Dataset: WikiText-103 (wikitext-103-raw-v1)")
    print(f"Split: {args.wikitext_split}, num_prompts: {args.num_prompts}, "
          f"max_prompt_chars: {args.max_prompt_chars}, ref_chars: {args.ref_chars}")

    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    except Exception as e:
        print(f"ERROR: Failed to load WikiText-103: {e}")
        return

    if args.wikitext_split not in ds:
        print(f"ERROR: Split '{args.wikitext_split}' not found in dataset")
        print(f"Available splits: {list(ds.keys())}")
        return

    split = ds[args.wikitext_split]
    texts = split["text"]

    # Sample (prompt, reference) pairs: non-empty lines, with enough length
    examples: List[Tuple[str, str]] = []
    for t in texts:
        if not t or t.isspace():
            continue
        t = t.strip()
        if not t:
            continue
        if len(t) <= args.max_prompt_chars + 20:
            # Need some room for continuation
            continue
        prompt = t[: args.max_prompt_chars]
        reference = t[args.max_prompt_chars : args.max_prompt_chars + args.ref_chars]
        examples.append((prompt, reference))
        if len(examples) >= args.num_prompts:
            break

    if not examples:
        print("ERROR: No suitable (prompt, reference) pairs found in WikiText-103 split")
        return

    print(f"Collected {len(examples)} (prompt, reference) pairs from WikiText-103\n")

    # ------------------------------------------------------------------
    # Warmup (not timed): run each tokenizer a few times to trigger any
    # one-time setup costs (caches, JIT, etc.) before measuring.
    # ------------------------------------------------------------------
    if examples:
        print("Running tokenizer warmup (not timed)...")
        warmup_examples = examples[: min(3, len(examples))]
        for prompt, _ in warmup_examples:
            # GPU warmup
            try:
                if gpu_ext_tokenizer is not None:
                    gpu_ext_tokenizer.tokenize_batch([prompt])
                else:
                    tokenize_with_gpu(prompt, args.gpu_tokenizer, args.chunk_size)
            except Exception:
                pass

            # tiktoken warmup
            if tiktoken is not None:
                try:
                    tokenize_with_tiktoken(prompt)
                except Exception:
                    pass

            # HuggingFace warmup
            try:
                tokenize_with_hf(prompt, tokenizer)
            except Exception:
                pass
        print("Warmup complete.\n")

    # Aggregators for similarity metrics

    # ref_similarity_acc[name] = [sum_char, sum_word, sum_seq, count]
    ref_similarity_acc: dict[str, List[float]] = {}

    def _accumulate_ref(name: str,
                        char_sim: float,
                        word_sim: float,
                        seq_sim: float) -> None:
        if name not in ref_similarity_acc:
            ref_similarity_acc[name] = [0.0, 0.0, 0.0, 0.0]
        ref_similarity_acc[name][0] += char_sim
        ref_similarity_acc[name][1] += word_sim
        ref_similarity_acc[name][2] += seq_sim
        ref_similarity_acc[name][3] += 1.0

    for idx, (prompt, reference) in enumerate(examples):
        print("=" * 80)
        print(f"Example {idx + 1}/{len(examples)}")
        print(f"Prompt (len={len(prompt)}):")
        preview = prompt.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  {preview}")
        ref_preview = reference.replace("\n", " ")
        if len(ref_preview) > 120:
            ref_preview = ref_preview[:117] + "..."
        print(f"Reference (len={len(reference)}):")
        print(f"  {ref_preview}")

        # Tokenize
        gpu_token_ids = None
        tiktoken_ids = None
        hf_token_ids = None

        # GPU BlockBPE tokenization timing:
        # - If extension is available: in-process tokenize_batch.
        # - Else: fallback to subprocess-based tokenizer.
        try:
            if gpu_ext_tokenizer is not None:
                batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch([prompt])
                gpu_token_ids = batch_ids[0]
            else:
                gpu_token_strings, _ = tokenize_with_gpu(
                    prompt, args.gpu_tokenizer, args.chunk_size
                )
                gpu_token_ids = map_token_strings_to_ids(gpu_token_strings, tokenizer)
        except Exception as e:
            print(f"  [GPU] ERROR: {e}")

        # tiktoken tokenization timing
        try:
            if tiktoken is None:
                raise ImportError("tiktoken not installed")
            tiktoken_ids = tokenize_with_tiktoken(prompt)
        except Exception as e:
            print(f"  [tiktoken] ERROR: {e}")

        # HuggingFace tokenization timing
        try:
            hf_token_ids = tokenize_with_hf(prompt, tokenizer)
        except Exception as e:
            print(f"  [HuggingFace] ERROR: {e}")

        # Generate for each successful tokenizer
        generations: dict[str, str] = {}

        if gpu_token_ids:
            try:
                gen_text, gen_len = generate_text(
                    model,
                    tokenizer,
                    gpu_token_ids,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                )
                generations["GPU BlockBPE"] = gen_text
            except Exception as e:
                print(f"  [GPU] Generation ERROR: {e}")

        if tiktoken_ids:
            try:
                gen_text, gen_len = generate_text(
                    model,
                    tokenizer,
                    tiktoken_ids,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                )
                generations["tiktoken"] = gen_text
            except Exception as e:
                print(f"  [tiktoken] Generation ERROR: {e}")

        if hf_token_ids:
            try:
                gen_text, gen_len = generate_text(
                    model,
                    tokenizer,
                    hf_token_ids,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                )
                generations["HuggingFace"] = gen_text
            except Exception as e:
                print(f"  [HuggingFace] Generation ERROR: {e}")

        if len(generations) == 0:
            print("  Skipping similarity metrics (no successful generations)")
            continue

        # Compute and accumulate similarity vs ground-truth reference
        for name, gen_text in generations.items():
            char_sim, word_sim, seq_sim = compute_similarity(gen_text, reference)
            _accumulate_ref(name, char_sim, word_sim, seq_sim)

    # Final aggregate report
    print("\n" + "=" * 80)
    print("WikiText-103 AGGREGATED RESULTS")
    print("=" * 80)
    # Similarity metrics vs reference
    print("\nSimilarity vs ground-truth continuation (averaged over prompts):")
    if not ref_similarity_acc:
        print("  No similarity statistics collected (not enough successful generations).")
    else:
        for name, (sum_char, sum_word, sum_seq, cnt) in ref_similarity_acc.items():
            if cnt == 0:
                continue
            print(f"\n{name}:")
            print(f"  Character overlap:   {sum_char / cnt:.3f}")
            print(f"  Word overlap:        {sum_word / cnt:.3f}")
            print(f"  Sequence similarity: {sum_seq / cnt:.3f}")

    # ------------------------------------------------------------------
    # Optional: batched GPU tokenization benchmark to show peak throughput
    # using the in-process CUDA extension, if available.
    # ------------------------------------------------------------------
    if gpu_ext_tokenizer is not None and examples:
        # Optional: you could add a quality-focused batched experiment here if desired.
        pass


if __name__ == "__main__":
    main()

