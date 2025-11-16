#!/usr/bin/env python3
"""
Benchmark how tokenizer speed scales with input length for:
- GPU BlockBPE (gputok_binding.cu, via pybind11 extension)
- tiktoken (GPT-2)
- HuggingFace GPT-2 tokenizer

We use WikiText-103 as the source corpus, build long token streams with the
HuggingFace tokenizer, then extract fixed-length windows of:
  256, 512, 1024, 2048, 4096, 8192, 16384 HF tokens

For each length we:
- Build a small batch of prompts (decoded from HF token windows)
- Warm up each tokenizer (not timed)
- Measure:
    * GPU BlockBPE batch end-to-end tokens/sec and chars/sec
    * GPU BlockBPE batch kernel-only tokens/sec
    * tiktoken batch tokens/sec and chars/sec
    * HuggingFace batch tokens/sec and chars/sec
- Compute speedup factors for GPU vs tiktoken and vs HuggingFace.
"""

import math
import time
from pathlib import Path
from typing import List, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from transformers import AutoTokenizer
    import torch
except ImportError:
    AutoTokenizer = None
    torch = None

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None

try:
    from torch.utils.cpp_extension import load as load_extension  # type: ignore
except ImportError:
    load_extension = None


def build_gpu_extension(chunk_tokens: int = 2048, max_iters: int = 50):
    """
    Build/load the gputok_gpu CUDA extension and construct a GpuTokenizer.
    Returns (gpu_ext, gpu_tokenizer) or (None, None) on failure.
    """
    if load_extension is None:
        print("torch.utils.cpp_extension.load not available; GPU extension disabled.")
        return None, None

    ext_path = Path(__file__).parent / "gputok_binding.cu"
    if not ext_path.exists():
        print(f"gputok_binding.cu not found at {ext_path}; GPU extension disabled.")
        return None, None

    try:
        print("Building/loading gputok_gpu CUDA extension (pybind11) for scaling benchmark...")
        gpu_ext = load_extension(
            name="gputok_gpu",
            sources=[str(ext_path)],
            extra_include_paths=[
                str(Path(__file__).parent / "externals/cuCollections/include"),
                str(Path(__file__).parent / "externals/cccl/include"),
            ],
            extra_cflags=["-O3"],
            # cuCollections requires extended device lambdas support
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-extended-lambda"],
            verbose=False,
        )
        gpu_tokenizer = gpu_ext.GpuTokenizer(
            str(Path("data/gpt2_tokenizer/merges.txt")), chunk_tokens, max_iters
        )
        print("Loaded gputok_gpu extension.\n")
        return gpu_ext, gpu_tokenizer
    except Exception as e:
        print(f"WARNING: Failed to build/load gputok_gpu extension: {e}")
        print("GPU BlockBPE extension disabled for this benchmark.\n")
        return None, None


def build_hf_and_tiktoken():
    """Build the HuggingFace and tiktoken GPT-2 tokenizers."""
    if AutoTokenizer is None:
        raise ImportError("transformers not installed; run `pip install transformers torch`.")
    hf_tok = AutoTokenizer.from_pretrained("gpt2")
    if hf_tok is None:
        raise RuntimeError("Failed to load HuggingFace GPT-2 tokenizer.")

    tk_enc = None
    if tiktoken is not None:
        tk_enc = tiktoken.get_encoding("gpt2")
    else:
        print("tiktoken not installed; tiktoken benchmarks will be skipped.")

    return hf_tok, tk_enc


def load_wikitext_stream(hf_tok) -> List[int]:
    """
    Load WikiText-103 (test split), concatenate text lines, and encode with
    HuggingFace GPT-2 tokenizer to get a long token stream.
    """
    if load_dataset is None:
        raise ImportError("datasets not installed; run `pip install datasets`.")

    print("Loading WikiText-103 (wikitext-103-raw-v1, test split) for scaling benchmark...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    split = ds["test"]
    texts = [t for t in split["text"] if t and not t.isspace()]
    full_text = "\n".join(texts)
    print(f"Full text length (chars): {len(full_text)}")

    # For this benchmark we only need a manageable slice of the corpus to keep
    # GPU shared-memory usage under control. Take the first ~200k characters.
    max_chars = 200_000
    sliced_text = full_text[:max_chars]
    print(f"Encoding first {len(sliced_text)} chars with HuggingFace GPT-2 tokenizer...")
    hf_ids = hf_tok.encode(sliced_text, add_special_tokens=False)
    print(f"Total HF tokens in stream (sliced): {len(hf_ids)}\n")
    return hf_ids


def make_prompts_for_length(
    hf_tok, hf_ids: List[int], length: int, num_prompts: int
) -> List[str]:
    """
    Build `num_prompts` prompts, each with exactly `length` HF tokens, by taking
    sliding windows over the hf_ids stream and decoding back to text.
    """
    if len(hf_ids) < length:
        raise ValueError(
            f"Requested length {length} exceeds available HF tokens ({len(hf_ids)})."
        )
    if len(hf_ids) < length * num_prompts:
        # Fallback: reduce number of prompts if the stream is too short
        num_prompts = max(1, len(hf_ids) // length)
    if num_prompts <= 0:
        raise ValueError("Not enough HF tokens to build any prompts for this length.")

    max_start = len(hf_ids) - length
    stride = max(1, max_start // num_prompts)

    prompts: List[str] = []
    for i in range(num_prompts):
        start = i * stride
        if start + length > len(hf_ids):
            start = max_start
        span = hf_ids[start : start + length]
        prompts.append(hf_tok.decode(span))

    return prompts


def benchmark_tokenizers_for_length(
    length: int,
    prompts: List[str],
    hf_tok,
    tk_enc,
    gpu_ext_tokenizer,
) -> Tuple[dict, dict]:
    """
    Benchmark all tokenizers for a given HF token length.

    Returns:
      (metrics, speedups)
      where metrics and speedups are dicts keyed by tokenizer name.
    """
    metrics = {}
    speedups = {}

    total_chars = sum(len(p) for p in prompts)

    # Warmup (not timed) for CPU tokenizers
    if tk_enc is not None:
        try:
            _ = [tk_enc.encode(p) for p in prompts]
        except Exception:
            pass

    try:
        _ = hf_tok.batch_encode_plus(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception:
        pass

    # GPU BlockBPE (batch)
    if gpu_ext_tokenizer is not None:
        try:
            t0 = time.perf_counter()
            batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch(prompts)
            t1 = time.perf_counter()
            gpu_time = t1 - t0
            gpu_tokens = sum(len(ids) for ids in batch_ids)
            gpu_chars = total_chars
            gpu_kernel_s = (kernel_ms or 0.0) / 1000.0

            metrics["gpu_blockbpe"] = {
                "tokens": gpu_tokens,
                "chars": gpu_chars,
                "time_s": gpu_time,
                "kernel_s": gpu_kernel_s,
                "tokens_per_s": gpu_tokens / gpu_time if gpu_time > 0 else float("inf"),
                "chars_per_s": gpu_chars / gpu_time if gpu_time > 0 else float("inf"),
                "kernel_tokens_per_s": (
                    gpu_tokens / gpu_kernel_s if gpu_kernel_s > 0 else float("inf")
                ),
            }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE ERROR: {e}")

    # CPU BlockBPE (batch, reference)
    if gpu_ext_tokenizer is not None:
        try:
            t0 = time.perf_counter()
            cpu_batch_ids = gpu_ext_tokenizer.tokenize_batch_cpu(prompts)
            t1 = time.perf_counter()
            cpu_time = t1 - t0
            cpu_tokens = sum(len(ids) for ids in cpu_batch_ids)
            cpu_chars = total_chars

            metrics["cpu_blockbpe"] = {
                "tokens": cpu_tokens,
                "chars": cpu_chars,
                "time_s": cpu_time,
                "tokens_per_s": cpu_tokens / cpu_time if cpu_time > 0 else float("inf"),
                "chars_per_s": cpu_chars / cpu_time if cpu_time > 0 else float("inf"),
            }
        except Exception as e:
            print(f"[len={length}] CPU BlockBPE ERROR: {e}")

    # tiktoken
    if tk_enc is not None:
        try:
            t0 = time.perf_counter()
            tk_ids_list = [tk_enc.encode(p) for p in prompts]
            t1 = time.perf_counter()
            tk_time = t1 - t0
            tk_tokens = sum(len(ids) for ids in tk_ids_list)
            tk_chars = total_chars

            metrics["tiktoken"] = {
                "tokens": tk_tokens,
                "chars": tk_chars,
                "time_s": tk_time,
                "tokens_per_s": tk_tokens / tk_time if tk_time > 0 else float("inf"),
                "chars_per_s": tk_chars / tk_time if tk_time > 0 else float("inf"),
            }
        except Exception as e:
            print(f"[len={length}] tiktoken ERROR: {e}")

    # HuggingFace GPT-2
    try:
        t0 = time.perf_counter()
        hf_out = hf_tok.batch_encode_plus(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        t1 = time.perf_counter()
        hf_time = t1 - t0
        hf_tokens = sum(len(ids) for ids in hf_out["input_ids"])
        hf_chars = total_chars

        metrics["huggingface"] = {
            "tokens": hf_tokens,
            "chars": hf_chars,
            "time_s": hf_time,
            "tokens_per_s": hf_tokens / hf_time if hf_time > 0 else float("inf"),
            "chars_per_s": hf_chars / hf_time if hf_time > 0 else float("inf"),
        }
    except Exception as e:
        print(f"[len={length}] HuggingFace ERROR: {e}")

    # Speedups: GPU vs others
    if "gpu_blockbpe" in metrics:
        g = metrics["gpu_blockbpe"]
        g_tps = g["tokens_per_s"]
        if "cpu_blockbpe" in metrics:
            tps = metrics["cpu_blockbpe"]["tokens_per_s"]
            speedups["gpu_vs_cpu_blockbpe"] = g_tps / tps if tps > 0 else float("inf")
        if "tiktoken" in metrics:
            tps = metrics["tiktoken"]["tokens_per_s"]
            speedups["gpu_vs_tiktoken"] = g_tps / tps if tps > 0 else float("inf")
        if "huggingface" in metrics:
            tps = metrics["huggingface"]["tokens_per_s"]
            speedups["gpu_vs_hf"] = g_tps / tps if tps > 0 else float("inf")

    return metrics, speedups


def benchmark_tokenizers_latency_for_length(
    length: int,
    prompts: List[str],
    hf_tok,
    tk_enc,
    gpu_ext_tokenizer,
) -> Tuple[dict, dict]:
    """
    Benchmark all tokenizers for a given HF token length in per-prompt
    (batch size 1) latency mode.

    Returns:
      (metrics, speedups) in the same format as the batched benchmark.
    """
    metrics = {}
    speedups = {}

    total_chars = sum(len(p) for p in prompts)

    # GPU BlockBPE latency: one prompt per call
    if gpu_ext_tokenizer is not None:
        try:
            total_tokens = 0
            total_time = 0.0
            total_kernel_s = 0.0
            for p in prompts:
                t0 = time.perf_counter()
                batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch([p])
                t1 = time.perf_counter()
                total_time += t1 - t0
                if batch_ids:
                    total_tokens += len(batch_ids[0])
                if kernel_ms is not None:
                    total_kernel_s += (kernel_ms / 1000.0)

            metrics["gpu_blockbpe"] = {
                "tokens": total_tokens,
                "chars": total_chars,
                "time_s": total_time,
                "kernel_s": total_kernel_s,
                "tokens_per_s": total_tokens / total_time if total_time > 0 else float("inf"),
                "chars_per_s": total_chars / total_time if total_time > 0 else float("inf"),
                "kernel_tokens_per_s": (
                    total_tokens / total_kernel_s if total_kernel_s > 0 else float("inf")
                ),
            }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE (latency) ERROR: {e}")

    # CPU BlockBPE latency: one prompt per call
    if gpu_ext_tokenizer is not None:
        try:
            total_tokens = 0
            total_time = 0.0
            for p in prompts:
                t0 = time.perf_counter()
                cpu_batch_ids = gpu_ext_tokenizer.tokenize_batch_cpu([p])
                t1 = time.perf_counter()
                total_time += t1 - t0
                if cpu_batch_ids:
                    total_tokens += len(cpu_batch_ids[0])

            metrics["cpu_blockbpe"] = {
                "tokens": total_tokens,
                "chars": total_chars,
                "time_s": total_time,
                "tokens_per_s": total_tokens / total_time if total_time > 0 else float("inf"),
                "chars_per_s": total_chars / total_time if total_time > 0 else float("inf"),
            }
        except Exception as e:
            print(f"[len={length}] CPU BlockBPE (latency) ERROR: {e}")

    # tiktoken latency: one prompt per encode() call
    if tk_enc is not None:
        try:
            total_tokens = 0
            total_time = 0.0
            for p in prompts:
                t0 = time.perf_counter()
                ids = tk_enc.encode(p)
                t1 = time.perf_counter()
                total_time += t1 - t0
                total_tokens += len(ids)

            metrics["tiktoken"] = {
                "tokens": total_tokens,
                "chars": total_chars,
                "time_s": total_time,
                "tokens_per_s": total_tokens / total_time if total_time > 0 else float("inf"),
                "chars_per_s": total_chars / total_time if total_time > 0 else float("inf"),
            }
        except Exception as e:
            print(f"[len={length}] tiktoken (latency) ERROR: {e}")

    # HuggingFace latency: one prompt per encode() call
    try:
        total_tokens = 0
        total_time = 0.0
        for p in prompts:
            t0 = time.perf_counter()
            ids = hf_tok.encode(p, add_special_tokens=False)
            t1 = time.perf_counter()
            total_time += t1 - t0
            total_tokens += len(ids)

        metrics["huggingface"] = {
            "tokens": total_tokens,
            "chars": total_chars,
            "time_s": total_time,
            "tokens_per_s": total_tokens / total_time if total_time > 0 else float("inf"),
            "chars_per_s": total_chars / total_time if total_time > 0 else float("inf"),
        }
    except Exception as e:
        print(f"[len={length}] HuggingFace (latency) ERROR: {e}")

    # Speedups: GPU vs others
    if "gpu_blockbpe" in metrics:
        g = metrics["gpu_blockbpe"]
        g_tps = g["tokens_per_s"]
        if "cpu_blockbpe" in metrics:
            tps = metrics["cpu_blockbpe"]["tokens_per_s"]
            speedups["gpu_vs_cpu_blockbpe"] = g_tps / tps if tps > 0 else float("inf")
        if "tiktoken" in metrics:
            tps = metrics["tiktoken"]["tokens_per_s"]
            speedups["gpu_vs_tiktoken"] = g_tps / tps if tps > 0 else float("inf")
        if "huggingface" in metrics:
            tps = metrics["huggingface"]["tokens_per_s"]
            speedups["gpu_vs_hf"] = g_tps / tps if tps > 0 else float("inf")

    return metrics, speedups


def main():
    if AutoTokenizer is None or load_dataset is None:
        print("ERROR: transformers and/or datasets not installed.")
        print("Install with: pip install transformers torch datasets")
        return

    hf_tok, tk_enc = build_hf_and_tiktoken()
    gpu_ext, gpu_ext_tokenizer = build_gpu_extension(chunk_tokens=2048, max_iters=50)

    hf_ids = load_wikitext_stream(hf_tok)

    # One short warmup for GPU tokenizer (safe small prompt), if available.
    if gpu_ext_tokenizer is not None:
        try:
            gpu_ext_tokenizer.tokenize_batch(["Hello world!"] * 4)
        except Exception:
            pass

    # Target HF-token lengths for the scaling experiment.
    # With chunking in GpuTokenizer (at most chunk_tokens_ per chunk), the GPU
    # kernel can safely handle arbitrarily long texts by splitting them into
    # multiple chunks. We therefore enable GPU benchmarking for all lengths.
    target_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 131072]
    num_prompts_per_length = 16  # per length

    print("=" * 80)
    print("Tokenizer scaling benchmark (WikiText-103, GPT-2 vocabulary)")
    print("=" * 80)
    print(f"Target HF token lengths: {target_lengths}")
    print(f"Prompts per length (max): {num_prompts_per_length}\n")

    results = {}

    for length in target_lengths:
        print("-" * 80)
        print(f"Length bucket: {length} HF tokens")
        try:
            prompts = make_prompts_for_length(
                hf_tok, hf_ids, length, num_prompts_per_length
            )
        except Exception as e:
            print(f"  Skipping length {length}: {e}")
            continue

        # Run GPU BlockBPE for all lengths when the extension is available; the
        # underlying implementation chunks long inputs internally.
        metrics_batch, speedups_batch = benchmark_tokenizers_for_length(
            length, prompts, hf_tok, tk_enc, gpu_ext_tokenizer
        )
        metrics_lat, speedups_lat = benchmark_tokenizers_latency_for_length(
            length, prompts, hf_tok, tk_enc, gpu_ext_tokenizer
        )
        results[length] = (metrics_batch, speedups_batch, metrics_lat, speedups_lat)

        total_chars = sum(len(p) for p in prompts)
        print(f"  Built {len(prompts)} prompts, total chars={total_chars}")

        # Pretty-print batched metrics
        if "gpu_blockbpe" in metrics_batch:
            m = metrics_batch["gpu_blockbpe"]
            print(
                f"  GPU BlockBPE (batch): "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )
            if m["kernel_s"] > 0:
                print(
                    f"    kernel-only: {m['kernel_tokens_per_s']:.1f} tokens/s "
                    f"(kernel_time={m['kernel_s']:.6f}s)"
                )

        if "tiktoken" in metrics_batch:
            m = metrics_batch["tiktoken"]
            print(
                f"  tiktoken:           "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )

        if "huggingface" in metrics_batch:
            m = metrics_batch["huggingface"]
            print(
                f"  HuggingFace GPT-2:  "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )

        # CPU BlockBPE (batch)
        if "cpu_blockbpe" in metrics_batch:
            m = metrics_batch["cpu_blockbpe"]
            print(
                f"  CPU BlockBPE (batch):"
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )

        # Pretty-print latency metrics
        if "cpu_blockbpe" in metrics_lat:
            m = metrics_lat["cpu_blockbpe"]
            print(
                f"  CPU BlockBPE (lat):  "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )

        if "gpu_blockbpe" in metrics_lat:
            m = metrics_lat["gpu_blockbpe"]
            print(
                f"  GPU BlockBPE (lat):   "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )
        if "tiktoken" in metrics_lat:
            m = metrics_lat["tiktoken"]
            print(
                f"  tiktoken (lat):      "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )
        if "huggingface" in metrics_lat:
            m = metrics_lat["huggingface"]
            print(
                f"  HF GPT-2 (lat):      "
                f"{m['tokens_per_s']:.1f} tokens/s, {m['chars_per_s']:.1f} chars/s "
                f"(tokens={m['tokens']}, chars={m['chars']}, time={m['time_s']:.4f}s)"
            )

        # Speedups (batched)
        if speedups_batch:
            if "gpu_vs_cpu_blockbpe" in speedups_batch:
                print(f"  Speedup GPU vs CPU BlockBPE (batch): {speedups_batch['gpu_vs_cpu_blockbpe']:.2f}x")
            if "gpu_vs_tiktoken" in speedups_batch:
                print(f"  Speedup GPU vs tiktoken (batch):  {speedups_batch['gpu_vs_tiktoken']:.2f}x")
            if "gpu_vs_hf" in speedups_batch:
                print(f"  Speedup GPU vs HF (batch):        {speedups_batch['gpu_vs_hf']:.2f}x")

        # Speedups (latency)
        if speedups_lat:
            if "gpu_vs_cpu_blockbpe" in speedups_lat:
                print(f"  Speedup GPU vs CPU BlockBPE (lat):   {speedups_lat['gpu_vs_cpu_blockbpe']:.2f}x")
            if "gpu_vs_tiktoken" in speedups_lat:
                print(f"  Speedup GPU vs tiktoken (lat):    {speedups_lat['gpu_vs_tiktoken']:.2f}x")
            if "gpu_vs_hf" in speedups_lat:
                print(f"  Speedup GPU vs HF (lat):          {speedups_lat['gpu_vs_hf']:.2f}x")

    # ------------------------------------------------------------------
    # Summary tables (tokens/sec) for easy comparison across lengths
    # ------------------------------------------------------------------
    if results:
        # Batched throughput
        print("\n" + "=" * 80)
        print("Summary (BATCH): tokenizer throughput vs input length (tokens/sec)")
        print("=" * 80)

        header = (
            f"{'Len (HF tok)':>12}  "
            f"{'GPU tok/s':>12}  "
            f"{'CPU BPE':>12}  "
            f"{'tiktoken':>12}  "
            f"{'HF tok/s':>12}  "
            f"{'GPU/CPU':>8}  "
            f"{'GPU/TK':>8}  "
            f"{'GPU/HF':>8}"
        )
        print(header)
        print("-" * len(header))

        def fmt_tps(metrics: dict, key: str) -> str:
            if key not in metrics:
                return "     -     "
            v = metrics[key]["tokens_per_s"]
            return f"{v/1e6:6.2f}M"

        def fmt_speed(speedups: dict, key: str) -> str:
            if key not in speedups:
                return "   -   "
            v = speedups[key]
            return f"{v:6.2f}x"

        for length in sorted(results.keys()):
            metrics_batch, speedups_batch, _, _ = results[length]
            gpu = fmt_tps(metrics_batch, "gpu_blockbpe")
            cpu = fmt_tps(metrics_batch, "cpu_blockbpe")
            tk = fmt_tps(metrics_batch, "tiktoken")
            hf = fmt_tps(metrics_batch, "huggingface")
            gpu_cpu = fmt_speed(speedups_batch, "gpu_vs_cpu_blockbpe")
            gpu_tk = fmt_speed(speedups_batch, "gpu_vs_tiktoken")
            gpu_hf = fmt_speed(speedups_batch, "gpu_vs_hf")
            print(
                f"{length:12d}  "
                f"{gpu:>12}  "
                f"{cpu:>12}  "
                f"{tk:>12}  "
                f"{hf:>12}  "
                f"{gpu_cpu:>8}  "
                f"{gpu_tk:>8}  "
                f"{gpu_hf:>8}"
            )

        # Latency throughput
        print("\n" + "=" * 80)
        print("Summary (LATENCY): tokenizer throughput vs input length (tokens/sec)")
        print("=" * 80)

        print(header)
        print("-" * len(header))

        for length in sorted(results.keys()):
            _, _, metrics_lat, speedups_lat = results[length]
            gpu = fmt_tps(metrics_lat, "gpu_blockbpe")
            cpu = fmt_tps(metrics_lat, "cpu_blockbpe")
            tk = fmt_tps(metrics_lat, "tiktoken")
            hf = fmt_tps(metrics_lat, "huggingface")
            gpu_cpu = fmt_speed(speedups_lat, "gpu_vs_cpu_blockbpe")
            gpu_tk = fmt_speed(speedups_lat, "gpu_vs_tiktoken")
            gpu_hf = fmt_speed(speedups_lat, "gpu_vs_hf")
            print(
                f"{length:12d}  "
                f"{gpu:>12}  "
                f"{cpu:>12}  "
                f"{tk:>12}  "
                f"{hf:>12}  "
                f"{gpu_cpu:>8}  "
                f"{gpu_tk:>8}  "
                f"{gpu_hf:>8}"
            )

    print("\n" + "=" * 80)
    print("Scaling benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()


