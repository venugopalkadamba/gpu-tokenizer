#!/usr/bin/env python3
"""
Benchmark how tokenizer speed scales with input length for:
- GPU BlockBPE (gputok_binding.cu, via pybind11 extension)
- tiktoken (GPT-2)
- HuggingFace GPT-2 tokenizer

We use WikiText-103 as the source corpus, build long token streams with the
HuggingFace tokenizer, then extract fixed-length windows of:
  256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 HF tokens

For each length we:
- Build a small batch of prompts (decoded from HF token windows)
- Warm up each tokenizer (not timed)
- Run each tokenizer multiple times (configurable) and measure:
    * GPU BlockBPE batch end-to-end pybind call time and kernel-only time
    * CPU BlockBPE batch time (reference)
    * tiktoken batch time
    * HuggingFace batch time
- Compute speedup factors for GPU vs tiktoken and vs HuggingFace based on tokens/sec
  (even though the summary table reports times).
"""

import math
import time
import subprocess
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional

# Disable HuggingFace tokenizers fork/parallelism warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


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


def run_blockbpe_binary_on_prompts(
    prompts: List[str],
    exe_path: Path,
    chunk_tokens: int,
    timeout_s: int = 300,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Run the standalone BlockBPE GPU tokenizer binary (built from gputok_blockbpe.cu)
    on a batch of prompts by concatenating them into a single input file.

    Returns:
        (total_tokens, call_time_s, kernel_time_s) where any element may be None on failure.
    """
    exe_path = exe_path.resolve()
    if not exe_path.exists():
        print(f"BlockBPE binary not found at {exe_path}; skipping baseline BlockBPE benchmark.")
        return None, None, None

    if not prompts:
        return None, None, None

    input_path = Path("data/input/blockbpe_scaling_input.txt")
    output_path = Path("data/output/gpu_tokens_blockbpe.txt")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate prompts with newlines; the binary treats the whole file as one sequence.
    text = "\n".join(prompts)
    input_path.write_text(text, encoding="utf-8")

    if output_path.exists():
        output_path.unlink()

    cmd = [str(exe_path), str(chunk_tokens), str(input_path.resolve())]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(exe_path.parent),
    )
    t1 = time.perf_counter()
    call_time_s = t1 - t0

    if result.returncode != 0:
        print(f"[BlockBPE baseline] stderr: {result.stderr[:500]}")
        print(f"[BlockBPE baseline] stdout: {result.stdout[:500]}")
        print(f"[BlockBPE baseline] binary failed with return code {result.returncode}")
        return None, None, None

    kernel_ms: Optional[float] = None
    for line in result.stdout.splitlines():
        m = re.search(
            r"GPU BlockBPE-style greedy kernel time.*:\s*([0-9.]+)\s*ms", line
        )
        if m:
            try:
                kernel_ms = float(m.group(1))
            except ValueError:
                kernel_ms = None
            break

    if not output_path.exists():
        print("[BlockBPE baseline] Output token file not found; skipping.")
        return None, None, None

    total_tokens = 0
    with output_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                total_tokens += len(line.split())

    if total_tokens == 0:
        print("[BlockBPE baseline] No tokens produced; skipping.")
        return None, None, None

    kernel_s = (kernel_ms / 1000.0) if kernel_ms is not None else None
    return total_tokens, call_time_s, kernel_s


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

    # Encode the full test split to ensure we have enough tokens for the
    # largest target length in the scaling benchmark. The GPU tokenizer
    # internally chunks long inputs, so we are not constrained by shared
    # memory limits here.
    print("Encoding full WikiText-103 test split with HuggingFace GPT-2 tokenizer...")
    hf_ids = hf_tok.encode(full_text, add_special_tokens=False)
    print(f"Total HF tokens in stream: {len(hf_ids)}\n")
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
    num_runs: int = 1,
) -> Tuple[dict, dict]:
    """
    Benchmark all tokenizers for a given HF token length.

    Returns:
      (metrics, speedups)
      where metrics and speedups are dicts keyed by tokenizer name.
    """
    metrics: dict = {}
    speedups: dict = {}

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

    # GPU BlockBPE (batch, optimized pybind implementation)
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            total_kernel_s = 0.0
            gpu_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                t0 = time.perf_counter()
                batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch(prompts)
                t1 = time.perf_counter()
                run_time = t1 - t0
                if batch_ids:
                    gpu_tokens = sum(len(ids) for ids in batch_ids)
                total_time += run_time
                if kernel_ms is not None:
                    total_kernel_s += (kernel_ms / 1000.0)
                runs_ok += 1

            if runs_ok > 0 and gpu_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                avg_kernel_s = total_kernel_s / runs_ok if total_kernel_s > 0 else 0.0
                metrics["gpu_blockbpe"] = {
                    "tokens": gpu_tokens,
                    "time_s": avg_time,
                    "kernel_s": avg_kernel_s,
                    "tokens_per_s": gpu_tokens / avg_time if avg_time > 0 else float("inf"),
                    "kernel_tokens_per_s": (
                        gpu_tokens / avg_kernel_s if avg_kernel_s > 0 else float("inf")
                    ),
                }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE ERROR: {e}")

    # GPU BlockBPE baseline (batch, pybind implementation of gputok_blockbpe.cu)
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            total_kernel_s = 0.0
            gpu_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                t0 = time.perf_counter()
                batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch_blockbpe_base(prompts)
                t1 = time.perf_counter()
                run_time = t1 - t0
                if batch_ids:
                    gpu_tokens = sum(len(ids) for ids in batch_ids)
                total_time += run_time
                if kernel_ms is not None:
                    total_kernel_s += (kernel_ms / 1000.0)
                runs_ok += 1

            if runs_ok > 0 and gpu_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                avg_kernel_s = total_kernel_s / runs_ok if total_kernel_s > 0 else 0.0
                metrics["gpu_blockbpe_baseline"] = {
                    "tokens": gpu_tokens,
                    "time_s": avg_time,
                    "kernel_s": avg_kernel_s,
                    "tokens_per_s": gpu_tokens / avg_time if avg_time > 0 else float("inf"),
                    "kernel_tokens_per_s": (
                        gpu_tokens / avg_kernel_s if avg_kernel_s > 0 else float("inf")
                    ),
                }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE baseline ERROR: {e}")

    # CPU BlockBPE (batch, reference)
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            cpu_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                t0 = time.perf_counter()
                cpu_batch_ids = gpu_ext_tokenizer.tokenize_batch_cpu(prompts)
                t1 = time.perf_counter()
                run_time = t1 - t0
                if cpu_batch_ids:
                    cpu_tokens = sum(len(ids) for ids in cpu_batch_ids)
                total_time += run_time
                runs_ok += 1

            if runs_ok > 0 and cpu_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                metrics["cpu_blockbpe"] = {
                    "tokens": cpu_tokens,
                    "time_s": avg_time,
                    "tokens_per_s": cpu_tokens / avg_time if avg_time > 0 else float("inf"),
                }
        except Exception as e:
            print(f"[len={length}] CPU BlockBPE ERROR: {e}")

    # tiktoken
    if tk_enc is not None:
        try:
            total_time = 0.0
            tk_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                t0 = time.perf_counter()
                tk_ids_list = [tk_enc.encode(p) for p in prompts]
                t1 = time.perf_counter()
                run_time = t1 - t0
                tk_tokens = sum(len(ids) for ids in tk_ids_list)
                total_time += run_time
                runs_ok += 1

            if runs_ok > 0 and tk_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                metrics["tiktoken"] = {
                    "tokens": tk_tokens,
                    "time_s": avg_time,
                    "tokens_per_s": tk_tokens / avg_time if avg_time > 0 else float("inf"),
                }
        except Exception as e:
            print(f"[len={length}] tiktoken ERROR: {e}")

    # HuggingFace GPT-2
    try:
        total_time = 0.0
        hf_tokens = 0
        runs_ok = 0
        for _ in range(num_runs):
            t0 = time.perf_counter()
            hf_out = hf_tok.batch_encode_plus(
                prompts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            t1 = time.perf_counter()
            run_time = t1 - t0
            hf_tokens = sum(len(ids) for ids in hf_out["input_ids"])
            total_time += run_time
            runs_ok += 1

        if runs_ok > 0 and hf_tokens > 0 and total_time > 0.0:
            avg_time = total_time / runs_ok
            metrics["huggingface"] = {
                "tokens": hf_tokens,
                "time_s": avg_time,
                "tokens_per_s": hf_tokens / avg_time if avg_time > 0 else float("inf"),
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
    num_runs: int = 1,
) -> Tuple[dict, dict]:
    """
    Benchmark all tokenizers for a given HF token length in per-prompt
    (batch size 1) latency mode.

    Returns:
      (metrics, speedups) in the same format as the batched benchmark.
    """
    metrics: dict = {}
    speedups: dict = {}

    # GPU BlockBPE latency: one prompt per call
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            total_kernel_s = 0.0
            total_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                run_tokens = 0
                run_time = 0.0
                run_kernel_s = 0.0
                for p in prompts:
                    t0 = time.perf_counter()
                    batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch([p])
                    t1 = time.perf_counter()
                    run_time += t1 - t0
                    if batch_ids:
                        run_tokens += len(batch_ids[0])
                    if kernel_ms is not None:
                        run_kernel_s += (kernel_ms / 1000.0)
                if run_tokens > 0 and run_time > 0.0:
                    total_time += run_time
                    total_kernel_s += run_kernel_s
                    total_tokens = run_tokens
                    runs_ok += 1

            if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                avg_kernel_s = total_kernel_s / runs_ok if total_kernel_s > 0 else 0.0
                metrics["gpu_blockbpe"] = {
                    "tokens": total_tokens,
                    "time_s": avg_time,
                    "kernel_s": avg_kernel_s,
                    "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
                    "kernel_tokens_per_s": (
                        total_tokens / avg_kernel_s if avg_kernel_s > 0 else float("inf")
                    ),
                }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE (latency) ERROR: {e}")

    # CPU BlockBPE latency: one prompt per call
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            total_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                run_tokens = 0
                run_time = 0.0
                for p in prompts:
                    t0 = time.perf_counter()
                    cpu_batch_ids = gpu_ext_tokenizer.tokenize_batch_cpu([p])
                    t1 = time.perf_counter()
                    run_time += t1 - t0
                    if cpu_batch_ids:
                        run_tokens += len(cpu_batch_ids[0])
                if run_tokens > 0 and run_time > 0.0:
                    total_time += run_time
                    total_tokens = run_tokens
                    runs_ok += 1

            if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                metrics["cpu_blockbpe"] = {
                    "tokens": total_tokens,
                    "time_s": avg_time,
                    "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
                }
        except Exception as e:
            print(f"[len={length}] CPU BlockBPE (latency) ERROR: {e}")

    # Baseline BlockBPE latency: one prompt per call.
    # Prefer pybind11 baseline kernel; fall back to standalone binary if needed.
    if gpu_ext_tokenizer is not None:
        try:
            total_time = 0.0
            total_kernel_s = 0.0
            total_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                run_tokens = 0
                run_time = 0.0
                run_kernel_s = 0.0
                for p in prompts:
                    t0 = time.perf_counter()
                    batch_ids, kernel_ms = gpu_ext_tokenizer.tokenize_batch_blockbpe_base([p])
                    t1 = time.perf_counter()
                    run_time += t1 - t0
                    if batch_ids:
                        run_tokens += len(batch_ids[0])
                    if kernel_ms is not None:
                        run_kernel_s += (kernel_ms / 1000.0)
                if run_tokens > 0 and run_time > 0.0:
                    total_time += run_time
                    total_kernel_s += run_kernel_s
                    total_tokens = run_tokens
                    runs_ok += 1

            if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                avg_kernel_s = total_kernel_s / runs_ok if total_kernel_s > 0 else 0.0
                metrics["gpu_blockbpe_baseline"] = {
                    "tokens": total_tokens,
                    "time_s": avg_time,
                    "kernel_s": avg_kernel_s,
                    "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
                    "kernel_tokens_per_s": (
                        total_tokens / avg_kernel_s if avg_kernel_s > 0 else float("inf")
                    ),
                }
        except Exception as e:
            print(f"[len={length}] GPU BlockBPE baseline (latency, pybind) ERROR: {e}")
    else:
        blockbpe_exe = (Path(__file__).parent / "gpu_tokenizer_blockbpe").resolve()
        if blockbpe_exe.exists():
            try:
                total_time = 0.0
                total_kernel_s = 0.0
                total_tokens = 0
                runs_ok = 0
                for _ in range(num_runs):
                    run_tokens = 0
                    run_time = 0.0
                    run_kernel_s = 0.0
                    for p in prompts:
                        tokens, call_time_s, kernel_s = run_blockbpe_binary_on_prompts(
                            [p], blockbpe_exe, chunk_tokens=2048
                        )
                        if tokens is None or call_time_s is None:
                            continue
                        run_tokens += tokens
                        run_time += call_time_s
                        if kernel_s is not None:
                            run_kernel_s += kernel_s
                    if run_tokens > 0 and run_time > 0.0:
                        total_time += run_time
                        total_kernel_s += run_kernel_s
                        total_tokens = run_tokens
                        runs_ok += 1

                if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
                    avg_time = total_time / runs_ok
                    avg_kernel_s = total_kernel_s / runs_ok if total_kernel_s > 0 else 0.0
                    metrics["gpu_blockbpe_baseline"] = {
                        "tokens": total_tokens,
                        "time_s": avg_time,
                        "kernel_s": avg_kernel_s,
                        "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
                        "kernel_tokens_per_s": (
                            total_tokens / avg_kernel_s if avg_kernel_s > 0 else float("inf")
                        ),
                    }
            except Exception as e:
                print(f"[len={length}] GPU BlockBPE baseline (latency, binary) ERROR: {e}")

    # tiktoken latency: one prompt per encode() call
    if tk_enc is not None:
        try:
            total_time = 0.0
            total_tokens = 0
            runs_ok = 0
            for _ in range(num_runs):
                run_tokens = 0
                run_time = 0.0
                for p in prompts:
                    t0 = time.perf_counter()
                    ids = tk_enc.encode(p)
                    t1 = time.perf_counter()
                    run_time += t1 - t0
                    run_tokens += len(ids)
                if run_tokens > 0 and run_time > 0.0:
                    total_time += run_time
                    total_tokens = run_tokens
                    runs_ok += 1

            if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
                avg_time = total_time / runs_ok
                metrics["tiktoken"] = {
                    "tokens": total_tokens,
                    "time_s": avg_time,
                    "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
                }
        except Exception as e:
            print(f"[len={length}] tiktoken (latency) ERROR: {e}")

    # HuggingFace latency: one prompt per encode() call
    try:
        total_time = 0.0
        total_tokens = 0
        runs_ok = 0
        for _ in range(num_runs):
            run_tokens = 0
            run_time = 0.0
            for p in prompts:
                t0 = time.perf_counter()
                ids = hf_tok.encode(p, add_special_tokens=False)
                t1 = time.perf_counter()
                run_time += t1 - t0
                run_tokens += len(ids)
            if run_tokens > 0 and run_time > 0.0:
                total_time += run_time
                total_tokens = run_tokens
                runs_ok += 1

        if runs_ok > 0 and total_tokens > 0 and total_time > 0.0:
            avg_time = total_time / runs_ok
            metrics["huggingface"] = {
                "tokens": total_tokens,
                "time_s": avg_time,
                "tokens_per_s": total_tokens / avg_time if avg_time > 0 else float("inf"),
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

        # Additional speedups for opt GPU BlockBPE vs CPU tokenizers
        if "tiktoken" in metrics:
            tps = metrics["tiktoken"]["tokens_per_s"]
            speedups["gpu_opt_vs_tiktoken"] = g_tps / tps if tps > 0 else float("inf")
        if "huggingface" in metrics:
            tps = metrics["huggingface"]["tokens_per_s"]
            speedups["gpu_opt_vs_hf"] = g_tps / tps if tps > 0 else float("inf")

    if "gpu_blockbpe_baseline" in metrics:
        b = metrics["gpu_blockbpe_baseline"]
        b_tps = b["tokens_per_s"]
        if "tiktoken" in metrics:
            tps = metrics["tiktoken"]["tokens_per_s"]
            speedups["gpu_base_vs_tiktoken"] = b_tps / tps if tps > 0 else float("inf")
        if "huggingface" in metrics:
            tps = metrics["huggingface"]["tokens_per_s"]
            speedups["gpu_base_vs_hf"] = b_tps / tps if tps > 0 else float("inf")

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
    target_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    num_prompts_per_length = 16  # per length
    num_runs_per_length = 10  # average over this many runs per tokenizer/length

    print("=" * 80)
    print("Tokenizer scaling benchmark (WikiText-103, GPT-2 vocabulary)")
    print("=" * 80)
    print(f"Target HF token lengths: {target_lengths}")
    print(f"Prompts per length (max): {num_prompts_per_length}\n")

    results = {}

    length_iter = target_lengths
    if tqdm is not None:
        length_iter = tqdm(target_lengths, desc="Seq length buckets", unit="bucket")

    for length in length_iter:
        print("-" * 80)
        print(f"Length bucket: {length} HF tokens")
        try:
            prompts = make_prompts_for_length(
                hf_tok, hf_ids, length, num_prompts_per_length
            )
        except Exception as e:
            print(f"  Skipping length {length}: {e}")
            continue

        # Single-sequence (latency) benchmarks only: one prompt at a time.
        metrics_lat, speedups_lat = benchmark_tokenizers_latency_for_length(
            length, prompts, hf_tok, tk_enc, gpu_ext_tokenizer, num_runs=num_runs_per_length
        )
        results[length] = (None, None, metrics_lat, speedups_lat)

        total_chars = sum(len(p) for p in prompts)
        print(f"  Built {len(prompts)} prompts, total chars={total_chars}")

        # Pretty-print latency metrics (single-sequence)
        if "cpu_blockbpe" in metrics_lat:
            m = metrics_lat["cpu_blockbpe"]
            print(
                f"  CPU BlockBPE (lat):  "
                f"{m['tokens_per_s']:.1f} tokens/s "
                f"(tokens={m['tokens']}, time={m['time_s']:.4f}s)"
            )

        if "gpu_blockbpe" in metrics_lat:
            m = metrics_lat["gpu_blockbpe"]
            print(
                f"  GPU BlockBPE (lat):   "
                f"{m['tokens_per_s']:.1f} tokens/s "
                f"(tokens={m['tokens']}, time={m['time_s']:.4f}s)"
            )
        if "gpu_blockbpe_baseline" in metrics_lat:
            m = metrics_lat["gpu_blockbpe_baseline"]
            print(
                f"  GPU BlockBPE base (lat):"
                f"{m['tokens_per_s']:.1f} tokens/s "
                f"(tokens={m['tokens']}, time={m['time_s']:.4f}s)"
            )
        if "tiktoken" in metrics_lat:
            m = metrics_lat["tiktoken"]
            print(
                f"  tiktoken (lat):      "
                f"{m['tokens_per_s']:.1f} tokens/s "
                f"(tokens={m['tokens']}, time={m['time_s']:.4f}s)"
            )
        if "huggingface" in metrics_lat:
            m = metrics_lat["huggingface"]
            print(
                f"  HF GPT-2 (lat):      "
                f"{m['tokens_per_s']:.1f} tokens/s "
                f"(tokens={m['tokens']}, time={m['time_s']:.4f}s)"
            )

        # Speedups (latency)
        if speedups_lat:
            if "gpu_vs_cpu_blockbpe" in speedups_lat:
                print(f"  Speedup GPU vs CPU BlockBPE (lat):   {speedups_lat['gpu_vs_cpu_blockbpe']:.2f}x")
            if "gpu_vs_tiktoken" in speedups_lat:
                print(f"  Speedup GPU vs tiktoken (lat):    {speedups_lat['gpu_vs_tiktoken']:.2f}x")
            if "gpu_vs_hf" in speedups_lat:
                print(f"  Speedup GPU vs HF (lat):          {speedups_lat['gpu_vs_hf']:.2f}x")

    # ------------------------------------------------------------------
    # Summary tables (latency-only, single-sequence) for easy comparison
    # ------------------------------------------------------------------
    if results:
        print("\n" + "=" * 80)
        print("Summary (LATENCY, single sequence): average time vs input length")
        print("=" * 80)

        def fmt_time_ms(metrics: dict, key: str) -> str:
            if key not in metrics:
                return "     -     "
            v = metrics[key]["time_s"]
            return f"{v*1000:7.2f}ms"

        def fmt_kernel_ms(metrics: dict, key: str) -> str:
            if key not in metrics or "kernel_s" not in metrics[key]:
                return "     -     "
            v = metrics[key]["kernel_s"]
            return f"{v*1000:7.2f}ms"

        def fmt_speed(speedups: dict, key: str) -> str:
            if key not in speedups:
                return "   -   "
            v = speedups[key]
            return f"{v:6.2f}x"

        # Table 1: absolute times
        header_time = (
            f"{'Len (HF tok)':>12}  "
            f"{'GPU opt':>12}  "
            f"{'GPU opt k':>12}  "
            f"{'GPU base':>12}  "
            f"{'GPU base k':>12}  "
            f"{'CPU BPE':>12}  "
            f"{'tiktoken':>12}  "
            f"{'HF time':>12}"
        )
        print(header_time)
        print("-" * len(header_time))

        for length in sorted(results.keys()):
            _, _, metrics_lat, _ = results[length]
            gpu_opt = fmt_time_ms(metrics_lat, "gpu_blockbpe")
            gpu_opt_k = fmt_kernel_ms(metrics_lat, "gpu_blockbpe")
            gpu_base = fmt_time_ms(metrics_lat, "gpu_blockbpe_baseline")
            gpu_base_k = fmt_kernel_ms(metrics_lat, "gpu_blockbpe_baseline")
            cpu = fmt_time_ms(metrics_lat, "cpu_blockbpe")
            tk = fmt_time_ms(metrics_lat, "tiktoken")
            hf = fmt_time_ms(metrics_lat, "huggingface")
            print(
                f"{length:12d}  "
                f"{gpu_opt:>12}  "
                f"{gpu_opt_k:>12}  "
                f"{gpu_base:>12}  "
                f"{gpu_base_k:>12}  "
                f"{cpu:>12}  "
                f"{tk:>12}  "
                f"{hf:>12}"
            )

        # Table 2: speedups of opt/base BlockBPE vs tiktoken and HF
        print("\n" + "=" * 80)
        print("Summary (LATENCY, single sequence): BlockBPE speedup vs tiktoken / HF")
        print("=" * 80)

        header_speed = (
            f"{'Len (HF tok)':>12}  "
            f"{'opt/TK':>8}  "
            f"{'opt/HF':>8}  "
            f"{'base/TK':>8}  "
            f"{'base/HF':>8}"
        )
        print(header_speed)
        print("-" * len(header_speed))

        for length in sorted(results.keys()):
            _, _, _, speedups_lat = results[length]
            opt_tk = fmt_speed(speedups_lat, "gpu_opt_vs_tiktoken")
            opt_hf = fmt_speed(speedups_lat, "gpu_opt_vs_hf")
            base_tk = fmt_speed(speedups_lat, "gpu_base_vs_tiktoken")
            base_hf = fmt_speed(speedups_lat, "gpu_base_vs_hf")
            print(
                f"{length:12d}  "
                f"{opt_tk:>8}  "
                f"{opt_hf:>8}  "
                f"{base_tk:>8}  "
                f"{base_hf:>8}"
            )

    print("\n" + "=" * 80)
    print("Scaling benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()



