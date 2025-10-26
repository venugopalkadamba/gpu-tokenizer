#!/usr/bin/env python3
"""
GPU Environment Validation Script
Tests CUDA availability, GPU memory, and basic compute capabilities
"""

import sys
import subprocess
import platform

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    print_section("Python Version")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

def check_nvidia_smi():
    print_section("NVIDIA GPU Check (nvidia-smi)")
    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              check=True)
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: nvidia-smi failed: {e}")
        return False

def check_nvcc():
    print_section("CUDA Compiler Check (nvcc)")
    try:
        result = subprocess.run(['nvcc', '--version'],
                              capture_output=True,
                              text=True,
                              check=True)
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: nvcc not found. You need to install CUDA toolkit.")
        print("Try: module load cuda  OR  install CUDA toolkit manually")
        return False

def check_pytorch():
    print_section("PyTorch CUDA Support")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
                print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

            # Simple GPU test
            print("\nRunning simple GPU computation test...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            print(f" GPU computation successful! Result shape: {y.shape}")
            return True
        else:
            print("WARNING: PyTorch installed but CUDA not available")
            return False
    except ImportError:
        print("PyTorch not installed. Install with:")
        print("  pip install torch torchvision torchaudio")
        return False

def check_cupy():
    print_section("CuPy Check (Python CUDA library)")
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

        # Simple test
        x = cp.array([1, 2, 3])
        print(f" CuPy working! Test array: {x}")
        return True
    except ImportError:
        print("CuPy not installed. Install with:")
        print("  pip install cupy-cuda11x  # for CUDA 11.x")
        return False
    except Exception as e:
        print(f"CuPy error: {e}")
        return False

def check_numba():
    print_section("Numba CUDA Check")
    try:
        from numba import cuda
        print(f"Numba CUDA available: {cuda.is_available()}")

        if cuda.is_available():
            gpus = cuda.gpus
            print(f"Number of GPUs detected: {len(gpus)}")
            for gpu in gpus:
                with gpu:
                    print(f"  {gpu.name.decode()}")
            return True
        else:
            print("WARNING: Numba installed but CUDA not detected")
            return False
    except ImportError:
        print("Numba not installed. Install with:")
        print("  pip install numba")
        return False

def check_huggingface_tokenizers():
    print_section("HuggingFace Tokenizers Check")
    try:
        import tokenizers
        print(f"Tokenizers version: {tokenizers.__version__}")

        from transformers import GPT2Tokenizer
        print(" Transformers library available")
        return True
    except ImportError:
        print("HuggingFace libraries not installed. Install with:")
        print("  pip install transformers tokenizers")
        return False

def print_recommendations():
    print_section("Recommendations for Your Project")
    print("""
Based on your GPU tokenization project, here's what you need:

CRITICAL (Must Have):
  1. CUDA Toolkit (nvcc) - for compiling CUDA code
  2. PyTorch with CUDA OR CuPy - for GPU programming in Python
  3. HuggingFace Tokenizers - for baseline comparison

RECOMMENDED:
  4. Numba - easier CUDA kernel prototyping
  5. CuPy - NumPy-like GPU arrays
  6. Jupyter - for experiments and visualization

NEXT STEPS:
  1. Install missing components above
  2. Implement matrix multiplication in CUDA (warmup exercise)
  3. Study BPE algorithm and implement CPU baseline
  4. Read BlockBPE paper carefully
  5. Start with simple parallel BPE on GPU

LEARNING RESOURCES:
  - CUDA Programming Guide: https://docs.nvidia.com/cuda/
  - Numba CUDA docs: https://numba.readthedocs.io/
  - BlockBPE paper: https://arxiv.org/abs/2507.11941
    """)

def main():
    print("""
    TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
    Q   GPU Tokenizer - Environment Validation Script           Q
    Q   CS Project: GPU-Accelerated BPE Tokenization            Q
    ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
    """)

    # Run all checks
    results = {
        'Python': check_python_version(),
        'nvidia-smi': check_nvidia_smi(),
        'nvcc': check_nvcc(),
        'PyTorch': check_pytorch(),
        'CuPy': check_cupy(),
        'Numba': check_numba(),
        'HuggingFace': check_huggingface_tokenizers(),
    }

    print_recommendations()

    # Summary
    print_section("Summary")
    for component, status in results.items():
        if status is None:
            continue
        status_str = " PASS" if status else " MISSING"
        print(f"{component:20s}: {status_str}")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()