# gpu-tokenizer

GPU-accelerated tokenizer using cuCollections for efficient hash-based operations.

## Setup

### Install cuCollections

Clone the required libraries into the `externals` folder:

```bash
mkdir -p externals
cd externals

# Clone cuCollections
git clone https://github.com/NVIDIA/cuCollections.git

# Clone CCCL (dependency)
git clone https://github.com/NVIDIA/cccl.git

cd ..
```

## Compilation

### Method 1: Using Make (Recommended)

```bash
make                    # Build test_cuco
make test              # Build and run test
make clean             # Clean build artifacts
```

### Method 2: Using Environment Variables

```bash
source cuco_config.sh
nvcc $CUCO_FLAGS your_program.cu -o your_program
```

### Method 3: Manual nvcc Command

```bash
nvcc -std=c++17 \
    --expt-extended-lambda \
    -I./externals/cuCollections/include \
    -I./externals/cccl/libcudacxx/include \
    -I./externals/cccl/thrust \
    -I./externals/cccl/cub \
    your_program.cu -o your_program
```

**Note:** Ensure CUDA module is loaded first: `module load cuda/13.0`

See `CUCO_SETUP.md` for detailed API documentation.