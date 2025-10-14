# cuCollections (CUCO) Setup Guide

## Installation Summary

The cuCollections library has been installed in the `externals` directory:
- **cuCollections**: `./externals/cuCollections/`
- **CCCL (dependency)**: `./externals/cccl/`

Both are header-only libraries, so no compilation is needed for the library itself.

## Required Include Paths

When compiling any CUDA program that uses cuCollections, you need to include these paths:

```bash
-I./externals/cuCollections/include
-I./externals/cccl/libcudacxx/include
-I./externals/cccl/thrust
-I./externals/cccl/cub
```

## Compilation Examples

### Method 1: Using Make (Recommended ⭐)

The easiest way to compile is using the provided Makefile:

```bash
# Compile test_cuco
make

# Compile and run test
make test

# Clean build artifacts
make clean

# Compile any .cu file
make your_program.out SOURCE=your_program.cu
```

### Method 2: Using Environment Variables

Source the config file and use the environment variables:

```bash
source cuco_config.sh
nvcc $CUCO_FLAGS your_program.cu -o your_program
```

### Method 3: Using nvcc Directly

```bash
nvcc -std=c++17 \
    --expt-extended-lambda \
    -I./externals/cuCollections/include \
    -I./externals/cccl/libcudacxx/include \
    -I./externals/cccl/thrust \
    -I./externals/cccl/cub \
    your_program.cu -o your_program
```

**Note:** The `--expt-extended-lambda` flag is **required** by cuCollections.

## Running the Test

After compilation:

```bash
./test_cuco
```

Expected output:
- Should show successful insertion of 50,000 keys
- Query results showing which keys are present/absent in the set

## Using cuCollections in Your Code

### Example: Static Set

```cpp
#include <cuco/static_set.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Create a set with capacity for 100,000 elements
// -1 is the empty_key sentinel (must not appear in your data)
cuco::static_set<int> set{
    100'000,                    // capacity
    cuco::empty_key<int>{-1}    // sentinel value for empty slots
};

// Insert keys
thrust::device_vector<int> keys = {...};
set.insert(keys.begin(), keys.end());

// Check membership
thrust::device_vector<bool> results(keys.size());
set.contains(keys.begin(), keys.end(), results.begin());
```

**Important:** You must specify an `empty_key` sentinel value that won't appear in your actual data.

### Example: Static Map

```cpp
#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>

// Create a map with capacity for 100,000 key-value pairs
cuco::static_map<int, int> map{100'000};

// Insert key-value pairs
thrust::device_vector<cuco::pair<int, int>> pairs = {...};
map.insert(pairs.begin(), pairs.end());

// Find values
thrust::device_vector<int> values(keys.size());
map.find(keys.begin(), keys.end(), values.begin());
```

## Available Data Structures

- `cuco::static_set` - Fixed-size hash set
- `cuco::static_map` - Fixed-size hash map
- `cuco::static_multiset` - Fixed-size hash multiset (allows duplicates)
- `cuco::static_multimap` - Fixed-size hash multimap
- `cuco::dynamic_map` - Growable hash map
- `cuco::bloom_filter` - Bloom filter for approximate membership
- `cuco::hyperloglog` - Cardinality estimation

## System Requirements

- CUDA 13.0+ (you have 13.0.88 ✓)
- C++17 or newer
- GPU Architecture: Volta or newer (RTX 4070 ✓)
- CMake 3.30.4+ (only needed if building with CMake)

## References

- [cuCollections GitHub](https://github.com/NVIDIA/cuCollections/)
- [Documentation](https://nvidia.github.io/cuCollections/)

