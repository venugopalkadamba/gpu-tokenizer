#!/bin/bash
# cuCollections configuration file
# Source this file to get CUCO compilation flags in your environment
# Usage: source cuco_config.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export cuCollections include paths
export CUCO_INCLUDES="-I${SCRIPT_DIR}/externals/cuCollections/include -I${SCRIPT_DIR}/externals/cccl/libcudacxx/include -I${SCRIPT_DIR}/externals/cccl/thrust -I${SCRIPT_DIR}/externals/cccl/cub"

# Export NVCC flags needed for cuCollections
export CUCO_NVCCFLAGS="-std=c++17 --expt-extended-lambda"

# Combined flags
export CUCO_FLAGS="${CUCO_NVCCFLAGS} ${CUCO_INCLUDES}"

echo "cuCollections environment configured!"
echo "Use: nvcc \$CUCO_FLAGS your_program.cu -o your_program"

