# Makefile for GPU projects using cuCollections
# Usage: make <target>
# Note: Ensure CUDA module is loaded before running make

# Compiler - try to find nvcc
NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Compiler flags
CXXFLAGS := -std=c++17
NVCCFLAGS := --expt-extended-lambda

# Include paths for external libraries
CUCO_INCLUDE := -I./externals/cuCollections/include
CCCL_INCLUDE := -I./externals/cccl/libcudacxx/include \
                -I./externals/cccl/thrust \
                -I./externals/cccl/cub

# Combined include paths
INCLUDES := $(CUCO_INCLUDE) $(CCCL_INCLUDE)

# All flags combined
ALL_FLAGS := $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDES)

# Targets
.PHONY: all clean test help check-nvcc

# Check if nvcc is available
check-nvcc:
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found. Please load CUDA module first: module load cuda/13.0" && exit 1)

# Default target
all: check-nvcc test_cuco

# Build the test program
test_cuco: test_cuco.cu
	@echo "Compiling test_cuco.cu..."
	$(NVCC) $(ALL_FLAGS) $< -o $@
	@echo "Build successful! Run with: ./test_cuco"

# Run the test
test: test_cuco
	@echo "Running test..."
	./test_cuco

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f test_cuco *.o

# Help message
help:
	@echo "Available targets:"
	@echo "  make              - Build test_cuco (default)"
	@echo "  make test_cuco    - Build test_cuco"
	@echo "  make test         - Build and run test_cuco"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "To compile your own .cu file:"
	@echo "  make your_program.out SOURCE=your_program.cu"

# Generic rule for compiling any .cu file
%.out: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(ALL_FLAGS) $< -o $@
	@echo "Build successful! Run with: ./$@"

# Allow custom source compilation
ifdef SOURCE
custom: $(SOURCE)
	@echo "Compiling $(SOURCE)..."
	$(NVCC) $(ALL_FLAGS) $(SOURCE) -o $(basename $(SOURCE))
	@echo "Build successful! Run with: ./$(basename $(SOURCE))"
endif

