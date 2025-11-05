/**
 * BPE Tokenization CUDA Kernels
 *
 * This is YOUR MAIN CUDA CODE for the assignment!
 *
 * Key kernels:
 * 1. count_pairs_kernel() - Count adjacent token pairs in parallel
 * 2. find_max_pair_kernel() - Find most frequent pair using reduction
 * 3. merge_pairs_kernel() - Merge all occurrences of a pair
 * 4. batch_tokenize_kernel() - Process multiple texts in parallel
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Parallel Pair Counting
// ============================================================================

/**
 * Count all adjacent pairs in a token sequence
 *
 * Strategy: Each thread checks one position and atomically increments count
 *
 * @param tokens        Input token sequence
 * @param length        Length of token sequence
 * @param pair_counts   Output: counts for each pair (vocab_size x vocab_size)
 * @param vocab_size    Size of vocabulary
 */
__global__ void count_pairs_kernel(
    const int* tokens,
    int length,
    int* pair_counts,
    int vocab_size
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one position
    if (idx < length - 1) {
        int token1 = tokens[idx];
        int token2 = tokens[idx + 1];

        // Bounds checking
        if (token1 >= 0 && token1 < vocab_size &&
            token2 >= 0 && token2 < vocab_size) {

            // Map 2D pair to 1D index
            int pair_idx = token1 * vocab_size + token2;

            // Atomically increment (multiple threads might access same pair)
            atomicAdd(&pair_counts[pair_idx], 1);
        }
    }
}

// ============================================================================
// KERNEL 2: Find Maximum Pair (Parallel Reduction)
// ============================================================================

/**
 * Find the pair with maximum count using parallel reduction
 *
 * This is a CLASSIC CUDA pattern - reduction!
 *
 * @param pair_counts   Input: counts for each pair
 * @param size          Number of pairs (vocab_size^2)
 * @param max_pair      Output: index of most frequent pair
 * @param max_count     Output: count of most frequent pair
 */
__global__ void find_max_pair_kernel(
    const int* pair_counts,
    int size,
    int* max_pair,
    int* max_count
) {
    // Shared memory for block-level reduction
    __shared__ int shared_counts[256];
    __shared__ int shared_indices[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        shared_counts[tid] = pair_counts[idx];
        shared_indices[tid] = idx;
    } else {
        shared_counts[tid] = -1;
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_counts[tid] < shared_counts[tid + stride]) {
                shared_counts[tid] = shared_counts[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // First thread writes block result
    if (tid == 0) {
        atomicMax(max_count, shared_counts[0]);
        // Note: This is simplified. Full implementation needs atomic compare-and-swap
        // for both max_count and max_pair together
        if (shared_counts[0] == *max_count) {
            *max_pair = shared_indices[0];
        }
    }
}

// ============================================================================
// KERNEL 3: Merge Pairs in Parallel
// ============================================================================

/**
 * Merge all occurrences of a specific pair
 *
 * This is HARDER - merge conflicts!
 * Example: "aaa" with merge rule ('a','a')
 *   - Position 0-1: 'a'+'a' → merge
 *   - Position 1-2: 'a'+'a' → merge  ❌ CONFLICT! Position 1 already used
 *
 * Strategy: Mark positions to merge, then compact
 *
 * @param tokens_in     Input token sequence
 * @param tokens_out    Output token sequence
 * @param valid_mask    Output: 1 if position produces output, 0 if consumed
 * @param length        Length of input
 * @param pair_first    First token of pair to merge
 * @param pair_second   Second token of pair to merge
 * @param merged_token  The merged result token
 */
__global__ void merge_pairs_kernel(
    const int* tokens_in,
    int* tokens_tmp,
    int* valid_mask,
    int* claimed_flags,
    int length,
    int pair_first,
    int pair_second,
    int merged_token
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }

    bool merged_here = false;
    if (idx < length - 1 &&
        tokens_in[idx] == pair_first &&
        tokens_in[idx + 1] == pair_second) {

        if (atomicCAS(&claimed_flags[idx], 0, 1) == 0) {
            if (atomicCAS(&claimed_flags[idx + 1], 0, 1) == 0) {
                merged_here = true;
            } else {
                atomicExch(&claimed_flags[idx], 0);
            }
        }
    }

    if (merged_here) {
        tokens_tmp[idx] = merged_token;
        valid_mask[idx] = 1;
    } else {
        if (claimed_flags[idx] == 0) {
            tokens_tmp[idx] = tokens_in[idx];
            valid_mask[idx] = 1;
        } else {
            valid_mask[idx] = 0;
        }
    }
}

// ============================================================================
// KERNEL 4: Stream Compaction (Remove Consumed Tokens)
// ============================================================================

/**
 * Compact array by removing invalid positions
 * Uses parallel prefix sum (scan)
 *
 * @param tokens_in    Input tokens (sparse, has gaps)
 * @param tokens_out   Output tokens (compacted)
 * @param valid_mask   1 if position is valid, 0 if should be removed
 * @param prefix_sum   Cumulative sum of valid_mask (output positions)
 * @param length       Input length
 */
__global__ void compact_tokens_kernel(
    const int* tokens_in,
    int* tokens_out,
    const int* valid_mask,
    const int* prefix_sum,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length && valid_mask[idx] == 1) {
        // This position is valid - write to compacted position
        int out_pos = prefix_sum[idx];
        tokens_out[out_pos] = tokens_in[idx];
    }
}

// ============================================================================
// KERNEL 5: Batch Processing (Process Multiple Texts in Parallel)
// ============================================================================

/**
 * Count pairs across multiple texts in parallel
 * Each block processes one text
 *
 * @param batch_tokens    All token sequences concatenated
 * @param batch_offsets   Starting position of each text
 * @param batch_lengths   Length of each text
 * @param num_texts       Number of texts in batch
 * @param pair_counts     Output counts (shared across batch)
 * @param vocab_size      Vocabulary size
 */
__global__ void batch_count_pairs_kernel(
    const int* batch_tokens,
    const int* batch_offsets,
    const int* batch_lengths,
    int num_texts,
    int* pair_counts,
    int vocab_size
) {
    // Each block processes one text
    int text_idx = blockIdx.x;

    if (text_idx >= num_texts) return;

    int offset = batch_offsets[text_idx];
    int length = batch_lengths[text_idx];

    // Each thread in block processes positions in this text
    for (int i = threadIdx.x; i < length - 1; i += blockDim.x) {
        int token1 = batch_tokens[offset + i];
        int token2 = batch_tokens[offset + i + 1];

        if (token1 >= 0 && token1 < vocab_size &&
            token2 >= 0 && token2 < vocab_size) {
            int pair_idx = token1 * vocab_size + token2;
            atomicAdd(&pair_counts[pair_idx], 1);
        }
    }
}

// ============================================================================
// HOST FUNCTIONS (Called from Python via ctypes)
// ============================================================================

extern "C" {

/**
 * Count pairs - Host wrapper
 */
void cuda_count_pairs(
    const int* h_tokens,
    int length,
    int* h_pair_counts,
    int vocab_size
) {
    // Allocate device memory
    int* d_tokens;
    int* d_pair_counts;

    size_t tokens_size = length * sizeof(int);
    size_t pairs_size = vocab_size * vocab_size * sizeof(int);

    cudaMalloc(&d_tokens, tokens_size);
    cudaMalloc(&d_pair_counts, pairs_size);

    // Copy input to device
    cudaMemcpy(d_tokens, h_tokens, tokens_size, cudaMemcpyHostToDevice);
    cudaMemset(d_pair_counts, 0, pairs_size);

    // Launch kernel
    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    count_pairs_kernel<<<blocks, threads>>>(
        d_tokens, length, d_pair_counts, vocab_size
    );

    // Copy result back
    cudaMemcpy(h_pair_counts, d_pair_counts, pairs_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_pair_counts);
}

int cuda_merge_pair(
    const int* h_tokens,
    int length,
    int pair_first,
    int pair_second,
    int merged_token,
    int* h_out_tokens
) {
    if (length <= 0) {
        return 0;
    }

    const int threads = 256;
    const int blocks = (length + threads - 1) / threads;

    size_t bytes = sizeof(int) * length;

    int* d_tokens_in = nullptr;
    int* d_tokens_tmp = nullptr;
    int* d_tokens_out = nullptr;
    int* d_valid_mask = nullptr;
    int* d_prefix = nullptr;
    int* d_claimed = nullptr;

    cudaMalloc(&d_tokens_in, bytes);
    cudaMalloc(&d_tokens_tmp, bytes);
    cudaMalloc(&d_tokens_out, bytes);
    cudaMalloc(&d_valid_mask, bytes);
    cudaMalloc(&d_prefix, bytes);
    cudaMalloc(&d_claimed, bytes);

    cudaMemcpy(d_tokens_in, h_tokens, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_tokens_tmp, 0, bytes);
    cudaMemset(d_valid_mask, 0, bytes);
    cudaMemset(d_prefix, 0, bytes);
    cudaMemset(d_claimed, 0, bytes);

    merge_pairs_kernel<<<blocks, threads>>>(
        d_tokens_in,
        d_tokens_tmp,
        d_valid_mask,
        d_claimed,
        length,
        pair_first,
        pair_second,
        merged_token
    );

    // Build exclusive offsets for compaction.
    thrust::device_ptr<int> valid_ptr(d_valid_mask);
    thrust::device_ptr<int> prefix_ptr(d_prefix);
    thrust::exclusive_scan(valid_ptr, valid_ptr + length, prefix_ptr);

    int new_length = thrust::reduce(valid_ptr, valid_ptr + length);

    compact_tokens_kernel<<<blocks, threads>>>(
        d_tokens_tmp,
        d_tokens_out,
        d_valid_mask,
        d_prefix,
        length
    );

    cudaDeviceSynchronize();

    if (new_length > 0) {
        cudaMemcpy(h_out_tokens, d_tokens_out, sizeof(int) * new_length, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_tokens_in);
    cudaFree(d_tokens_tmp);
    cudaFree(d_tokens_out);
    cudaFree(d_valid_mask);
    cudaFree(d_prefix);
    cudaFree(d_claimed);

    return new_length;
}

/**
 * Simple test function to verify CUDA is working
 */
void cuda_hello() {
    printf("CUDA Hello from GPU tokenizer!\n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
}

} // extern "C"
