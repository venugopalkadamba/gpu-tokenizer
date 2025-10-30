/**
 * BPE CUDA Kernels - Header File
 */

#ifndef BPE_KERNELS_H
#define BPE_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Count pairs in a token sequence on GPU
 *
 * @param h_tokens       Host array of tokens
 * @param length         Number of tokens
 * @param h_pair_counts  Host array for output counts (vocab_size^2)
 * @param vocab_size     Size of vocabulary
 */
void cuda_count_pairs(
    const int* h_tokens,
    int length,
    int* h_pair_counts,
    int vocab_size
);

/**
 * Test function - prints GPU info
 */
void cuda_hello();

#ifdef __cplusplus
}
#endif

#endif // BPE_KERNELS_H
