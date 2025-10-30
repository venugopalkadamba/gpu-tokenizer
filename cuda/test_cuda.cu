/**
 * Simple C++ test for CUDA kernels
 * Compile: make test
 * Run: ./test_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include "bpe_kernels.h"

int main() {
    printf("=== BPE CUDA Kernels Test ===\n\n");

    // Test 1: GPU info
    printf("Test 1: GPU Information\n");
    printf("-----------------------\n");
    cuda_hello();

    // Test 2: Count pairs
    printf("\n\nTest 2: Pair Counting\n");
    printf("---------------------\n");

    // Example: tokens = [1, 2, 3, 2, 3]
    // Pairs: (1,2), (2,3), (3,2), (2,3)
    // Expected counts: (1,2)=1, (2,3)=2, (3,2)=1
    int tokens[] = {1, 2, 3, 2, 3};
    int length = 5;
    int vocab_size = 10;

    // Allocate pair counts array
    int* pair_counts = (int*)calloc(vocab_size * vocab_size, sizeof(int));

    printf("Input tokens: [");
    for (int i = 0; i < length; i++) {
        printf("%d%s", tokens[i], i < length-1 ? ", " : "");
    }
    printf("]\n");

    // Call CUDA function
    cuda_count_pairs(tokens, length, pair_counts, vocab_size);

    // Print results
    printf("\nPair counts:\n");
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            int count = pair_counts[i * vocab_size + j];
            if (count > 0) {
                printf("  (%d, %d): %d\n", i, j, count);
            }
        }
    }

    // Verify
    int expected_12 = 1;
    int expected_23 = 2;
    int expected_32 = 1;

    int actual_12 = pair_counts[1 * vocab_size + 2];
    int actual_23 = pair_counts[2 * vocab_size + 3];
    int actual_32 = pair_counts[3 * vocab_size + 2];

    printf("\nVerification:\n");
    printf("  (1,2): expected=%d, actual=%d %s\n",
           expected_12, actual_12, expected_12 == actual_12 ? "✓" : "✗");
    printf("  (2,3): expected=%d, actual=%d %s\n",
           expected_23, actual_23, expected_23 == actual_23 ? "✓" : "✗");
    printf("  (3,2): expected=%d, actual=%d %s\n",
           expected_32, actual_32, expected_32 == actual_32 ? "✓" : "✗");

    free(pair_counts);

    printf("\n=== Test Complete ===\n");

    return 0;
}
