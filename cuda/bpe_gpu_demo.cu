#include <iostream>
#include <vector>
#include "bpe_kernels.h"

struct MergeRule {
    int first;
    int second;
    int result;
    const char* description;
};

static void print_tokens(const std::vector<int>& tokens) {
    std::cout << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < tokens.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

int main() {
    std::cout << "\n=== GPU BPE Merge Demo ===\n";

    // Toy vocabulary for the tokenized word "hello"
    // 1=h, 2=e, 3=l, 4=o
    std::vector<int> tokens = {1, 2, 3, 3, 4};

    std::vector<MergeRule> merges = {
        {3, 3, 5, "Merge 'l' + 'l' -> token 5"},
        {5, 4, 6, "Merge token 5 + 'o' -> token 6"},
        {1, 2, 7, "Merge 'h' + 'e' -> token 7"},
        {7, 6, 8, "Merge token 7 + token 6 -> token 8"}
    };

    std::vector<int> output(tokens.size());

    std::cout << "Initial tokens: ";
    print_tokens(tokens);
    std::cout << "\n";

    for (const auto& merge : merges) {
        const int old_size = static_cast<int>(tokens.size());

        int new_length = cuda_merge_pair(
            tokens.data(),
            old_size,
            merge.first,
            merge.second,
            merge.result,
            output.data()
        );

        if (new_length == old_size) {
            std::cout << "No occurrences for " << merge.description << ", skipping." << std::endl;
            continue;
        }

        output.resize(new_length);
        std::copy(output.begin(), output.end(), tokens.begin());
        tokens.resize(new_length);

        std::cout << merge.description << "\n  -> tokens: ";
        print_tokens(tokens);
        std::cout << "\n";

        output.resize(tokens.size());
    }

    std::cout << "\nFinal token sequence: ";
    print_tokens(tokens);
    std::cout << "\n\nDone!\n";

    return 0;
}
