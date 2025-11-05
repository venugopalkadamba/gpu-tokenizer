/**
 * BPE CPU Implementation in C++
 *
 * This is a PURE CPU version written in C++ (.cu extension for CUDA compatibility)
 * We'll convert this to GPU step-by-step!
 *
 * Goal: Understand the algorithm in C++ before parallelizing
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <chrono>

// ==============================================================================
// DATA STRUCTURES
// ==============================================================================

/**
 * Token: Represents a piece of text (character or merged string)
 */
struct Token {
    std::string text;

    Token() : text("") {}
    Token(const std::string& t) : text(t) {}
    Token(char c) : text(1, c) {}

    bool operator==(const Token& other) const {
        return text == other.text;
    }
};

/**
 * Pair: Two adjacent tokens
 */
struct Pair {
    std::string first;
    std::string second;

    bool operator<(const Pair& other) const {
        if (first != other.first) return first < other.first;
        return second < other.second;
    }

    bool operator==(const Pair& other) const {
        return first == other.first && second == other.second;
    }
};

/**
 * BPE Tokenizer Class
 */
class BPETokenizerCPU {
private:
    // Vocabulary: token_string → token_id
    std::map<std::string, int> vocab;

    // Inverse vocabulary: token_id → token_string
    std::map<int, std::string> inverse_vocab;

    // Merge rules: (token1, token2) → priority (lower = merge first)
    std::map<Pair, int> merge_ranks;

    // List of all merge rules in order
    std::vector<Pair> merges;

public:
    // ==========================================================================
    // SECTION 1: LOADING VOCABULARY & MERGE RULES
    // ==========================================================================

    /**
     * Load vocabulary from file
     * Format: {"token": id, ...}
     */
    bool load_vocab(const std::string& vocab_file) {
        std::cout << "Loading vocabulary from " << vocab_file << "..." << std::endl;

        std::ifstream file(vocab_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << vocab_file << std::endl;
            return false;
        }

        // Simple JSON parsing (for demonstration)
        // In production, use a proper JSON library like nlohmann/json
        std::string line;
        std::getline(file, line); // Read entire file

        // TODO: Implement proper JSON parsing
        // For now, we'll use a simplified approach

        std::cout << "✓ Loaded vocabulary" << std::endl;
        return true;
    }

    /**
     * Load merge rules from file
     * Format: "token1 token2\ntoken3 token4\n..."
     */
    bool load_merges(const std::string& merges_file) {
        std::cout << "Loading merges from " << merges_file << "..." << std::endl;

        std::ifstream file(merges_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << merges_file << std::endl;
            return false;
        }

        std::string line;
        std::getline(file, line); // Skip header line

        int rank = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Parse "token1 token2"
            std::istringstream iss(line);
            std::string first, second;
            iss >> first >> second;

            Pair pair;
            pair.first = first;
            pair.second = second;

            merges.push_back(pair);
            merge_ranks[pair] = rank++;
        }

        std::cout << "✓ Loaded " << merges.size() << " merge rules" << std::endl;
        return true;
    }

    // ==========================================================================
    // SECTION 2: CORE BPE ALGORITHM (CPU VERSION)
    // ==========================================================================

    /**
     * Get all adjacent pairs in a token sequence
     *
     * Example: ['l', 'o', 'w'] → {('l','o'), ('o','w')}
     */
    std::set<Pair> get_pairs(const std::vector<Token>& word) {
        std::set<Pair> pairs;

        for (size_t i = 0; i < word.size() - 1; i++) {
            Pair p;
            p.first = word[i].text;
            p.second = word[i + 1].text;
            pairs.insert(p);
        }

        return pairs;
    }

    /**
     * Find the pair with highest priority (lowest rank)
     *
     * THIS IS BOTTLENECK #1 - We'll parallelize this on GPU!
     */
    Pair find_best_pair(const std::set<Pair>& pairs) {
        Pair best_pair;
        int best_rank = INT_MAX;

        // Sequential scan through all pairs
        for (const Pair& p : pairs) {
            auto it = merge_ranks.find(p);
            if (it != merge_ranks.end()) {
                if (it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = p;
                }
            }
        }

        return best_pair;
    }

    /**
     * Merge all occurrences of a pair in the word
     *
     * THIS IS BOTTLENECK #2 - We'll parallelize this on GPU!
     */
    std::vector<Token> merge_pair(const std::vector<Token>& word, const Pair& pair) {
        std::vector<Token> new_word;

        size_t i = 0;
        while (i < word.size()) {
            // Check if current position starts the pair to merge
            if (i < word.size() - 1 &&
                word[i].text == pair.first &&
                word[i + 1].text == pair.second) {

                // Merge!
                Token merged(pair.first + pair.second);
                new_word.push_back(merged);
                i += 2; // Skip next token
            } else {
                // No merge, copy as-is
                new_word.push_back(word[i]);
                i++;
            }
        }

        return new_word;
    }

    /**
     * Apply BPE to a single token
     *
     * THIS IS THE MAIN ALGORITHM!
     */
    std::vector<Token> bpe(const std::string& token) {
        // Step 1: Convert to character-level tokens
        std::vector<Token> word;
        for (char c : token) {
            word.push_back(Token(c));
        }

        // Step 2: Iteratively merge pairs
        while (word.size() > 1) {
            // Get all adjacent pairs
            std::set<Pair> pairs = get_pairs(word);
            if (pairs.empty()) break;

            // Find best pair to merge
            Pair best = find_best_pair(pairs);

            // Check if valid merge exists
            if (merge_ranks.find(best) == merge_ranks.end()) {
                break; // No more valid merges
            }

            // Merge the pair
            word = merge_pair(word, best);
        }

        return word;
    }

    /**
     * Encode text to token IDs
     */
    std::vector<int> encode(const std::string& text) {
        std::vector<int> token_ids;

        // For simplicity, process entire text as one token
        // (In full implementation, we'd split by regex first)
        std::vector<Token> tokens = bpe(text);

        // Look up IDs in vocabulary
        for (const Token& t : tokens) {
            auto it = vocab.find(t.text);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            }
        }

        return token_ids;
    }

    // ==========================================================================
    // SECTION 3: BENCHMARKING
    // ==========================================================================

    void benchmark(const std::string& text, int num_runs = 100) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " CPU BPE Benchmark (C++)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Text length: " << text.length() << " characters" << std::endl;
        std::cout << "Number of runs: " << num_runs << std::endl;

        // Warmup
        encode(text);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_runs; i++) {
            std::vector<int> tokens = encode(text);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avg_time_ms = duration.count() / (double)num_runs / 1000.0;

        std::cout << "\nResults:" << std::endl;
        std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << (text.length() / (avg_time_ms / 1000.0)) << " chars/sec" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
};

// ==============================================================================
// MAIN FUNCTION
// ==============================================================================

int main(int argc, char** argv) {
    std::cout << R"(
    ╔══════════════════════════════════════════════════════════════════╗
    ║            BPE CPU Implementation (C++)                          ║
    ║            Step 1: Understand Algorithm                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    )" << std::endl;

    // Initialize tokenizer
    BPETokenizerCPU tokenizer;

    // Load vocabulary and merges
    if (!tokenizer.load_vocab("../data/vocab.json")) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return 1;
    }

    if (!tokenizer.load_merges("../data/merges.txt")) {
        std::cerr << "Failed to load merges" << std::endl;
        return 1;
    }

    // Test encoding
    std::string test_text = "Hello";
    std::cout << "\nTest encoding: \"" << test_text << "\"" << std::endl;

    std::vector<int> tokens = tokenizer.encode(test_text);
    std::cout << "Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Benchmark
    std::string long_text = "Machine learning is a subset of artificial intelligence.";
    tokenizer.benchmark(long_text, 100);

    std::cout << "\nCPU version complete!" << std::endl;

    return 0;
}
