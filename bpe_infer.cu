// Byte-level BPE tokenization for inference (CUDA)
// - Loads GPT-2 `vocab.json` and `merges.txt` from data/gpt2_tokenizer
// - Implements byte-level BPE merges on GPU without regex pre-tokenization
// - Provides CPU reference and GPU vs CPU correctness tests

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

using namespace std;

// ------------------------------
// Utilities
// ------------------------------

static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

// Convert a Unicode code point to UTF-8 string
static string codepoint_to_utf8(uint32_t cp) {
    string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// GPT-2 byte encoder mapping (bytes_to_unicode), returns 256 UTF-8 strings
static vector<string> build_gpt2_byte_encoder() {
    vector<int> bs;
    bs.reserve(256);
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n += 1;
        }
    }
    vector<string> encoder(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        encoder[bs[i]] = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
    }
    return encoder;
}

// Minimal JSON parser to read a flat object {"token": id, ...}
class FlatJsonVocab {
public:
    // token string (UTF-8) -> id
    unordered_map<string, int> token_to_id;

    bool load(const string &path) {
        ifstream in(path);
        if (!in.is_open()) {
            cerr << "Failed to open vocab.json: " << path << endl;
            return false;
        }
        string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        size_t i = 0;
        auto skip_ws = [&]() {
            while (i < s.size() && isspace(static_cast<unsigned char>(s[i]))) ++i;
        };
        auto parse_string = [&]() -> string {
            string out;
            if (i >= s.size() || s[i] != '"') return string();
            ++i; // skip opening quote
            while (i < s.size()) {
                char c = s[i++];
                if (c == '"') break;
                if (c == '\\') {
                    if (i >= s.size()) break;
                    char e = s[i++];
                    if (e == '"' || e == '\\' || e == '/') out.push_back(e);
                    else if (e == 'b') out.push_back('\b');
                    else if (e == 'f') out.push_back('\f');
                    else if (e == 'n') out.push_back('\n');
                    else if (e == 'r') out.push_back('\r');
                    else if (e == 't') out.push_back('\t');
                    else if (e == 'u') {
                        if (i + 4 <= s.size()) {
                            unsigned int cp = 0;
                            for (int k = 0; k < 4; ++k) {
                                char h = s[i++];
                                cp <<= 4;
                                if (h >= '0' && h <= '9') cp |= (h - '0');
                                else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            }
                            out += codepoint_to_utf8(cp);
                        }
                    }
                } else {
                    out.push_back(c);
                }
            }
            return out;
        };
        auto parse_int = [&]() -> long long {
            skip_ws();
            bool neg = false;
            if (i < s.size() && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; ++i; }
            long long val = 0;
            while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) {
                val = val * 10 + (s[i++] - '0');
            }
            return neg ? -val : val;
        };

        skip_ws();
        if (i >= s.size() || s[i] != '{') {
            cerr << "Invalid vocab.json: expected '{'" << endl;
            return false;
        }
        ++i;
        while (true) {
            skip_ws();
            if (i < s.size() && s[i] == '}') { ++i; break; }
            skip_ws();
            string key = parse_string();
            skip_ws();
            if (i >= s.size() || s[i] != ':') { cerr << "Invalid vocab.json: expected ':'" << endl; return false; }
            ++i;
            skip_ws();
            long long id = parse_int();
            token_to_id[key] = static_cast<int>(id);
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            skip_ws();
            if (i < s.size() && s[i] == '}') { ++i; break; }
        }
        return true;
    }
};

struct PairInfo { int rank; int new_token; };

// Host-side: build pair -> (rank,new_token) from merges.txt + vocab.json
static bool load_merges_and_vocab(
    const string &merges_path,
    const string &vocab_path,
    unordered_map<uint64_t, PairInfo> &pairinfo,
    vector<int> &byte_to_token_id
) {
    FlatJsonVocab vocab;
    if (!vocab.load(vocab_path)) return false;

    // Build GPT-2 byte encoder mapping and map each byte to its vocab id
    vector<string> byte_encoder = build_gpt2_byte_encoder();
    byte_to_token_id.assign(256, -1);
    for (int b = 0; b < 256; ++b) {
        auto it = vocab.token_to_id.find(byte_encoder[b]);
        if (it == vocab.token_to_id.end()) {
            cerr << "Byte encoder token not found in vocab for byte " << b << endl;
            return false;
        }
        byte_to_token_id[b] = it->second;
    }

    ifstream in(merges_path);
    if (!in.is_open()) {
        cerr << "Failed to open merges.txt: " << merges_path << endl;
        return false;
    }
    string line;
    int rank = 0;
    // Some files have a header line starting with "#"; skip such lines
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        // Split by space into two tokens
        size_t sp = line.find(' ');
        if (sp == string::npos) continue;
        string a = line.substr(0, sp);
        string b = line.substr(sp + 1);
        // Lookup token ids
        auto ia = vocab.token_to_id.find(a);
        auto ib = vocab.token_to_id.find(b);
        if (ia == vocab.token_to_id.end() || ib == vocab.token_to_id.end()) {
            cerr << "Merge tokens not in vocab: '" << a << "' '" << b << "'" << endl;
            return false;
        }
        // New token is concatenation
        string merged = a + b;
        auto im = vocab.token_to_id.find(merged);
        if (im == vocab.token_to_id.end()) {
            cerr << "Merged token not found in vocab: '" << merged << "'" << endl;
            return false;
        }
        uint64_t key = pack_pair(static_cast<uint32_t>(ia->second), static_cast<uint32_t>(ib->second));
        pairinfo[key] = PairInfo{rank, im->second};
        rank += 1;
    }
    return true;
}

// ------------------------------
// Device hash table for pair->(rank,new_token)
// Linear probing, power-of-two size, empty key = 0xFFFFFFFFFFFFFFFF
// ------------------------------

struct DevicePairMap {
    uint64_t *keys;   // size = capacity
    int *ranks;       // size = capacity
    int *new_tokens;  // size = capacity
    int capacity;
};

__device__ inline uint32_t hash64(uint64_t x) {
    // SplitMix64 inspired hashing, then fold to 32-bit
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<uint32_t>(x ^ (x >> 32));
}

__device__ inline int d_map_find(const DevicePairMap &m, uint64_t key, int &out_new_token) {
    const uint64_t empty = 0xFFFFFFFFFFFFFFFFULL;
    int mask = m.capacity - 1;
    int idx = hash64(key) & mask;
    for (int probe = 0; probe < m.capacity; ++probe) {
        uint64_t k = m.keys[idx];
        if (k == key) {
            out_new_token = m.new_tokens[idx];
            return m.ranks[idx];
        }
        if (k == empty) break; // not found
        idx = (idx + 1) & mask;
    }
    out_new_token = -1;
    return INT_MAX; // treat as no-merge
}

// ------------------------------
// GPU tokenization kernel (one block per sequence)
// ------------------------------

// For simplicity, we bound max length per sequence for shared mem path.
#ifndef BPE_MAX_TOKENS
#define BPE_MAX_TOKENS 8192
#endif

__global__ void bpe_tokenize_block_kernel(
    const int *seq_offsets,   // [B+1]
    const int *in_tokens,     // flattened initial byte tokens (ids)
    int *out_tokens,          // flattened output buffer (same max size as input)
    int *out_lengths,         // [B]
    DevicePairMap pairmap
) {
    extern __shared__ int smem[]; // dynamic shared memory
    int *tokens = smem;           // length up to (end-start)
    int *ranks = tokens + BPE_MAX_TOKENS; // ranks for adjacent pairs
    int *new_ids = ranks + BPE_MAX_TOKENS; // new token ids for pairs
    __shared__ int s_len;         // shared current length

    int b = blockIdx.x;
    int start = seq_offsets[b];
    int end = seq_offsets[b + 1];
    int initial_len = end - start;
    if (initial_len <= 0) {
        if (threadIdx.x == 0) out_lengths[b] = 0;
        return;
    }
    if (initial_len > BPE_MAX_TOKENS) {
        // Fallback: just copy (no merge) to avoid OOB; real implementation should chunk
        for (int i = threadIdx.x; i < initial_len; i += blockDim.x) {
            out_tokens[start + i] = in_tokens[start + i];
        }
        if (threadIdx.x == 0) out_lengths[b] = initial_len;
        return;
    }

    // Load to shared memory
    for (int i = threadIdx.x; i < initial_len; i += blockDim.x) {
        tokens[i] = in_tokens[start + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) s_len = initial_len;
    __syncthreads();

    // Iterative merge passes
    while (true) {
        // 1) compute ranks for all adjacent pairs
        int len = s_len;
        for (int i = threadIdx.x; i < len - 1; i += blockDim.x) {
            uint64_t key = (static_cast<uint64_t>(tokens[i]) << 32) | static_cast<uint64_t>(tokens[i + 1]);
            int nid;
            int r = d_map_find(pairmap, key, nid);
            ranks[i] = r;
            new_ids[i] = nid;
        }
        if (len - 1 <= 0) {
            // nothing to merge
            break;
        }
        __syncthreads();

        // 2) decide which positions to merge: find global min rank and merge all non-overlapping occurrences
        __shared__ int any_selected;
        if (threadIdx.x == 0) any_selected = 0;
        __syncthreads();

        len = s_len;
        if (threadIdx.x == 0) {
            int min_rank = INT_MAX;
            for (int i = 0; i < len - 1; ++i) {
                int ri = ranks[i];
                if (ri < min_rank) min_rank = ri;
            }
            if (min_rank == INT_MAX) {
                any_selected = 0;
            } else {
                int any = 0;
                int last_merged = -2;
                for (int i = 0; i < len - 1; ++i) {
                    int ri = ranks[i];
                    int selected = (ri == min_rank && i != last_merged + 1) ? 1 : 0;
                    if (selected) { last_merged = i; any = 1; }
                    // reuse ranks as selection mask (1 selected, 0 otherwise)
                    ranks[i] = selected;
                }
                any_selected = any;
            }
        }
        __syncthreads();
        if (any_selected == 0) {
            break; // no more merges possible
        }

        // 3) scatter to new sequence applying merges
        // simple two-phase: compute output length and then write; do it serially by one thread for simplicity
        if (threadIdx.x == 0) {
            int w = 0;
            int i = 0;
            int cur_len = s_len;
            while (i < cur_len) {
                if (i < cur_len - 1 && ranks[i] == 1) {
                    tokens[w++] = new_ids[i];
                    i += 2; // skip right partner
                } else {
                    tokens[w++] = tokens[i];
                    i += 1;
                }
            }
            s_len = w;
        }
        __syncthreads();
    }

    // Write back result
    __syncthreads();
    int final_len = s_len;
    for (int i = threadIdx.x; i < final_len; i += blockDim.x) {
        out_tokens[start + i] = tokens[i];
    }
    if (threadIdx.x == 0) out_lengths[b] = final_len;
}

// ------------------------------
// CPU reference implementation (greedy BPE merges)
// ------------------------------

static vector<int> bpe_cpu_greedy(
    const vector<int> &initial_ids,
    const unordered_map<uint64_t, PairInfo> &pairinfo
) {
    vector<int> ids = initial_ids;
    if (ids.size() <= 1) return ids;
    while (true) {
        size_t n = ids.size();
        vector<int> ranks(n - 1, INT_MAX);
        for (size_t i = 0; i + 1 < n; ++i) {
            auto it = pairinfo.find(pack_pair(static_cast<uint32_t>(ids[i]), static_cast<uint32_t>(ids[i + 1])));
            if (it != pairinfo.end()) ranks[i] = it->second.rank;
        }
        // select merges: merge all non-overlapping occurrences of the global min rank
        int min_rank = INT_MAX;
        for (size_t i = 0; i + 1 < n; ++i) if (ranks[i] < min_rank) min_rank = ranks[i];
        if (min_rank == INT_MAX) break;
        vector<char> select(n - 1, 0);
        bool any = false;
        size_t last_merged = static_cast<size_t>(-2);
        for (size_t i = 0; i + 1 < n; ++i) {
            if (ranks[i] == min_rank && i != last_merged + 1) {
                select[i] = 1; any = true; last_merged = i;
            }
        }
        if (!any) break;
        // apply merges
        vector<int> out;
        out.reserve(n);
        size_t i = 0;
        while (i < n) {
            if (i + 1 < n && select[i]) {
                auto it = pairinfo.find(pack_pair(static_cast<uint32_t>(ids[i]), static_cast<uint32_t>(ids[i + 1])));
                out.push_back(it->second.new_token);
                i += 2;
            } else {
                out.push_back(ids[i]);
                i += 1;
            }
        }
        ids.swap(out);
        if (ids.size() <= 1) break;
    }
    return ids;
}

// ------------------------------
// Host runner
// ------------------------------

static int next_pow2(int x) {
    int p = 1; while (p < x) p <<= 1; return p;
}

static inline uint32_t host_hash64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<uint32_t>(x ^ (x >> 32));
}

static bool build_device_map(const unordered_map<uint64_t, PairInfo> &pairinfo, DevicePairMap &m, vector<uint64_t> &h_keys, vector<int> &h_ranks, vector<int> &h_new) {
    int needed = static_cast<int>(pairinfo.size());
    int cap = next_pow2(max(1, needed * 2));
    h_keys.assign(cap, 0xFFFFFFFFFFFFFFFFULL);
    h_ranks.assign(cap, INT_MAX);
    h_new.assign(cap, -1);
    auto probe = [&](uint64_t key) {
        int mask = cap - 1;
        int idx = static_cast<int>(host_hash64(key)) & mask;
        while (h_keys[idx] != 0xFFFFFFFFFFFFFFFFULL) {
            idx = (idx + 1) & mask;
        }
        return idx;
    };
    for (auto &kv : pairinfo) {
        int idx = probe(kv.first);
        h_keys[idx] = kv.first;
        h_ranks[idx] = kv.second.rank;
        h_new[idx] = kv.second.new_token;
    }
    // allocate device arrays
    cudaMalloc(&m.keys, sizeof(uint64_t) * cap);
    cudaMalloc(&m.ranks, sizeof(int) * cap);
    cudaMalloc(&m.new_tokens, sizeof(int) * cap);
    cudaMemcpy(m.keys, h_keys.data(), sizeof(uint64_t) * cap, cudaMemcpyHostToDevice);
    cudaMemcpy(m.ranks, h_ranks.data(), sizeof(int) * cap, cudaMemcpyHostToDevice);
    cudaMemcpy(m.new_tokens, h_new.data(), sizeof(int) * cap, cudaMemcpyHostToDevice);
    m.capacity = cap;
    return true;
}

static void free_device_map(DevicePairMap &m) {
    cudaFree(m.keys); cudaFree(m.ranks); cudaFree(m.new_tokens);
    m.keys = nullptr; m.ranks = nullptr; m.new_tokens = nullptr; m.capacity = 0;
}

static vector<int> text_to_initial_token_ids(const string &text, const vector<string> &byte_encoder, const unordered_map<string,int> &tok2id) {
    vector<int> ids; ids.reserve(text.size());
    for (unsigned char c : text) {
        const string &sym = byte_encoder[c];
        auto it = tok2id.find(sym);
        if (it == tok2id.end()) {
            cerr << "Initial symbol not in vocab: code=" << (int)c << endl;
            continue;
        }
        ids.push_back(it->second);
    }
    return ids;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string base = string("/home/vk2636/gpu-tokenizer/data/gpt2_tokenizer");
    string vocab_path = base + "/vocab.json";
    string merges_path = base + "/merges.txt";
    if (argc >= 3) {
        merges_path = argv[1];
        vocab_path = argv[2];
    }

    unordered_map<uint64_t, PairInfo> pairinfo;
    vector<int> byte_to_token_id;
    if (!load_merges_and_vocab(merges_path, vocab_path, pairinfo, byte_to_token_id)) {
        return 1;
    }

    // Build encoder again for CPU initial ids
    vector<string> byte_encoder = build_gpt2_byte_encoder();

    // Load some inputs
    vector<string> inputs;
    {
        ifstream f("/home/vk2636/gpu-tokenizer/data/input/input.txt");
        if (f.is_open()) {
            string line; int cnt = 0;
            while (getline(f, line) && cnt < 64) { inputs.push_back(line); cnt++; }
        } else {
            inputs.push_back("Hello world!");
            inputs.push_back("The quick brown fox jumps over the lazy dog.");
        }
    }

    // Prepare host flattened initial ids and offsets
    FlatJsonVocab vocab; vocab.load(vocab_path);
    vector<int> h_offsets(inputs.size() + 1, 0);
    vector<int> h_init_ids;
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<int> ids = text_to_initial_token_ids(inputs[i], byte_encoder, vocab.token_to_id);
        h_offsets[i + 1] = static_cast<int>(h_offsets[i] + ids.size());
        h_init_ids.insert(h_init_ids.end(), ids.begin(), ids.end());
    }

    // CPU reference
    vector<vector<int>> cpu_tokens;
    cpu_tokens.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<int> ids(h_init_ids.begin() + h_offsets[i], h_init_ids.begin() + h_offsets[i + 1]);
        cpu_tokens.push_back(bpe_cpu_greedy(ids, pairinfo));
    }

    // Device map
    DevicePairMap dmap{};
    vector<uint64_t> h_keys; vector<int> h_ranks, h_new;
    build_device_map(pairinfo, dmap, h_keys, h_ranks, h_new);

    // Copy inputs to device
    int total_init = h_offsets.back();
    int B = static_cast<int>(inputs.size());
    int *d_offsets = nullptr; int *d_init = nullptr; int *d_out = nullptr; int *d_outlen = nullptr;
    cudaMalloc(&d_offsets, sizeof(int) * (B + 1));
    cudaMalloc(&d_init, sizeof(int) * total_init);
    cudaMalloc(&d_out, sizeof(int) * total_init);
    cudaMalloc(&d_outlen, sizeof(int) * B);
    cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * (B + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init, h_init_ids.data(), sizeof(int) * total_init, cudaMemcpyHostToDevice);

    // Launch kernel: one block per sequence
    dim3 grid(B);
    dim3 block(128);
    size_t smem_bytes = (size_t)(BPE_MAX_TOKENS * 3) * sizeof(int);
    auto t0 = chrono::high_resolution_clock::now();
    bpe_tokenize_block_kernel<<<grid, block, smem_bytes>>>(d_offsets, d_init, d_out, d_outlen, dmap);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();

    // Gather results
    vector<int> h_outlen(B);
    cudaMemcpy(h_outlen.data(), d_outlen, sizeof(int) * B, cudaMemcpyDeviceToHost);
    vector<int> h_out(total_init);
    cudaMemcpy(h_out.data(), d_out, sizeof(int) * total_init, cudaMemcpyDeviceToHost);

    // Compare with CPU
    bool all_ok = true;
    for (int i = 0, off = 0; i < B; ++i) {
        int L = h_outlen[i];
        vector<int> gpu_ids(h_out.begin() + h_offsets[i], h_out.begin() + h_offsets[i] + L);
        if (gpu_ids != cpu_tokens[i]) {
            all_ok = false;
            cerr << "Mismatch at sample " << i << ": CPU(" << cpu_tokens[i].size() << ") vs GPU(" << gpu_ids.size() << ")" << endl;
        }
        off += L;
    }

    auto ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0;
    cout << "GPU tokenization done in " << ms << " ms for " << B << " sequences" << endl;
    cout << (all_ok ? "GPU matches CPU" : "GPU differs from CPU") << endl;

    // Cleanup
    free_device_map(dmap);
    cudaFree(d_offsets); cudaFree(d_init); cudaFree(d_out); cudaFree(d_outlen);
    return all_ok ? 0 : 2;
}


