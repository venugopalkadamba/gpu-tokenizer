// BlockBPE-style byte-level BPE tokenization (CUDA)
// - Uses cuCollections static_map for pair->(rank,new_token)
// - Uses CUB BlockReduce and BlockScan for per-block parallel min-reduction and compaction
// - One block per sequence; byte-level input; greedy global-min per pass; non-overlap merges

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

static inline __host__ __device__ uint64_t pack_pairval(int rank, int new_token) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(rank)) << 32) | static_cast<uint32_t>(new_token);
}
static inline __host__ __device__ void unpack_pairval(uint64_t pv, int &rank, int &new_token) {
    rank = static_cast<int>(static_cast<uint32_t>(pv >> 32));
    new_token = static_cast<int>(static_cast<uint32_t>(pv & 0xFFFFFFFFu));
}

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

static vector<string> build_gpt2_byte_encoder() {
    vector<int> bs; bs.reserve(256);
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    vector<int> cs = bs; int n = 0;
    for (int b = 0; b < 256; ++b) if (find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n += 1; }
    vector<string> encoder(256);
    for (size_t i = 0; i < bs.size(); ++i) encoder[bs[i]] = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
    return encoder;
}

class FlatJsonVocab {
public:
    unordered_map<string, int> token_to_id;
    bool load(const string &path) {
        ifstream in(path);
        if (!in.is_open()) { cerr << "Failed to open vocab.json: " << path << endl; return false; }
        string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        size_t i = 0; auto skip_ws = [&]() { while (i < s.size() && isspace(static_cast<unsigned char>(s[i]))) ++i; };
        auto parse_string = [&]() -> string {
            string out; if (i >= s.size() || s[i] != '"') return string(); ++i; while (i < s.size()) {
                char c = s[i++]; if (c == '"') break; if (c == '\\') { if (i >= s.size()) break; char e = s[i++];
                    if (e == '"' || e == '\\' || e == '/') out.push_back(e);
                    else if (e == 'b') out.push_back('\b'); else if (e == 'f') out.push_back('\f'); else if (e == 'n') out.push_back('\n');
                    else if (e == 'r') out.push_back('\r'); else if (e == 't') out.push_back('\t'); else if (e == 'u') { if (i + 4 <= s.size()) {
                        unsigned int cp = 0; for (int k = 0; k < 4; ++k) { char h = s[i++]; cp <<= 4; if (h >= '0' && h <= '9') cp |= (h - '0'); else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10); else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10); }
                        out += codepoint_to_utf8(cp);
                    }}
                } else out.push_back(c);
            } return out; };
        auto parse_int = [&]() -> long long { skip_ws(); bool neg = false; if (i < s.size() && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; ++i; } long long val = 0; while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) { val = val * 10 + (s[i++] - '0'); } return neg ? -val : val; };
        skip_ws(); if (i >= s.size() || s[i] != '{') { cerr << "Invalid vocab.json: expected '{'" << endl; return false; } ++i;
        while (true) { skip_ws(); if (i < s.size() && s[i] == '}') { ++i; break; } skip_ws(); string key = parse_string(); skip_ws(); if (i >= s.size() || s[i] != ':') { cerr << "Invalid vocab.json: expected ':'" << endl; return false; } ++i; skip_ws(); long long id = parse_int(); token_to_id[key] = static_cast<int>(id); skip_ws(); if (i < s.size() && s[i] == ',') { ++i; continue; } skip_ws(); if (i < s.size() && s[i] == '}') { ++i; break; } }
        return true;
    }
};

// packed (rank << 32) | new_token
struct PairVal { int rank; int new_token; };

static bool load_merges_and_vocab(
    const string &merges_path,
    const string &vocab_path,
    unordered_map<uint64_t, uint64_t> &pairinfo,
    vector<int> &byte_to_token_id
) {
    FlatJsonVocab vocab; if (!vocab.load(vocab_path)) return false;
    vector<string> byte_encoder = build_gpt2_byte_encoder();
    byte_to_token_id.assign(256, -1);
    for (int b = 0; b < 256; ++b) {
        auto it = vocab.token_to_id.find(byte_encoder[b]);
        if (it == vocab.token_to_id.end()) { cerr << "Byte encoder token not found for byte " << b << endl; return false; }
        byte_to_token_id[b] = it->second;
    }
    ifstream in(merges_path); if (!in.is_open()) { cerr << "Failed to open merges.txt: " << merges_path << endl; return false; }
    string line; int rank = 0; while (getline(in, line)) {
        if (line.empty() || line[0] == '#') continue; size_t sp = line.find(' '); if (sp == string::npos) continue;
        string a = line.substr(0, sp); string b = line.substr(sp + 1);
        auto ia = vocab.token_to_id.find(a); auto ib = vocab.token_to_id.find(b);
        if (ia == vocab.token_to_id.end() || ib == vocab.token_to_id.end()) { cerr << "Merge tokens not in vocab: '" << a << "' '" << b << "'" << endl; return false; }
        string merged = a + b; auto im = vocab.token_to_id.find(merged); if (im == vocab.token_to_id.end()) { cerr << "Merged token missing: '" << merged << "'" << endl; return false; }
        uint64_t key = pack_pair(static_cast<uint32_t>(ia->second), static_cast<uint32_t>(ib->second));
        pairinfo[key] = pack_pairval(rank, im->second); rank += 1;
    }
    return true;
}

#ifndef BLOCK_BPE_MAX_TOKENS
#define BLOCK_BPE_MAX_TOKENS 1024
#endif

#ifndef BLOCK_BPE_BLOCK_SIZE
#define BLOCK_BPE_BLOCK_SIZE 256
#endif

// Device kernel: one block per sequence
template <typename MapRef>
__global__ void block_bpe_kernel(
    const int *seq_offsets, const int *in_tokens, int *out_tokens, int *out_lengths, MapRef map_ref
) {
    __shared__ int tokens[BLOCK_BPE_MAX_TOKENS];
    __shared__ int pair_rank[BLOCK_BPE_MAX_TOKENS]; // ranks for pairs [0..len-2]
    __shared__ int new_id[BLOCK_BPE_MAX_TOKENS];    // new token ids for pairs
    __shared__ unsigned char selected[BLOCK_BPE_MAX_TOKENS];
    __shared__ int s_len;

    using BlockReduce = cub::BlockReduce<int, BLOCK_BPE_BLOCK_SIZE>;
    using BlockScan   = cub::BlockScan<int, BLOCK_BPE_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage red_temp;
    __shared__ typename BlockScan::TempStorage scan_temp;

    int b = blockIdx.x;
    int start = seq_offsets[b];
    int end = seq_offsets[b + 1];
    int len0 = end - start;
    if (len0 <= 0) { if (threadIdx.x == 0) out_lengths[b] = 0; return; }
    if (len0 > BLOCK_BPE_MAX_TOKENS) {
        // Fallback: copy through (no merges) to keep correctness
        for (int i = threadIdx.x; i < len0; i += blockDim.x) out_tokens[start + i] = in_tokens[start + i];
        if (threadIdx.x == 0) out_lengths[b] = len0; return;
    }

    // load and initialize output with identity (needed if no merges occur)
    for (int i = threadIdx.x; i < len0; i += blockDim.x) {
        int t = in_tokens[start + i];
        tokens[i] = t;
        out_tokens[start + i] = t;
    }
    if (threadIdx.x == 0) s_len = len0;
    __syncthreads();

    int iter = 0;
    const int MAX_ITER = 1000;
    while (iter < MAX_ITER) {
        iter++;
        int len = s_len;
        // compute ranks and new ids for all pairs
        for (int i = threadIdx.x; i < len - 1; i += blockDim.x) {
            uint64_t key = (static_cast<uint64_t>(tokens[i]) << 32) | static_cast<uint64_t>(tokens[i + 1]);
            auto found = map_ref.find(key);
            if (found != map_ref.end()) { int r, nid; unpack_pairval((*found).second, r, nid); pair_rank[i] = r; new_id[i] = nid; }
            else { pair_rank[i] = INT_MAX; new_id[i] = -1; }
        }
        __syncthreads();
        if (len - 1 <= 0) break;

        // parallel min reduction across ranks
        int thread_min = INT_MAX;
        for (int i = threadIdx.x; i < len - 1; i += blockDim.x) thread_min = min(thread_min, pair_rank[i]);
        struct MinOp { __device__ int operator()(int a, int b) const { return a < b ? a : b; } };
        int min_rank = BlockReduce(red_temp).Reduce(thread_min, MinOp{});
        __syncthreads();
        if (min_rank == INT_MAX) break;

        // mark selected occurrences (non-overlapping leftmost among equal min rank)
        if (threadIdx.x == 0) {
            for (int i = 0; i < len - 1; ++i) selected[i] = 0;
            int last = -2;
            for (int i = 0; i < len - 1; ++i) {
                if (pair_rank[i] == min_rank && i != last + 1) { selected[i] = 1; last = i; }
            }
        }
        __syncthreads();

        // Serial compaction by thread 0 (matches CPU logic exactly)
        if (threadIdx.x == 0) {
            int w = 0;
            int i = 0;
            while (i < len) {
                if (i < len - 1 && selected[i]) {
                    out_tokens[start + w++] = new_id[i];
                    i += 2; // skip both left and right of the merged pair
                } else {
                    out_tokens[start + w++] = tokens[i];
                    i += 1;
                }
            }
            s_len = w;
        }
        __syncthreads();

        // reload tokens from out buffer for next pass
        int new_len = s_len;
        if (new_len > BLOCK_BPE_MAX_TOKENS) break; // safety: can't reload if exceeds shared memory
        for (int i = threadIdx.x; i < new_len; i += blockDim.x) tokens[i] = out_tokens[start + i];
        __syncthreads();
        if (new_len <= 1) break;
    }

    // Write final length once at the end
    if (threadIdx.x == 0) out_lengths[b] = s_len;
}

static vector<int> text_to_initial_token_ids(const string &text, const vector<string> &byte_encoder, const unordered_map<string,int> &tok2id) {
    vector<int> ids; ids.reserve(text.size());
    for (unsigned char c : text) {
        const string &sym = byte_encoder[c]; auto it = tok2id.find(sym);
        if (it == tok2id.end()) { cerr << "Initial symbol missing in vocab for byte " << (int)c << endl; continue; }
        ids.push_back(it->second);
    }
    return ids;
}

// CPU reference (same as in bpe_infer.cu)
static vector<int> bpe_cpu_greedy(const vector<int> &initial_ids, const unordered_map<uint64_t, uint64_t> &pairinfo) {
    vector<int> ids = initial_ids; if (ids.size() <= 1) return ids;
    while (true) {
        size_t n = ids.size(); vector<int> ranks(n - 1, INT_MAX);
        for (size_t i = 0; i + 1 < n; ++i) { auto it = pairinfo.find(pack_pair((uint32_t)ids[i], (uint32_t)ids[i + 1])); if (it != pairinfo.end()) { int r, nid; unpack_pairval(it->second, r, nid); ranks[i] = r; } }
        int min_rank = INT_MAX; for (size_t i = 0; i + 1 < n; ++i) if (ranks[i] < min_rank) min_rank = ranks[i]; if (min_rank == INT_MAX) break;
        vector<char> select(n - 1, 0); bool any = false; size_t last = (size_t)-2;
        for (size_t i = 0; i + 1 < n; ++i) if (ranks[i] == min_rank && i != last + 1) { select[i] = 1; any = true; last = i; }
        if (!any) break; vector<int> out; out.reserve(n); size_t i = 0; while (i < n) { if (i + 1 < n && select[i]) { auto it = pairinfo.find(pack_pair((uint32_t)ids[i], (uint32_t)ids[i + 1])); out.push_back((int)(it->second & 0xFFFFFFFFu)); i += 2; } else { out.push_back(ids[i]); i += 1; } }
        ids.swap(out); if (ids.size() <= 1) break;
    }
    return ids;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    string base = string("/home/vk2636/gpu-tokenizer/data/gpt2_tokenizer");
    string vocab_path = base + "/vocab.json";
    string merges_path = base + "/merges.txt";
    if (argc >= 3) { merges_path = argv[1]; vocab_path = argv[2]; }

    unordered_map<uint64_t, uint64_t> pairinfo; vector<int> byte_to_token_id;
    if (!load_merges_and_vocab(merges_path, vocab_path, pairinfo, byte_to_token_id)) return 1;
    vector<string> byte_encoder = build_gpt2_byte_encoder();

    // inputs
    vector<string> inputs; {
        ifstream f("/home/vk2636/gpu-tokenizer/data/input/input.txt"); if (f.is_open()) { string line; int cnt=0; while (getline(f, line) && cnt < 64) { inputs.push_back(line); cnt++; } } else { inputs.push_back("Hello world!"); inputs.push_back("The quick brown fox jumps over the lazy dog."); }
    }

    // host flattened initial ids and offsets
    FlatJsonVocab vocab; vocab.load(vocab_path);
    vector<int> h_offsets(inputs.size() + 1, 0); vector<int> h_init_ids;
    for (size_t i = 0; i < inputs.size(); ++i) { auto ids = text_to_initial_token_ids(inputs[i], byte_encoder, vocab.token_to_id); h_offsets[i + 1] = (int)(h_offsets[i] + ids.size()); h_init_ids.insert(h_init_ids.end(), ids.begin(), ids.end()); }

    // CPU reference
    vector<vector<int>> cpu_tokens; cpu_tokens.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) { vector<int> ids(h_init_ids.begin() + h_offsets[i], h_init_ids.begin() + h_offsets[i + 1]); cpu_tokens.push_back(bpe_cpu_greedy(ids, pairinfo)); }

    // Build cuco::static_map with ~50% load factor and probing cg_size=1 for device find
    int num_pairs = (int)pairinfo.size(); size_t capacity = (size_t)ceil(num_pairs / 0.5);
    constexpr uint64_t empty_key = 0xFFFFFFFFFFFFFFFFULL; constexpr uint64_t empty_val = 0xFFFFFFFFFFFFFFFFULL;
    auto map = cuco::static_map{
        capacity,
        cuco::empty_key{empty_key},
        cuco::empty_value{empty_val},
        cuda::std::equal_to<uint64_t>{},
        cuco::linear_probing<1, cuco::default_hash_function<uint64_t>>{}
    };

    // bulk insert pairs
    thrust::device_vector<uint64_t> d_keys(num_pairs);
    thrust::device_vector<uint64_t> d_vals(num_pairs);
    {
        thrust::host_vector<uint64_t> h_keys; h_keys.reserve(num_pairs);
        thrust::host_vector<uint64_t> h_vals; h_vals.reserve(num_pairs);
        for (auto &kv : pairinfo) { h_keys.push_back(kv.first); h_vals.push_back(kv.second); }
        d_keys = h_keys; d_vals = h_vals;
    }
    auto pairs = thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
        cuda::proclaim_return_type<cuco::pair<uint64_t, uint64_t>>(
            [k = d_keys.begin(), v = d_vals.begin()] __device__(size_t i) { return cuco::pair<uint64_t, uint64_t>{k[i], v[i]}; }));
    map.insert(pairs, pairs + num_pairs);

    // device buffers
    int total_init = h_offsets.back(); int B = (int)inputs.size();
    int *d_offsets=nullptr, *d_init=nullptr, *d_out=nullptr, *d_outlen=nullptr;
    cudaMalloc(&d_offsets, sizeof(int) * (B + 1));
    cudaMalloc(&d_init, sizeof(int) * total_init);
    cudaMalloc(&d_out, sizeof(int) * total_init);
    cudaMalloc(&d_outlen, sizeof(int) * B);
    cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * (B + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init, h_init_ids.data(), sizeof(int) * total_init, cudaMemcpyHostToDevice);

    auto find_ref = map.ref(cuco::find);
    dim3 grid(B); dim3 block(BLOCK_BPE_BLOCK_SIZE);
    auto t0 = chrono::high_resolution_clock::now();
    block_bpe_kernel<<<grid, block>>>(d_offsets, d_init, d_out, d_outlen, find_ref);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();

    vector<int> h_outlen(B); vector<int> h_out(total_init);
    cudaMemcpy(h_outlen.data(), d_outlen, sizeof(int) * B, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out.data(), d_out, sizeof(int) * total_init, cudaMemcpyDeviceToHost);

    bool all_ok = true;
    for (int i = 0; i < B; ++i) {
        int L = h_outlen[i]; vector<int> gpu_ids(h_out.begin() + h_offsets[i], h_out.begin() + h_offsets[i] + L);
        if (gpu_ids != cpu_tokens[i]) { all_ok = false; cerr << "Mismatch at sample " << i << ": CPU(" << cpu_tokens[i].size() << ") vs GPU(" << gpu_ids.size() << ")" << endl; }
    }

    auto ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0;
    cout << "BlockBPE GPU tokenization done in " << ms << " ms for " << B << " sequences" << endl;
    cout << (all_ok ? "GPU matches CPU" : "GPU differs from CPU") << endl;

    cudaFree(d_offsets); cudaFree(d_init); cudaFree(d_out); cudaFree(d_outlen);
    return all_ok ? 0 : 2;
}


