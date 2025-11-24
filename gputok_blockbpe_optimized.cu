// GPU Byte-level BPE Tokenizer (BlockBPE-style, optimized variant)
//
// This file implements the same greedy BPE semantics as gputok_blockbpe.cu
// (Algorithm 1: global minimum-rank pair, merge left-most occurrence, repeat),
// but optimizes the GPU kernel in two ways:
//   1. Uses a simpler, fully double-buffered compaction scheme for all
//      sequence lengths (short and long), avoiding BlockScan and reducing
//      shared-memory usage and synchronization overhead.
//   2. Keeps the same thread-coarsened argmin behavior for long sequences
//      (each thread processes multiple positions with stride = blockDim.x),
//      improving compatibility with long contexts.
//
// Semantics are preserved: we still use one block per sequence, a global
// argmin over pair ranks per pass, and write exactly one merged pair per
// iteration. CPU and GPU results are checked for exact equality.

#include <cuco/static_map.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------- Utility: CUDA error checking -------------------------

inline void check_cuda(cudaError_t err, const char* msg)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// ----------------- GPT-2 style byte encoder reconstruction -------------------
//
// This reproduces the Python logic used by GPT-2 to build a reversible mapping
// from bytes 0..255 to a "safe" set of unicode chars (to make BPE operate at
// the byte level while staying Unicode-friendly).

std::string codepoint_to_utf8(int codepoint)
{
  // Minimal UTF-8 encoder for codepoints in [0, 0x10FFFF].
  std::string out;
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
  return out;
}

struct ByteEncoder {
  // byte -> encoded symbol string
  std::vector<std::string> byte_to_symbol;  // size 256

  explicit ByteEncoder()
  {
    std::vector<int> bs;
    std::vector<int> cs;
    bs.reserve(512);
    cs.reserve(512);

    auto push_range = [&](int start, int end) {
      for (int c = start; c <= end; ++c) {
        bs.push_back(c);
        cs.push_back(c);
      }
    };

    // Ranges from GPT-2 code
    push_range(static_cast<int>('!'), static_cast<int>('~'));
    push_range(0xA1, 0xAC);  // '¡' to '¬'
    push_range(0xAE, 0xFF);  // '®' to 'ÿ'

    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }

    byte_to_symbol.assign(256, std::string());
    for (std::size_t i = 0; i < bs.size(); ++i) {
      int b  = bs[i];
      int cp = cs[i];
      if (b < 0 || b > 255) continue;
      byte_to_symbol[b] = codepoint_to_utf8(cp);
    }
  }
};

// -------------------------- BPE vocabulary / merges --------------------------

struct PairInfo {
  std::uint32_t rank;
  std::int32_t new_token;
};

using PairKey    = std::uint64_t;  // pack (token_a, token_b)
using PairVal    = std::uint64_t;  // pack (rank, new_token)
using CpuPairMap = std::unordered_map<PairKey, PairInfo>;

__host__ __device__ inline PairKey pack_pair(std::int32_t a, std::int32_t b)
{
  return (static_cast<PairKey>(static_cast<std::uint32_t>(a)) << 32) |
         (static_cast<PairKey>(static_cast<std::uint32_t>(b)));
}

inline PairVal pack_val(PairInfo info)
{
  return (static_cast<PairVal>(static_cast<std::uint32_t>(info.new_token)) << 32) |
         (static_cast<PairVal>(info.rank));
}

__host__ __device__ inline PairInfo unpack_val(PairVal v)
{
  PairInfo info;
  info.rank      = static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
  info.new_token = static_cast<std::int32_t>((v >> 32) & 0xFFFFFFFFu);
  return info;
}

struct BPEVocab {
  // Token string <-> id
  std::unordered_map<std::string, std::int32_t> token_to_id;
  std::vector<std::string> id_to_token;

  // Pair -> (rank, new_token) on CPU
  CpuPairMap cpu_pairs;

  std::int32_t add_token(const std::string& tok)
  {
    auto it = token_to_id.find(tok);
    if (it != token_to_id.end()) return it->second;
    std::int32_t id = static_cast<std::int32_t>(id_to_token.size());
    id_to_token.push_back(tok);
    token_to_id.emplace(tok, id);
    return id;
  }
};

void build_vocab_and_merges(const std::string& merges_path, BPEVocab& vocab, const ByteEncoder& enc)
{
  // 1. Seed vocabulary with base 256 byte-encoder symbols
  for (int b = 0; b < 256; ++b) {
    const auto& sym = enc.byte_to_symbol[b];
    vocab.add_token(sym);
  }

  // 2. Parse merges.txt and create pair->(rank,new_token) mapping
  std::ifstream in(merges_path);
  if (!in) {
    throw std::runtime_error("Failed to open merges file: " + merges_path);
  }

  std::string line;
  // First line is a version comment, skip
  if (!std::getline(in, line)) {
    throw std::runtime_error("Empty merges file");
  }

  std::uint32_t rank = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string a, b;
    if (!(iss >> a >> b)) {
      continue;
    }

    auto it_a = vocab.token_to_id.find(a);
    if (it_a == vocab.token_to_id.end()) {
      // Should not happen for valid GPT-2 merges, but be robust.
      it_a = vocab.token_to_id.emplace(a, vocab.add_token(a)).first;
    }
    auto it_b = vocab.token_to_id.find(b);
    if (it_b == vocab.token_to_id.end()) {
      it_b = vocab.token_to_id.emplace(b, vocab.add_token(b)).first;
    }

    std::int32_t id_a = it_a->second;
    std::int32_t id_b = it_b->second;

    // New merged symbol is concatenation of symbol strings
    std::string merged = vocab.id_to_token[id_a] + vocab.id_to_token[id_b];
    std::int32_t new_id = vocab.add_token(merged);

    PairInfo info;
    info.rank      = rank;
    info.new_token = new_id;

    PairKey key = pack_pair(id_a, id_b);
    vocab.cpu_pairs.emplace(key, info);

    ++rank;
  }

  std::cerr << "Loaded BPE merges: " << vocab.cpu_pairs.size()
            << ", vocab size: " << vocab.id_to_token.size() << std::endl;
}

// ------------------------------ CPU BPE (greedy) -----------------------------
//
// Exact implementation of Algorithm 1 from the BlockBPE paper:
//   - While there exists a mergeable pair in the sequence:
//       * Find the pair (a, b) occurring in the sequence whose merge rank
//         in the table is minimal.
//       * Among occurrences of that pair in the sequence, merge the
//         left-most occurrence only.
//   - Repeat until no such pair exists.

void bpe_merge_cpu_greedy(std::vector<std::int32_t>& tokens,
                          CpuPairMap const& pair_map,
                          int /*max_iters_hint*/ = 0)
{
  if (tokens.empty()) return;

  constexpr std::uint32_t INF_RANK = std::numeric_limits<std::uint32_t>::max();

  while (true) {
    int n = static_cast<int>(tokens.size());
    if (n < 2) break;

    std::uint32_t best_rank = INF_RANK;
    int best_pos            = -1;
    PairInfo best_info{};

    // 1. Find globally best pair (lowest rank) and left-most occurrence
    for (int i = 0; i < n - 1; ++i) {
      PairKey key = pack_pair(tokens[i], tokens[i + 1]);
      auto it     = pair_map.find(key);
      if (it == pair_map.end()) continue;

      const auto& info = it->second;
      if (info.rank < best_rank) {
        best_rank = info.rank;
        best_pos  = i;
        best_info = info;
      }
      // If ranks are unique (as in GPT-2 merges), no tie-break is needed
    }

    if (best_pos < 0) {
      // No mergeable pair exists
      break;
    }

    // 2. Merge the left-most occurrence at best_pos
    std::vector<std::int32_t> new_tokens;
    new_tokens.reserve(n - 1);

    for (int i = 0; i < n;) {
      if (i == best_pos) {
        new_tokens.push_back(best_info.new_token);
        i += 2;  // skip tokens[i+1]
      } else {
        new_tokens.push_back(tokens[i]);
        ++i;
      }
    }

    tokens.swap(new_tokens);
  }
}

// -------------------------- GPU BPE with cuCollections -----------------------

using PairProbing      = cuco::linear_probing<1, cuco::default_hash_function<PairKey>>;
using DevicePairMap    = cuco::static_map<PairKey,
                                       PairVal,
                                       cuco::extent<std::size_t>,
                                       cuda::thread_scope_device,
                                       cuda::std::equal_to<PairKey>,
                                       PairProbing>;
using DevicePairMapRef = decltype(std::declval<DevicePairMap>().ref(cuco::find));

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

// Kernel that mirrors Algorithm 1 on GPU:
//   - One block per sequence.
//   - Each iteration:
//       * All threads scan subsets of adjacent pairs to find their local
//         best (rank, position).
//       * A block-wide reduction yields the global best (rank, position).
//       * If a merge exists, we merge that single pair and shift tokens
//         left in parallel.

__global__ void bpe_merge_kernel_blockbpe(DevicePairMapRef map_ref,
                                          const std::int32_t* __restrict__ d_tokens_in,
                                          const int* __restrict__ d_offsets,
                                          const int* __restrict__ d_lengths,
                                          std::int32_t* __restrict__ d_tokens_out,
                                          int* __restrict__ d_out_lengths,
                                          int max_seq_len)
{
  int seq_id = blockIdx.x;
  int tid    = threadIdx.x;

  int offset = d_offsets[seq_id];
  int len    = d_lengths[seq_id];
  if (len <= 0 || len > max_seq_len) return;

  extern __shared__ unsigned char smem[];
  // Double-buffered shared-storage for tokens: [in | out]
  std::int32_t* sh_tokens_in  = reinterpret_cast<std::int32_t*>(smem);
  std::int32_t* sh_tokens_out = sh_tokens_in + max_seq_len;

  __shared__ int sh_len;

  // Load tokens into shared memory
  for (int i = tid; i < len; i += blockDim.x) {
    sh_tokens_in[i] = d_tokens_in[offset + i];
  }

  if (tid == 0) {
    sh_len = len;
  }
  __syncthreads();

  const std::uint32_t INF_RANK = 0xFFFFFFFFu;

  // Scratch for block-wide argmin reduction
  __shared__ std::uint32_t sh_best_rank[BLOCK_SIZE];
  __shared__ int           sh_best_pos[BLOCK_SIZE];
  __shared__ PairInfo      sh_best_info;

  while (true) {
    int cur_len = sh_len;
    if (cur_len < 2) break;

    // 1. Each thread scans a strided subset of pairs to find its local best
    std::uint32_t local_best_rank = INF_RANK;
    int           local_best_pos  = -1;

    for (int i = tid; i < cur_len - 1; i += blockDim.x) {
      PairKey key = pack_pair(sh_tokens_in[i], sh_tokens_in[i + 1]);
      auto    it  = map_ref.find(key);
      if (it == map_ref.end()) continue;

      PairVal v     = (*it).second;
      PairInfo info = unpack_val(v);

      if (info.rank < local_best_rank) {
        local_best_rank = info.rank;
        local_best_pos  = i;
      }
    }

    // 2. Block-wide argmin reduction on (rank, position), breaking ties
    //    by choosing the left-most position.
    sh_best_rank[tid] = local_best_rank;
    sh_best_pos[tid]  = local_best_pos;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        std::uint32_t r1 = sh_best_rank[tid];
        int           p1 = sh_best_pos[tid];
        std::uint32_t r2 = sh_best_rank[tid + stride];
        int           p2 = sh_best_pos[tid + stride];

        bool take_r2 = false;
        if (r2 < r1) {
          take_r2 = true;
        } else if (r2 == r1) {
          // Both INF or same rank; prefer valid and left-most position.
          if (p2 >= 0 && (p1 < 0 || p2 < p1)) {
            take_r2 = true;
          }
        }

        if (take_r2) {
          sh_best_rank[tid] = r2;
          sh_best_pos[tid]  = p2;
        }
      }
      __syncthreads();
    }

    std::uint32_t best_rank = sh_best_rank[0];
    int           best_pos  = sh_best_pos[0];

    // 3. If no mergeable pair exists, we are done.
    if (best_pos < 0 || best_rank == INF_RANK) {
      break;
    }

    // 4. Look up the new token id for the best pair once (by thread 0)
    if (tid == 0) {
      PairKey key = pack_pair(sh_tokens_in[best_pos], sh_tokens_in[best_pos + 1]);
      auto    it  = map_ref.find(key);
      if (it != map_ref.end()) {
        PairVal v = (*it).second;
        sh_best_info = unpack_val(v);
      } else {
        // Fallback (should not happen if maps are consistent)
        sh_best_info.rank      = best_rank;
        sh_best_info.new_token = sh_tokens_in[best_pos];
      }
    }
    __syncthreads();

    // 5. Thread-coarsened, double-buffered compaction for all sequence
    //    lengths. Each thread processes indices idx = tid, idx += blockDim.x:
    //      * idx == best_pos     -> write new_token at dst = best_pos
    //      * idx == best_pos + 1 -> dropped
    //      * idx <  best_pos + 1 -> dst = idx
    //      * idx >  best_pos + 1 -> dst = idx - 1
    for (int idx = tid; idx < cur_len; idx += blockDim.x) {
      if (idx == best_pos + 1) {
        continue;  // removed token
      }

      int val = sh_tokens_in[idx];
      if (idx == best_pos) {
        val = sh_best_info.new_token;
      }

      int dst = (idx <= best_pos) ? idx : (idx - 1);
      sh_tokens_out[dst] = val;
    }
    __syncthreads();

    if (tid == 0) {
      sh_len = cur_len - 1;
    }
    __syncthreads();

    // Swap buffers for next iteration
    std::int32_t* tmp = sh_tokens_in;
    sh_tokens_in      = sh_tokens_out;
    sh_tokens_out     = tmp;
    __syncthreads();
  }

  // Write back output tokens and length
  len = sh_len;
  for (int i = tid; i < len; i += blockDim.x) {
    d_tokens_out[offset + i] = sh_tokens_in[i];
  }
  if (tid == 0) {
    d_out_lengths[seq_id] = len;
  }
}

// ---------------------------- I/O helpers ------------------------------------

std::vector<std::string> read_lines(const std::string& path, int max_lines = 1000)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
    if (max_lines > 0 && static_cast<int>(lines.size()) >= max_lines) break;
  }
  return lines;
}

std::vector<std::int32_t> encode_text_to_tokens(const std::string& text,
                                                const ByteEncoder& enc,
                                                const BPEVocab& vocab)
{
  std::vector<std::int32_t> tokens;
  tokens.reserve(text.size());

  for (unsigned char c : text) {
    const std::string& sym = enc.byte_to_symbol[static_cast<int>(c)];
    auto               it  = vocab.token_to_id.find(sym);
    if (it == vocab.token_to_id.end()) {
      throw std::runtime_error("Byte encoder symbol not found in vocab");
    }
    tokens.push_back(it->second);
  }
  return tokens;
}

// ----------------------------- Main driver -----------------------------------

int main(int argc, char** argv)
{
  try {
    const std::string merges_path = "data/gpt2_tokenizer/merges.txt";
    // Default input path; can be overridden via argv[2]
    std::string input_path = "data/input/pride_and_prejudice.txt";

    ByteEncoder encoder;
    BPEVocab   vocab;
    build_vocab_and_merges(merges_path, vocab, encoder);

    // Optional command-line arg: chunk size in tokens (default 2048)
    int chunk_tokens = 2048;
    if (argc > 1) {
      try {
        chunk_tokens = std::max(1, std::stoi(argv[1]));
      } catch (...) {
        std::cerr << "Warning: failed to parse chunk size from argv[1]; using default "
                  << chunk_tokens << std::endl;
      }
    }

    // Optional command-line arg: input text file path (default as above)
    if (argc > 2) {
      input_path = argv[2];
    }

    // Read entire file as a single text sequence
    std::ifstream in(input_path);
    if (!in) {
      std::cerr << "Failed to open input file: " << input_path << std::endl;
      return 1;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    std::string text = oss.str();
    if (text.empty()) {
      std::cerr << "Input file is empty: " << input_path << std::endl;
      return 0;
    }

    // Encode full text into initial token sequence (byte-level)
    std::vector<std::int32_t> all_tokens = encode_text_to_tokens(text, encoder, vocab);

    // Split into chunks of up to chunk_tokens tokens (production-style sequence chunking)
    std::vector<std::vector<std::int32_t>> cpu_seqs;
    cpu_seqs.reserve((all_tokens.size() + chunk_tokens - 1) / chunk_tokens);

    int max_seq_len = 0;
    for (std::size_t pos = 0; pos < all_tokens.size();) {
      std::size_t len = std::min<std::size_t>(chunk_tokens, all_tokens.size() - pos);
      cpu_seqs.emplace_back(all_tokens.begin() + static_cast<std::ptrdiff_t>(pos),
                            all_tokens.begin() + static_cast<std::ptrdiff_t>(pos + len));
      max_seq_len = std::max(max_seq_len, static_cast<int>(len));
      pos += len;
    }

    std::cerr << "Tokenized full text into " << cpu_seqs.size()
              << " chunk(s), max tokens per chunk: " << max_seq_len << std::endl;

    // Flatten sequences for GPU
    int num_seqs = static_cast<int>(cpu_seqs.size());
    std::vector<int>          h_offsets(num_seqs);
    std::vector<int>          h_lengths(num_seqs);
    std::vector<std::int32_t> h_tokens_flat;

    int offset = 0;
    for (int i = 0; i < num_seqs; ++i) {
      h_offsets[i] = offset;
      int len      = static_cast<int>(cpu_seqs[i].size());
      h_lengths[i] = len;
      h_tokens_flat.insert(h_tokens_flat.end(), cpu_seqs[i].begin(), cpu_seqs[i].end());
      offset += len;
    }
    int total_tokens = offset;

    std::cerr << "Total initial tokens: " << total_tokens << std::endl;

    // Prepare GPU static_map for pair->(rank,new_token)
    std::size_t num_pairs   = vocab.cpu_pairs.size();
    double      load_factor = 0.5;
    std::size_t capacity    = static_cast<std::size_t>(std::ceil(num_pairs / load_factor));
    PairKey     empty_key   = std::numeric_limits<PairKey>::max();
    PairVal     empty_val   = std::numeric_limits<PairVal>::max();

    DevicePairMap d_map{
      cuco::extent<std::size_t>{capacity},
      cuco::empty_key{empty_key},
      cuco::empty_value{empty_val}};

    // Copy pairs to device and bulk-insert
    std::vector<cuco::pair<PairKey, PairVal>> h_pairs;
    h_pairs.reserve(num_pairs);

    for (auto const& kv : vocab.cpu_pairs) {
      PairKey  key  = kv.first;
      PairInfo info = kv.second;
      h_pairs.emplace_back(key, pack_val(info));
    }

    cuco::pair<PairKey, PairVal>* d_pairs = nullptr;
    check_cuda(cudaMalloc(&d_pairs, num_pairs * sizeof(cuco::pair<PairKey, PairVal>)),
               "cudaMalloc d_pairs");
    check_cuda(cudaMemcpy(d_pairs,
                          h_pairs.data(),
                          num_pairs * sizeof(cuco::pair<PairKey, PairVal>),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy pairs");

    d_map.insert(d_pairs, d_pairs + num_pairs);

    cudaFree(d_pairs);

    DevicePairMapRef d_map_ref = d_map.ref(cuco::find);

    // Allocate device buffers for tokens/offsets/lengths
    std::int32_t* d_tokens_in  = nullptr;
    std::int32_t* d_tokens_out = nullptr;
    int*          d_offsets    = nullptr;
    int*          d_lengths    = nullptr;
    int*          d_out_lengths = nullptr;

    check_cuda(cudaMalloc(&d_tokens_in, total_tokens * sizeof(std::int32_t)), "cudaMalloc tokens_in");
    check_cuda(cudaMalloc(&d_tokens_out, total_tokens * sizeof(std::int32_t)), "cudaMalloc tokens_out");
    check_cuda(cudaMalloc(&d_offsets, num_seqs * sizeof(int)), "cudaMalloc offsets");
    check_cuda(cudaMalloc(&d_lengths, num_seqs * sizeof(int)), "cudaMalloc lengths");
    check_cuda(cudaMalloc(&d_out_lengths, num_seqs * sizeof(int)), "cudaMalloc out_lengths");

    check_cuda(cudaMemcpy(d_tokens_in,
                          h_tokens_flat.data(),
                          total_tokens * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy tokens_in");
    check_cuda(cudaMemcpy(d_offsets,
                          h_offsets.data(),
                          num_seqs * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy offsets");
    check_cuda(cudaMemcpy(d_lengths,
                          h_lengths.data(),
                          num_seqs * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy lengths");

    // CPU BPE timing (greedy Algorithm 1)
    auto cpu_tokens_merged = cpu_seqs;  // copy
    auto cpu_start         = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_seqs; ++i) {
      bpe_merge_cpu_greedy(cpu_tokens_merged[i], vocab.cpu_pairs);
    }
    auto  cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // GPU BPE timing (BlockBPE-style greedy kernel)
    int block_size = BLOCK_SIZE;
    int grid_size  = num_seqs;

    std::size_t shared_bytes =
      static_cast<std::size_t>(2) * static_cast<std::size_t>(max_seq_len) * sizeof(std::int32_t);

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start);
    bpe_merge_kernel_blockbpe<<<grid_size, block_size, shared_bytes>>>(
      d_map_ref, d_tokens_in, d_offsets, d_lengths, d_tokens_out, d_out_lengths, max_seq_len);
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_end);

    check_cuda(cudaGetLastError(), "bpe_merge_kernel_blockbpe");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    // Copy GPU outputs back
    std::vector<std::int32_t> h_tokens_out(total_tokens);
    std::vector<int>          h_out_lengths(num_seqs);
    check_cuda(cudaMemcpy(h_tokens_out.data(),
                          d_tokens_out,
                          total_tokens * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy tokens_out");
    check_cuda(cudaMemcpy(h_out_lengths.data(),
                          d_out_lengths,
                          num_seqs * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy out_lengths");

    // Check correctness: compare CPU vs GPU for each sequence
    bool all_ok = true;
    for (int i = 0; i < num_seqs; ++i) {
      int off = h_offsets[i];
      int len = h_out_lengths[i];

      std::vector<std::int32_t> gpu_seq;
      gpu_seq.reserve(len);
      for (int j = 0; j < len; ++j) {
        gpu_seq.push_back(h_tokens_out[off + j]);
      }

      if (static_cast<int>(cpu_tokens_merged[i].size()) != len) {
        all_ok = false;
        std::cerr << "[Seq " << i << "] Length mismatch: CPU "
                  << cpu_tokens_merged[i].size() << " vs GPU " << len << std::endl;

        // Print a small prefix of both sequences for debugging
        int debug_len = std::min<int>(static_cast<int>(cpu_tokens_merged[i].size()), len);
        debug_len     = std::min(debug_len, 64);
        std::cerr << "CPU tokens (prefix up to " << debug_len << "): ";
        for (int j = 0; j < debug_len; ++j) {
          std::cerr << cpu_tokens_merged[i][j] << " ";
        }
        std::cerr << std::endl;
        std::cerr << "GPU tokens (prefix up to " << debug_len << "): ";
        for (int j = 0; j < debug_len; ++j) {
          std::cerr << gpu_seq[j] << " ";
        }
        std::cerr << std::endl;
        break;
      }

      for (int j = 0; j < len; ++j) {
        if (cpu_tokens_merged[i][j] != gpu_seq[j]) {
          all_ok = false;
          std::cerr << "[Seq " << i << "] Token mismatch at position " << j
                    << ": CPU " << cpu_tokens_merged[i][j]
                    << " vs GPU " << gpu_seq[j] << std::endl;

          int debug_len = std::min<int>(len, j + 32);
          debug_len     = std::min(debug_len, 128);
          std::cerr << "CPU tokens (prefix up to " << debug_len << "): ";
          for (int k = 0; k < debug_len && k < static_cast<int>(cpu_tokens_merged[i].size()); ++k) {
            std::cerr << cpu_tokens_merged[i][k] << " ";
          }
          std::cerr << std::endl;
          std::cerr << "GPU tokens (prefix up to " << debug_len << "): ";
          for (int k = 0; k < debug_len; ++k) {
            std::cerr << gpu_seq[k] << " ";
          }
          std::cerr << std::endl;
          break;
        }
      }
      if (!all_ok) break;
    }

    std::cout << "CPU BPE (greedy Algorithm 1) time: " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU BlockBPE-style greedy kernel time (excl. H2D/D2H): "
              << gpu_ms << " ms" << std::endl;
    if (gpu_ms > 0.0f) {
      double speedup       = cpu_ms / gpu_ms;
      double gpu_tok_per_s = (static_cast<double>(total_tokens) / (gpu_ms / 1000.0));
      double cpu_tok_per_s = (static_cast<double>(total_tokens) / (cpu_ms / 1000.0));
      std::cout << "Approximate speedup (CPU / GPU greedy kernel): " << speedup << "x" << std::endl;
      std::cout << "CPU throughput: " << cpu_tok_per_s << " tokens/s" << std::endl;
      std::cout << "GPU greedy kernel throughput: " << gpu_tok_per_s << " tokens/s" << std::endl;
    }
    std::cout << "CPU vs GPU token match (greedy Algorithm 1): "
              << (all_ok ? "OK" : "MISMATCH") << std::endl;

    // Write final tokenized output (GPU result) to file in data/output.
    {
      const std::string out_path = "data/output/gpu_tokens_blockbpe.txt";
      std::ofstream     out(out_path);
      if (!out) {
        std::cerr << "Failed to open output file for writing: " << out_path << std::endl;
      } else {
        for (int i = 0; i < num_seqs; ++i) {
          int off = h_offsets[i];
          int len = h_out_lengths[i];
          for (int j = 0; j < len; ++j) {
            std::int32_t tid = h_tokens_out[off + j];
            if (tid < 0 || static_cast<std::size_t>(tid) >= vocab.id_to_token.size()) {
              // Fallback: write raw id if out of range (should not happen)
              out << tid;
            } else {
              out << vocab.id_to_token[static_cast<std::size_t>(tid)];
            }
            if (j + 1 < len) out << ' ';
          }
          out << '\n';
        }
        std::cerr << "Wrote GPU BlockBPE tokens to " << out_path << std::endl;
      }
    }

    // Cleanup
    cudaFree(d_tokens_in);
    cudaFree(d_tokens_out);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_out_lengths);

    return all_ok ? 0 : 1;

  } catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
    return 1;
  }
}



