// GPU Byte-level BPE Tokenizer (Block-style) using cuCollections
//
// Optimized variant:
// - Same overall algorithm as main.cu, but reduces redundant cuco::static_map
//   lookups in the GPU kernel by caching both rank and new_token for each pair
//   in shared memory. This removes one hash-table lookup per merged pair.
//
// High-level design (same as original):
// - Implements a byte-level BPE tokenizer similar in spirit to BlockBPE:
//   * No regex pre-tokenization, we operate directly on bytes.
//   * Uses GPT-2 style byte encoder to map bytes -> initial symbol alphabet.
//   * Builds BPE merge table from GPT-2 `merges.txt`.
//   * Stores (token_a, token_b) -> (rank, new_token_id) in a GPU-resident
//     cuco::static_map for fast pair lookup in kernels.
//   * Performs parallel merge passes inside each CUDA block over sequences.
// - Also implements a CPU reference tokenizer that uses the same merge policy
//   as the GPU kernel (local, non-overlapping merges selected by rank).
// - The main() function:
//   * Loads merges from data/gpt2_tokenizer/merges.txt
//   * Reads input lines from data/input/input.txt
//   * Encodes them to initial tokens with GPT-2 byte encoder
//   * Runs CPU and GPU BPE, checks for equality, and times both.
//

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

// ---------------------- Utility: CUDA error checking -----------------------

inline void check_cuda(cudaError_t err, const char* msg)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// ----------------- GPT-2 style byte encoder reconstruction -----------------
//
// This reproduces the Python logic used by GPT-2 to build a reversible mapping
// from bytes 0..255 to a "safe" set of unicode chars (to make BPE operate at
// the byte level while staying Unicode-friendly).
//

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

// -------------------------- BPE vocabulary / merges ------------------------

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

// ------------------------------ CPU BPE ------------------------------------

void bpe_merge_cpu(std::vector<std::int32_t>& tokens,
                   CpuPairMap const& pair_map,
                   int max_iters = 50)
{
  if (tokens.empty()) return;

  constexpr std::uint32_t INF_RANK = std::numeric_limits<std::uint32_t>::max();

  bool changed = true;
  int  iter    = 0;
  while (changed && iter < max_iters) {
    int n = static_cast<int>(tokens.size());
    if (n < 2) break;

    std::vector<std::uint32_t> ranks(n - 1, INF_RANK);

    // 1. Compute ranks for each adjacent pair
    for (int i = 0; i < n - 1; ++i) {
      PairKey key = pack_pair(tokens[i], tokens[i + 1]);
      auto it     = pair_map.find(key);
      if (it != pair_map.end()) {
        ranks[i] = it->second.rank;
      }
    }

    // 2. Decide which positions to merge using non-overlapping, rank-based rule
    std::vector<char> merge_here(n - 1, 0);
    changed = false;

    for (int i = 0; i < n - 1; ++i) {
      auto r = ranks[i];
      if (r == INF_RANK) continue;

      // Left neighbor has strictly better rank -> don't merge here
      if (i > 0 && ranks[i - 1] < r) continue;
      // Right neighbor has better or equal rank -> don't merge here
      if (i + 1 < n - 1 && ranks[i + 1] <= r) continue;

      merge_here[i] = 1;
      changed       = true;
    }

    if (!changed) break;

    // 3. Build new token sequence
    std::vector<std::int32_t> new_tokens;
    new_tokens.reserve(n);

    int i = 0;
    while (i < n) {
      if (i < n - 1 && merge_here[i]) {
        PairKey key = pack_pair(tokens[i], tokens[i + 1]);
        auto it     = pair_map.find(key);
        if (it == pair_map.end()) {
          // Fallback: no merge (should not happen)
          new_tokens.push_back(tokens[i]);
          ++i;
        } else {
          new_tokens.push_back(it->second.new_token);
          i += 2;
        }
      } else {
        new_tokens.push_back(tokens[i]);
        ++i;
      }
    }

    tokens.swap(new_tokens);
    ++iter;
  }
}

// -------------------------- GPU BPE with cuCollections ---------------------

using PairProbing       = cuco::linear_probing<1, cuco::default_hash_function<PairKey>>;
using DevicePairMap     = cuco::static_map<PairKey,
                                       PairVal,
                                       cuco::extent<std::size_t>,
                                       cuda::thread_scope_device,
                                       cuda::std::equal_to<PairKey>,
                                       PairProbing>;
using DevicePairMapRef  = decltype(std::declval<DevicePairMap>().ref(cuco::find));

__global__ void bpe_merge_kernel_optimized(DevicePairMapRef map_ref,
                                           const std::int32_t* __restrict__ d_tokens_in,
                                           const int* __restrict__ d_offsets,
                                           const int* __restrict__ d_lengths,
                                           std::int32_t* __restrict__ d_tokens_out,
                                           int* __restrict__ d_out_lengths,
                                           int max_seq_len,
                                           int max_iters)
{
  int seq_id = blockIdx.x;
  int tid    = threadIdx.x;

  int offset = d_offsets[seq_id];
  int len    = d_lengths[seq_id];
  if (len <= 0 || len > max_seq_len) return;

  extern __shared__ unsigned char smem[];
  std::int32_t*  sh_tokens     = reinterpret_cast<std::int32_t*>(smem);
  std::uint32_t* sh_ranks      = reinterpret_cast<std::uint32_t*>(sh_tokens + max_seq_len);
  std::int32_t*  sh_new_token  = reinterpret_cast<std::int32_t*>(sh_ranks + max_seq_len);
  unsigned char* sh_mergehere  = reinterpret_cast<unsigned char*>(sh_new_token + max_seq_len);

  __shared__ int sh_len;
  __shared__ int sh_changed;

  // Load tokens into shared memory
  for (int i = tid; i < len; i += blockDim.x) {
    sh_tokens[i] = d_tokens_in[offset + i];
  }

  if (tid == 0) {
    sh_len     = len;
    sh_changed = 1;
  }
  __syncthreads();

  const std::uint32_t INF_RANK = 0xFFFFFFFFu;

  int iter = 0;
  while (iter < max_iters) {
    if (sh_len < 2) break;

    // Reset changed flag
    if (tid == 0) sh_changed = 0;
    __syncthreads();

    int cur_len = sh_len;

    // 1. Compute ranks and new_token for each adjacent pair
    for (int i = tid; i < cur_len - 1; i += blockDim.x) {
      PairKey key = pack_pair(sh_tokens[i], sh_tokens[i + 1]);
      auto    it  = map_ref.find(key);
      if (it != map_ref.end()) {
        PairVal v     = (*it).second;
        PairInfo info = unpack_val(v);
        sh_ranks[i]   = info.rank;
        sh_new_token[i] = info.new_token;
      } else {
        sh_ranks[i]    = INF_RANK;
        sh_new_token[i] = -1;
      }
    }

    // For convenience, set last rank to INF (no pair)
    if (tid == 0 && cur_len - 1 >= 0) {
      sh_ranks[cur_len - 1]   = INF_RANK;
      sh_new_token[cur_len - 1] = -1;
    }
    __syncthreads();

    // 2. Decide which positions to merge
    for (int i = tid; i < cur_len - 1; i += blockDim.x) {
      std::uint32_t r = sh_ranks[i];
      unsigned char m = 0;
      if (r != INF_RANK) {
        if (i > 0 && sh_ranks[i - 1] < r) {
          m = 0;
        } else if (i + 1 < cur_len - 1 && sh_ranks[i + 1] <= r) {
          m = 0;
        } else {
          m = 1;
        }
      }
      sh_mergehere[i] = m;
      if (m) {
        atomicExch(&sh_changed, 1);
      }
    }
    __syncthreads();

    if (sh_changed == 0) {
      break;  // no merges this iteration
    }

    // 3. Sequential compaction per block (small per-sequence lengths)
    if (tid == 0) {
      int write_idx = 0;
      int i         = 0;
      while (i < cur_len) {
        if (i < cur_len - 1 && sh_mergehere[i]) {
          // Use cached new_token instead of looking into the map again
          sh_tokens[write_idx++] = sh_new_token[i];
          i += 2;
        } else if (i > 0 && sh_mergehere[i - 1]) {
          // This position was merged with previous, skip
          ++i;
        } else {
          sh_tokens[write_idx++] = sh_tokens[i];
          ++i;
        }
      }
      sh_len = write_idx;
    }
    __syncthreads();

    ++iter;
  }

  // Write back output tokens and length
  len = sh_len;
  for (int i = tid; i < len; i += blockDim.x) {
    d_tokens_out[offset + i] = sh_tokens[i];
  }
  if (tid == 0) {
    d_out_lengths[seq_id] = len;
  }
}

// ---------------------------- I/O helpers ----------------------------------

std::vector<std::string> read_lines(const std::string& path, int max_lines = 1000)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  std::vector<std::string> lines;
  std::string              line;
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

// ----------------------------- Main driver ---------------------------------

int main(int argc, char** argv)
{
  try {
    const std::string merges_path = "data/gpt2_tokenizer/merges.txt";
    const std::string input_path  = "data/input/pride_and_prejudice.txt";

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
    int                             num_seqs = static_cast<int>(cpu_seqs.size());
    std::vector<int>                h_offsets(num_seqs);
    std::vector<int>                h_lengths(num_seqs);
    std::vector<std::int32_t>       h_tokens_flat;

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

    // Build CPU pair map already inside vocab.cpu_pairs
    // Prepare GPU static_map for pair->(rank,new_token)
    std::size_t num_pairs  = vocab.cpu_pairs.size();
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

    // CPU BPE timing
    auto cpu_tokens_merged = cpu_seqs;  // copy
    auto cpu_start         = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_seqs; ++i) {
      bpe_merge_cpu(cpu_tokens_merged[i], vocab.cpu_pairs);
    }
    auto  cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // GPU BPE timing (optimized kernel)
    int block_size = 128;
    int grid_size  = num_seqs;
    int max_iters  = 50;

    std::size_t shared_bytes =
      static_cast<std::size_t>(max_seq_len) *
      (sizeof(std::int32_t) + sizeof(std::uint32_t) + sizeof(std::int32_t) + sizeof(unsigned char));

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start);
    bpe_merge_kernel_optimized<<<grid_size, block_size, shared_bytes>>>(
      d_map_ref, d_tokens_in, d_offsets, d_lengths, d_tokens_out, d_out_lengths, max_seq_len, max_iters);
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_end);

    check_cuda(cudaGetLastError(), "bpe_merge_kernel_optimized");

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
        std::cerr << "Length mismatch in sequence " << i << ": CPU "
                  << cpu_tokens_merged[i].size() << " vs GPU " << len << std::endl;
        break;
      }

      for (int j = 0; j < len; ++j) {
        if (cpu_tokens_merged[i][j] != gpu_seq[j]) {
          all_ok = false;
          std::cerr << "Token mismatch in sequence " << i << " at position " << j << ": CPU "
                    << cpu_tokens_merged[i][j] << " vs GPU " << gpu_seq[j] << std::endl;
          break;
        }
      }
      if (!all_ok) break;
    }

    std::cout << "CPU BPE time: " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU BPE kernel time (optimized, excl. H2D/D2H): " << gpu_ms << " ms" << std::endl;
    if (gpu_ms > 0.0f) {
      double speedup       = cpu_ms / gpu_ms;
      double gpu_tok_per_s = (static_cast<double>(total_tokens) / (gpu_ms / 1000.0));
      double cpu_tok_per_s = (static_cast<double>(total_tokens) / (cpu_ms / 1000.0));
      std::cout << "Approximate speedup (CPU / GPU optimized kernel): " << speedup << "x" << std::endl;
      std::cout << "CPU throughput: " << cpu_tok_per_s << " tokens/s" << std::endl;
      std::cout << "GPU optimized kernel throughput: " << gpu_tok_per_s << " tokens/s" << std::endl;
    }
    std::cout << "CPU vs GPU token match (optimized): " << (all_ok ? "OK" : "MISMATCH") << std::endl;

    // Write final tokenized output (GPU result) to file in data/output.
    {
      const std::string out_path = "data/output/gpu_tokens_optimized.txt";
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
        std::cerr << "Wrote GPU optimized tokens to " << out_path << std::endl;
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


