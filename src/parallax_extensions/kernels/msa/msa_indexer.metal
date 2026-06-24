#include "../common/utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES>
[[kernel]] void msa_block_scores(
    device float *block_scores [[buffer(0)]],
    device const T *index_query [[buffer(1)]],
    device const T *index_key_cache [[buffer(2)]],
    device const uint32_t *block_tables [[buffer(3)]],
    device const uint32_t *context_lens [[buffer(4)]],
    const constant int &index_heads [[buffer(5)]],
    const constant int &index_key_heads [[buffer(6)]],
    const constant int &cache_block_size [[buffer(7)]],
    const constant int &sparse_block_size [[buffer(8)]],
    const constant int &max_num_sparse_blocks [[buffer(9)]],
    const constant int &max_num_blocks_per_seq [[buffer(10)]],
    const constant int &sparse_init_blocks [[buffer(11)]],
    const constant int &sparse_local_blocks [[buffer(12)]],
    const constant float &scale [[buffer(13)]],
    const constant int &q_stride [[buffer(14)]],
    const constant int &q_head_stride [[buffer(15)]],
    const constant int &cache_block_stride [[buffer(16)]],
    const constant int &cache_head_stride [[buffer(17)]],
    const constant int &cache_token_stride [[buffer(18)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int sparse_block_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int thread_idx = thread_position_in_threadgroup.x;
  const int context_len = context_lens[seq_idx];
  const int cur_block = max((context_len - 1) / sparse_block_size, 0);
  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  float local_max = -INFINITY;
  if (sparse_block_idx <= cur_block) {
    const int total_items = index_heads * sparse_block_size;
    for (int item = thread_idx; item < total_items; item += NUM_THREADS) {
      const int query_head_idx = item / sparse_block_size;
      const int token_offset_in_sparse_block = item - query_head_idx * sparse_block_size;
      const int token_pos = sparse_block_idx * sparse_block_size + token_offset_in_sparse_block;
      if (token_pos >= context_len) {
        continue;
      }

      const int logical_cache_block = token_pos / cache_block_size;
      if (logical_cache_block < 0 ||
          logical_cache_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int cache_token_offset = token_pos - logical_cache_block * cache_block_size;
      const int64_t physical_cache_block =
          static_cast<int64_t>(block_table[logical_cache_block]);
      const int key_head_idx =
          index_key_heads == 1 ? 0 : min(query_head_idx, index_key_heads - 1);

      const device T *q_ptr =
          index_query + seq_idx * q_stride + query_head_idx * q_head_stride;
      const device T *k_ptr =
          index_key_cache + physical_cache_block * cache_block_stride +
          key_head_idx * cache_head_stride + cache_token_offset * cache_token_stride;

      float score = 0.f;
#pragma unroll
      for (int i = 0; i < HEAD_SIZE; ++i) {
        score += (float)q_ptr[i] * (float)k_ptr[i];
      }
      local_max = max(local_max, score * scale);
    }
  }

  threadgroup float red_smem[NUM_WARPS];
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    local_max = max(local_max, simd_shuffle_xor(local_max, mask));
  }
  if (simd_lid == 0) {
    red_smem[simd_tid] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float block_max = simd_lid < NUM_WARPS ? red_smem[simd_lid] : -INFINITY;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    block_max = max(block_max, simd_shuffle_xor(block_max, mask));
  }
  block_max = simd_shuffle(block_max, 0);

  if (thread_idx == 0) {
    float final_score = sparse_block_idx <= cur_block ? block_max : -INFINITY;
    if (sparse_block_idx <= cur_block && sparse_init_blocks > 0 &&
        sparse_block_idx < sparse_init_blocks) {
      final_score = 1e30f;
    }
    if (sparse_block_idx <= cur_block && sparse_local_blocks > 0) {
      const int local_start = max(cur_block - sparse_local_blocks + 1, 0);
      if (sparse_block_idx >= local_start) {
        final_score = 1e29f;
      }
    }
    block_scores[seq_idx * max_num_sparse_blocks + sparse_block_idx] = final_score;
  }
}

[[kernel]] void msa_block_topk_tokens(
    device int32_t *out_positions [[buffer(0)]],
    device const float *block_scores [[buffer(1)]],
    device const uint32_t *context_lens [[buffer(2)]],
    const constant int &max_num_sparse_blocks [[buffer(3)]],
    const constant int &max_topk_blocks [[buffer(4)]],
    const constant int &sparse_block_size [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int seq_idx = threadgroup_position_in_grid.x;
  float best_scores[32];
  int best_blocks[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    best_scores[i] = -INFINITY;
    best_blocks[i] = -1;
  }

  const device float *scores = block_scores + seq_idx * max_num_sparse_blocks;
  for (int block_idx = 0; block_idx < max_num_sparse_blocks; ++block_idx) {
    const float score = scores[block_idx];
    if (score == -INFINITY || !(score == score)) {
      continue;
    }

    for (int rank = 0; rank < max_topk_blocks; ++rank) {
      if (score > best_scores[rank]) {
        for (int j = max_topk_blocks - 1; j > rank; --j) {
          best_scores[j] = best_scores[j - 1];
          best_blocks[j] = best_blocks[j - 1];
        }
        best_scores[rank] = score;
        best_blocks[rank] = block_idx;
        break;
      }
    }
  }

  for (int i = 0; i < max_topk_blocks; ++i) {
    for (int j = i + 1; j < max_topk_blocks; ++j) {
      if (best_blocks[j] >= 0 &&
          (best_blocks[i] < 0 || best_blocks[j] < best_blocks[i])) {
        const int tmp_block = best_blocks[i];
        best_blocks[i] = best_blocks[j];
        best_blocks[j] = tmp_block;
      }
    }
  }

  const int context_len = context_lens[seq_idx];
  device int32_t *out =
      out_positions + seq_idx * max_topk_blocks * sparse_block_size;
  for (int rank = 0; rank < max_topk_blocks; ++rank) {
    const int block_idx = best_blocks[rank];
    for (int offset = 0; offset < sparse_block_size; ++offset) {
      const int out_idx = rank * sparse_block_size + offset;
      const int token_pos = block_idx * sparse_block_size + offset;
      out[out_idx] =
          (block_idx >= 0 && token_pos < context_len) ? token_pos : -1;
    }
  }
}

#define instantiate_msa_block_scores_inner(type, head_size, num_threads,    \
                                              num_simd_lanes)                 \
  template [[host_name("msa_block_scores_" #type "_hs" #head_size "_nt"    \
                       #num_threads "_nsl" #num_simd_lanes)]] [[kernel]] void \
  msa_block_scores<type, head_size, num_threads, num_simd_lanes>(          \
      device float *block_scores [[buffer(0)]],                               \
      device const type *index_query [[buffer(1)]],                           \
      device const type *index_key_cache [[buffer(2)]],                       \
      device const uint32_t *block_tables [[buffer(3)]],                      \
      device const uint32_t *context_lens [[buffer(4)]],                      \
      const constant int &index_heads [[buffer(5)]],                          \
      const constant int &index_key_heads [[buffer(6)]],                      \
      const constant int &cache_block_size [[buffer(7)]],                     \
      const constant int &sparse_block_size [[buffer(8)]],                    \
      const constant int &max_num_sparse_blocks [[buffer(9)]],                \
      const constant int &max_num_blocks_per_seq [[buffer(10)]],              \
      const constant int &sparse_init_blocks [[buffer(11)]],                  \
      const constant int &sparse_local_blocks [[buffer(12)]],                 \
      const constant float &scale [[buffer(13)]],                             \
      const constant int &q_stride [[buffer(14)]],                            \
      const constant int &q_head_stride [[buffer(15)]],                       \
      const constant int &cache_block_stride [[buffer(16)]],                  \
      const constant int &cache_head_stride [[buffer(17)]],                   \
      const constant int &cache_token_stride [[buffer(18)]],                  \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_msa_block_scores_heads(type, num_threads,               \
                                              num_simd_lanes)                 \
  instantiate_msa_block_scores_inner(type, 4, num_threads, num_simd_lanes); \
  instantiate_msa_block_scores_inner(type, 8, num_threads, num_simd_lanes); \
  instantiate_msa_block_scores_inner(type, 16, num_threads,                 \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 32, num_threads,                 \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 64, num_threads,                 \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 80, num_threads,                 \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 96, num_threads,                 \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 112, num_threads,                \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 120, num_threads,                \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 128, num_threads,                \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 192, num_threads,                \
                                        num_simd_lanes);                      \
  instantiate_msa_block_scores_inner(type, 256, num_threads,                \
                                        num_simd_lanes);

#define instantiate_msa_block_scores(type, num_simd_lanes)                 \
  instantiate_msa_block_scores_heads(type, 256, num_simd_lanes);

instantiate_msa_block_scores(float, 32);
instantiate_msa_block_scores(bfloat16_t, 32);
instantiate_msa_block_scores(half, 32);
