#include "./utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void store_indexer_cache(
    device const T *index_key [[buffer(0)]],
    device T *index_key_cache [[buffer(1)]],
    device const int64_t *slot_mapping [[buffer(2)]],
    const constant int &key_stride [[buffer(3)]],
    const constant int &key_head_stride [[buffer(4)]],
    const constant int &cache_block_stride [[buffer(5)]],
    const constant int &cache_head_stride [[buffer(6)]],
    const constant int &cache_token_stride [[buffer(7)]],
    const constant int &index_key_heads [[buffer(8)]],
    const constant int &head_size [[buffer(9)]],
    const constant int &block_size [[buffer(10)]],
    const constant int &num_blocks [[buffer(11)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  if (block_idx < 0 || block_idx >= num_blocks) {
    return;
  }
  const int64_t block_offset = slot_idx - block_idx * block_size;

  const int total = index_key_heads * head_size;
  for (int i = tid; i < total; i += threads_per_threadgroup) {
    const int head_idx = i / head_size;
    const int dim_idx = i - head_idx * head_size;
    const int64_t src_idx =
        token_idx * key_stride + head_idx * key_head_stride + dim_idx;
    const int64_t dst_idx =
        block_idx * cache_block_stride + head_idx * cache_head_stride +
        block_offset * cache_token_stride + dim_idx;
    index_key_cache[dst_idx] = index_key[src_idx];
  }
}

#define instantiate_store_indexer_cache(type)                                 \
  template [[host_name("store_indexer_cache_" #type)]] [[kernel]] void        \
  store_indexer_cache<type>(                                                  \
      device const type *index_key [[buffer(0)]],                             \
      device type *index_key_cache [[buffer(1)]],                             \
      device const int64_t *slot_mapping [[buffer(2)]],                       \
      const constant int &key_stride [[buffer(3)]],                           \
      const constant int &key_head_stride [[buffer(4)]],                      \
      const constant int &cache_block_stride [[buffer(5)]],                   \
      const constant int &cache_head_stride [[buffer(6)]],                    \
      const constant int &cache_token_stride [[buffer(7)]],                   \
      const constant int &index_key_heads [[buffer(8)]],                      \
      const constant int &head_size [[buffer(9)]],                            \
      const constant int &block_size [[buffer(10)]],                          \
      const constant int &num_blocks [[buffer(11)]],                          \
      uint token_idx [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                            \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_store_indexer_cache(float);
instantiate_store_indexer_cache(bfloat16_t);
instantiate_store_indexer_cache(half);

template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float dsa_indexer_block_sum(threadgroup float *red_smem, float sum,
                                   uint simd_tid, uint simd_lid) {
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  } else {
    sum = 0.f;
  }

#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  return simd_shuffle(sum, 0);
}

template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES>
[[kernel]] void dsa_indexer_scores(
    device float *scores [[buffer(0)]],
    device const T *index_query [[buffer(1)]],
    device const T *index_key_cache [[buffer(2)]],
    device const int32_t *block_tables [[buffer(3)]],
    device const int32_t *context_lens [[buffer(4)]],
    device const float *index_weights [[buffer(5)]],
    const constant int &index_heads [[buffer(6)]],
    const constant int &index_key_heads [[buffer(7)]],
    const constant int &cache_block_size [[buffer(8)]],
    const constant int &max_context_len [[buffer(9)]],
    const constant int &max_num_blocks_per_seq [[buffer(10)]],
    const constant int &q_stride [[buffer(11)]],
    const constant int &q_head_stride [[buffer(12)]],
    const constant int &cache_block_stride [[buffer(13)]],
    const constant int &cache_head_stride [[buffer(14)]],
    const constant int &cache_token_stride [[buffer(15)]],
    const constant int &weights_stride [[buffer(16)]],
    const constant int &weights_head_stride [[buffer(17)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  threadgroup float red_smem[NUM_WARPS];

  const int token_pos = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int thread_idx = thread_position_in_threadgroup.x;
  const int context_len = context_lens[seq_idx];
  device float *out = scores + seq_idx * max_context_len + token_pos;

  if (token_pos >= context_len || context_len <= 0) {
    if (thread_idx == 0) {
      *out = -INFINITY;
    }
    return;
  }

  const int logical_block_idx = token_pos / cache_block_size;
  if (logical_block_idx < 0 || logical_block_idx >= max_num_blocks_per_seq) {
    if (thread_idx == 0) {
      *out = -INFINITY;
    }
    return;
  }

  const int block_offset = token_pos - logical_block_idx * cache_block_size;
  const int physical_block =
      block_tables[seq_idx * max_num_blocks_per_seq + logical_block_idx];

  float weighted_score = 0.f;
  for (int query_head_idx = 0; query_head_idx < index_heads; ++query_head_idx) {
    const int key_head_idx =
        index_key_heads == 1 ? 0 : min(query_head_idx, index_key_heads - 1);
    const device T *q_ptr =
        index_query + seq_idx * q_stride + query_head_idx * q_head_stride;
    const device T *k_ptr =
        index_key_cache + (int64_t)physical_block * cache_block_stride +
        key_head_idx * cache_head_stride + block_offset * cache_token_stride;

    float dot = 0.f;
#pragma unroll
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
      dot += (float)q_ptr[i] * (float)k_ptr[i];
    }
    dot = dsa_indexer_block_sum<NUM_WARPS, NUM_SIMD_LANES>(
        red_smem, dot, simd_tid, simd_lid);

    if (thread_idx == 0) {
      const float weight =
          index_weights[seq_idx * weights_stride +
                        query_head_idx * weights_head_stride];
      weighted_score += max(dot, 0.f) * weight;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (thread_idx == 0) {
    *out = weighted_score;
  }
}

#define instantiate_dsa_indexer_scores_inner(type, head_size, num_threads,    \
                                             num_simd_lanes)                  \
  template [[host_name("dsa_indexer_scores_" #type "_hs" #head_size "_nt"    \
                       #num_threads "_nsl" #num_simd_lanes)]] [[kernel]]     \
  void dsa_indexer_scores<type, head_size, num_threads, num_simd_lanes>(      \
      device float *scores [[buffer(0)]],                                     \
      device const type *index_query [[buffer(1)]],                           \
      device const type *index_key_cache [[buffer(2)]],                       \
      device const int32_t *block_tables [[buffer(3)]],                       \
      device const int32_t *context_lens [[buffer(4)]],                       \
      device const float *index_weights [[buffer(5)]],                        \
      const constant int &index_heads [[buffer(6)]],                          \
      const constant int &index_key_heads [[buffer(7)]],                      \
      const constant int &cache_block_size [[buffer(8)]],                     \
      const constant int &max_context_len [[buffer(9)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(10)]],              \
      const constant int &q_stride [[buffer(11)]],                            \
      const constant int &q_head_stride [[buffer(12)]],                       \
      const constant int &cache_block_stride [[buffer(13)]],                  \
      const constant int &cache_head_stride [[buffer(14)]],                   \
      const constant int &cache_token_stride [[buffer(15)]],                  \
      const constant int &weights_stride [[buffer(16)]],                      \
      const constant int &weights_head_stride [[buffer(17)]],                 \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_dsa_indexer_scores_heads(type, num_threads,               \
                                             num_simd_lanes)                  \
  instantiate_dsa_indexer_scores_inner(type, 4, num_threads, num_simd_lanes); \
  instantiate_dsa_indexer_scores_inner(type, 8, num_threads, num_simd_lanes); \
  instantiate_dsa_indexer_scores_inner(type, 16, num_threads,                 \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 32, num_threads,                 \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 64, num_threads,                 \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 80, num_threads,                 \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 96, num_threads,                 \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 112, num_threads,                \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 120, num_threads,                \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 128, num_threads,                \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 192, num_threads,                \
                                       num_simd_lanes);                       \
  instantiate_dsa_indexer_scores_inner(type, 256, num_threads,                \
                                       num_simd_lanes);

#define instantiate_dsa_indexer_scores(type, num_simd_lanes)                  \
  instantiate_dsa_indexer_scores_heads(type, 256, num_simd_lanes);

instantiate_dsa_indexer_scores(float, 32);
instantiate_dsa_indexer_scores(bfloat16_t, 32);
instantiate_dsa_indexer_scores(half, 32);
