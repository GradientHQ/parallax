#include "./utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
                       uint simd_lid) {
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
  }

#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  return simd_shuffle(sum, 0);
}

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

template <typename T, typename CACHE_T, int HEAD_SIZE, int NUM_THREADS,
          int NUM_SIMD_LANES>
[[kernel]] void sparse_paged_attention(
    device T *out [[buffer(0)]], device const T *q [[buffer(1)]],
    device const CACHE_T *k_cache [[buffer(2)]],
    device const CACHE_T *v_cache [[buffer(3)]],
    device const uint32_t *block_tables [[buffer(4)]],
    device const uint32_t *context_lens [[buffer(5)]],
    device const int32_t *token_positions [[buffer(6)]],
    device const int32_t *token_positions_valid [[buffer(7)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant int &block_size [[buffer(10)]],
    const constant int &max_num_blocks_per_seq [[buffer(11)]],
    const constant int &max_num_positions [[buffer(12)]],
    const constant int &q_stride [[buffer(13)]],
    const constant int &kv_block_stride [[buffer(14)]],
    const constant int &kv_head_stride [[buffer(15)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int seq_idx = threadgroup_position_in_grid.y;
  const int head_idx = threadgroup_position_in_grid.x;
  const int thread_idx = thread_position_in_threadgroup.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const int context_len = context_lens[seq_idx];

  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float *q_smem = logits + max_num_positions;
  threadgroup float red_smem[2 * (NUM_THREADS / NUM_SIMD_LANES)];
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;
  const device T *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
    q_smem[i] = (float)q_ptr[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float qk_max = -FLT_MAX;
  constexpr int x = 16 / sizeof(CACHE_T);
  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  const int position_base = seq_idx * max_num_positions;

  for (int p = thread_idx; p < max_num_positions; p += NUM_THREADS) {
    const int pos_offset = position_base + p;
    float qk = -INFINITY;

    if (token_positions_valid[pos_offset] != 0) {
      const int token_pos = token_positions[pos_offset];
      if (token_pos >= 0 && token_pos < context_len) {
        const int logical_block_idx = token_pos / block_size;
        if (logical_block_idx >= 0 &&
            logical_block_idx < max_num_blocks_per_seq) {
          const int block_offset = token_pos - logical_block_idx * block_size;
          const int64_t physical_block_number =
              static_cast<int64_t>(block_table[logical_block_idx]);

          const device CACHE_T *k_ptr =
              k_cache + physical_block_number * kv_block_stride +
              kv_head_idx * kv_head_stride;
          qk = 0.f;
          for (int i = 0; i < HEAD_SIZE; ++i) {
            const int x_idx = i / x;
            const int x_offset = i % x;
            const int64_t k_offset =
                x_idx * block_size * x + block_offset * x + x_offset;
            qk += q_smem[i] * (float)k_ptr[k_offset];
          }
          qk *= scale;
        }
      }
    }
    logits[p] = qk;
    qk_max = max(qk_max, qk);
  }

#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  qk_max = simd_shuffle(qk_max, 0);

  float exp_sum = 0.f;
  for (int p = thread_idx; p < max_num_positions; p += NUM_THREADS) {
    float val = exp(logits[p] - qk_max);
    logits[p] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum,
                                                 simd_tid, simd_lid);
  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int p = thread_idx; p < max_num_positions; p += NUM_THREADS) {
    logits[p] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_SIMD_LANES);
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
    accs[i] = 0.f;
  }

  for (int p = warp_idx; p < max_num_positions; p += NUM_WARPS) {
    const float weight = logits[p];
    if (weight == 0.f) {
      continue;
    }

    const int pos_offset = position_base + p;
    if (token_positions_valid[pos_offset] == 0) {
      continue;
    }

    const int token_pos = token_positions[pos_offset];
    if (token_pos < 0 || token_pos >= context_len) {
      continue;
    }

    const int logical_block_idx = token_pos / block_size;
    if (logical_block_idx < 0 ||
        logical_block_idx >= max_num_blocks_per_seq) {
      continue;
    }
    const int block_offset = token_pos - logical_block_idx * block_size;
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[logical_block_idx]);
    const device CACHE_T *v_ptr =
        v_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride;

#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      const int row_idx = lane + i * NUM_SIMD_LANES;
      if (row_idx < HEAD_SIZE) {
        accs[i] += weight * (float)v_ptr[row_idx * block_size + block_offset];
      }
    }
  }

  // Reuse dynamic shared memory to reduce partial V accumulators across SIMD
  // groups, matching the reduction pattern used by paged_attention_v1.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float *out_smem =
      reinterpret_cast<threadgroup float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    const int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; ++j) {
        const int row_idx = lane + j * NUM_SIMD_LANES;
        if (row_idx < HEAD_SIZE) {
          dst[row_idx] = accs[j];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; ++j) {
        const int row_idx = lane + j * NUM_SIMD_LANES;
        if (row_idx < HEAD_SIZE) {
          accs[j] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_idx == 0) {
    device T *out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      const int row_idx = lane + i * NUM_SIMD_LANES;
      if (row_idx < HEAD_SIZE) {
        out_ptr[row_idx] = T(accs[i]);
      }
    }
  }
}

#define instantiate_sparse_paged_attention_inner(                             \
    type, cache_type, head_size, num_threads, num_simd_lanes)                  \
  template [[host_name("sparse_paged_attention_" #type "_cache_" #cache_type  \
                       "_hs" #head_size "_nt" #num_threads                    \
                       "_nsl" #num_simd_lanes)]] [[kernel]] void               \
  sparse_paged_attention<type, cache_type, head_size, num_threads,            \
                          num_simd_lanes>(                                    \
      device type *out [[buffer(0)]], device const type *q [[buffer(1)]],      \
      device const cache_type *k_cache [[buffer(2)]],                          \
      device const cache_type *v_cache [[buffer(3)]],                          \
      device const uint32_t *block_tables [[buffer(4)]],                       \
      device const uint32_t *context_lens [[buffer(5)]],                       \
      device const int32_t *token_positions [[buffer(6)]],                     \
      device const int32_t *token_positions_valid [[buffer(7)]],               \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant int &block_size [[buffer(10)]],                           \
      const constant int &max_num_blocks_per_seq [[buffer(11)]],               \
      const constant int &max_num_positions [[buffer(12)]],                    \
      const constant int &q_stride [[buffer(13)]],                             \
      const constant int &kv_block_stride [[buffer(14)]],                      \
      const constant int &kv_head_stride [[buffer(15)]],                       \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_sparse_paged_attention_heads(type, cache_type,            \
                                                  num_threads, num_simd_lanes) \
  instantiate_sparse_paged_attention_inner(type, cache_type, 4, num_threads,  \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 8, num_threads,  \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 16, num_threads, \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 32, num_threads, \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 64, num_threads, \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 80, num_threads, \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 96, num_threads, \
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 112, num_threads,\
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 120, num_threads,\
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 128, num_threads,\
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 192, num_threads,\
                                            num_simd_lanes);                   \
  instantiate_sparse_paged_attention_inner(type, cache_type, 256, num_threads,\
                                            num_simd_lanes);

#define instantiate_sparse_paged_attention(type, cache_type, num_simd_lanes)  \
  instantiate_sparse_paged_attention_heads(type, cache_type, 256,             \
                                            num_simd_lanes);

instantiate_sparse_paged_attention(float, float, 32);
instantiate_sparse_paged_attention(bfloat16_t, bfloat16_t, 32);
instantiate_sparse_paged_attention(half, half, 32);
