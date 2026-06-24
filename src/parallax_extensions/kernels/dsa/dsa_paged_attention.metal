#include "../common/utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float dsa_block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
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

template <typename T, typename CACHE_T, int LATENT_DIM, int ROPE_DIM,
          int NUM_THREADS, int NUM_SIMD_LANES>
[[kernel]] void dsa_paged_attention(
    device T *out [[buffer(0)]],
    device const T *q_latent [[buffer(1)]],
    device const T *q_pe [[buffer(2)]],
    device const CACHE_T *latent_cache [[buffer(3)]],
    device const CACHE_T *rope_cache [[buffer(4)]],
    device const int32_t *block_tables [[buffer(5)]],
    device const int32_t *context_lens [[buffer(6)]],
    device const int32_t *topk_indices [[buffer(7)]],
    const constant float &scale [[buffer(8)]],
    const constant int &block_size [[buffer(9)]],
    const constant int &max_num_blocks_per_seq [[buffer(10)]],
    const constant int &max_num_positions [[buffer(11)]],
    const constant int &q_latent_seq_stride [[buffer(12)]],
    const constant int &q_latent_head_stride [[buffer(13)]],
    const constant int &q_pe_seq_stride [[buffer(14)]],
    const constant int &q_pe_head_stride [[buffer(15)]],
    const constant int &latent_block_stride [[buffer(16)]],
    const constant int &latent_token_stride [[buffer(17)]],
    const constant int &rope_block_stride [[buffer(18)]],
    const constant int &rope_token_stride [[buffer(19)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int seq_idx = threadgroup_position_in_grid.y;
  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int thread_idx = thread_position_in_threadgroup.x;
  const int context_len = context_lens[seq_idx];
  const int position_base = seq_idx * max_num_positions;
  const bool dense_context = topk_indices[position_base] < 0;

  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float *q_latent_smem = logits + max_num_positions;
  threadgroup float *q_pe_smem = q_latent_smem + LATENT_DIM;
  threadgroup float red_smem[2 * (NUM_THREADS / NUM_SIMD_LANES)];

  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const device T *q_latent_ptr =
      q_latent + seq_idx * q_latent_seq_stride + head_idx * q_latent_head_stride;
  const device T *q_pe_ptr =
      q_pe + seq_idx * q_pe_seq_stride + head_idx * q_pe_head_stride;

  for (int i = thread_idx; i < LATENT_DIM; i += NUM_THREADS) {
    q_latent_smem[i] = (float)q_latent_ptr[i];
  }
  for (int i = thread_idx; i < ROPE_DIM; i += NUM_THREADS) {
    q_pe_smem[i] = (float)q_pe_ptr[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device int32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  float qk_max = -FLT_MAX;
  for (int p = thread_idx; p < max_num_positions; p += NUM_THREADS) {
    int token_pos = dense_context ? p : topk_indices[position_base + p];
    float qk = -INFINITY;

    if (token_pos >= 0 && token_pos < context_len) {
      const int logical_block_idx = token_pos / block_size;
      if (logical_block_idx >= 0 && logical_block_idx < max_num_blocks_per_seq) {
        const int block_offset = token_pos - logical_block_idx * block_size;
        const int physical_block_number = block_table[logical_block_idx];

        const device CACHE_T *latent_ptr =
            latent_cache + (int64_t)physical_block_number * latent_block_stride +
            block_offset * latent_token_stride;
        const device CACHE_T *rope_ptr =
            rope_cache + (int64_t)physical_block_number * rope_block_stride +
            block_offset * rope_token_stride;

        qk = 0.f;
#pragma unroll
        for (int i = 0; i < LATENT_DIM; ++i) {
          qk += q_latent_smem[i] * (float)latent_ptr[i];
        }
#pragma unroll
        for (int i = 0; i < ROPE_DIM; ++i) {
          qk += q_pe_smem[i] * (float)rope_ptr[i];
        }
        qk *= scale;
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
  exp_sum = dsa_block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], exp_sum, simd_tid, simd_lid);
  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int p = thread_idx; p < max_num_positions; p += NUM_THREADS) {
    logits[p] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(LATENT_DIM, NUM_SIMD_LANES);
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

    const int token_pos = dense_context ? p : topk_indices[position_base + p];
    if (token_pos < 0 || token_pos >= context_len) {
      continue;
    }

    const int logical_block_idx = token_pos / block_size;
    if (logical_block_idx < 0 || logical_block_idx >= max_num_blocks_per_seq) {
      continue;
    }
    const int block_offset = token_pos - logical_block_idx * block_size;
    const int physical_block_number = block_table[logical_block_idx];
    const device CACHE_T *latent_ptr =
        latent_cache + (int64_t)physical_block_number * latent_block_stride +
        block_offset * latent_token_stride;

#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      const int row_idx = lane + i * NUM_SIMD_LANES;
      if (row_idx < LATENT_DIM) {
        accs[i] += weight * (float)latent_ptr[row_idx];
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float *out_smem = reinterpret_cast<threadgroup float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    const int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float *dst = &out_smem[(warp_idx - mid) * LATENT_DIM];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; ++j) {
        const int row_idx = lane + j * NUM_SIMD_LANES;
        if (row_idx < LATENT_DIM) {
          dst[row_idx] = accs[j];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * LATENT_DIM];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; ++j) {
        const int row_idx = lane + j * NUM_SIMD_LANES;
        if (row_idx < LATENT_DIM) {
          accs[j] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_idx == 0) {
    device T *out_ptr =
        out + seq_idx * num_heads * LATENT_DIM + head_idx * LATENT_DIM;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      const int row_idx = lane + i * NUM_SIMD_LANES;
      if (row_idx < LATENT_DIM) {
        out_ptr[row_idx] = T(accs[i]);
      }
    }
  }
}

#define instantiate_dsa_paged_attention_inner(                                \
    type, cache_type, latent_dim, rope_dim, num_threads, num_simd_lanes)       \
  template [[host_name("dsa_paged_attention_" #type "_cache_" #cache_type     \
                       "_ld" #latent_dim "_rd" #rope_dim "_nt" #num_threads  \
                       "_nsl" #num_simd_lanes)]] [[kernel]] void              \
  dsa_paged_attention<type, cache_type, latent_dim, rope_dim, num_threads,    \
                      num_simd_lanes>(                                        \
      device type *out [[buffer(0)]], device const type *q_latent             \
      [[buffer(1)]], device const type *q_pe [[buffer(2)]],                   \
      device const cache_type *latent_cache [[buffer(3)]],                    \
      device const cache_type *rope_cache [[buffer(4)]],                      \
      device const int32_t *block_tables [[buffer(5)]],                       \
      device const int32_t *context_lens [[buffer(6)]],                       \
      device const int32_t *topk_indices [[buffer(7)]],                       \
      const constant float &scale [[buffer(8)]],                              \
      const constant int &block_size [[buffer(9)]],                           \
      const constant int &max_num_blocks_per_seq [[buffer(10)]],              \
      const constant int &max_num_positions [[buffer(11)]],                   \
      const constant int &q_latent_seq_stride [[buffer(12)]],                 \
      const constant int &q_latent_head_stride [[buffer(13)]],                \
      const constant int &q_pe_seq_stride [[buffer(14)]],                     \
      const constant int &q_pe_head_stride [[buffer(15)]],                    \
      const constant int &latent_block_stride [[buffer(16)]],                 \
      const constant int &latent_token_stride [[buffer(17)]],                 \
      const constant int &rope_block_stride [[buffer(18)]],                   \
      const constant int &rope_token_stride [[buffer(19)]],                   \
      threadgroup char *shared_mem [[threadgroup(0)]],                        \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                  \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_dsa_rope_dims(type, cache_type, latent_dim, num_threads,  \
                                  num_simd_lanes)                             \
  instantiate_dsa_paged_attention_inner(type, cache_type, latent_dim, 2,      \
                                        num_threads, num_simd_lanes);         \
  instantiate_dsa_paged_attention_inner(type, cache_type, latent_dim, 32,     \
                                        num_threads, num_simd_lanes);         \
  instantiate_dsa_paged_attention_inner(type, cache_type, latent_dim, 64,     \
                                        num_threads, num_simd_lanes);         \
  instantiate_dsa_paged_attention_inner(type, cache_type, latent_dim, 128,    \
                                        num_threads, num_simd_lanes);

#define instantiate_dsa_latent_dims(type, cache_type, num_threads,            \
                                    num_simd_lanes)                           \
  instantiate_dsa_rope_dims(type, cache_type, 4, num_threads,                 \
                            num_simd_lanes);                                  \
  instantiate_dsa_rope_dims(type, cache_type, 64, num_threads,                \
                            num_simd_lanes);                                  \
  instantiate_dsa_rope_dims(type, cache_type, 128, num_threads,               \
                            num_simd_lanes);                                  \
  instantiate_dsa_rope_dims(type, cache_type, 256, num_threads,               \
                            num_simd_lanes);                                  \
  instantiate_dsa_rope_dims(type, cache_type, 512, num_threads,               \
                            num_simd_lanes);

#define instantiate_dsa_paged_attention(type, cache_type, num_simd_lanes)     \
  instantiate_dsa_latent_dims(type, cache_type, 256, num_simd_lanes);

instantiate_dsa_paged_attention(float, float, 32);
instantiate_dsa_paged_attention(bfloat16_t, bfloat16_t, 32);
instantiate_dsa_paged_attention(half, half, 32);
