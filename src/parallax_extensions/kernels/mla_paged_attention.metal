#include "./utils.metal"
#include <metal_stdlib>

using namespace metal;

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

template <typename T, typename CACHE_T, int LATENT_DIM, int ROPE_DIM,
          int NUM_THREADS, int NUM_SIMD_LANES>
[[kernel]] void mla_paged_attention(
    device T *out [[buffer(0)]],
    device const T *q_latent [[buffer(1)]],
    device const T *q_pe [[buffer(2)]],
    device const CACHE_T *latent_cache [[buffer(3)]],
    device const CACHE_T *rope_cache [[buffer(4)]],
    device const int32_t *block_tables [[buffer(5)]],
    device const int32_t *context_lens [[buffer(6)]],
    const constant float &scale [[buffer(7)]],
    const constant int &block_size [[buffer(8)]],
    const constant int &max_num_blocks_per_seq [[buffer(9)]],
    const constant int &q_latent_seq_stride [[buffer(10)]],
    const constant int &q_latent_head_stride [[buffer(11)]],
    const constant int &q_pe_seq_stride [[buffer(12)]],
    const constant int &q_pe_head_stride [[buffer(13)]],
    const constant int &latent_block_stride [[buffer(14)]],
    const constant int &latent_token_stride [[buffer(15)]],
    const constant int &rope_block_stride [[buffer(16)]],
    const constant int &rope_token_stride [[buffer(17)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  const int head_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int tid = thread_position_in_threadgroup.x;
  const int context_len = context_lens[seq_idx];
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, block_size);

  const device T *q_latent_ptr =
      q_latent + seq_idx * q_latent_seq_stride + head_idx * q_latent_head_stride;
  const device T *q_pe_ptr =
      q_pe + seq_idx * q_pe_seq_stride + head_idx * q_pe_head_stride;

  float q_latent_vec[16] = {0.0f};
  float q_pe_vec[4] = {0.0f};
  float acc_vec[16] = {0.0f};

  for (int i = tid; i < LATENT_DIM; i += NUM_SIMD_LANES) {
    q_latent_vec[i / NUM_SIMD_LANES] = (float)q_latent_ptr[i];
  }
  for (int i = tid; i < ROPE_DIM; i += NUM_SIMD_LANES) {
    q_pe_vec[i / NUM_SIMD_LANES] = (float)q_pe_ptr[i];
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  const device int32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = 0; block_idx < num_context_blocks; ++block_idx) {
    const int physical_block_number = block_table[block_idx];
    const int block_start = block_idx * block_size;
    const int tokens_in_block = min(block_size, context_len - block_start);

    const device CACHE_T *latent_block =
        latent_cache + (int64_t)physical_block_number * latent_block_stride;
    const device CACHE_T *rope_block =
        rope_cache + (int64_t)physical_block_number * rope_block_stride;

    for (int token_offset = 0; token_offset < tokens_in_block; ++token_offset) {
      const device CACHE_T *latent_ptr =
          latent_block + token_offset * latent_token_stride;
      const device CACHE_T *rope_ptr =
          rope_block + token_offset * rope_token_stride;

      float score = 0.0f;
      for (int i = tid; i < LATENT_DIM; i += NUM_SIMD_LANES) {
        score += q_latent_vec[i / NUM_SIMD_LANES] * (float)latent_ptr[i];
      }
      for (int i = tid; i < ROPE_DIM; i += NUM_SIMD_LANES) {
        score += q_pe_vec[i / NUM_SIMD_LANES] * (float)rope_ptr[i];
      }
      score = simd_sum(score) * scale;

      const float m_prev = m_i;
      m_i = max(m_prev, score);
      const float alpha = exp(m_prev - m_i);
      const float beta = exp(score - m_i);
      l_i = l_i * alpha + beta;

      for (int i = tid; i < LATENT_DIM; i += NUM_SIMD_LANES) {
        const int idx = i / NUM_SIMD_LANES;
        acc_vec[idx] = acc_vec[idx] * alpha + (float)latent_ptr[i] * beta;
      }
    }
  }

  device T *out_ptr =
      out + (seq_idx * threadgroups_per_grid.x + head_idx) * LATENT_DIM;

  const float inv_l = l_i > 0.0f ? 1.0f / l_i : 0.0f;
  for (int i = tid; i < LATENT_DIM; i += NUM_SIMD_LANES) {
    out_ptr[i] = T(acc_vec[i / NUM_SIMD_LANES] * inv_l);
  }
}

#define instantiate_mla_paged_attention_inner(                                \
    type, cache_type, latent_dim, rope_dim, num_threads, num_simd_lanes)       \
  template [[host_name("mla_paged_attention_" #type "_cache_" #cache_type     \
                       "_ld" #latent_dim "_rd" #rope_dim "_nt" #num_threads  \
                       "_nsl" #num_simd_lanes)]] [[kernel]] void              \
  mla_paged_attention<type, cache_type, latent_dim, rope_dim, num_threads,    \
                      num_simd_lanes>(                                        \
      device type *out [[buffer(0)]], device const type *q_latent             \
      [[buffer(1)]], device const type *q_pe [[buffer(2)]],                   \
      device const cache_type *latent_cache [[buffer(3)]],                    \
      device const cache_type *rope_cache [[buffer(4)]],                      \
      device const int32_t *block_tables [[buffer(5)]],                       \
      device const int32_t *context_lens [[buffer(6)]],                       \
      const constant float &scale [[buffer(7)]],                              \
      const constant int &block_size [[buffer(8)]],                           \
      const constant int &max_num_blocks_per_seq [[buffer(9)]],               \
      const constant int &q_latent_seq_stride [[buffer(10)]],                 \
      const constant int &q_latent_head_stride [[buffer(11)]],                \
      const constant int &q_pe_seq_stride [[buffer(12)]],                     \
      const constant int &q_pe_head_stride [[buffer(13)]],                    \
      const constant int &latent_block_stride [[buffer(14)]],                 \
      const constant int &latent_token_stride [[buffer(15)]],                 \
      const constant int &rope_block_stride [[buffer(16)]],                   \
      const constant int &rope_token_stride [[buffer(17)]],                   \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                  \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]);

#define instantiate_mla_rope_dims(type, cache_type, latent_dim, num_threads,  \
                                  num_simd_lanes)                             \
  instantiate_mla_paged_attention_inner(type, cache_type, latent_dim, 2,      \
                                        num_threads, num_simd_lanes);         \
  instantiate_mla_paged_attention_inner(type, cache_type, latent_dim, 32,     \
                                        num_threads, num_simd_lanes);         \
  instantiate_mla_paged_attention_inner(type, cache_type, latent_dim, 64,     \
                                        num_threads, num_simd_lanes);         \
  instantiate_mla_paged_attention_inner(type, cache_type, latent_dim, 128,    \
                                        num_threads, num_simd_lanes);

#define instantiate_mla_latent_dims(type, cache_type, num_threads,            \
                                    num_simd_lanes)                           \
  instantiate_mla_rope_dims(type, cache_type, 4, num_threads,                 \
                            num_simd_lanes);                                  \
  instantiate_mla_rope_dims(type, cache_type, 64, num_threads,                \
                            num_simd_lanes);                                  \
  instantiate_mla_rope_dims(type, cache_type, 128, num_threads,               \
                            num_simd_lanes);                                  \
  instantiate_mla_rope_dims(type, cache_type, 256, num_threads,               \
                            num_simd_lanes);                                  \
  instantiate_mla_rope_dims(type, cache_type, 512, num_threads,               \
                            num_simd_lanes);

#define instantiate_mla_paged_attention(type, cache_type, num_simd_lanes)     \
  instantiate_mla_latent_dims(type, cache_type, 32, num_simd_lanes);

instantiate_mla_paged_attention(float, float, 32);
instantiate_mla_paged_attention(bfloat16_t, bfloat16_t, 32);
instantiate_mla_paged_attention(half, half, 32);
