#include "../common/utils.metal"
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
