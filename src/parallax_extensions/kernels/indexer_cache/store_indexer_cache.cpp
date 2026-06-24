#include <algorithm>
#include <string>

#include "store_indexer_cache.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "../common/utils.h"

namespace parallax_ext {

mx::array store_indexer_cache(
    const mx::array& index_key,
    const mx::array& index_key_cache,
    const mx::array& slot_mapping,
    mx::StreamOrDevice s /* = {} */
) {
    const std::vector<mx::array> inputs = {
        index_key,
        index_key_cache,
        slot_mapping};
    return mx::array(
        index_key.shape(),
        index_key.dtype(),
        std::make_shared<StoreIndexerCache>(to_stream(s)),
        inputs);
}

void StoreIndexerCache::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void StoreIndexerCache::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 3);
    auto& index_key = inputs[0];
    auto& index_key_cache = inputs[1];
    auto& slot_mapping = inputs[2];
    auto& out = outputs[0];

    if (index_key.dtype() != index_key_cache.dtype()) {
      throw std::runtime_error(
          "StoreIndexerCache requires index key and cache to share dtype");
    }
    if (index_key.shape().size() != 3 || index_key_cache.shape().size() != 5) {
      throw std::runtime_error(
          "StoreIndexerCache expects index_key [tokens, heads, dim] and "
          "cache [1, blocks, heads, block_size, dim]");
    }
    if (index_key_cache.shape(0) != 1) {
      throw std::runtime_error("StoreIndexerCache expects a single-layer cache tensor");
    }
    if (slot_mapping.shape().size() != 1 ||
        slot_mapping.shape(0) != index_key.shape(0)) {
      throw std::runtime_error(
          "StoreIndexerCache slot_mapping must be shaped [num_tokens]");
    }
    if (index_key.shape(1) != index_key_cache.shape(2)) {
      throw std::runtime_error(
          "StoreIndexerCache index key heads must match cache heads; "
          "shared-key caches should be allocated with one key head");
    }
    if (index_key.shape(2) != index_key_cache.shape(4)) {
      throw std::runtime_error(
          "StoreIndexerCache index key dim must match cache dim");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    const int64_t num_tokens = index_key.shape(0);
    const int64_t index_key_heads = index_key.shape(1);
    const int64_t head_size = index_key.shape(2);
    const int64_t block_size = index_key_cache.shape(3);
    const int64_t num_blocks = index_key_cache.shape(1);

    std::string kname =
        "store_indexer_cache_" + get_type_string(index_key.dtype());
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    int32_t key_stride_32 = static_cast<int32_t>(index_key.strides(0));
    int32_t key_head_stride_32 = static_cast<int32_t>(index_key.strides(1));
    int32_t cache_block_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(1));
    int32_t cache_head_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(2));
    int32_t cache_token_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(3));
    int32_t index_key_heads_32 = static_cast<int32_t>(index_key_heads);
    int32_t head_size_32 = static_cast<int32_t>(head_size);
    int32_t block_size_32 = static_cast<int32_t>(block_size);
    int32_t num_blocks_32 = static_cast<int32_t>(num_blocks);

    compute_encoder.set_input_array(index_key, 0);
    compute_encoder.set_input_array(index_key_cache, 1);
    compute_encoder.register_output_array(index_key_cache);
    compute_encoder.set_input_array(slot_mapping, 2);
    compute_encoder.set_bytes(key_stride_32, 3);
    compute_encoder.set_bytes(key_head_stride_32, 4);
    compute_encoder.set_bytes(cache_block_stride_32, 5);
    compute_encoder.set_bytes(cache_head_stride_32, 6);
    compute_encoder.set_bytes(cache_token_stride_32, 7);
    compute_encoder.set_bytes(index_key_heads_32, 8);
    compute_encoder.set_bytes(head_size_32, 9);
    compute_encoder.set_bytes(block_size_32, 10);
    compute_encoder.set_bytes(num_blocks_32, 11);

    const uint64_t num_threads =
        std::min<uint64_t>(512, index_key_heads * head_size);
    MTL::Size grid = MTL::Size(num_tokens, 1, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

bool StoreIndexerCache::is_equivalent(const mx::Primitive& other) const {
  const StoreIndexerCache& r_other =
      static_cast<const StoreIndexerCache&>(other);
  return true;
}

} // namespace parallax_ext
