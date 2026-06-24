#include <algorithm>
#include <sstream>
#include <string>

#include "dsa_indexer.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "utils.h"

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

mx::array dsa_indexer_scores_with_update(
    const mx::array& index_query,
    const mx::array& index_key_update,
    const mx::array& index_key_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const mx::array& index_weights,
    const int64_t max_context_len,
    mx::StreamOrDevice s /* = {} */
) {
    mx::Shape out_shape{
        index_query.shape(0),
        static_cast<mx::ShapeElem>(max_context_len)};
    const std::vector<mx::array> inputs = {
        index_query,
        index_key_update,
        index_key_cache,
        block_tables,
        seq_lens,
        index_weights};
    return mx::array(
        out_shape,
        mx::float32,
        std::make_shared<DSAIndexerScoresWithUpdate>(
            to_stream(s), max_context_len),
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

void DSAIndexerScoresWithUpdate::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void DSAIndexerScoresWithUpdate::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 6);
    auto& index_query = inputs[0];
    auto& index_key_update = inputs[1];
    auto& index_key_cache = inputs[2];
    auto& block_tables = inputs[3];
    auto& seq_lens = inputs[4];
    auto& index_weights = inputs[5];
    auto& out = outputs[0];

    const int head_size = index_query.shape(2);
    if (head_size > 256) {
      std::ostringstream msg;
      msg << "DSAIndexerScoresWithUpdate supports head_size <= 256, got "
          << head_size;
      throw std::runtime_error(msg.str());
    }
    if (index_query.dtype() != index_key_cache.dtype() ||
        index_key_update.dtype() != index_key_cache.dtype()) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate requires query, update, and cache to share dtype");
    }
    if (index_weights.dtype() != mx::float32) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate requires float32 index_weights");
    }
    if (index_key_cache.shape(4) != head_size ||
        index_key_update.shape(2) != head_size) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate cache/update dims must match query dim");
    }
    if (index_key_update.shape(0) != index_query.shape(0) ||
        index_weights.shape(0) != index_query.shape(0)) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate batch dimensions must match");
    }
    if (index_weights.shape(1) != index_query.shape(1)) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate weights heads must match query heads");
    }
    if (index_key_update.shape(1) != index_key_cache.shape(2)) {
      throw std::runtime_error(
          "DSAIndexerScoresWithUpdate update heads must match cache heads");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto& compute_encoder = mx::metal::get_command_encoder(s);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    constexpr int num_threads = 256;
    constexpr int num_simd_lanes = 32;
    const int64_t num_seqs = index_query.shape(0);
    const int64_t index_heads = index_query.shape(1);
    const int64_t index_key_heads = index_key_cache.shape(2);
    const int64_t cache_block_size = index_key_cache.shape(3);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);

    std::string store_name =
        "store_sparse_index_key_" + get_type_string(index_key_cache.dtype());
    auto store_kernel = d.get_kernel(store_name, lib);
    compute_encoder.set_compute_pipeline_state(store_kernel);

    int32_t update_heads_32 = static_cast<int32_t>(index_key_update.shape(1));
    int32_t head_size_32 = static_cast<int32_t>(head_size);
    int32_t cache_block_size_32 = static_cast<int32_t>(cache_block_size);
    int32_t max_num_blocks_per_seq_32 =
        static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t update_stride_32 = static_cast<int32_t>(index_key_update.strides(0));
    int32_t update_head_stride_32 =
        static_cast<int32_t>(index_key_update.strides(1));
    int32_t cache_block_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(1));
    int32_t cache_head_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(2));
    int32_t cache_token_stride_32 =
        static_cast<int32_t>(index_key_cache.strides(3));

    compute_encoder.set_input_array(index_key_update, 0);
    compute_encoder.set_input_array(index_key_cache, 1);
    compute_encoder.register_output_array(index_key_cache);
    compute_encoder.set_input_array(block_tables, 2);
    compute_encoder.set_input_array(seq_lens, 3);
    compute_encoder.set_bytes(update_heads_32, 4);
    compute_encoder.set_bytes(head_size_32, 5);
    compute_encoder.set_bytes(cache_block_size_32, 6);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 7);
    compute_encoder.set_bytes(update_stride_32, 8);
    compute_encoder.set_bytes(update_head_stride_32, 9);
    compute_encoder.set_bytes(cache_block_stride_32, 10);
    compute_encoder.set_bytes(cache_head_stride_32, 11);
    compute_encoder.set_bytes(cache_token_stride_32, 12);

    const uint64_t store_threads =
        std::min<uint64_t>(512, index_key_update.shape(1) * head_size);
    MTL::Size store_grid = MTL::Size(num_seqs, 1, 1);
    MTL::Size store_threadgroup = MTL::Size(store_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(store_grid, store_threadgroup);
    compute_encoder.barrier();

    std::string score_name =
        "dsa_indexer_scores_" + get_type_string(index_query.dtype());
    score_name += "_hs" + std::to_string(head_size);
    score_name += "_nt" + std::to_string(num_threads);
    score_name += "_nsl" + std::to_string(num_simd_lanes);
    auto score_kernel = d.get_kernel(score_name, lib);
    compute_encoder.set_compute_pipeline_state(score_kernel);

    int32_t index_heads_32 = static_cast<int32_t>(index_heads);
    int32_t index_key_heads_32 = static_cast<int32_t>(index_key_heads);
    int32_t max_context_len_32 = static_cast<int32_t>(max_context_len_);
    int32_t q_stride_32 = static_cast<int32_t>(index_query.strides(0));
    int32_t q_head_stride_32 = static_cast<int32_t>(index_query.strides(1));
    int32_t weights_stride_32 = static_cast<int32_t>(index_weights.strides(0));
    int32_t weights_head_stride_32 =
        static_cast<int32_t>(index_weights.strides(1));

    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(index_query, 1);
    compute_encoder.set_input_array(index_key_cache, 2);
    compute_encoder.set_input_array(block_tables, 3);
    compute_encoder.set_input_array(seq_lens, 4);
    compute_encoder.set_input_array(index_weights, 5);
    compute_encoder.set_bytes(index_heads_32, 6);
    compute_encoder.set_bytes(index_key_heads_32, 7);
    compute_encoder.set_bytes(cache_block_size_32, 8);
    compute_encoder.set_bytes(max_context_len_32, 9);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 10);
    compute_encoder.set_bytes(q_stride_32, 11);
    compute_encoder.set_bytes(q_head_stride_32, 12);
    compute_encoder.set_bytes(cache_block_stride_32, 13);
    compute_encoder.set_bytes(cache_head_stride_32, 14);
    compute_encoder.set_bytes(cache_token_stride_32, 15);
    compute_encoder.set_bytes(weights_stride_32, 16);
    compute_encoder.set_bytes(weights_head_stride_32, 17);

    MTL::Size score_grid = MTL::Size(max_context_len_, num_seqs, 1);
    MTL::Size score_threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(score_grid, score_threadgroup);
}

bool DSAIndexerScoresWithUpdate::is_equivalent(
    const mx::Primitive& other) const {
  const DSAIndexerScoresWithUpdate& r_other =
      static_cast<const DSAIndexerScoresWithUpdate&>(other);
  return max_context_len_ == r_other.max_context_len_;
}

} // namespace parallax_ext
