#include <algorithm>
#include <sstream>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "sparse_indexer.h"
#include "utils.h"

namespace parallax_ext {

mx::array sparse_token_indexer(
    const mx::array& index_query,
    const mx::array& index_key_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const int64_t max_context_len,
    const int64_t sparse_block_size,
    const int64_t sparse_topk_blocks,
    const int64_t sparse_init_blocks,
    const int64_t sparse_local_blocks,
    const float scale,
    mx::StreamOrDevice s /* = {} */
) {
    const int64_t max_num_sparse_blocks =
        std::max<int64_t>(1, (max_context_len + sparse_block_size - 1) / sparse_block_size);
    const int64_t max_topk_blocks =
        std::max<int64_t>(1, std::min<int64_t>(sparse_topk_blocks, max_num_sparse_blocks));
    mx::Shape out_shape{
        index_query.shape(0),
        static_cast<mx::ShapeElem>(max_topk_blocks * sparse_block_size)};
    const std::vector<mx::array> inputs = {
        index_query, index_key_cache, block_tables, seq_lens};
    return mx::array(
        out_shape,
        mx::int32,
        std::make_shared<SparseTokenIndexer>(
            to_stream(s), max_context_len, sparse_block_size, sparse_topk_blocks,
            sparse_init_blocks, sparse_local_blocks, scale),
        inputs);
}

mx::array sparse_token_indexer_with_update(
    const mx::array& index_query,
    const mx::array& index_key_update,
    const mx::array& index_key_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const int64_t max_context_len,
    const int64_t sparse_block_size,
    const int64_t sparse_topk_blocks,
    const int64_t sparse_init_blocks,
    const int64_t sparse_local_blocks,
    const float scale,
    mx::StreamOrDevice s /* = {} */
) {
    const int64_t max_num_sparse_blocks =
        std::max<int64_t>(1, (max_context_len + sparse_block_size - 1) / sparse_block_size);
    const int64_t max_topk_blocks =
        std::max<int64_t>(1, std::min<int64_t>(sparse_topk_blocks, max_num_sparse_blocks));
    mx::Shape out_shape{
        index_query.shape(0),
        static_cast<mx::ShapeElem>(max_topk_blocks * sparse_block_size)};
    const std::vector<mx::array> inputs = {
        index_query, index_key_cache, block_tables, seq_lens, index_key_update};
    return mx::array(
        out_shape,
        mx::int32,
        std::make_shared<SparseTokenIndexer>(
            to_stream(s), max_context_len, sparse_block_size, sparse_topk_blocks,
            sparse_init_blocks, sparse_local_blocks, scale, true),
        inputs);
}

void SparseTokenIndexer::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void SparseTokenIndexer::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 4 || inputs.size() == 5);
    auto& index_query = inputs[0];
    auto& index_key_cache = inputs[1];
    auto& block_tables = inputs[2];
    auto& seq_lens = inputs[3];
    auto& out = outputs[0];

    const int head_size = index_query.shape(2);
    if (head_size > 256) {
      std::ostringstream msg;
      msg << "SparseTokenIndexer supports head_size <= 256, got " << head_size;
      throw std::runtime_error(msg.str());
    }
    if (index_key_cache.dtype() != index_query.dtype()) {
      throw std::runtime_error(
          "SparseTokenIndexer requires index_query and index_key_cache to share dtype");
    }
    if (sparse_topk_blocks_ > 32) {
      std::ostringstream msg;
      msg << "SparseTokenIndexer supports sparse_topk_blocks <= 32, got "
          << sparse_topk_blocks_;
      throw std::runtime_error(msg.str());
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
    const int64_t max_num_sparse_blocks =
        std::max<int64_t>(1, (max_context_len_ + sparse_block_size_ - 1) / sparse_block_size_);
    const int64_t max_topk_blocks =
        std::max<int64_t>(1, out.shape(1) / sparse_block_size_);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);

    mx::Shape score_shape{
        static_cast<mx::ShapeElem>(num_seqs),
        static_cast<mx::ShapeElem>(max_num_sparse_blocks)};
    mx::array block_scores(score_shape, mx::float32, nullptr, {});
    block_scores.set_data(mlx::core::allocator::malloc(block_scores.nbytes()));
    compute_encoder.add_temporary(block_scores);

    std::string score_name = "sparse_block_scores_" + get_type_string(index_query.dtype());
    score_name += "_hs" + std::to_string(head_size);
    score_name += "_nt" + std::to_string(num_threads);
    score_name += "_nsl" + std::to_string(num_simd_lanes);

    if (store_current_index_key_) {
      auto& index_key_update = inputs[4];
      if (index_key_update.dtype() != index_key_cache.dtype()) {
        throw std::runtime_error(
            "SparseTokenIndexer index key update and cache must share dtype");
      }
      if (index_key_update.shape(0) != num_seqs ||
          index_key_update.shape(2) != head_size) {
        throw std::runtime_error(
            "SparseTokenIndexer index key update must be shaped "
            "[num_seqs, index_key_heads, head_size]");
      }
      std::string store_name =
          "store_sparse_index_key_" + get_type_string(index_key_cache.dtype());
      auto store_kernel = d.get_kernel(store_name, lib);
      compute_encoder.set_compute_pipeline_state(store_kernel);

      int32_t index_key_heads_32 = static_cast<int32_t>(index_key_update.shape(1));
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
      compute_encoder.set_bytes(index_key_heads_32, 4);
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
    }

    auto score_kernel = d.get_kernel(score_name, lib);
    compute_encoder.set_compute_pipeline_state(score_kernel);

    int32_t index_heads_32 = static_cast<int32_t>(index_heads);
    int32_t index_key_heads_32 = static_cast<int32_t>(index_key_heads);
    int32_t cache_block_size_32 = static_cast<int32_t>(cache_block_size);
    int32_t sparse_block_size_32 = static_cast<int32_t>(sparse_block_size_);
    int32_t max_num_sparse_blocks_32 = static_cast<int32_t>(max_num_sparse_blocks);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t sparse_init_blocks_32 = static_cast<int32_t>(sparse_init_blocks_);
    int32_t sparse_local_blocks_32 = static_cast<int32_t>(sparse_local_blocks_);
    float scale_32 = static_cast<float>(scale_);
    int32_t q_stride_32 = static_cast<int32_t>(index_query.strides(0));
    int32_t q_head_stride_32 = static_cast<int32_t>(index_query.strides(1));
    int32_t cache_block_stride_32 = static_cast<int32_t>(index_key_cache.strides(1));
    int32_t cache_head_stride_32 = static_cast<int32_t>(index_key_cache.strides(2));
    int32_t cache_token_stride_32 = static_cast<int32_t>(index_key_cache.strides(3));

    compute_encoder.set_output_array(block_scores, 0);
    compute_encoder.set_input_array(index_query, 1);
    compute_encoder.set_input_array(index_key_cache, 2);
    compute_encoder.set_input_array(block_tables, 3);
    compute_encoder.set_input_array(seq_lens, 4);
    compute_encoder.set_bytes(index_heads_32, 5);
    compute_encoder.set_bytes(index_key_heads_32, 6);
    compute_encoder.set_bytes(cache_block_size_32, 7);
    compute_encoder.set_bytes(sparse_block_size_32, 8);
    compute_encoder.set_bytes(max_num_sparse_blocks_32, 9);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 10);
    compute_encoder.set_bytes(sparse_init_blocks_32, 11);
    compute_encoder.set_bytes(sparse_local_blocks_32, 12);
    compute_encoder.set_bytes(scale_32, 13);
    compute_encoder.set_bytes(q_stride_32, 14);
    compute_encoder.set_bytes(q_head_stride_32, 15);
    compute_encoder.set_bytes(cache_block_stride_32, 16);
    compute_encoder.set_bytes(cache_head_stride_32, 17);
    compute_encoder.set_bytes(cache_token_stride_32, 18);

    MTL::Size score_grid = MTL::Size(max_num_sparse_blocks, num_seqs, 1);
    MTL::Size score_threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(score_grid, score_threadgroup);

    auto topk_kernel = d.get_kernel("sparse_block_topk_tokens", lib);
    compute_encoder.set_compute_pipeline_state(topk_kernel);

    int32_t max_topk_blocks_32 = static_cast<int32_t>(max_topk_blocks);

    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(block_scores, 1);
    compute_encoder.set_input_array(seq_lens, 2);
    compute_encoder.set_bytes(max_num_sparse_blocks_32, 3);
    compute_encoder.set_bytes(max_topk_blocks_32, 4);
    compute_encoder.set_bytes(sparse_block_size_32, 5);

    MTL::Size topk_grid = MTL::Size(num_seqs, 1, 1);
    MTL::Size topk_threadgroup = MTL::Size(1, 1, 1);
    compute_encoder.dispatch_threadgroups(topk_grid, topk_threadgroup);
}

bool SparseTokenIndexer::is_equivalent(const mx::Primitive& other) const {
  const SparseTokenIndexer& r_other = static_cast<const SparseTokenIndexer&>(other);
  return max_context_len_ == r_other.max_context_len_ &&
         sparse_block_size_ == r_other.sparse_block_size_ &&
         sparse_topk_blocks_ == r_other.sparse_topk_blocks_ &&
         sparse_init_blocks_ == r_other.sparse_init_blocks_ &&
         sparse_local_blocks_ == r_other.sparse_local_blocks_ &&
         scale_ == r_other.scale_ &&
         store_current_index_key_ == r_other.store_current_index_key_;
}

} // namespace parallax_ext
