#include <algorithm>
#include <sstream>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "sparse_paged_attention.h"
#include "utils.h"

namespace parallax_ext {

mx::array sparse_paged_attention(
    const mx::array& query,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const mx::array& token_positions,
    const mx::array& token_positions_valid,
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t max_num_positions,
    const float scale,
    mx::StreamOrDevice s /* = {} */
) {
    auto out_dtype = query.dtype();
    auto out_shape = query.shape();
    const std::vector<mx::array> inputs = {
        query, key_cache, value_cache, block_tables, seq_lens,
        token_positions, token_positions_valid};
    return mx::array(
        out_shape,
        out_dtype,
        std::make_shared<SparsePagedAttention>(
            to_stream(s), num_kv_heads, block_size, max_num_positions, scale),
        inputs);
}

void SparsePagedAttention::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void SparsePagedAttention::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 7);
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& block_tables = inputs[3];
    auto& seq_lens = inputs[4];
    auto& token_positions = inputs[5];
    auto& token_positions_valid = inputs[6];
    auto& out = outputs[0];

    const int head_size = q.shape(2);
    if (head_size > 256) {
      std::ostringstream msg;
      msg << "SparsePagedAttention supports head_size <= 256, got " << head_size;
      throw std::runtime_error(msg.str());
    }
    if (k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
      throw std::runtime_error(
          "SparsePagedAttention requires query, key_cache, and value_cache to share dtype");
    }
    if (v.shape(2) != head_size) {
      std::ostringstream msg;
      msg << "SparsePagedAttention requires value head dim to match query head dim, got "
          << v.shape(2) << " and " << head_size;
      throw std::runtime_error(msg.str());
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto& compute_encoder = mx::metal::get_command_encoder(s);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    constexpr int num_threads = 256;
    constexpr int num_simd_lanes = 32;
    constexpr int num_simds = num_threads / num_simd_lanes;
    const int64_t num_seqs = q.shape(0);
    const int64_t num_heads = q.shape(1);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);

    std::string kname = "sparse_paged_attention_" + get_type_string(out.dtype());
    kname += "_cache_" + get_type_string(k.dtype());
    kname += "_hs" + std::to_string(head_size);
    kname += "_nt" + std::to_string(num_threads);
    kname += "_nsl" + std::to_string(num_simd_lanes);

    auto kernel = d.get_kernel(kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    const int logits_size = (max_num_positions_ + head_size) * sizeof(float);
    const int outputs_size = (num_simds / 2) * head_size * sizeof(float);
    const size_t shared_memory_size = std::max(logits_size, outputs_size);
    compute_encoder.set_threadgroup_memory_length(shared_memory_size, 0);

    int32_t num_kv_heads_32 = static_cast<int32_t>(num_kv_heads_);
    float scale_32 = static_cast<float>(scale_);
    int32_t block_size_32 = static_cast<int32_t>(block_size_);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t max_num_positions_32 = static_cast<int32_t>(max_num_positions_);
    int32_t q_stride_32 = static_cast<int32_t>(q.strides(0));
    int32_t kv_block_stride_32 = static_cast<int32_t>(k.strides(0));
    int32_t kv_head_stride_32 = static_cast<int32_t>(k.strides(1));

    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(q, 1);
    compute_encoder.set_input_array(k, 2);
    compute_encoder.set_input_array(v, 3);
    compute_encoder.set_input_array(block_tables, 4);
    compute_encoder.set_input_array(seq_lens, 5);
    compute_encoder.set_input_array(token_positions, 6);
    compute_encoder.set_input_array(token_positions_valid, 7);
    compute_encoder.set_bytes(num_kv_heads_32, 8);
    compute_encoder.set_bytes(scale_32, 9);
    compute_encoder.set_bytes(block_size_32, 10);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 11);
    compute_encoder.set_bytes(max_num_positions_32, 12);
    compute_encoder.set_bytes(q_stride_32, 13);
    compute_encoder.set_bytes(kv_block_stride_32, 14);
    compute_encoder.set_bytes(kv_head_stride_32, 15);

    MTL::Size grid = MTL::Size(num_heads, num_seqs, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

bool SparsePagedAttention::is_equivalent(const mx::Primitive& other) const {
  const SparsePagedAttention& r_other = static_cast<const SparsePagedAttention&>(other);
  return num_kv_heads_ == r_other.num_kv_heads_ && block_size_ == r_other.block_size_ &&
         max_num_positions_ == r_other.max_num_positions_ && scale_ == r_other.scale_;
}

} // namespace parallax_ext
