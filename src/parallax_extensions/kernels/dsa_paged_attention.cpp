#include <algorithm>
#include <sstream>
#include <string>

#include "dsa_paged_attention.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "utils.h"

namespace parallax_ext {

mx::array dsa_paged_attention(
    const mx::array& q_latent,
    const mx::array& q_pe,
    const mx::array& latent_cache,
    const mx::array& rope_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const mx::array& topk_indices,
    const int64_t block_size,
    const int64_t max_num_positions,
    const float scale,
    mx::StreamOrDevice s /* = {} */
) {
    auto out_dtype = q_latent.dtype();
    auto out_shape = q_latent.shape();
    const std::vector<mx::array> inputs = {
        q_latent, q_pe, latent_cache, rope_cache, block_tables, seq_lens, topk_indices};
    return mx::array(
        out_shape,
        out_dtype,
        std::make_shared<DSAPagedAttention>(
            to_stream(s), block_size, max_num_positions, scale),
        inputs);
}

void DSAPagedAttention::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void DSAPagedAttention::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 7);
    auto& q_latent = inputs[0];
    auto& q_pe = inputs[1];
    auto& latent_cache = inputs[2];
    auto& rope_cache = inputs[3];
    auto& block_tables = inputs[4];
    auto& seq_lens = inputs[5];
    auto& topk_indices = inputs[6];
    auto& out = outputs[0];

    const int latent_dim = q_latent.shape(2);
    const int rope_dim = q_pe.shape(2);
    if (latent_dim > 512) {
      std::ostringstream msg;
      msg << "DSAPagedAttention supports latent_dim <= 512, got " << latent_dim;
      throw std::runtime_error(msg.str());
    }
    if (rope_dim > 128) {
      std::ostringstream msg;
      msg << "DSAPagedAttention supports rope_dim <= 128, got " << rope_dim;
      throw std::runtime_error(msg.str());
    }
    if (q_pe.shape(0) != q_latent.shape(0) || q_pe.shape(1) != q_latent.shape(1)) {
      throw std::runtime_error("DSAPagedAttention requires q_latent and q_pe batch/heads to match");
    }
    if (latent_cache.dtype() != q_latent.dtype() || rope_cache.dtype() != q_latent.dtype()) {
      throw std::runtime_error(
          "DSAPagedAttention requires q_latent, latent_cache, and rope_cache to share dtype");
    }
    if (latent_cache.shape(4) != latent_dim || rope_cache.shape(4) != rope_dim) {
      throw std::runtime_error("DSAPagedAttention cache dims must match query dims");
    }
    if (topk_indices.shape(1) != max_num_positions_) {
      throw std::runtime_error("DSAPagedAttention topk width must match max_num_positions");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto& compute_encoder = mx::metal::get_command_encoder(s);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    constexpr int num_threads = 256;
    constexpr int num_simd_lanes = 32;
    constexpr int num_simds = num_threads / num_simd_lanes;
    const int64_t num_seqs = q_latent.shape(0);
    const int64_t num_heads = q_latent.shape(1);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);

    std::string kname = "dsa_paged_attention_" + get_type_string(out.dtype());
    kname += "_cache_" + get_type_string(latent_cache.dtype());
    kname += "_ld" + std::to_string(latent_dim);
    kname += "_rd" + std::to_string(rope_dim);
    kname += "_nt" + std::to_string(num_threads);
    kname += "_nsl" + std::to_string(num_simd_lanes);

    auto kernel = d.get_kernel(kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    const int logits_size = (max_num_positions_ + latent_dim + rope_dim) * sizeof(float);
    const int outputs_size = (num_simds / 2) * latent_dim * sizeof(float);
    const size_t shared_memory_size = std::max(logits_size, outputs_size);
    compute_encoder.set_threadgroup_memory_length(shared_memory_size, 0);

    float scale_32 = static_cast<float>(scale_);
    int32_t block_size_32 = static_cast<int32_t>(block_size_);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t max_num_positions_32 = static_cast<int32_t>(max_num_positions_);
    int32_t q_latent_seq_stride_32 = static_cast<int32_t>(q_latent.strides(0));
    int32_t q_latent_head_stride_32 = static_cast<int32_t>(q_latent.strides(1));
    int32_t q_pe_seq_stride_32 = static_cast<int32_t>(q_pe.strides(0));
    int32_t q_pe_head_stride_32 = static_cast<int32_t>(q_pe.strides(1));
    int32_t latent_block_stride_32 = static_cast<int32_t>(latent_cache.strides(1));
    int32_t latent_token_stride_32 = static_cast<int32_t>(latent_cache.strides(3));
    int32_t rope_block_stride_32 = static_cast<int32_t>(rope_cache.strides(1));
    int32_t rope_token_stride_32 = static_cast<int32_t>(rope_cache.strides(3));

    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(q_latent, 1);
    compute_encoder.set_input_array(q_pe, 2);
    compute_encoder.set_input_array(latent_cache, 3);
    compute_encoder.set_input_array(rope_cache, 4);
    compute_encoder.set_input_array(block_tables, 5);
    compute_encoder.set_input_array(seq_lens, 6);
    compute_encoder.set_input_array(topk_indices, 7);
    compute_encoder.set_bytes(scale_32, 8);
    compute_encoder.set_bytes(block_size_32, 9);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 10);
    compute_encoder.set_bytes(max_num_positions_32, 11);
    compute_encoder.set_bytes(q_latent_seq_stride_32, 12);
    compute_encoder.set_bytes(q_latent_head_stride_32, 13);
    compute_encoder.set_bytes(q_pe_seq_stride_32, 14);
    compute_encoder.set_bytes(q_pe_head_stride_32, 15);
    compute_encoder.set_bytes(latent_block_stride_32, 16);
    compute_encoder.set_bytes(latent_token_stride_32, 17);
    compute_encoder.set_bytes(rope_block_stride_32, 18);
    compute_encoder.set_bytes(rope_token_stride_32, 19);

    MTL::Size grid = MTL::Size(num_heads, num_seqs, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

bool DSAPagedAttention::is_equivalent(const mx::Primitive& other) const {
  const DSAPagedAttention& r_other = static_cast<const DSAPagedAttention&>(other);
  return block_size_ == r_other.block_size_ &&
         max_num_positions_ == r_other.max_num_positions_ && scale_ == r_other.scale_;
}

} // namespace parallax_ext
