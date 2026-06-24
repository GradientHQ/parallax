#include <sstream>
#include <string>

#include "mla_paged_attention.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "../common/utils.h"

namespace parallax_ext {

mx::array mla_paged_attention(
    const mx::array& q_latent,
    const mx::array& q_pe,
    const mx::array& latent_cache,
    const mx::array& rope_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const int64_t block_size,
    const float scale,
    mx::StreamOrDevice s /* = {} */
) {
    auto out_dtype = q_latent.dtype();
    auto out_shape = q_latent.shape();
    const std::vector<mx::array> inputs = {
        q_latent, q_pe, latent_cache, rope_cache, block_tables, seq_lens};
    return mx::array(
        out_shape,
        out_dtype,
        std::make_shared<MLAPagedAttention>(to_stream(s), block_size, scale),
        inputs);
}

void MLAPagedAttention::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void MLAPagedAttention::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 6);
    auto& q_latent = inputs[0];
    auto& q_pe = inputs[1];
    auto& latent_cache = inputs[2];
    auto& rope_cache = inputs[3];
    auto& block_tables = inputs[4];
    auto& seq_lens = inputs[5];
    auto& out = outputs[0];

    const int latent_dim = q_latent.shape(2);
    const int rope_dim = q_pe.shape(2);
    if (latent_dim > 512) {
      std::ostringstream msg;
      msg << "MLAPagedAttention supports latent_dim <= 512, got " << latent_dim;
      throw std::runtime_error(msg.str());
    }
    if (rope_dim > 128) {
      std::ostringstream msg;
      msg << "MLAPagedAttention supports rope_dim <= 128, got " << rope_dim;
      throw std::runtime_error(msg.str());
    }
    if (q_pe.shape(0) != q_latent.shape(0) || q_pe.shape(1) != q_latent.shape(1)) {
      throw std::runtime_error("MLAPagedAttention requires q_latent and q_pe batch/heads to match");
    }
    if (latent_cache.dtype() != q_latent.dtype() || rope_cache.dtype() != q_latent.dtype()) {
      throw std::runtime_error(
          "MLAPagedAttention requires q_latent, latent_cache, and rope_cache to share dtype");
    }
    if (latent_cache.shape(4) != latent_dim || rope_cache.shape(4) != rope_dim) {
      throw std::runtime_error("MLAPagedAttention cache dims must match query dims");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto& compute_encoder = mx::metal::get_command_encoder(s);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    constexpr int num_threads = 32;
    constexpr int num_simd_lanes = 32;
    const int64_t num_seqs = q_latent.shape(0);
    const int64_t num_heads = q_latent.shape(1);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);

    std::string kname = "mla_paged_attention_" + get_type_string(out.dtype());
    kname += "_cache_" + get_type_string(latent_cache.dtype());
    kname += "_ld" + std::to_string(latent_dim);
    kname += "_rd" + std::to_string(rope_dim);
    kname += "_nt" + std::to_string(num_threads);
    kname += "_nsl" + std::to_string(num_simd_lanes);

    auto kernel = d.get_kernel(kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    float scale_32 = static_cast<float>(scale_);
    int32_t block_size_32 = static_cast<int32_t>(block_size_);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
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
    compute_encoder.set_bytes(scale_32, 7);
    compute_encoder.set_bytes(block_size_32, 8);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 9);
    compute_encoder.set_bytes(q_latent_seq_stride_32, 10);
    compute_encoder.set_bytes(q_latent_head_stride_32, 11);
    compute_encoder.set_bytes(q_pe_seq_stride_32, 12);
    compute_encoder.set_bytes(q_pe_head_stride_32, 13);
    compute_encoder.set_bytes(latent_block_stride_32, 14);
    compute_encoder.set_bytes(latent_token_stride_32, 15);
    compute_encoder.set_bytes(rope_block_stride_32, 16);
    compute_encoder.set_bytes(rope_token_stride_32, 17);

    MTL::Size grid = MTL::Size(num_heads, num_seqs, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

bool MLAPagedAttention::is_equivalent(const mx::Primitive& other) const {
  const MLAPagedAttention& r_other = static_cast<const MLAPagedAttention&>(other);
  return block_size_ == r_other.block_size_ && scale_ == r_other.scale_;
}

} // namespace parallax_ext
