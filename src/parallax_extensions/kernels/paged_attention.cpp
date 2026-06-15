#include <dlfcn.h>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>

#include "utils.h"
#include "paged_attention.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace parallax_ext {


static size_t calculate_shared_memory_size(int max_seq_len, int head_size,
                                        int num_threads, int num_simd_lanes) {
  size_t logits_size = max_seq_len * sizeof(float);
  size_t reduction_size = 2 * (num_threads / num_simd_lanes) * sizeof(float);
  size_t output_size = head_size * sizeof(float);
  return std::max(logits_size + reduction_size, output_size);
}

mx::array paged_attention_v1(
    const mx::array& query,         // [num_seqs, num_heads, head_size]
    const mx::array& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
    const mx::array& value_cache,   // [num_blocks, num_heads, head_size, block_size]
    const mx::array& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,      // [num_seqs]
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t max_seq_len,
    const float scale,
    const int window_size,          // Sliding window size (0 = no window)
    const mx::array& sinks,         // Attention sink biases [num_heads]
    const int has_sink,             // 1 = use sinks, 0 = ignore sinks
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    auto out_dtype = query.dtype();
    auto out_shape = query.shape();
    const std::vector<mx::array> inputs = {query, key_cache, value_cache, block_tables, seq_lens, sinks};
    // Construct the array as the output of the PagedAttentionV1 primitive
    return mx::array(
        /* const std::vector<int>& shape = */ out_shape,
        /* Dtype dtype = */ out_dtype,
        /* std::unique_ptr<Primitive> primitive = */
        std::make_shared<PagedAttentionV1>(to_stream(s), num_kv_heads, block_size,
                                           max_seq_len, scale, window_size, has_sink),
        /* const std::vector<array>& inputs = */ inputs);
}

mx::array paged_attention_v2(
    const mx::array& query,         // [num_seqs, num_heads, head_size]
    const mx::array& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
    const mx::array& value_cache,   // [num_blocks, num_heads, head_size, block_size]
    const mx::array& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,      // [num_seqs]
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t max_seq_len,
    const float scale,
    const int window_size,          // Sliding window size (0 = no window)
    const mx::array& sinks,         // Attention sink biases [num_heads]
    const int has_sink,             // 1 = use sinks, 0 = ignore sinks
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    auto out_dtype = query.dtype();
    auto out_shape = query.shape();
    const std::vector<mx::array> inputs = {query, key_cache, value_cache, block_tables, seq_lens, sinks};
    return mx::array(
        out_shape,
        out_dtype,
        std::make_shared<PagedAttentionV2>(to_stream(s), num_kv_heads, block_size,
                                           max_seq_len, scale, window_size, has_sink),
        inputs);
}

mx::array sparse_paged_attention(
    const mx::array& query,                  // [num_seqs, num_heads, head_size]
    const mx::array& key_cache,              // [num_blocks, num_heads, head_size/x, block_size, x]
    const mx::array& value_cache,            // [num_blocks, num_heads, head_size, block_size]
    const mx::array& block_tables,           // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,               // [num_seqs]
    const mx::array& token_positions,        // [num_seqs, max_num_positions]
    const mx::array& token_positions_valid,  // [num_seqs, max_num_positions]
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t max_num_positions,
    const float scale,
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
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

/** Evaluate primitive on CPU */
void PagedAttentionV1::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Currently not implemented
    return;
}

/** Evaluate primitive on GPU */
void PagedAttentionV1::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Prepare inputs
    assert(inputs.size() == 6);
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& block_tables = inputs[3];
    auto& seq_lens = inputs[4];
    auto& sinks = inputs[5];
    auto& out = outputs[0];
    int head_size = q.shape(2);

    // Each primitive carries the stream it should execute on
    // and each stream carries its device identifiers
    auto& s = stream();
    // We get the needed metal device using the stream
    auto& d = mx::metal::device(s.device);

    // Allocate output memory
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    // Set kernel paramas
    int num_threads = 256;
    if (window_size_ > 0) {
      if (window_size_ <= block_size_) {
        num_threads = 64;
      } else if (window_size_ <= 8 * block_size_) {
        num_threads = 128;
      }
    }
    const int num_simd_lanes = 32;
    const int partition_size = 0; // v1 doesn't use partitioning

    // Function constants
    bool use_partitioning = false;
    bool use_alibi = false;
    bool use_fp8_scales = false;
    mx::metal::MTLFCList func_consts = {
      {&use_partitioning, MTL::DataType::DataTypeBool, 10},
      {&use_alibi, MTL::DataType::DataTypeBool, 20},
      {&use_fp8_scales, MTL::DataType::DataTypeBool, 30},
    };

    // Resolve name of kernel
    std::string kname;
    std::string hash_name = "";
    kname = "paged_attention_" + get_type_string(out.dtype());
    kname += "_cache_" + get_type_string(k.dtype());
    kname += "_hs" + std::to_string(head_size);
    kname += "_bs" + std::to_string(block_size_);
    kname += "_nt" + std::to_string(num_threads);
    kname += "_nsl" + std::to_string(num_simd_lanes);
    kname += "_ps" + std::to_string(partition_size);

    // Load the metal library
    auto lib = d.get_library("parallax_ext", current_binary_dir());

    // Make a kernel from this metal library
    auto kernel = d.get_kernel(kname, lib, hash_name, func_consts);

    // Prepare to encode kernel
    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Shared Memory
    int effective_max_context_len = max_seq_len_;
    if (window_size_ > 0) {
      // Windowed attention only needs logits for a small tail of the sequence.
      effective_max_context_len =
          std::min<int>(max_seq_len_, window_size_ + block_size_);
    }
    const int padded_max_context_len =
        ((effective_max_context_len + block_size_ - 1) / block_size_) * block_size_;
    const int num_simds = num_threads / num_simd_lanes;
    const int logits_size = padded_max_context_len * sizeof(float);
    const int outputs_size = (num_simds / 2) * head_size * sizeof(float);
    const size_t shared_memory_size = std::max(logits_size, outputs_size);

    // set Threadgroup Memory (index 0)
    compute_encoder.set_threadgroup_memory_length(shared_memory_size, 0);


    // Calculate parameters
    float softcapping_ = 1.0;       // hard code for not use
    const int64_t num_seqs = q.shape(0);
    const int64_t num_heads = q.shape(1);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);
    int32_t q_stride = static_cast<int32_t>(q.strides(0));
    int32_t kv_block_stride = static_cast<int32_t>(k.strides(0));
    int32_t kv_head_stride = static_cast<int32_t>(k.strides(1));

    // Encode arrays to kernel
    // Skip exp_sums and max_logits for v1 (buffers 0, 1)
    compute_encoder.set_output_array(out, 2);
    compute_encoder.set_input_array(q, 3);
    compute_encoder.set_input_array(k, 4);
    compute_encoder.set_input_array(v, 5);
    // Skip k_scale and v_scale for non-fp8 (buffers 6, 7)
    int32_t num_kv_heads_32 = static_cast<int32_t>(num_kv_heads_);
    float scale_32 = static_cast<float>(scale_);
    float softcapping_32 = static_cast<float>(softcapping_);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t q_stride_32 = static_cast<int32_t>(q_stride);
    int32_t kv_block_stride_32 = static_cast<int32_t>(kv_block_stride);
    int32_t kv_head_stride_32 = static_cast<int32_t>(kv_head_stride);
    compute_encoder.set_bytes(num_kv_heads_32, 8);
    compute_encoder.set_bytes(scale_32, 9);
    compute_encoder.set_bytes(softcapping_32, 10);
    compute_encoder.set_input_array(block_tables, 11);
    compute_encoder.set_input_array(seq_lens, 12);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 13);
    // Skip alibi_slopes (buffer 14)
    compute_encoder.set_bytes(q_stride_32, 15);
    compute_encoder.set_bytes(kv_block_stride_32, 16);
    compute_encoder.set_bytes(kv_head_stride_32, 17);
    int32_t window_size_32 = static_cast<int32_t>(window_size_);
    int32_t has_sink_32 = static_cast<int32_t>(has_sink_);
    compute_encoder.set_bytes(window_size_32, 18);
    compute_encoder.set_input_array(sinks, 19);
    compute_encoder.set_bytes(has_sink_32, 20);

    // Dispatch configuration
    // Grid: (num_heads, num_seqs, 1) - no partitioning for v1
    MTL::Size grid = MTL::Size(num_heads, num_seqs, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);

    // Launch the grid with the given number of threads divided among
    // the given threadgroups
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

/** Evaluate primitive on CPU */
void SparsePagedAttention::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Currently not implemented
    return;
}

/** Evaluate primitive on GPU */
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

    constexpr int num_threads = 32;
    constexpr int num_simd_lanes = 32;
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

/** Evaluate primitive on CPU */
void PagedAttentionV2::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Currently not implemented
    return;
}

/** Evaluate primitive on GPU */
void PagedAttentionV2::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 6);
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& block_tables = inputs[3];
    auto& seq_lens = inputs[4];
    auto& sinks = inputs[5];
    auto& out = outputs[0];

    constexpr int partition_size = 512;
    const int head_size = q.shape(2);
    const int num_threads = 256;
    const int num_simd_lanes = 32;
    const int num_simds = num_threads / num_simd_lanes;

    if (block_size_ <= 0 || partition_size % block_size_ != 0) {
      std::ostringstream msg;
      msg << "PagedAttentionV2 requires block_size to divide partition_size="
          << partition_size << ", got block_size=" << block_size_;
      throw std::runtime_error(msg.str());
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto& compute_encoder = mx::metal::get_command_encoder(s);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    const int64_t num_seqs = q.shape(0);
    const int64_t num_heads = q.shape(1);
    const int64_t max_num_blocks_per_seq = block_tables.shape(1);
    const int max_num_partitions =
        std::max<int>(1, (max_seq_len_ + partition_size - 1) / partition_size);

    mx::Shape tmp_shape{
        static_cast<mx::ShapeElem>(num_seqs),
        static_cast<mx::ShapeElem>(num_heads),
        static_cast<mx::ShapeElem>(max_num_partitions),
        static_cast<mx::ShapeElem>(head_size)};
    mx::array tmp_out(tmp_shape, out.dtype(), nullptr, {});

    mx::Shape stat_shape{
        static_cast<mx::ShapeElem>(num_seqs),
        static_cast<mx::ShapeElem>(num_heads),
        static_cast<mx::ShapeElem>(max_num_partitions)};
    mx::array exp_sums(stat_shape, mx::float32, nullptr, {});
    mx::array max_logits(stat_shape, mx::float32, nullptr, {});

    tmp_out.set_data(mlx::core::allocator::malloc(tmp_out.nbytes()));
    exp_sums.set_data(mlx::core::allocator::malloc(exp_sums.nbytes()));
    max_logits.set_data(mlx::core::allocator::malloc(max_logits.nbytes()));
    compute_encoder.add_temporary(tmp_out);
    compute_encoder.add_temporary(exp_sums);
    compute_encoder.add_temporary(max_logits);

    bool use_partitioning = true;
    bool use_alibi = false;
    bool use_fp8_scales = false;
    mx::metal::MTLFCList func_consts = {
      {&use_partitioning, MTL::DataType::DataTypeBool, 10},
      {&use_alibi, MTL::DataType::DataTypeBool, 20},
      {&use_fp8_scales, MTL::DataType::DataTypeBool, 30},
    };

    std::string kname = "paged_attention_" + get_type_string(out.dtype());
    kname += "_cache_" + get_type_string(k.dtype());
    kname += "_hs" + std::to_string(head_size);
    kname += "_bs" + std::to_string(block_size_);
    kname += "_nt" + std::to_string(num_threads);
    kname += "_nsl" + std::to_string(num_simd_lanes);
    kname += "_ps" + std::to_string(partition_size);

    auto kernel = d.get_kernel(kname, lib, "", func_consts);
    compute_encoder.set_compute_pipeline_state(kernel);

    const int padded_partition_len =
        ((std::min<int64_t>(max_seq_len_, partition_size) + block_size_ - 1) / block_size_) *
        block_size_;
    const int logits_size = padded_partition_len * sizeof(float);
    const int outputs_size = (num_simds / 2) * head_size * sizeof(float);
    const size_t shared_memory_size = std::max(logits_size, outputs_size);
    compute_encoder.set_threadgroup_memory_length(shared_memory_size, 0);

    float softcapping_ = 1.0;       // hard code for not use
    int32_t q_stride = static_cast<int32_t>(q.strides(0));
    int32_t kv_block_stride = static_cast<int32_t>(k.strides(0));
    int32_t kv_head_stride = static_cast<int32_t>(k.strides(1));
    int32_t num_kv_heads_32 = static_cast<int32_t>(num_kv_heads_);
    float scale_32 = static_cast<float>(scale_);
    float softcapping_32 = static_cast<float>(softcapping_);
    int32_t max_num_blocks_per_seq_32 = static_cast<int32_t>(max_num_blocks_per_seq);
    int32_t q_stride_32 = static_cast<int32_t>(q_stride);
    int32_t kv_block_stride_32 = static_cast<int32_t>(kv_block_stride);
    int32_t kv_head_stride_32 = static_cast<int32_t>(kv_head_stride);
    int32_t window_size_32 = static_cast<int32_t>(window_size_);
    int32_t has_sink_32 = static_cast<int32_t>(has_sink_);

    compute_encoder.set_output_array(exp_sums, 0);
    compute_encoder.set_output_array(max_logits, 1);
    compute_encoder.set_output_array(tmp_out, 2);
    compute_encoder.set_input_array(q, 3);
    compute_encoder.set_input_array(k, 4);
    compute_encoder.set_input_array(v, 5);
    compute_encoder.set_bytes(num_kv_heads_32, 8);
    compute_encoder.set_bytes(scale_32, 9);
    compute_encoder.set_bytes(softcapping_32, 10);
    compute_encoder.set_input_array(block_tables, 11);
    compute_encoder.set_input_array(seq_lens, 12);
    compute_encoder.set_bytes(max_num_blocks_per_seq_32, 13);
    compute_encoder.set_bytes(q_stride_32, 15);
    compute_encoder.set_bytes(kv_block_stride_32, 16);
    compute_encoder.set_bytes(kv_head_stride_32, 17);
    compute_encoder.set_bytes(window_size_32, 18);
    compute_encoder.set_input_array(sinks, 19);
    compute_encoder.set_bytes(has_sink_32, 20);

    MTL::Size grid = MTL::Size(num_heads, num_seqs, max_num_partitions);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);

    std::string reduce_name = "paged_attention_v2_reduce_" + get_type_string(out.dtype());
    reduce_name += "_hs" + std::to_string(head_size);
    reduce_name += "_nt" + std::to_string(num_threads);
    reduce_name += "_nsl" + std::to_string(num_simd_lanes);
    reduce_name += "_ps" + std::to_string(partition_size);
    auto reduce_kernel = d.get_kernel(reduce_name, lib);
    compute_encoder.set_compute_pipeline_state(reduce_kernel);

    const int32_t max_num_partitions_32 = static_cast<int32_t>(max_num_partitions);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(exp_sums, 1);
    compute_encoder.set_input_array(max_logits, 2);
    compute_encoder.set_input_array(tmp_out, 3);
    compute_encoder.set_input_array(seq_lens, 4);
    compute_encoder.set_bytes(max_num_partitions_32, 5);

    const size_t reduce_shared_memory_size =
        2 * static_cast<size_t>(max_num_partitions) * sizeof(float);
    compute_encoder.set_threadgroup_memory_length(reduce_shared_memory_size, 0);

    MTL::Size reduce_grid = MTL::Size(num_heads, num_seqs, 1);
    compute_encoder.dispatch_threadgroups(reduce_grid, threadgroup);
}

/** Equivalence check **/
bool PagedAttentionV1::is_equivalent(const mx::Primitive& other) const {
  const PagedAttentionV1& r_other = static_cast<const PagedAttentionV1&>(other);
  return num_kv_heads_ == r_other.num_kv_heads_ && block_size_ == r_other.block_size_ &&
         max_seq_len_ == r_other.max_seq_len_ && scale_ == r_other.scale_ &&
         window_size_ == r_other.window_size_ && has_sink_ == r_other.has_sink_;
}

/** Equivalence check **/
bool PagedAttentionV2::is_equivalent(const mx::Primitive& other) const {
  const PagedAttentionV2& r_other = static_cast<const PagedAttentionV2&>(other);
  return num_kv_heads_ == r_other.num_kv_heads_ && block_size_ == r_other.block_size_ &&
         max_seq_len_ == r_other.max_seq_len_ && scale_ == r_other.scale_ &&
         window_size_ == r_other.window_size_ && has_sink_ == r_other.has_sink_;
}

bool SparsePagedAttention::is_equivalent(const mx::Primitive& other) const {
  const SparsePagedAttention& r_other = static_cast<const SparsePagedAttention&>(other);
  return num_kv_heads_ == r_other.num_kv_heads_ && block_size_ == r_other.block_size_ &&
         max_num_positions_ == r_other.max_num_positions_ && scale_ == r_other.scale_;
}

} // namespace parallax_ext
