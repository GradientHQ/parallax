#include <dlfcn.h>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>

#include "utils.h"
#include "reshape_and_cache.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace parallax_ext {


mx::array reshape_and_cache(
    const mx::array& key,          // [num_tokens, num_heads, head_size]
    const mx::array& value,        // [num_tokens, num_heads, head_size]
    const mx::array& key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const mx::array& value_cache,  // [num_blocks, num_heads, head_size/x, block_size]
    const mx::array& slot_mapping, // [num_tokens]
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
    auto key_shape = key.shape();
    auto key_dtype = key.dtype();
    const std::vector<mx::array> inputs = {key, value, key_cache, value_cache, slot_mapping};
    // Construct the array as the dummy output of ReshapeAndCache kernel
    return mx::array(
        /* const std::vector<int>& shape = */ key_shape,
        /* Dtype dtype = */ key_dtype,
        /* std::unique_ptr<Primitive> primitive = */
        std::make_shared<ReshapeAndCache>(to_stream(s)),
        /* const std::vector<array>& inputs = */ inputs);
}

mx::array dsa_reshape_and_cache(
    const mx::array& key,
    const mx::array& value,
    const mx::array& key_cache,
    const mx::array& value_cache,
    const mx::array& slot_mapping,
    mx::StreamOrDevice s /* = {} */
) {
    auto key_shape = key.shape();
    auto key_dtype = key.dtype();
    const std::vector<mx::array> inputs = {key, value, key_cache, value_cache, slot_mapping};
    return mx::array(
        key_shape,
        key_dtype,
        std::make_shared<DSAReshapeAndCache>(to_stream(s)),
        inputs);
}

/** Evaluate primitive on CPU */
void ReshapeAndCache::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Currently not implemented
    return;
}

/** Evaluate primitive on GPU */
void ReshapeAndCache::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    // Prepare inputs
    assert(inputs.size() == 5);
    auto& key = inputs[0];
    auto& value = inputs[1];
    auto& key_cache = inputs[2];
    auto& value_cache = inputs[3];
    auto& slot_mapping = inputs[4];
    auto& out = outputs[0];

    // Each primitive carries the stream it should execute on
    // and each stream carries its device identifiers
    auto& s = stream();
    // We get the needed metal device using the stream
    auto& d = mx::metal::device(s.device);

    // Allocate output memory
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    // Set kernel paramas
    const int64_t num_tokens = key.shape(0);
    const int64_t num_heads = key.shape(1);
    const int64_t head_size = key.shape(2);
    const int64_t block_size = key_cache.shape(3);
    const int64_t x = key_cache.shape(4);
    bool use_fp8_scales = false;

    // Function constants
    mx::metal::MTLFCList func_consts = {
      {&use_fp8_scales, MTL::DataType::DataTypeBool, 10},
    };

    // Resolve name of kernel
    std::string kname;
    std::string hash_name = "";
    kname = "reshape_and_cache_kv_" + get_type_string(key.dtype());
    kname += "_cache_" + get_type_string(key_cache.dtype());

    // Load the metal library
    auto lib = d.get_library("parallax_ext", current_binary_dir());

    // Make a kernel from this metal library
    auto kernel = d.get_kernel(kname, lib, hash_name, func_consts);

    // Prepare to encode kernel
    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Calculate parameters
    int32_t key_stride = static_cast<int32_t>(key.strides(0));
    int32_t value_stride = static_cast<int32_t>(value.strides(0));

    // Encode arrays to kernel
    compute_encoder.set_input_array(key, 0);
    compute_encoder.set_input_array(value, 1);
    compute_encoder.set_input_array(key_cache, 2);
    compute_encoder.set_input_array(value_cache, 3);
    compute_encoder.set_input_array(slot_mapping, 4);
    // Skip k_scale and v_scale for non-fp8 (buffers 5, 6)
    compute_encoder.set_bytes(key_stride, 7);
    compute_encoder.set_bytes(value_stride, 8);
    int32_t num_heads_32 = static_cast<int32_t>(num_heads);
    int32_t head_size_32 = static_cast<int32_t>(head_size);
    int32_t block_size_32 = static_cast<int32_t>(block_size);
    int32_t x_32 = static_cast<int32_t>(x);
    compute_encoder.set_bytes(num_heads_32, 9);
    compute_encoder.set_bytes(head_size_32, 10);
    compute_encoder.set_bytes(block_size_32, 11);
    compute_encoder.set_bytes(x_32, 12);

    // Dispatch configuration
    const uint64_t num_threads = std::min<uint64_t>(512, num_heads * head_size);
    MTL::Size grid = MTL::Size(num_tokens, 1, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);

    // Launch the grid with the given number of threads divided among
    // the given threadgroups
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

/** Equivalence check **/
bool ReshapeAndCache::is_equivalent(const mx::Primitive& other) const {
  const ReshapeAndCache& r_other =
      static_cast<const ReshapeAndCache&>(other);
  return true;
}

void DSAReshapeAndCache::eval_cpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    return;
}

void DSAReshapeAndCache::eval_gpu(
  const std::vector<mx::array>& inputs,
  std::vector<mx::array>& outputs) {
    assert(inputs.size() == 5);
    auto& key = inputs[0];
    auto& value = inputs[1];
    auto& key_cache = inputs[2];
    auto& value_cache = inputs[3];
    auto& slot_mapping = inputs[4];
    auto& out = outputs[0];

    if (key.dtype() != value.dtype() || key.dtype() != key_cache.dtype() ||
        key.dtype() != value_cache.dtype()) {
      throw std::runtime_error(
          "DSAReshapeAndCache requires key, value, and caches to share dtype");
    }
    if (key.shape().size() != 3 || value.shape().size() != 3 ||
        key_cache.shape().size() != 5 || value_cache.shape().size() != 5) {
      throw std::runtime_error(
          "DSAReshapeAndCache expects key/value [tokens, heads, dim] and "
          "caches [1, blocks, heads, block_size, dim]");
    }
    if (key_cache.shape(0) != 1 || value_cache.shape(0) != 1) {
      throw std::runtime_error(
          "DSAReshapeAndCache expects single-layer cache tensors");
    }
    if (key.shape(0) != value.shape(0) || key.shape(1) != value.shape(1)) {
      throw std::runtime_error(
          "DSAReshapeAndCache key/value token and head dimensions must match");
    }
    if (key_cache.shape(2) != key.shape(1) ||
        value_cache.shape(2) != value.shape(1)) {
      throw std::runtime_error(
          "DSAReshapeAndCache cache head dimensions must match inputs");
    }
    if (key_cache.shape(4) != key.shape(2) ||
        value_cache.shape(4) != value.shape(2)) {
      throw std::runtime_error(
          "DSAReshapeAndCache cache dims must match input dims");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    const int64_t num_tokens = key.shape(0);
    const int64_t num_heads = key.shape(1);
    const int64_t key_dim = key.shape(2);
    const int64_t value_dim = value.shape(2);
    const int64_t block_size = key_cache.shape(3);

    std::string kname = "dsa_reshape_and_cache_kv_" + get_type_string(key.dtype());
    kname += "_cache_" + get_type_string(key_cache.dtype());
    auto lib = d.get_library("parallax_ext", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    int32_t key_stride = static_cast<int32_t>(key.strides(0));
    int32_t key_head_stride = static_cast<int32_t>(key.strides(1));
    int32_t value_stride = static_cast<int32_t>(value.strides(0));
    int32_t value_head_stride = static_cast<int32_t>(value.strides(1));
    int32_t key_cache_block_stride = static_cast<int32_t>(key_cache.strides(1));
    int32_t key_cache_head_stride = static_cast<int32_t>(key_cache.strides(2));
    int32_t key_cache_token_stride = static_cast<int32_t>(key_cache.strides(3));
    int32_t value_cache_block_stride = static_cast<int32_t>(value_cache.strides(1));
    int32_t value_cache_head_stride = static_cast<int32_t>(value_cache.strides(2));
    int32_t value_cache_token_stride = static_cast<int32_t>(value_cache.strides(3));
    int32_t num_heads_32 = static_cast<int32_t>(num_heads);
    int32_t key_dim_32 = static_cast<int32_t>(key_dim);
    int32_t value_dim_32 = static_cast<int32_t>(value_dim);
    int32_t block_size_32 = static_cast<int32_t>(block_size);

    compute_encoder.set_input_array(key, 0);
    compute_encoder.set_input_array(value, 1);
    compute_encoder.set_input_array(key_cache, 2);
    compute_encoder.set_input_array(value_cache, 3);
    compute_encoder.set_input_array(slot_mapping, 4);
    compute_encoder.set_bytes(key_stride, 5);
    compute_encoder.set_bytes(key_head_stride, 6);
    compute_encoder.set_bytes(value_stride, 7);
    compute_encoder.set_bytes(value_head_stride, 8);
    compute_encoder.set_bytes(key_cache_block_stride, 9);
    compute_encoder.set_bytes(key_cache_head_stride, 10);
    compute_encoder.set_bytes(key_cache_token_stride, 11);
    compute_encoder.set_bytes(value_cache_block_stride, 12);
    compute_encoder.set_bytes(value_cache_head_stride, 13);
    compute_encoder.set_bytes(value_cache_token_stride, 14);
    compute_encoder.set_bytes(num_heads_32, 15);
    compute_encoder.set_bytes(key_dim_32, 16);
    compute_encoder.set_bytes(value_dim_32, 17);
    compute_encoder.set_bytes(block_size_32, 18);

    const uint64_t max_dim = std::max<int64_t>(key_dim, value_dim);
    const uint64_t num_threads = std::min<uint64_t>(512, num_heads * max_dim);
    MTL::Size grid = MTL::Size(num_tokens, 1, 1);
    MTL::Size threadgroup = MTL::Size(num_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, threadgroup);
}

bool DSAReshapeAndCache::is_equivalent(const mx::Primitive& other) const {
  const DSAReshapeAndCache& r_other =
      static_cast<const DSAReshapeAndCache&>(other);
  return true;
}

} // namespace parallax_ext
