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
    mx::array& key_cache,          // [num_blocks, num_heads, head_size/x, block_size, x]
    mx::array& value_cache,        // [num_blocks, num_heads, head_size/x, block_size]
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
    auto& compute_encoder = d.get_command_encoder(s.index);
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
  const ReshapeAndCache& r_other = static_cast<const ReshapeAndCache&>(other);
  return true;
}

} // namespace parallax_ext
