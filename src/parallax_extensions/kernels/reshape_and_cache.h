#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array reshape_and_cache(
    const mx::array& key,           // [num_tokens, num_heads, head_size]
    const mx::array& value,         // [num_tokens, num_heads, head_size]
    mx::array& key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
    mx::array& value_cache,         // [num_blocks, num_heads, head_size/x, block_size]
    const mx::array& slot_mapping,  // [num_tokens]
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

class ReshapeAndCache : public mx::Primitive {
  public:
    explicit ReshapeAndCache(mx::Stream stream) : mx::Primitive(stream){};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    /** The name of primitive. */
    const char* name() const override {
      return "ReshapeAndCache";
    }

    /** Equivalence check **/
    bool is_equivalent(const mx::Primitive& other) const override;

  private:
};

} // namespace parallax_ext
