#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array store_indexer_cache(
    const mx::array& index_key,         // [num_tokens, index_key_heads, index_dim]
    const mx::array& index_key_cache,   // [1, num_blocks, index_key_heads, block_size, index_dim]
    const mx::array& slot_mapping,      // [num_tokens]
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

class StoreIndexerCache : public mx::Primitive {
  public:
    explicit StoreIndexerCache(mx::Stream stream) : mx::Primitive(stream) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return "StoreIndexerCache";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
};

} // namespace parallax_ext
