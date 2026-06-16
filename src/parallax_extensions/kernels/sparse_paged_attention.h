#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

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
);

class SparsePagedAttention : public mx::Primitive {
  public:
    explicit SparsePagedAttention(mx::Stream stream, int64_t num_kv_heads,
                                   int64_t block_size, int64_t max_num_positions,
                                   float scale)
        : mx::Primitive(stream), num_kv_heads_(num_kv_heads), block_size_(block_size),
          max_num_positions_(max_num_positions), scale_(scale) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return "SparsePagedAttention";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t num_kv_heads_;
    int64_t block_size_;
    int64_t max_num_positions_;
    float scale_;
};

} // namespace parallax_ext
