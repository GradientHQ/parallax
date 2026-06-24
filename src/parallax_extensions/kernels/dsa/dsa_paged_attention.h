#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array dsa_paged_attention(
    const mx::array& q_latent,        // [num_seqs, num_heads, latent_dim]
    const mx::array& q_pe,            // [num_seqs, num_heads, rope_dim]
    const mx::array& latent_cache,    // [1, num_blocks, 1, block_size, latent_dim]
    const mx::array& rope_cache,      // [1, num_blocks, 1, block_size, rope_dim]
    const mx::array& block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,        // [num_seqs]
    const mx::array& topk_indices,    // [num_seqs, max_num_positions]
    const int64_t block_size,
    const int64_t max_num_positions,
    const float scale,
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

class DSAPagedAttention : public mx::Primitive {
  public:
    explicit DSAPagedAttention(mx::Stream stream, int64_t block_size,
                               int64_t max_num_positions, float scale)
        : mx::Primitive(stream), block_size_(block_size),
          max_num_positions_(max_num_positions), scale_(scale) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return "DSAPagedAttention";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t block_size_;
    int64_t max_num_positions_;
    float scale_;
};

} // namespace parallax_ext
