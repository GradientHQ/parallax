#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array dsa_indexer_scores_with_update(
    const mx::array& index_query,       // [num_seqs, index_heads, index_dim]
    const mx::array& index_key_update,  // [num_seqs, index_key_heads, index_dim]
    const mx::array& index_key_cache,   // [1, num_blocks, index_key_heads, block_size, index_dim]
    const mx::array& block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,          // [num_seqs]
    const mx::array& index_weights,     // [num_seqs, index_heads]
    const mx::array& slot_mapping,      // [num_seqs]
    const int64_t max_context_len,
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

class DSAIndexerScoresWithUpdate : public mx::Primitive {
  public:
    explicit DSAIndexerScoresWithUpdate(mx::Stream stream, int64_t max_context_len)
        : mx::Primitive(stream), max_context_len_(max_context_len) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return "DSAIndexerScoresWithUpdate";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t max_context_len_;
};

} // namespace parallax_ext
