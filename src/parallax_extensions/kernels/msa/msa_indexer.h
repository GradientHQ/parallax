#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array msa_token_indexer(
    const mx::array& index_query,       // [num_seqs, index_heads, index_dim]
    const mx::array& index_key_cache,   // [1, num_blocks, index_kv_heads, block_size, index_dim]
    const mx::array& block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,          // [num_seqs]
    const int64_t max_context_len,
    const int64_t sparse_block_size,
    const int64_t sparse_topk_blocks,
    const int64_t sparse_init_blocks,
    const int64_t sparse_local_blocks,
    const float scale,
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

mx::array msa_token_indexer_with_update(
    const mx::array& index_query,       // [num_seqs, index_heads, index_dim]
    const mx::array& index_key_update,  // [num_seqs, index_kv_heads, index_dim]
    const mx::array& index_key_cache,   // [1, num_blocks, index_kv_heads, block_size, index_dim]
    const mx::array& block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const mx::array& seq_lens,          // [num_seqs]
    const mx::array& slot_mapping,      // [num_seqs]
    const int64_t max_context_len,
    const int64_t sparse_block_size,
    const int64_t sparse_topk_blocks,
    const int64_t sparse_init_blocks,
    const int64_t sparse_local_blocks,
    const float scale,
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
);

class MSATokenIndexer : public mx::Primitive {
  public:
    explicit MSATokenIndexer(mx::Stream stream, int64_t max_context_len,
                                 int64_t sparse_block_size,
                                 int64_t sparse_topk_blocks,
                                 int64_t sparse_init_blocks,
                                 int64_t sparse_local_blocks,
                                 float scale,
                                 bool store_indexer_cache = false)
        : mx::Primitive(stream), max_context_len_(max_context_len),
          sparse_block_size_(sparse_block_size), sparse_topk_blocks_(sparse_topk_blocks),
          sparse_init_blocks_(sparse_init_blocks), sparse_local_blocks_(sparse_local_blocks),
          scale_(scale), store_indexer_cache_(store_indexer_cache) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return store_indexer_cache_ ? "MSATokenIndexerWithUpdate" : "MSATokenIndexer";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t max_context_len_;
    int64_t sparse_block_size_;
    int64_t sparse_topk_blocks_;
    int64_t sparse_init_blocks_;
    int64_t sparse_local_blocks_;
    float scale_;
    bool store_indexer_cache_;
};

} // namespace parallax_ext
