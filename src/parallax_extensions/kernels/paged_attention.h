#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

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
);

class PagedAttentionV1 : public mx::Primitive {
  public:
    explicit PagedAttentionV1(mx::Stream stream, int64_t num_kv_heads, int64_t block_size,
                              int64_t max_seq_len, float scale, int window_size,
                              int has_sink)
        : mx::Primitive(stream), num_kv_heads_(num_kv_heads), block_size_(block_size),
          max_seq_len_(max_seq_len), scale_(scale), window_size_(window_size),
          has_sink_(has_sink) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    /** The name of primitive. */
    const char* name() const override {
      return "PagedAttentionV1";
    }

    /** Equivalence check **/
    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t num_kv_heads_;
    int64_t block_size_;
    int64_t max_seq_len_;
    float scale_;
    int window_size_;
    int has_sink_;
};

} // namespace parallax_ext
