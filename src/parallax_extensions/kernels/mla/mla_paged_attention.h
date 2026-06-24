#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

mx::array mla_paged_attention(
    const mx::array& q_latent,
    const mx::array& q_pe,
    const mx::array& latent_cache,
    const mx::array& rope_cache,
    const mx::array& block_tables,
    const mx::array& seq_lens,
    const int64_t block_size,
    const float scale,
    mx::StreamOrDevice s = {});

class MLAPagedAttention : public mx::Primitive {
  public:
    explicit MLAPagedAttention(mx::Stream stream, int64_t block_size, float scale)
        : mx::Primitive(stream), block_size_(block_size), scale_(scale) {};

    void eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override {
      return "MLAPagedAttention";
    }

    bool is_equivalent(const mx::Primitive& other) const override;

  private:
    int64_t block_size_;
    float scale_;
};

} // namespace parallax_ext
