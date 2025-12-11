#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "paged_attention/paged_attention.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "vLLM PagedAttentionV1";

  m.def(
      "paged_attention_v1",
      &parallax_ext::paged_attention_v1,
      "query"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "num_kv_heads"_a,
      "block_size"_a,
      "max_seq_len"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        vLLM PagedAttentionV1 operation

        Args:
            query (array): Input array [num_seqs, num_heads, head_size].
            key_cache (array): Input array [num_blocks, num_heads, head_size/x, block_size, x].
            value_cache (array): Input array [num_blocks, num_heads, head_size, block_size].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            num_kv_heads (int): Input parameter.
            block_size (int): Input parameter.
            max_seq_len (int): Input parameter.
            scale (float): Input parameter.

        Returns:
            array: ``Paged attention result``
      )");
}
