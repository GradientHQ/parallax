#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "kernels/paged_attention.h"
#include "kernels/reshape_and_cache.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Parallax extensions";

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
      "window_size"_a = 0,
      "sinks"_a,
      "has_sink"_a = 0,
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
            window_size (int): Sliding window size (0 = no window).
            sinks (array): Attention sink biases [num_heads].
            has_sink (int): 1 = use sinks, 0 = ignore sinks.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: ``Paged attention result``
      )");

  m.def(
      "paged_attention_v2",
      &parallax_ext::paged_attention_v2,
      "query"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "num_kv_heads"_a,
      "block_size"_a,
      "max_seq_len"_a,
      "scale"_a,
      "window_size"_a = 0,
      "sinks"_a,
      "has_sink"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Partitioned PagedAttentionV2 operation

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
            window_size (int): Sliding window size (0 = no window).
            sinks (array): Attention sink biases [num_heads].
            has_sink (int): 1 = use sinks, 0 = ignore sinks.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: ``Paged attention result``
      )");

  m.def(
      "sparse_paged_attention",
      &parallax_ext::sparse_paged_attention,
      "query"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "token_positions"_a,
      "token_positions_valid"_a,
      "num_kv_heads"_a,
      "block_size"_a,
      "max_num_positions"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Token-sparse paged attention operation

        Args:
            query (array): Input array [num_seqs, num_heads, head_size].
            key_cache (array): Input array [num_blocks, num_heads, head_size/x, block_size, x].
            value_cache (array): Input array [num_blocks, num_heads, head_size, block_size].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            token_positions (array): Token positions to attend over [num_seqs, max_num_positions].
            token_positions_valid (array): Validity mask for token_positions.
            num_kv_heads (int): Input parameter.
            block_size (int): KV cache block size.
            max_num_positions (int): Number of sparse token positions per sequence.
            scale (float): Attention scale.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: ``Sparse paged attention result``
      )");

  m.def(
      "reshape_and_cache",
      &parallax_ext::reshape_and_cache,
      "key"_a,
      "value"_a,
      "key_cache"_a,
      "value_cache"_a,
      "slot_mapping"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        vLLM ReshapeAndCache operation

        Args:
            key (array): Input array [num_tokens, num_heads, head_size].
            value (array): Input array [num_tokens, num_heads, head_size].
            key_cache (array): Input array [num_blocks, num_heads, head_size/x, block_size, x].
            value_cache (array): Input array [num_blocks, num_heads, head_size, block_size].
            slot_mapping (array): Input array [num_tokens].

        Returns:
            array: ``Dummy output``
      )");
}
