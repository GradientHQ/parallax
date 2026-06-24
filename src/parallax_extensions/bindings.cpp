#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "kernels/dsa/dsa_indexer.h"
#include "kernels/dsa/dsa_paged_attention.h"
#include "kernels/indexer_cache/store_indexer_cache.h"
#include "kernels/mla/mla_paged_attention.h"
#include "kernels/msa/msa_indexer.h"
#include "kernels/msa/msa_paged_attention.h"
#include "kernels/paged_attention/paged_attention.h"
#include "kernels/reshape_and_cache/reshape_and_cache.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Parallax extensions";

  m.def(
      "store_indexer_cache",
      &parallax_ext::store_indexer_cache,
      "index_key"_a,
      "index_key_cache"_a,
      "slot_mapping"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Store index keys into paged index cache.

        Args:
            index_key (array): Input array [num_tokens, index_key_heads, index_dim].
            index_key_cache (array): Paged index-key cache [1, num_blocks, index_key_heads, block_size, index_dim].
            slot_mapping (array): Physical cache slots [num_tokens].

        Returns:
            array: ``Dummy output``
      )");

  m.def(
      "mla_paged_attention",
      &parallax_ext::mla_paged_attention,
      "q_latent"_a,
      "q_pe"_a,
      "latent_cache"_a,
      "rope_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "block_size"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode dense MLA paged attention over compressed MLA cache.

        Args:
            q_latent (array): MLA latent query [num_seqs, num_heads, latent_dim].
            q_pe (array): RoPE query [num_seqs, num_heads, rope_dim].
            latent_cache (array): Paged MLA latent cache [1, num_blocks, 1, block_size, latent_dim].
            rope_cache (array): Paged RoPE key cache [1, num_blocks, 1, block_size, rope_dim].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            block_size (int): Cache block size.
            scale (float): Attention scale.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: latent-space attention result [num_seqs, num_heads, latent_dim]
      )");

  m.def(
      "dsa_paged_attention",
      &parallax_ext::dsa_paged_attention,
      "q_latent"_a,
      "q_pe"_a,
      "latent_cache"_a,
      "rope_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "topk_indices"_a,
      "block_size"_a,
      "max_num_positions"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode DSA paged attention for compressed MLA cache.

        Args:
            q_latent (array): MLA latent query [num_seqs, num_heads, latent_dim].
            q_pe (array): RoPE query [num_seqs, num_heads, rope_dim].
            latent_cache (array): Paged MLA latent cache [1, num_blocks, 1, block_size, latent_dim].
            rope_cache (array): Paged RoPE key cache [1, num_blocks, 1, block_size, rope_dim].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            topk_indices (array): Sparse logical token positions [num_seqs, max_num_positions].
            block_size (int): Cache block size.
            max_num_positions (int): Number of sparse token positions per sequence.
            scale (float): Attention scale.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: latent-space attention result [num_seqs, num_heads, latent_dim]
      )");

  m.def(
      "dsa_indexer_scores_with_update",
      &parallax_ext::dsa_indexer_scores_with_update,
      "index_query"_a,
      "index_key_update"_a,
      "index_key_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "index_weights"_a,
      "slot_mapping"_a,
      "max_context_len"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode DSA indexer score operation with index-cache update.

        Args:
            index_query (array): Index query [num_seqs, index_heads, index_dim].
            index_key_update (array): Current index key [num_seqs, index_key_heads, index_dim].
            index_key_cache (array): Paged index-key cache [1, num_blocks, index_key_heads, block_size, index_dim].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            index_weights (array): Per-query-head weights [num_seqs, index_heads].
            slot_mapping (array): Physical cache slots [num_seqs].
            max_context_len (int): Maximum scored context length in this batch.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: token scores [num_seqs, max_context_len]
      )");

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
      "msa_paged_attention",
      &parallax_ext::msa_paged_attention,
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
        MSA paged attention operation

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
            array: ``MSA paged attention result``
      )");

  m.def(
      "msa_token_indexer",
      &parallax_ext::msa_token_indexer,
      "index_query"_a,
      "index_key_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "max_context_len"_a,
      "sparse_block_size"_a,
      "sparse_topk_blocks"_a,
      "sparse_init_blocks"_a,
      "sparse_local_blocks"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MSA token indexer operation

        Args:
            index_query (array): Index query [num_seqs, index_heads, index_dim].
            index_key_cache (array): Paged index-key cache [1, num_blocks, index_key_heads, block_size, index_dim].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            max_context_len (int): Maximum context length in this batch.
            sparse_block_size (int): Sparse index block size.
            sparse_topk_blocks (int): Number of sparse index blocks to select.
            sparse_init_blocks (int): Initial sparse blocks to force include.
            sparse_local_blocks (int): Local tail sparse blocks to force include.
            scale (float): Index attention scale.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: ``MSA token positions, with -1 for invalid slots``
      )");

  m.def(
      "msa_token_indexer_with_update",
      &parallax_ext::msa_token_indexer_with_update,
      "index_query"_a,
      "index_key_update"_a,
      "index_key_cache"_a,
      "block_tables"_a,
      "seq_lens"_a,
      "slot_mapping"_a,
      "max_context_len"_a,
      "sparse_block_size"_a,
      "sparse_topk_blocks"_a,
      "sparse_init_blocks"_a,
      "sparse_local_blocks"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MSA token indexer operation with decode index-cache update

        Args:
            index_query (array): Index query [num_seqs, index_heads, index_dim].
            index_key_update (array): Current index key [num_seqs, index_key_heads, index_dim].
            index_key_cache (array): Paged index-key cache [1, num_blocks, index_key_heads, block_size, index_dim].
            block_tables (array): Input array [num_seqs, max_num_blocks_per_seq].
            seq_lens (array): Input array [num_seqs].
            slot_mapping (array): Physical cache slots [num_seqs].
            max_context_len (int): Maximum context length in this batch.
            sparse_block_size (int): Sparse index block size.
            sparse_topk_blocks (int): Number of sparse index blocks to select.
            sparse_init_blocks (int): Initial sparse blocks to force include.
            sparse_local_blocks (int): Local tail sparse blocks to force include.
            scale (float): Index attention scale.
            stream (Stream or Device): Stream on which to schedule the operation.

        Returns:
            array: ``MSA token positions, with -1 for invalid slots``
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

  m.def(
      "dsa_reshape_and_cache",
      &parallax_ext::dsa_reshape_and_cache,
      "key"_a,
      "value"_a,
      "key_cache"_a,
      "value_cache"_a,
      "slot_mapping"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Store KV/MLA tensors into DeepSeek/GLM DSA paged cache layout.

        Args:
            key (array): Input array [num_tokens, num_heads, key_dim].
            value (array): Input array [num_tokens, num_heads, value_dim].
            key_cache (array): Input array [1, num_blocks, num_heads, block_size, key_dim].
            value_cache (array): Input array [1, num_blocks, num_heads, block_size, value_dim].
            slot_mapping (array): Input array [num_tokens].

        Returns:
            array: ``Dummy output``
      )");
}
