"""Cache layer type labels shared by executor and cache manager."""

from typing import Final

ATTENTION: Final = "attention"
MLA_ATTENTION: Final = "mla_attention"
DSA_ATTENTION: Final = "dsa_attention"
MSA_ATTENTION: Final = "msa_attention"
LINEAR: Final = "linear"

ATTENTION_LAYER_TYPES: Final = frozenset(
    {
        ATTENTION,
        MLA_ATTENTION,
        DSA_ATTENTION,
        MSA_ATTENTION,
    }
)
MLA_CACHE_LAYER_TYPES: Final = frozenset({MLA_ATTENTION, DSA_ATTENTION})
INDEX_CACHE_LAYER_TYPES: Final = frozenset({DSA_ATTENTION, MSA_ATTENTION})
