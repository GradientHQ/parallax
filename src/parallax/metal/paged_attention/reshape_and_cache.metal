device {{T}} *key_cache_mut = (device {{T}} *)key_cache;
device {{T}} *value_cache_mut = (device {{T}} *)value_cache;
// reshape_and_cache logic
// Inputs are provided by MLX wrapper:
// key, value, key_cache, value_cache, slot_mapping, ...

// MLX provided variable for grid position
uint3 gid = thread_position_in_grid;

int kv_head_dim_idx = gid.x;
int token_idx = gid.y;

// Scalars are passed by value (int32), so no dereference needed
int n_kv_heads = num_kv_heads;
int h_dim = head_dim;

if (kv_head_dim_idx >= n_kv_heads * h_dim)
  return;

int head_idx = kv_head_dim_idx / h_dim;
int dim_idx = kv_head_dim_idx % h_dim;

int64_t slot = slot_mapping[token_idx];

// Handle padding tokens (slot == -1)
if (slot < 0) {
  return;
}

int b_size = block_size;
int64_t block_idx = slot / b_size;
int64_t block_offset = slot % b_size;

int l_idx = layer_idx;
int n_blocks = num_blocks;

// Calculate source index
// key shape: (num_tokens, num_kv_heads, head_dim)
int64_t src_idx =
    (int64_t)token_idx * n_kv_heads * h_dim + head_idx * h_dim + dim_idx;

// Calculate destination index
int64_t head_stride = b_size * h_dim;
int64_t block_stride = n_kv_heads * head_stride;
int64_t layer_stride = n_blocks * block_stride;

int64_t dest_idx = (int64_t)l_idx * layer_stride + block_idx * block_stride +
                   (int64_t)head_idx * head_stride + block_offset * h_dim +
                   dim_idx;

// Cast away const for cache updates
// 'key_cache' is 'const device {{T}}*' in inputs, we need 'device {{T}}*'

key_cache_mut[dest_idx] = key[src_idx];
value_cache_mut[dest_idx] = value[src_idx];
