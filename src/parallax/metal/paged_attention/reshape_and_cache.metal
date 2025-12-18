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
int k_dim = k_head_dim;
int v_dim = v_head_dim;
int max_dim = (k_dim > v_dim) ? k_dim : v_dim;

if (kv_head_dim_idx >= n_kv_heads * max_dim)
  return;

int head_idx = kv_head_dim_idx / max_dim;
int dim_idx = kv_head_dim_idx % max_dim;

int slot = slot_mapping[token_idx];

// Handle padding tokens (slot == -1)
if (slot < 0) {
  return;
}

int b_size = block_size;
int block_idx = slot / b_size;
int block_offset = slot % b_size;

// Handle Key
if (dim_idx < k_dim) {
    // Calculate source index
    // key shape: (num_tokens, num_kv_heads, k_head_dim)
    int64_t src_idx =
        (int64_t)token_idx * n_kv_heads * k_dim + head_idx * k_dim + dim_idx;

    // Calculate destination index
    int64_t head_stride = b_size * k_dim;
    int64_t block_stride = n_kv_heads * head_stride;

    int64_t dest_idx = (int64_t)block_idx * block_stride +
                       (int64_t)head_idx * head_stride + block_offset * k_dim +
                       dim_idx;

    key_cache_mut[dest_idx] = key[src_idx];
}

// Handle Value
if (dim_idx < v_dim) {
    // Calculate source index
    // value shape: (num_tokens, num_kv_heads, v_head_dim)
    int64_t src_idx =
        (int64_t)token_idx * n_kv_heads * v_dim + head_idx * v_dim + dim_idx;

    // Calculate destination index
    int64_t head_stride = b_size * v_dim;
    int64_t block_stride = n_kv_heads * head_stride;

    int64_t dest_idx = (int64_t)block_idx * block_stride +
                       (int64_t)head_idx * head_stride + block_offset * v_dim +
                       dim_idx;

    value_cache_mut[dest_idx] = value[src_idx];
}
