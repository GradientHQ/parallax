// Inputs provided by MLX wrapper:
// key, key_cache, slot_mapping (pointers)
// key_stride, num_heads, head_dim, block_size, num_layers, num_blocks (scalars)

device {{T}} *key_cache_mut = (device {{T}} *)key_cache;

uint3 gid = thread_position_in_grid;

int dim_idx = gid.x; // range [0, num_heads * head_dim)
int token_idx = gid.y; // range [0, num_tokens)

if (dim_idx >= num_heads * head_dim) return;

int head_idx = dim_idx / head_dim;
int d_idx = dim_idx % head_dim;

int slot = slot_mapping[token_idx];
if (slot < 0) return;

int block_idx = slot / block_size;
int block_offset = slot % block_size;

long src_idx = (long)token_idx * num_heads * head_dim + dim_idx;

long k_block_stride = num_heads * block_size * head_dim;
long k_head_stride = block_size * head_dim;

long k_layer_stride = (long)num_blocks * k_block_stride;

long dest_idx = block_idx * k_block_stride +
                head_idx * k_head_stride +
                block_offset * head_dim +
                d_idx;

key_cache_mut[dest_idx] = key[src_idx];
