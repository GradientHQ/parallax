
// Inputs:
// key, value, key_cache, value_cache, slot_mapping
// key_stride, value_stride, num_kv_heads, head_dim, block_size, layer_idx,
// num_layers, num_blocks (All are pointers)

// Cast away const for cache updates
device {{T}} *key_cache_mut = (device {{T}} *)key_cache;
device {{T}} *value_cache_mut = (device {{T}} *)value_cache;

uint3 gid = thread_position_in_grid;

int batch_idx = gid.y;
int flat_dim = gid.x;

// Dereference constants
// MLX seems to pass scalars as values if they are 0-D arrays?
// Or maybe constant int& ref?
// Let's try treating them as values first.
int _num_kv_heads = num_kv_heads;
int _head_dim = head_dim;
int _block_size = block_size;
int _layer_idx = layer_idx;
int _num_blocks = num_blocks;

int head_idx = flat_dim / _head_dim;
int dim_idx = flat_dim % _head_dim;

if (head_idx >= _num_kv_heads)
  return;

// Input Index
// key: [batch, num_kv_heads, head_dim]
int src_idx = batch_idx * (_num_kv_heads * _head_dim) + flat_dim;

long slot_idx = slot_mapping[batch_idx];
int block_idx = slot_idx / _block_size;
int block_offset = slot_idx % _block_size;

// Cache Layout: (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
long layer_stride = (long)_num_blocks * _num_kv_heads * _block_size * _head_dim;
long block_stride = _num_kv_heads * _block_size * _head_dim;
long head_stride = _block_size * _head_dim;
long offset_stride = _head_dim;

long dest_idx = _layer_idx * layer_stride + block_idx * block_stride +
                head_idx * head_stride + block_offset * offset_stride + dim_idx;

key_cache_mut[dest_idx] = key[src_idx];
value_cache_mut[dest_idx] = value[src_idx];
