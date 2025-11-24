
// Inputs:
// queries, key_cache, value_cache, block_tables, context_lengths
// output (output array)
// num_heads, num_kv_heads, head_dim, block_size, max_blocks, layer_idx,
// num_layers, num_total_blocks, scale (All pointers)

uint3 gid = thread_position_in_grid;
uint3 tid = thread_position_in_threadgroup;

// Each threadgroup handles one head.
// Assuming threadgroup size is 32 (SIMD width).
// head_idx comes from the group index.
// Or we can calculate it from gid if grid is linear.
// grid.x = num_heads * 32.

int head_idx = gid.x / 32;
int batch_idx = gid.y;

// Dereference constants
int _num_heads = num_heads;
int _num_kv_heads = num_kv_heads;
int _head_dim = head_dim;
int _block_size = block_size;
int _max_blocks = max_blocks;
int _layer_idx = layer_idx;
int _num_total_blocks = num_total_blocks;
float _scale = scale;

if (head_idx >= _num_heads)
  return;

int kv_head_idx = head_idx / (_num_heads / _num_kv_heads);

// Load Query
// Q: [batch, num_heads, head_dim]
// Thread i loads elements i, i+32, ...

float q_vec[4] = {0.0f, 0.0f, 0.0f, 0.0f};

int q_offset = batch_idx * _num_heads * _head_dim + head_idx * _head_dim;

for (int i = tid.x; i < _head_dim; i += 32) {
  if (i < 128) {
    q_vec[i / 32] = queries[q_offset + i];
  }
}

// Running statistics for Softmax
float m_i = -INFINITY;
float l_i = 0.0f;
float acc_vec[4] = {0.0f, 0.0f, 0.0f, 0.0f};

int context_len = context_lengths[batch_idx];
int num_context_blocks = (context_len + _block_size - 1) / _block_size;

// Strides
long layer_stride =
    (long)_num_total_blocks * _num_kv_heads * _block_size * _head_dim;
long block_stride = _num_kv_heads * _block_size * _head_dim;
long head_stride = _block_size * _head_dim;

long layer_offset = _layer_idx * layer_stride;

// Iterate over blocks
for (int b = 0; b < num_context_blocks; b++) {
  int block_num = block_tables[batch_idx * _max_blocks + b];

  long block_base =
      layer_offset + block_num * block_stride + kv_head_idx * head_stride;

  int tokens_in_block = _block_size;
  if (b == num_context_blocks - 1) {
    tokens_in_block = context_len % _block_size;
    if (tokens_in_block == 0)
      tokens_in_block = _block_size;
  }

  for (int t = 0; t < tokens_in_block; t++) {
    // Compute Dot Product Q * K[t]
    float score = 0.0f;
    for (int i = tid.x; i < _head_dim; i += 32) {
      // offset inside block: t * head_dim + i
      float k_val = key_cache[block_base + t * _head_dim + i];

      if (i < 128) {
        score += q_vec[i / 32] * k_val;
      }
    }

    // SIMD Reduction for score
    score = simd_sum(score);
    score *= _scale;

    // Softmax update
    float m_prev = m_i;
    m_i = max(m_prev, score);
    float alpha = exp(m_prev - m_i);
    float beta = exp(score - m_i);

    l_i = l_i * alpha + beta;

    // Accumulate V
    for (int i = tid.x; i < _head_dim; i += 32) {
      float v_val = value_cache[block_base + t * _head_dim + i];
      if (i < 128) {
        acc_vec[i / 32] = acc_vec[i / 32] * alpha + v_val * beta;
      }
    }
  }
}

// Finalize Output
for (int i = 0; i < 4; i++) {
  acc_vec[i] /= l_i;
}

int out_offset = batch_idx * _num_heads * _head_dim + head_idx * _head_dim;

for (int i = tid.x; i < _head_dim; i += 32) {
  if (i < 128) {
    output[out_offset + i] = ({{T}})acc_vec[i / 32];
  }
}
