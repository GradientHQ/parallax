
// Inputs:
// queries, key_cache, value_cache, block_tables, context_lengths, sinks
// output (output array)
// num_heads, num_kv_heads, k_head_dim, v_head_dim, block_size, max_blocks,
// num_layers, num_total_blocks, scale, window_size (All pointers)

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
int _k_head_dim = k_head_dim;
int _v_head_dim = v_head_dim;
int _block_size = block_size;
int _max_blocks = max_blocks;
int _num_total_blocks = num_total_blocks;
float _scale = scale;
int _window_size = window_size;

if (head_idx >= _num_heads)
  return;

int kv_head_idx = head_idx / (_num_heads / _num_kv_heads);

// Load Query
// Q: [batch, num_heads, k_head_dim]
// Thread i loads elements i, i+32, ...

// Support up to 256 head dim (8 * 32)
float q_vec[8] = {0.0f};

int q_offset = batch_idx * _num_heads * _k_head_dim + head_idx * _k_head_dim;

for (int i = tid.x; i < _k_head_dim; i += 32) {
  if (i < 256) {
    q_vec[i / 32] = queries[q_offset + i];
  }
}

// Running statistics for Softmax
float m_i = -INFINITY;
float l_i = 0.0f;
float acc_vec[8] = {0.0f};

// -------------------------------------------------------------------------
// Handle Implicit Sink Token (KV = 0, Not in Block Table)
// -------------------------------------------------------------------------
// The sink token is assumed to be an extra token (logical pos 0? or just extra)
// that is NOT stored in the paged KV blocks.
// Its Key is 0, Value is 0.
// Its Attention Score = (Q * K_sink) * scale + sinks_bias
//                     = (Q * 0) * scale + sinks[head_idx]
//                     = sinks[head_idx]

float sink_score = sinks[head_idx];

// Initialize Softmax stats with Sink Token if it's not masked out (-inf)
// We use a threshold slightly larger than -INFINITY to avoid precision issues
if (sink_score > -3.40282e+38f) { // roughly > -FLT_MAX
    m_i = sink_score;
    l_i = 1.0f;
    // acc_vec remains 0.0f because V_sink is 0.
}

int context_len = context_lengths[batch_idx];
int num_context_blocks = (context_len + _block_size - 1) / _block_size;

// Strides for Key
long k_block_stride = _num_kv_heads * _block_size * _k_head_dim;
long k_head_stride = _block_size * _k_head_dim;

// Strides for Value
long v_block_stride = _num_kv_heads * _block_size * _v_head_dim;
long v_head_stride = _block_size * _v_head_dim;

// Iterate over blocks
for (int b = 0; b < num_context_blocks; b++) {
  int block_num = block_tables[batch_idx * _max_blocks + b];

  // Calculate logic position range for this block
  int block_start_pos = b * _block_size;
  int block_end_pos = block_start_pos + _block_size; // exclusive

  // Handle Window Attention Skip
  // Current query position is at (context_len - 1)
  // Window range is [context_len - 1 - window_size, context_len - 1]

  // NOTE: Sink token is handled separately above, so we don't need to protect block 0
  // unless block 0 also contains other vital tokens (it usually does).

  if (_window_size > 0) {
      int window_start = context_len - 1 - _window_size;
      if (block_end_pos <= window_start) {
          // Entire block is out of window
          continue;
      }
  }

  long k_block_base =
      block_num * k_block_stride + kv_head_idx * k_head_stride;
  long v_block_base =
      block_num * v_block_stride + kv_head_idx * v_head_stride;

  int tokens_in_block = _block_size;
  if (b == num_context_blocks - 1) {
    tokens_in_block = context_len % _block_size;
    if (tokens_in_block == 0)
      tokens_in_block = _block_size;
  }

  for (int t = 0; t < tokens_in_block; t++) {
    int token_pos = block_start_pos + t;

    // Check if token is within window
    if (_window_size > 0) {
        int window_start = context_len - 1 - _window_size;
        if (token_pos < window_start) {
            continue;
        }
    }

    // Compute Dot Product Q * K[t]
    float score = 0.0f;
    for (int i = tid.x; i < _k_head_dim; i += 32) {
      // offset inside block: t * k_head_dim + i
      float k_val = key_cache[k_block_base + t * _k_head_dim + i];

      if (i < 256) {
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
    for (int i = tid.x; i < _v_head_dim; i += 32) {
      float v_val = value_cache[v_block_base + t * _v_head_dim + i];
      if (i < 256) {
        acc_vec[i / 32] = acc_vec[i / 32] * alpha + v_val * beta;
      }
    }
  }
}

// Finalize Output
for (int i = 0; i < 8; i++) {
  acc_vec[i] /= l_i;
}

int out_offset = batch_idx * _num_heads * _v_head_dim + head_idx * _v_head_dim;

for (int i = tid.x; i < _v_head_dim; i += 32) {
  if (i < 256) {
    output[out_offset + i] = ({{T}})acc_vec[i / 32];
  }
}
