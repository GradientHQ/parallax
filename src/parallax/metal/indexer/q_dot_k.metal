// Inputs provided by MLX wrapper:
// q, key_cache, block_table, output (pointers)
// context_len, block_size, num_heads, head_dim, num_layers, num_total_blocks, max_blocks (scalars)

uint3 gid = thread_position_in_grid;

int token_in_block = gid.x;
int head_idx = gid.y;

if (token_in_block >= block_size || head_idx >= num_heads) return;

int num_valid_blocks = (context_len + block_size - 1) / block_size;

long k_block_stride = num_heads * block_size * head_dim;
long k_head_stride = block_size * head_dim;
long k_layer_stride = (long)num_total_blocks * k_block_stride;

for (int b = 0; b < num_valid_blocks; b++) {
    int block_num = block_table[b];
    int logical_idx = b * block_size + token_in_block;

    if (logical_idx >= context_len) continue;

    long k_base = (long)block_num * k_block_stride +
                  head_idx * k_head_stride +
                  token_in_block * head_dim;

    float score = 0.0f;
    int q_base = head_idx * head_dim;

    for (int d = 0; d < head_dim; d++) {
        score += (float)q[q_base + d] * (float)key_cache[k_base + d];
    }

    output[head_idx * context_len + logical_idx] = score;
}
