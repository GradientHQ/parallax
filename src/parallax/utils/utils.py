"""Utility functions."""

import random
import socket
from typing import List

import mlx.core as mx
import numpy as np
import psutil
import torch
import zmq
from mlx_lm.utils import get_model_path, load_config


def is_cuda_available():
    """Check backend supports cuda"""
    return torch.cuda.is_available()


def is_mps_available():
    """Check backend supports mps"""
    return torch.mps.is_available()


def get_current_device():
    """
    Returns the backend device name.
    Parallax currently supports cuda, mlx, cpu
    """
    device = "cpu"
    if is_cuda_available():
        device = "cuda"
    if is_mps_available():
        device = "mlx"
    return device


def get_device_dtype(dtype_str: str, device: str):
    """Gets the real data type according to current device"""
    if device == "cuda":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
    else:
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
    return dtype_map[dtype_str]


def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool):
    """Create and configure a ZeroMQ socket.

    Ported from SGLang.
    """
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def get_infinite_value_by_dtype(dtype: mx.Dtype):
    """Returns infinite value according to mx dtype"""
    inf = 6e4
    if dtype in (mx.bfloat16, mx.float32):
        inf = 1e9
    return inf


def pad_prefix_caches(
    cache: List, input_lengths: List, dtype: mx.Dtype = mx.bfloat16
) -> tuple[mx.array, mx.array]:
    """
    Pads prefix kv caches.

    Returnas:
        - mx.array: The padded batch of caches with a shape of [B, max_input_seq_len].
        - mx.array: The corresponding 4D k mask with a shape of [B, 1, 1, max_output_seq_len].
    """
    caches_mx = [mx.array(i) if isinstance(i, np.ndarray) else i for i in cache]

    seq_len_axis = 2
    max_input_len = 0
    max_output_len = 0
    for i, tensor in enumerate(caches_mx):
        max_input_len = max(max_input_len, tensor.shape[seq_len_axis])
        max_output_len = max(max_output_len, input_lengths[i])

    padded_tensors = []
    k_masks = []
    for i, tensor in enumerate(caches_mx):
        cache_len = tensor.shape[seq_len_axis]
        num_kv_padding = max_input_len - cache_len
        input_seq_len = input_lengths[i] - 1
        num_mask_padding = max_output_len - input_seq_len - 1

        if num_kv_padding > 0:
            pad_shape = list(tensor.shape)
            pad_shape[seq_len_axis] = num_kv_padding
            padding = mx.zeros(tuple(pad_shape), dtype=tensor.dtype)
            padded_tensors.append(mx.concatenate([tensor, padding], axis=seq_len_axis))
        else:
            padded_tensors.append(tensor)

        k_masks.append([1] * (input_seq_len + 1) + [0] * num_mask_padding)

    padded_batch = mx.stack(padded_tensors, axis=0)
    attention_mask = mx.array(k_masks, dtype=dtype)[:, None, None, :]
    return padded_batch, attention_mask


def pad_inputs(
    pad_value: int, inputs: List, dtype: mx.Dtype = mx.bfloat16
) -> tuple[mx.array, mx.array]:
    """
    Pads a list of sequences (token ID lists or hidden state arrays) to the same length.
    # TODO: refactor this allow cumstomized dim.

    Args:
        pad_value: The value to use for padding. For token IDs, this should be the
                   tokenizer's pad_token_id. For hidden states, it's ignored (always 0).
        inputs: A list of sequences to pad. Each sequence can be a list of integers
                or an MLX/NumPy array of hidden states.
        dtype: The data type for the padded inputs.

    Returns:
        A tuple containing:
        - mx.array: The padded batch of inputs.
        - mx.array: The corresponding 4D attention mask.
    """
    if not inputs:
        return mx.array([]), mx.array([])

    max_len = 0
    attention_masks = []

    # Check the dimensionality of the input to handle KV cache padding
    is_kv_cache = isinstance(inputs[0], mx.array) and inputs[0].ndim == 4

    if isinstance(inputs[0], list):  # Assuming list of token IDs
        for tokens in inputs:
            max_len = max(max_len, len(tokens))

        padded_sequences = []
        for tokens in inputs:
            num_padding = max_len - len(tokens)
            padded_sequences.append(tokens + [pad_value] * num_padding)
            attention_masks.append([1] * len(tokens) + [0] * num_padding)

        padded_batch = mx.array(padded_sequences)

    elif isinstance(
        inputs[0], (mx.array, np.ndarray)
    ):  # Assuming list of hidden states or KV caches
        inputs_mx = [mx.array(i) if isinstance(i, np.ndarray) else i for i in inputs]

        # Determine sequence length axis based on input type
        # kv cache: (n_layers, n_kv_h, source_len, h_dim)
        seq_len_axis = 2 if is_kv_cache else 0
        for tensor in inputs_mx:
            max_len = max(max_len, tensor.shape[seq_len_axis])

        padded_tensors = []
        for tensor in inputs_mx:
            seq_len = tensor.shape[seq_len_axis]
            num_padding = max_len - seq_len

            if num_padding > 0:
                if is_kv_cache:
                    pad_shape = list(tensor.shape)
                    pad_shape[seq_len_axis] = num_padding
                    padding = mx.zeros(tuple(pad_shape), dtype=tensor.dtype)
                else:
                    # Hidden state shape: (seq_len, hidden_dim)
                    hidden_dim = tensor.shape[1]
                    padding = mx.zeros((num_padding, hidden_dim), dtype=tensor.dtype)
                padded_tensors.append(mx.concatenate([tensor, padding], axis=seq_len_axis))
            else:
                padded_tensors.append(tensor)
            attention_masks.append([1] * seq_len + [0] * num_padding)

        padded_batch = mx.stack(padded_tensors, axis=0)

    else:
        raise TypeError(f"Unsupported input type for padding: {type(inputs[0])}")

    # Create 4D attention mask, ensuring it's float
    attention_mask = mx.array(attention_masks, dtype=dtype)[:, None, None, :]
    return padded_batch, attention_mask


def create_causal_mask(seq_len: int, total_len: int, dtype=mx.bfloat16) -> mx.array:
    """
    Creates a causal attention mask of shape (input_seq, total_seq).

    Args:
        input_seq: The length of sequence.
        total_seq: The length of sequence + cached sequence.
        dtype: The data type for the mask.

    Returns:
        mx.array: A square matrix with -1e9 on the upper triangle (excluding the diagonal).
    """
    assert (
        total_len >= seq_len
    ), f"Total lengths {total_len} should be no less than input sequence {seq_len}."
    inf_value = get_infinite_value_by_dtype(dtype)
    mask = mx.triu(mx.full((seq_len, seq_len), -inf_value, dtype), k=1)
    if total_len == seq_len:
        return mask
    # total lengths is larger than input sequence length
    cached_zeros = mx.zeros((seq_len, total_len - seq_len), dtype)
    final_mask = mx.concatenate([cached_zeros, mask], axis=1)
    return final_mask


def combine_padding_and_causal_masks(
    padding_mask: mx.array, causal_mask: mx.array, dtype=mx.bfloat16
) -> mx.array:
    """
    Combines a padding mask and a causal mask.

    Args:
        padding_mask: A 4D padding mask of shape (B, 1, 1, total_seq)
                      where masked positions are 0 and unmasked are 1.
        causal_mask: A 2D causal mask of shape (input_seq, total_seq).
        dtype: The data type for the final mask.

    Returns:
        mx.array: A combined attention mask, typically of shape (B, 1, input_seq, total_seq).
    """
    inf_value = get_infinite_value_by_dtype(dtype)
    padding_mask_float = (padding_mask - 1) * inf_value
    padding_mask_float = padding_mask_float.astype(dtype)
    return causal_mask + padding_mask_float


def fetch_model_from_hf(name: str):
    """Fetch model from huggingface and returns model config"""
    model_path = get_model_path(name)[0]
    config = load_config(model_path)
    return config


def is_port_available(port: int):
    """
    Copied from SGLang.
    Return whether a port is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


def initialize_nccl_port():
    """Initialize nccl port for GPU"""
    nccl_port = random.randint(4000, 5000)
    while True:
        if is_port_available(nccl_port):
            break
        if nccl_port < 60000:
            nccl_port += 42
        else:
            nccl_port -= 43
    return nccl_port
