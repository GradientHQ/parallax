"""
Tests for the ShardedModel loader utilities on CUDA.
These tests are similar to test_model.py but use CUDA backend instead of MLX.
"""

from typing import List, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallax.utils.utils import is_cuda_available

CUDA_MODEL_REPO = "Qwen/Qwen3-0.6B"
TOTAL_LAYERS = 28


@pytest.fixture(scope="module")
def ref_model_and_tokenizer():
    """Load reference model and tokenizer for CUDA tests"""
    if not is_cuda_available():
        pytest.skip("CUDA not available")
    
    model = AutoModelForCausalLM.from_pretrained(
        CUDA_MODEL_REPO,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(CUDA_MODEL_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    yield model, tokenizer
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


@pytest.mark.cuda
def test_cuda_model_forward_pass(ref_model_and_tokenizer):
    """Test basic forward pass of CUDA model"""
    if not is_cuda_available():
        pytest.skip("CUDA not available")
    
    model, tokenizer = ref_model_and_tokenizer
    
    # Test with a simple prompt
    prompt = "The capital of China is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check output shape
    assert outputs.logits.shape[0] == 1  # batch size
    assert outputs.logits.shape[1] == inputs.input_ids.shape[1]  # sequence length
    assert outputs.logits.shape[2] == model.config.vocab_size  # vocab size


@pytest.mark.cuda
def test_cuda_model_generation(ref_model_and_tokenizer):
    """Test text generation with CUDA model"""
    if not is_cuda_available():
        pytest.skip("CUDA not available")
    
    model, tokenizer = ref_model_and_tokenizer
    
    prompts = [
        "The capital of China is",
        "Qwen is a large language model",
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Verify generation
        assert len(generated_text) > len(prompt)
        assert prompt in generated_text

