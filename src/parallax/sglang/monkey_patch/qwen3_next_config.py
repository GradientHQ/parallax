# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3Hybrid model configuration"""

import enum

from transformers.utils import logging

logger = logging.get_logger(__name__)


# NOTE: HybridLayerType
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


@property
def monkey_patch_layers_block_type(self):
    layer_type_list = []

    for l in range(self.num_hidden_layers):
        if l + 1 < self.start_layer or l + 1 >= self.end_layer:
            continue

        if (l + 1) % self.full_attention_interval == 0:
            layer_type_list.append(HybridLayerType.full_attention.value)
        else:
            layer_type_list.append(HybridLayerType.linear_attention.value)

    return layer_type_list
