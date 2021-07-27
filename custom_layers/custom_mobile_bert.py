# MIT License
#
# Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.mobilebert.configuration_mobilebert import MobileBertConfig

from custom_layers.custom_embedding import CustomEmbedding
from custom_layers.custom_linear import CustomLinear
from custom_layers.custom_layernorm import CustomLayerNorm, CustomNoNorm
from custom_layers.custom_bert import BertEmbeddings
from copy import deepcopy
from loss import CrossEntropyLossSoft
from loss import *

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"
_CONFIG_FOR_DOC = "MobileBertConfig"
_TOKENIZER_FOR_DOC = "MobileBertTokenizer"

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["google/mobilebert-uncased"]


def load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.replace("ffn_layer", "ffn")
        name = name.replace("FakeLayerNorm", "LayerNorm")
        name = name.replace("extra_output_weights", "dense/kernel")
        name = name.replace("bert", "mobilebert")
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class NoNorm(nn.Module):
    def __init__(self, feat_size, eps=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


NORM2FN = {"layer_norm": CustomLayerNorm, "no_norm": CustomNoNorm}


def calc_dropout(dropout, sample_hidden_size, super_hidden_size):
    return dropout * 1.0 * sample_hidden_size / super_hidden_size


class MobileBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.word_embeddings = CustomEmbedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = CustomEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = CustomEmbedding(
            config.type_vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = CustomLinear(
            embedded_input_size, config.hidden_size
        )

        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def set_sample_config(self, config):
        # we name all param inside sampling ocnfig as sample_*
        # hidden_size -> sample_emb_hidden_size
        sample_hidden_size = config.sample_hidden_size
        embed_dim_multiplier = 3 if self.trigram_input else 1
        sample_embedded_input_size = config.sample_embedding_size * embed_dim_multiplier

        self.word_embeddings.set_sample_config(sample_hidden_size, part="encoder")
        self.position_embeddings.set_sample_config(sample_hidden_size, part="encoder")
        self.token_type_embeddings.set_sample_config(sample_hidden_size, part="encoder")

        self.LayerNorm.set_sample_config(sample_hidden_size)
        self.embedding_transformation.set_sample_config(
            sample_embedded_input_size, sample_hidden_size
        )
        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=sample_hidden_size,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    def get_active_subnet(self, config):
        sublayer = MobileBertEmbeddings(config)
        sublayer.word_embeddings = self.word_embeddings.get_active_subnet(
            part="encoder"
        )
        sublayer.position_embeddings = self.position_embeddings.get_active_subnet(
            part="encoder"
        )
        sublayer.token_type_embeddings = self.token_type_embeddings.get_active_subnet(
            part="encoder"
        )
        self.embedding_transformation.get_active_subnet()

        sublayer.LayerNorm = self.LayerNorm.get_active_subnet()
        return sublayer

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = torch.cat(
                [
                    nn.functional.pad(
                        inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0
                    ),
                    inputs_embeds,
                    nn.functional.pad(
                        inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0
                    ),
                ],
                dim=2,
            )
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# same
# differnce is only true_hidden_size
class MobileBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.intra_bottleneck_size / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = CustomLinear(config.true_hidden_size, self.all_head_size)
        self.key = CustomLinear(config.true_hidden_size, self.all_head_size)
        self.value = CustomLinear(
            config.true_hidden_size
            if config.use_bottleneck_attention
            else config.hidden_size,
            self.all_head_size,
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.sample_num_attention_heads,
            self.sample_attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def set_sample_config(self, config, tiny_attn=False):
        if tiny_attn:
            self.sample_num_attention_heads = 1
            # no sampling
            sample_true_hidden_size = config.sample_true_hidden_size
            sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
            sample_hidden_size = config.sample_hidden_size
            self.sample_attention_head_size = int(
                sample_intra_bottleneck_size / self.sample_num_attention_heads
            )
        else:
            self.sample_num_attention_heads = config.sample_num_attention_heads
            sample_true_hidden_size = config.sample_true_hidden_size
            sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
            sample_hidden_size = config.sample_hidden_size
            self.sample_attention_head_size = int(
                sample_intra_bottleneck_size / self.sample_num_attention_heads
            )

        self.sample_all_head_size = (
            self.sample_num_attention_heads * self.sample_attention_head_size
        )

        # print("Sample IntraBottleneck size: ", sample_intra_bottleneck_size)
        # print("Sample num heads size: ", self.sample_num_attention_heads)
        # print("Sample attention head size: ", self.sample_attention_head_size)
        # print(
        #     f"Changing num_attention heads from {self.num_attention_heads} -> {self.sample_num_attention_heads}"
        # )
        # print(
        #     f"Changing attention head size from {self.attention_head_size} -> {self.sample_attention_head_size}"
        # )

        self.query.set_sample_config(sample_true_hidden_size, self.sample_all_head_size)
        self.key.set_sample_config(sample_true_hidden_size, self.sample_all_head_size)
        if config.use_bottleneck_attention:
            self.value.set_sample_config(
                sample_true_hidden_size, self.sample_all_head_size
            )
        else:
            self.value.set_sample_config(sample_hidden_size, self.sample_all_head_size)
        sample_attention_probs_dropout_prob = calc_dropout(
            config.attention_probs_dropout_prob,
            super_hidden_size=config.num_attention_heads,
            sample_hidden_size=config.sample_num_attention_heads,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_attention_probs_dropout_prob)

    def get_active_subnet(self, config):
        sublayer = MobileBertSelfAttention(config)
        sublayer.set_sample_config(config)  ## Necessary evil
        sublayer.query = self.query.get_active_subnet()
        sublayer.key = self.key.get_active_subnet()
        sublayer.value = self.value.get_active_subnet()

        return sublayer

    def forward(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # assert (torch.isfinite(attention_scores).all(), "NaNs in attention scores 1")
        attention_scores = attention_scores / math.sqrt(self.sample_attention_head_size)
        # assert (torch.isfinite(attention_scores).all(), "NaNs in attention scores 2")
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # adding 1e-8 for preventing of small values in attention scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # assert (
        #     torch.isfinite(attention_probs).all(),
        #     "NaNs in attention probabilities after softmax",
        # )
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # assert (
        #     torch.isfinite(attention_probs).all(),
        #     "NaNs in attention probabilities after dropout",
        # )
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.sample_all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


# same
# differnce is only true_hidden_size
class MobileBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = CustomLinear(
            config.intra_bottleneck_size, config.intra_bottleneck_size
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, eps=config.layer_norm_eps
        )
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, config):
        sample_true_hidden_size = config.sample_true_hidden_size
        sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        self.dense.set_sample_config(
            sample_intra_bottleneck_size, sample_intra_bottleneck_size
        )
        self.LayerNorm.set_sample_config(sample_intra_bottleneck_size)

        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        if not self.use_bottleneck:
            sample_hidden_dropout_prob = calc_dropout(
                config.hidden_dropout_prob,
                super_hidden_size=config.intra_bottleneck_size,
                sample_hidden_size=sample_intra_bottleneck_size,
            )
            self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    def get_active_subnet(self, config):
        sublayer = MobileBertSelfOutput(config)

        sublayer.dense = self.dense.get_active_subnet()
        sublayer.LayerNorm = self.LayerNorm.get_active_subnet()

        return sublayer

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


# same
class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MobileBertSelfAttention(config)
        self.output = MobileBertSelfOutput(config)
        self.pruned_heads = set()

    def set_sample_config(self, config):
        self.self.set_sample_config(config)
        self.output.set_sample_config(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def get_active_subnet(self, config):
        sublayer = MobileBertAttention(config)
        sublayer.self = self.self.get_active_subnet(config)
        sublayer.output = self.output.get_active_subnet(config)

        return sublayer

    def forward(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        attention_output = self.output(self_outputs[0], layer_input)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# same
# true_hidden_size
class MobileBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(
            config.intra_bottleneck_size, config.intermediate_size
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def set_sample_config(self, config):
        sample_intermediate_size = config.sample_intermediate_size
        sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        sample_true_hidden_size = config.sample_true_hidden_size
        self.dense.set_sample_config(
            sample_intra_bottleneck_size, sample_intermediate_size
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# diff
class OutputBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.intra_bottleneck_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # add linear scaler to weight the red lines
        self.linear_scaler = nn.Parameter(
            # torch.zeros(config.true_hidden_size) + config.layer_norm_eps
            torch.zeros(config.true_hidden_size)
            # torch.zeros(1)
        )
        self.register_parameter("linear_scaler", self.linear_scaler)

    def set_sample_config(self, config):
        sample_true_hidden_size = config.sample_true_hidden_size
        sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        sample_hidden_size = config.sample_hidden_size
        self.dense.set_sample_config(sample_intra_bottleneck_size, sample_hidden_size)
        self.LayerNorm.set_sample_config(sample_hidden_size)

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=sample_hidden_size,
        )
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    def forward(self, hidden_states, residual_tensor):
        # make sure linear scaler is clamped between [0, 1]
        w = self.linear_scaler.data
        w = torch.clamp(w, min=0.0, max=1.0)

        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs) * (1.0 - self.linear_scaler)
        layer_outputs = layer_outputs + self.linear_scaler * residual_tensor
        # layer_outputs = self.LayerNorm(
        #     layer_outputs + self.linear_scaler * residual_tensor
        # )
        return layer_outputs


class MobileBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = CustomLinear(
            config.intermediate_size, config.intra_bottleneck_size
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, eps=config.layer_norm_eps
        )
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)

    def set_sample_config(self, config):
        sample_intermediate_size = config.sample_intermediate_size
        sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        sample_true_hidden_size = config.sample_true_hidden_size
        self.LayerNorm.set_sample_config(sample_intra_bottleneck_size)
        self.dense.set_sample_config(
            sample_intermediate_size, sample_intra_bottleneck_size
        )
        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.intermediate_size,
            sample_hidden_size=sample_intermediate_size,
        )

        if not self.use_bottleneck:
            self.dropout = nn.Dropout(sample_hidden_dropout_prob)
        else:
            self.bottleneck.set_sample_config(config)

    def forward(self, intermediate_states, residual_tensor_1, residual_tensor_2):
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            # linear scale the embeddings
            # residual_tensor_2 = self.linear_scaler * residual_tensor_2
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output


class BottleneckLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, eps=config.layer_norm_eps
        )

    def set_sample_config(self, config):
        sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        sample_hidden_size = config.sample_hidden_size
        self.LayerNorm.set_sample_config(sample_intra_bottleneck_size)
        self.dense.set_sample_config(sample_hidden_size, sample_intra_bottleneck_size)

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        # layer_input = self.LayerNorm(layer_input)
        return layer_input


class Bottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def set_sample_config(self, config):
        self.input.set_sample_config(config)

        if self.key_query_shared_bottleneck:
            self.attention.set_sample_config(config)

    def forward(self, hidden_states):
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (
                shared_attention_input,
                shared_attention_input,
                hidden_states,
                bottlenecked_hidden_states,
            )
        else:
            return (
                hidden_states,
                hidden_states,
                hidden_states,
                bottlenecked_hidden_states,
            )


class FFNOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, eps=config.layer_norm_eps
        )

    def set_sample_config(self, config):
        sample_intermediate_size = config.sample_intermediate_size
        sample_true_hidden_size = config.sample_true_hidden_size
        self.dense.set_sample_config(sample_intermediate_size, sample_true_hidden_size)

        self.LayerNorm.set_sample_config(sample_true_hidden_size)

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class FFNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def set_sample_config(self, config):
        self.intermediate.set_sample_config(config)
        self.output.set_sample_config(config)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class MobileBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList(
                [FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)]
            )
        self.is_identity_layer = False

    def set_sample_config(self, config, is_identity_layer=False):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.attention.set_sample_config(config)
        self.intermediate.set_sample_config(config)
        self.output.set_sample_config(config)
        if self.use_bottleneck:
            self.bottleneck.set_sample_config(config)
        if config.num_feedforward_networks > 1:
            for _ffn in self.ffn:
                _ffn.set_sample_config(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        if self.is_identity_layer:
            # print("Returning without any operations")
            return (hidden_states, None, None)
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(
                hidden_states
            )
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (
            (layer_output,)
            + outputs
            + (
                torch.tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs


class MobileBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # assert config.mixing == "mobilebert"
        self.layer = nn.ModuleList(
            [MobileBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def set_sample_config(self, config):

        self.sample_num_hidden_layers = config.sample_num_hidden_layers

        if isinstance(config.sample_intermediate_size, list):
            sample_intermediate_sizes = config.sample_intermediate_size
        else:
            sample_intermediate_sizes = [config.sample_intermediate_size] * len(
                self.layer
            )
        if isinstance(config.sample_num_attention_heads, list):
            sample_num_attention_heads_list = config.sample_num_attention_heads
        else:
            sample_num_attention_heads_list = [config.sample_num_attention_heads] * len(
                self.layer
            )
        if isinstance(config.sample_intra_bottleneck_size, list):
            sample_intra_bottleneck_size = config.sample_intra_bottleneck_size
        else:
            sample_intra_bottleneck_size = [
                config.sample_intra_bottleneck_size
            ] * config.sample_num_hidden_layers

        for i, layer in enumerate(self.layer):
            layer_config = deepcopy(config)

            if i < self.sample_num_hidden_layers:
                layer_config.sample_intermediate_size = sample_intermediate_sizes[i]
                layer_config.sample_num_attention_heads = (
                    sample_num_attention_heads_list[i]
                )
                layer_config.sample_intra_bottleneck_size = (
                    sample_intra_bottleneck_size[i]
                )
                layer.set_sample_config(layer_config, is_identity_layer=False)
            else:
                layer.set_sample_config(layer_config, is_identity_layer=True)

    def get_active_subnet(self, config):
        sublayer = MobileBertEncoder(config)

        if isinstance(config.sample_intermediate_size, list):
            sample_intermediate_sizes = config.sample_intermediate_size
        else:
            sample_intermediate_sizes = [
                config.sample_intermediate_size
            ] * config.sample_num_hidden_layers
        if isinstance(config.num_attention_heads, list):
            sample_num_attention_heads_list = config.num_attention_heads
        else:
            sample_num_attention_heads_list = [
                config.num_attention_heads
            ] * config.sample_num_hidden_layers

        if isinstance(config.intra_bottleneck_size, list):
            sample_intra_bottleneck_size_list = config.intra_bottleneck_size
        else:
            sample_intra_bottleneck_size = [
                config.intra_bottleneck_size
            ] * config.sample_num_hidden_layers

        ### Extracting the subnetworks
        for i in range(config.sample_num_hidden_layers):
            layer_config = deepcopy(config)

            layer_config.sample_intermediate_size = sample_intermediate_sizes[i]
            layer_config.sample_num_attention_heads = sample_num_attention_heads_list[i]
            layer_config.sample_intra_bottleneck_size = (
                sample_intra_bottleneck_size_list[i]
            )
            sublayer.layer[i].set_sample_config(layer_config, is_identity_layer=False)

            sublayer.layer[i] = self.layer[i].get_active_subnet(layer_config)

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class MobileBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # modifying this to be inline with bertpooler
        # self.do_activate = config.classifier_activation
        # if self.do_activate:
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, config):
        if isinstance(config.sample_hidden_size, list):
            sample_hidden_size = config.sample_hidden_size[-1]
        else:
            sample_hidden_size = config.sample_hidden_size
        # if self.do_activate:
        self.dense.set_sample_config(sample_hidden_size, sample_hidden_size)

    def get_active_subnet(self, config):
        sublayer = MobileBertPooler(config)
        sublayer.dense = self.dense.get_active_subnet()
        return sublayer

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        # if not self.do_activate:
        #     return first_token_tensor
        # else:
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MobileBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN["layer_norm"](
            config.hidden_size, eps=config.layer_norm_eps
        )

    def set_sample_config(self, config):
        self.dense.set_sample_config(
            config.sample_hidden_size, config.sample_hidden_size
        )
        self.LayerNorm.set_sample_config(config.sample_hidden_size)

    def get_active_subnet(self, config):
        subnet = MobileBertPredictionHeadTransform(config)

        subnet.dense = self.dense.get_active_subnet()
        subnet.LayerNorm = self.LayerNorm.get_active_subnet(config)

        return subnet

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# this class is changed a lot from bert
class MobileBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MobileBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = CustomLinear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def set_sample_config(self, config):
        self.transform.set_sample_config(config)
        # self.dense.set_sample_config(
        #     config.vocab_size, config.sample_hidden_size - config.sample_embedding_size
        # )
        self.decoder.set_sample_config(config.sample_hidden_size, config.vocab_size)

    def get_active_subnet(self, config):
        subnet = MobileBertLMPredictionHead(config)
        subnet.transform = self.transform.get_active_subnet(config)
        subnet.decoder = self.decoder.get_active_subnet()
        subnet.bias.data.copy_(self.bias)

        return subnet

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MobileBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)

    def set_sample_config(self, config):
        self.predictions.set_sample_config(config)

    def get_active_subnet(self, config):
        subnet = MobileBertOnlyMLMHead(config)
        subnet.predictions = self.predictions.get_active_subnet(config)

        return subnet

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MobileBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)
        self.seq_relationship = CustomLinear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    load_tf_weights = load_tf_weights_in_mobilebert
    base_model_prefix = "mobilebert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, CustomLinear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, CustomEmbedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, CustomNoNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CustomLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class MobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.MobileBertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


MOBILEBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.MobileBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

MOBILEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertModel(MobileBertPreTrainedModel):
    """
    https://arxiv.org/pdf/2004.02984.pdf
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        # self.embeddings = MobileBertEmbeddings(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = MobileBertEncoder(config)

        self.pooler = MobileBertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def set_sample_config(self, config):
        self.embeddings.set_sample_config(config)
        self.encoder.set_sample_config(config)
        if self.pooler is not None:
            self.pooler.set_sample_config(config)

    def get_active_subnet(self, config):
        subnet = MobileBertModel(config)

        subnet.embeddings = self.embeddings.get_active_subnet(config)
        subnet.encoder = self.encoder.get_active_subnet(config)
        if self.pooler is not None:
            subnet.pooler = self.pooler.get_active_subnet(config)

        return subnet

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time"
        #     )
        # elif input_ids is not None:
        #     input_shape = input_ids.size()
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")

        # modified to be inline with bertmodel
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # if attention_mask is None:
        #     attention_mask = torch.ones(input_shape, device=device)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.sample_num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertForPreTraining(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs

    # def resize_token_embeddings(
    #     self, new_num_tokens: Optional[int] = None
    # ) -> CustomEmbedding:
    #     # resize dense output embedings at first
    #     self.cls.predictions.dense = self._get_resized_lm_head(
    #         self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
    #     )

    #     return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=MobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples::

            >>> from transformers import MobileBertTokenizer, MobileBertForPreTraining
            >>> import torch

            >>> tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
            >>> model = MobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids)

            >>> prediction_logits = outptus.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MobileBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MobileBert Model with a `language modeling` head on top. """,
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertForMaskedLM(MobileBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        self.cls = MobileBertOnlyMLMHead(config)
        self.config = config

        self.init_weights()

    def set_sample_config(self, config):
        self.mobilebert.set_sample_config(config)
        self.cls.set_sample_config(config)

    def get_active_subnet(self, config):
        subnet = MobileBertForMaskedLM(config)
        # subnet.set_sample_config(config)
        subnet.bert = self.bert.get_active_subnet(config)
        subnet.cls = self.cls.get_active_subnet(config)
        # subnet.classifier = self.classifier.get_active_subnet()

        return subnet

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs

    # def resize_token_embeddings(
    #     self, new_num_tokens: Optional[int] = None
    # ) -> CustomEmbedding:
    #     # resize dense output embedings at first
    #     self.cls.predictions.dense = self._get_resized_lm_head(
    #         self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
    #     )
    #     return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_soft_loss=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            if use_soft_loss:
                loss_fct = CrossEntropyLossSoft()
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1, self.config.vocab_size),
                )
            else:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
                )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MobileBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = CustomLinear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top. """,
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring) Indices should be in ``[0, 1]``.

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples::

            >>> from transformers import MobileBertTokenizer, MobileBertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = MobileBertForNextSentencePrediction.from_pretrained('google/mobilebert-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        seq_relationship_score = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), labels.view(-1)
            )

        if not return_dict:
            output = (seq_relationship_score,) + outputs[2:]
            return (
                ((next_sentence_loss,) + output)
                if next_sentence_loss is not None
                else output
            )

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.mobilebert = MobileBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering with Bert->MobileBert all-casing
class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        self.qa_outputs = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice with Bert->MobileBert all-casing
class MobileBertForMultipleChoice(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mobilebert = MobileBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        input_ids = (
            input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# Copied from transformers.models.bert.modeling_bert.BertForTokenClassification with Bert->MobileBert all-casing
class MobileBertForTokenClassification(MobileBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
