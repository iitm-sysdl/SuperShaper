# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CustomEmbedding(nn.Embedding):
    def __init__(self, vocab_size, super_hidden_size, padding_idx, *args, **kwargs):
        super().__init__(vocab_size, super_hidden_size, padding_idx, *args, **kwargs)

        # the largest embed dim
        self.super_hidden_size = {
            "encoder": super_hidden_size,
            "decoder": super_hidden_size,
        }

        # the current sampled embed dim
        self.sample_hidden_size = {"encoder": None, "decoder": None}

        self.vocab_size = vocab_size

        self.samples = {"encoder": {}, "decoder": {}}
        self.profiling = False
        self.reset_parameters()

    def profile(self, mode=True):
        self.profiling = mode

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_hidden_size, part):
        self.sample_hidden_size[part] = sample_hidden_size
        self._sample_parameters(part)

    def _sample_parameters(self, part):
        weight = self.weight[..., : self.sample_hidden_size[part]]
        self.samples[part]["weight"] = weight

        return self.samples

    def sample_parameters(self, part, resample=False):
        return (
            self._sample_parameters(part)
            if self.profiling or resample
            else self.samples
        )

    def get_active_subnet(self, part):
        sub_layer = nn.Embedding(
            self.vocab_size, self.sample_hidden_size[part], padding_idx=self.padding_idx
        )
        sub_layer.weight.data.copy_(self.samples[part]["weight"])

        return sub_layer

    def sampled_weight(self, part):
        return self.sample_parameters(part)[part]["weight"]

    def forward(self, input, part="encoder"):
        return F.embedding(
            input,
            self.sampled_weight(part),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )