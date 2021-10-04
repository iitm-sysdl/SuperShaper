# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import torch.nn as nn
import torch.nn.functional as F


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class CustomLayerNorm(torch.nn.LayerNorm):
    def __init__(self, super_hidden_size, eps):
        super().__init__(super_hidden_size)

        # the largest embed dim
        self.super_hidden_size = super_hidden_size

        # the current sampled embed dim
        self.sample_hidden_size = None

        self.samples = {}
        self.profiling = False
        self.eps = eps

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples["weight"] = self.weight[: self.sample_hidden_size]
        self.samples["bias"] = self.bias[: self.sample_hidden_size]
        return self.samples

    def set_sample_config(self, sample_hidden_size):
        self.sample_hidden_size = sample_hidden_size
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(
            x,
            (self.sample_hidden_size,),
            weight=self.samples["weight"],
            bias=self.samples["bias"],
            eps=self.eps,
        )

    # TODO: Make layerNorm cleaner
    def get_active_subnet(self, sub_sample_hidden_size=None):
        sub_layer = torch.nn.LayerNorm(self.sample_hidden_size, self.eps)
        sub_layer.weight.data.copy_(self.samples["weight"])
        sub_layer.bias.data.copy_(self.samples["bias"])

        return sub_layer

    def calc_sampled_param_num(self):
        assert "weight" in self.samples.keys()
        assert "bias" in self.samples.keys()
        return self.samples["weight"].numel() + self.samples["bias"].numel()


class CustomNoNorm(nn.Module):
    def __init__(self, super_feat_size, eps=None):
        super().__init__()
        self.super_feat_size = super_feat_size
        self.sample_feat_size = None
        self.bias = nn.Parameter(torch.zeros(super_feat_size))
        self.weight = nn.Parameter(torch.ones(super_feat_size))

        self.samples = {}
        self.profiling = False
        self.eps = eps

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples["weight"] = self.weight[: self.sample_feat_size]
        self.samples["bias"] = self.bias[: self.sample_feat_size]
        return self.samples

    def set_sample_config(self, sample_feat_size):
        self.sample_feat_size = sample_feat_size
        self._sample_parameters()

    def forward(self, input_tensor):
        self.sample_parameters()
        return input_tensor * self.weight + self.bias

    def get_active_subnet(self, sub_sample_feat_size=None):
        sub_layer = torch.nn.LayerNorm(self.sample_feat_size, self.eps)
        sub_layer.weight.data.copy_(self.samples["weight"])
        sub_layer.bias.data.copy_(self.samples["bias"])

        return sub_layer

    def calc_sampled_param_num(self):
        assert "weight" in self.samples.keys()
        assert "bias" in self.samples.keys()
        return self.samples["weight"].numel() + self.samples["bias"].numel()