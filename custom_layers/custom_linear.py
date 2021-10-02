# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinear(nn.Linear):
    def __init__(
        self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear="linear"
    ):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}
        self.sliced_samples = {}

        self.bias_sample = bias
        self.uniform_ = uniform_
        self.non_linear = non_linear

        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False
        self.bias_val = bias

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear
        )
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    def _sample_parameters(self):
        # memoize the sampled parameters
        if self.sliced_samples.get((self.sample_in_dim, self.sample_out_dim)) is None:
            self.samples["weight"] = sample_weight(
                self.weight, self.sample_in_dim, self.sample_out_dim
            )
            self.samples["bias"] = self.bias
            if self.bias is not None:
                self.samples["bias"] = sample_bias(self.bias, self.sample_out_dim)
            self.sliced_samples[
                (self.sample_in_dim, self.sample_out_dim)
            ] = self.samples
        else:
            self.samples = self.sliced_samples[
                (self.sample_in_dim, self.sample_out_dim)
            ]
        return self.samples

    def get_active_subnet(self):
        sub_layer = nn.Linear(
            self.sample_in_dim,
            self.sample_out_dim,
            self.bias_sample,
            # self.uniform_,  #Vin: Check if these cause some issues down the line
            # self.non_linear,
        )
        # self._sample_parameters()
        sub_layer.weight.data.copy_(self.samples["weight"])
        if self.bias_val:
            sub_layer.bias.data.copy_(self.samples["bias"])

        return sub_layer

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples["weight"], self.samples["bias"])

    def calc_sampled_param_num(self):
        assert "weight" in self.samples.keys()
        weight_numel = self.samples["weight"].numel()

        if self.samples["bias"] is not None:
            bias_numel = self.samples["bias"].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    @property
    def module_str(self):
        return "Linear(%d, %d)" % (self.sample_in_dim, self.sample_out_dim)


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias
