import argparse
import os
import random
from pprint import pprint

import datasets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import wandb
from custom_layers.custom_bert import BertForSequenceClassification
from custom_layers.custom_bert import BertForMaskedLM
from tasks.glue.prepare_task import GlueTask
from utils.module_proxy_wrapper import ModuleProxyWrapper
from utils.wipe_memory import free_memory, get_gpu_memory
from utils.early_stopping import EarlyStopping
from utils import count_parameters
import copy

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

# SUPPORTED_MODELS = ['bert-base-cased', 'bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased', 'bert-large-cased']


class Sampler:
    def __init__(
        self,
        sampling_type,
        sampling_rule,
        mixing,
        config,
        static_keys=None,
        layerwise_changing_keys=None,
        magic_sampling=False,
    ):
        self.config = config
        self.sampling_type = sampling_type
        self.sampling_rule = sampling_rule
        self.mixing = mixing
        self.magic_sampling = magic_sampling
        self.prev_subtransformer_configs = None
        self.static_keys = static_keys or [
            "sample_hidden_size",
            "sample_num_hidden_layers",
        ]
        if self.mixing == "mobilebert":
            self.layerwise_changing_keys = layerwise_changing_keys or [
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_intra_bottleneck_size",
            ]
        elif self.mixing == "bert-bottleneck":
            self.layerwise_changing_keys = layerwise_changing_keys or [
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_hidden_size",
            ]
            self.static_keys = static_keys or [
                "sample_num_hidden_layers",
            ]

        else:
            self.layerwise_changing_keys = layerwise_changing_keys or [
                "sample_num_attention_heads",
                "sample_intermediate_size",
            ]

        if self.magic_sampling:
            assert (
                config.magic_sampling_random_walk_prob is not None
                and config.magic_sampling_per_layer_change_prob is not None
            )
            assert (
                config.magic_sampling_random_walk_prob > 0
                and config.magic_sampling_random_walk_prob <= 1
            )
            assert (
                config.magic_sampling_per_layer_change_prob > 0
                and config.magic_sampling_per_layer_change_prob <= 1
            )
            self.random_walk_prob = config.magic_sampling_random_walk_prob
            self.layer_change_prob = config.magic_sampling_per_layer_change_prob

    # TODO: Replace this with a YAML file.
    def get_choices(self):
        choices = {
            "sample_hidden_size": [120, 240, 360, 480, 540, 600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            "sample_num_hidden_layers": list(range(6, self.config.num_hidden_layers, 2))
            + [self.config.num_hidden_layers],
        }
        choices["sample_hidden_size"] = (
            [120, 240, 360, 480, 512]
            if self.mixing == "gmlp"
            else choices["sample_hidden_size"]
        )
        if self.mixing == "mobilebert":
            choices["sample_hidden_size"] = [768]
            choices["sample_intra_bottleneck_size"] = [
                120,
                240,
                360,
                480,
                540,
                600,
                768,
            ]
            choices["sample_true_hidden_size"] = [768]
            choices["sample_intermediate_size"] = [3072]
            choices["sample_num_hidden_layers"] = [12]
            choices["sample_num_attention_heads"] = [12]
        elif self.mixing == "bert-bottleneck":
            choices = {
                "sample_hidden_size": [120, 240, 360, 480, 540, 600, 768],
                "sample_num_attention_heads": [12],
                "sample_intermediate_size": [3072],
                "sample_num_hidden_layers": [12],
            }

        return choices

    def get_diverse_subtransformers(self, elastic_variable):
        diverse_configs = []
        all_choices = self.get_choices()

        num_hidden_layers = int(self.config.num_hidden_layers)
        elastic_variable_choices = all_choices[elastic_variable]

        diverse_config = copy.deepcopy(self.config)

        # we now set the max possible values for single choices and layer wise chhoices

        for key in self.static_keys:
            if key == elastic_variable:
                continue
            value = max(all_choices[key])
            setattr(diverse_config, key, value)

        for key in self.layerwise_changing_keys:
            if key == elastic_variable:
                continue
            value = [max(all_choices[key])] * num_hidden_layers
            setattr(diverse_config, key, value)

        for choice in elastic_variable_choices:
            if elastic_variable in self.static_keys:
                value = choice
                setattr(diverse_config, elastic_variable, value)
            else:
                value = [choice] * num_hidden_layers
                setattr(diverse_config, elastic_variable, value)

            if self.mixing == "bert-bottleneck":
                hidden = getattr(diverse_config, "sample_hidden_size")[0]
                if hidden % getattr(diverse_config, "sample_num_attention_heads")[0]:
                    continue
            # TODO: add sample_intra_bottleneck_size later
            elif (
                getattr(diverse_config, "sample_hidden_size")
                % getattr(diverse_config, "sample_num_attention_heads")[0]
            ):
                continue
            diverse_configs.append(copy.deepcopy(diverse_config))

        def sorter(x):
            value = getattr(x, elastic_variable)
            if isinstance(value, list):
                return value[0]
            else:
                return value

        diverse_configs = sorted(diverse_configs, key=sorter)

        return diverse_configs

    def naive_params_sampling(self, population_size=30):

        config = copy.deepcopy(self.config)
        choices = self.get_choices()

        max_params = 0
        best_config = None

        assert population_size > 0

        ## We can replace this with a simple mathematical function to compute params given a config and a maximizing
        ## function to give precedence for params! That might be faster
        ## For now implemented as using the best params from a randomly sampled population!

        for i in range(population_size):
            config = self.weighted_params_sample(self.config)
            model = BertForMaskedLM(config)
            params = count_parameters(model)

            if max_params < params:
                max_params = params
                best_config = config

        return best_config

    def weighted_params_sample(self):
        config = copy.deepcopy(self.config)

        choices = self.get_choices()
        normalized_probs = self.calc_probs(choices)

        ### Figuring the number of hidden layers
        hidden_layers_list = choices["sample_num_hidden_layers"]
        num_hidden_layers = random.choices(
            hidden_layers_list,
            k=1,
            weights=normalized_probs["sample_num_hidden_layers"],
        )[0]
        setattr(config, "sample_num_hidden_layers", num_hidden_layers)

        if self.mixing != "bert-bottleneck":
            ## Figuring the hidden size for BERT embeddings
            hidden_size_embeddings_list = choices["sample_hidden_size"]
            num_hidden_size = random.choices(
                hidden_size_embeddings_list,
                k=1,
                weights=normalized_probs["sample_hidden_size"],
            )[0]
            setattr(config, "sample_hidden_size", num_hidden_size)

        config_dict = {
            "sample_num_attention_heads": [],
            "sample_intermediate_size": [],
            "sample_intra_bottleneck_size": [],
        }

        if not hasattr(config, "sample_intra_bottleneck_size"):
            _ = config_dict.pop("sample_intra_bottleneck_size")

        if self.mixing == "bert-bottleneck":
            config_dict = {
                "sample_num_attention_heads": [],
                "sample_intermediate_size": [],
                # we need to have diff hiddensizes for every layer
                "sample_hidden_size": [],
            }

        for i in range(num_hidden_layers):
            while True:
                for key in config_dict.keys():

                    choice_list = choices[key]
                    choice = random.choices(
                        choice_list, k=1, weights=normalized_probs[key]
                    )[0]
                    config_dict[key].append(choice)

                if self.mixing == "bert-bottleneck":
                    if (
                        config.sample_hidden_size[i]
                        % config_dict["sample_num_attention_heads"][i]
                    ):
                        for key in config_dict.keys():
                            config_dict[key] = config_dict[key][:-1]
                        continue
                else:
                    if (
                        config.sample_hidden_size
                        % config_dict["sample_num_attention_heads"][i]
                    ):
                        for key in config_dict.keys():
                            # we remove this element from the config dict
                            config_dict[key] = config_dict[key][:-1]
                        continue
                    else:
                        if hasattr(config, "sample_intra_bottleneck_size"):
                            if (
                                config.sample_intra_bottleneck_size[i]
                                % config_dict["sample_num_attention_heads"][i]
                            ):
                                for key in config_dict.keys():
                                    config_dict[key] = config_dict[key][:-1]
                                continue

                break

        for key in config_dict.keys():
            setattr(config, key, config_dict[key])

        return config

    def get_small_config(self):

        config = copy.deepcopy(self.config)
        choices = self.get_choices()

        config_dict = {}
        config_dict["sample_num_hidden_layers"] = min(
            choices["sample_num_hidden_layers"]
        )
        config_dict["sample_hidden_size"] = min(choices["sample_hidden_size"])

        if self.mixing == "bert-bottleneck":
            config_dict["sample_hidden_size"] = [
                min(choices["sample_hidden_size"])
            ] * config_dict["sample_num_hidden_layers"]
            config_dict["sample_num_attention_heads"] = [12] * config_dict[
                "sample_num_hidden_layers"
            ]

        elif self.mixing == "mobilebert":
            config_dict["sample_num_attention_heads"] = [12] * config_dict[
                "sample_num_hidden_layers"
            ]
            config_dict["sample_true_hidden_size"] = min(
                choices["sample_true_hidden_size"]
            )
        else:
            # 2 is selected as any even hidden size, or other dimensions we
            # choose will be divisible
            config_dict["sample_num_attention_heads"] = [2] * config_dict[
                "sample_num_hidden_layers"
            ]

        assigned_keys = [
            "sample_num_hidden_layers",
            "sample_num_attention_heads",
            "sample_hidden_size",
            "sample_true_hidden_size",
        ]

        for choice in choices.keys():
            if choice in assigned_keys:
                continue

            config_dict[choice] = [min(choices[choice])] * config_dict[
                "sample_num_hidden_layers"
            ]

        for key in config_dict.keys():
            setattr(config, key, config_dict[key])

        return config

    def sample_subtransformer(self, randomize=True, rand_seed=0, pop_size=1):
        # we store the previous subtransformer configs so that we can do random
        # walks (ie change some parameters on previous configs) instead of uniform
        # sampling
        if randomize:
            random.seed(rand_seed)

        smallest_config = None

        if self.sampling_rule == "sandwich":
            smallest_config = self.get_small_config()

        def _sample():
            configs = []
            for _ in range(pop_size):
                if (
                    self.sampling_type == "random"
                    or self.sampling_type == "biased_params"
                ):
                    _config = self.weighted_params_sample()
                elif self.sampling_type == "naive_params":
                    _config = self.naive_params_sampling()
                else:
                    raise NotImplementedError
                configs.append(_config)
            return configs

        if self.prev_subtransformer_configs is not None and self.magic_sampling:
            random_number = random.random()
            choices = self.get_choices()
            num_hidden_layers = len(choices["sample_num_hidden_layers"])
            sample_hidden_sizes = choices["sample_hidden_size"]
            if random_number <= self.random_walk_prob:
                configs = []
                for _config in self.prev_subtransformer_configs:
                    new_config = copy.deepcopy(_config)
                    to_change = (
                        np.random.uniform(0, 1, num_hidden_layers)
                        <= self.layer_change_prob
                    )
                    hidden_sizes = getattr(new_config, "sample_hidden_size")
                    for i in range(num_hidden_layers):
                        if to_change[i]:
                            hidden_sizes[i] = random.choice(sample_hidden_sizes)
                            setattr(new_config, "sample_hidden_size", hidden_sizes)
                    configs.append(new_config)
            else:
                configs = _sample()
        else:
            configs = _sample()

        self.prev_subtransformer_configs = configs
        return {
            "smallest_subtransformer": smallest_config,
            "random_subtransformers": configs,
        }

    def calc_probs(self, choices_dictionary):
        normalized_probs = {}
        for choice, v in choices_dictionary.items():
            _v = []
            _sum = sum(v)
            for i in v:
                if self.sampling_type == "biased_params":
                    _v.append(i / _sum)
                elif self.sampling_type == "random":
                    _v.append(1 / len(v))
            normalized_probs[choice] = _v
        return normalized_probs


def get_task(task_name):
    if task_name in GLUE_TASKS:
        return GlueTask


def show_random_elements(dataset, accelerator, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    accelerator.print(df)


def get_supertransformer_config(
    model_name_or_path="bert-base-cased",
    mixing="attention",
    additional_random_softmaxing=False,
    random_layer_selection_probability=0.1,
    custom_hidden_size=None,
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if custom_hidden_size is not None:
        config.hidden_size = custom_hidden_size
        config.true_hidden_size = custom_hidden_size
        # let intermediate size be 3072 as reducing ffn dim may have neg impact on training
        # config.intermediate_size = custom_hidden_size * 4

    if mixing == "gmlp":
        # gmlp needs twice the encoder layers to match bert param size
        config.num_hidden_layers = 36
        config.hidden_size = 512

    config.sample_hidden_size = config.hidden_size
    config.sample_num_hidden_layers = config.num_hidden_layers

    if mixing == "bert-bottleneck":
        config.sample_hidden_size = [
            config.hidden_size
        ] * config.sample_num_hidden_layers

    # for all networks we use layernorm and feedforwardnetworks 1
    config.normalization_type = "layer_norm"
    config.num_feedforward_networks = 1

    config.sample_num_attention_heads = [
        config.num_attention_heads
    ] * config.sample_num_hidden_layers

    config.sample_intermediate_size = [
        config.intermediate_size
    ] * config.sample_num_hidden_layers

    if mixing == "mobilebert":
        config.embedding_size = 768
        config.hidden_size = 768
        config.intra_bottleneck_size = 768
        config.true_hidden_size = 768

        config.sample_embedding_size = config.embedding_size
        config.sample_intra_bottleneck_size = [
            config.intra_bottleneck_size
        ] * config.sample_num_hidden_layers
        config.sample_true_hidden_size = config.true_hidden_size
        config.use_bottleneck = True
        config.use_bottleneck_attention = False
        config.key_query_shared_bottleneck = False

    config.mixing = mixing
    config.additional_random_softmaxing = additional_random_softmaxing
    config.random_layer_selection_probability = random_layer_selection_probability
    config.rewire = False
    return config


def show_args(accelerator, args):
    accelerator.print(
        f"Free gpu Memory ( in MBs) on each gpus before starting training: {get_gpu_memory()}"
    )
    accelerator.print(
        "==================================================================="
    )
    accelerator.print("Training Arguments:")
    for arg in vars(args):
        accelerator.print(f"{arg}: {getattr(args, arg)}")
    accelerator.print(
        "==================================================================="
    )
