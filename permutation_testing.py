import argparse
import logging
import math
import os
import random
import wandb
from copy import deepcopy


import plotly.express as px
from datetime import datetime
from collections import defaultdict, OrderedDict as OD
import loss

import numpy as np
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
from train_mlm import validate_subtransformer
import transformers
from transformers import (
    CONFIG_MAPPING,
    AdamW,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from utils.module_proxy_wrapper import ModuleProxyWrapper
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType

from sampling import (
    Sampler,
    get_supertransformer_config,
    show_random_elements,
    show_args,
)


from custom_layers import custom_bert, custom_mobile_bert

import plotly.graph_objects as go
from more_itertools import unique_everseen
from utils import (
    count_parameters,
    check_path,
    get_current_datetime,
    read_json,
    calculate_params_from_config,
    millify,
)
from loss import *
import transformers
from transformers.models.bert.modeling_bert import BertForMaskedLM

from torchinfo import summary


def inverse_permutation(permutation_order):
    inv = torch.empty_like(permutation_order)
    inv[permutation_order] = torch.arange(
        permutation_order.size(0), device=permutation_order.device
    )

    return inv


class BackHook:
    def __init__(self, max_steps=1000):
        self.grad_input = {}
        self.grad_output = {}
        self.layer_num = 0
        self.max_steps = max_steps
        self.steps = 0

    def __call__(self, module, grad_in, grad_out):
        print("calling")
        self.steps += 1

        grad_out = torch.abs(grad_out[0])
        if not hasattr(module, "name"):
            setattr(module, "name", self.layer_num)
            self.grad_output[self.layer_num] = grad_out
            self.layer_num += 1
        else:
            # take mean along batch dimension
            layer_num = getattr(module, "name")
            self.grad_output[layer_num] = torch.mean(
                torch.stack([self.grad_output[layer_num], grad_out]), dim=0
            )
        if self.steps == self.max_steps:
            # self.steps = 0
            layer_num = getattr(module, "name")
            grad_output = self.grad_output[layer_num]
            importance_order = torch.argsort(grad_output, descending=True)
            setattr(module, "importance_order", importance_order)
            setattr(
                module, "inv_importance_order", inverse_permutation(importance_order)
            )


def rewire_model(model, config):
    def permute_linear(
        W, permutation, dim="col", permute_weight=False, permute_bias=False
    ):
        """
        Permute linear layer

        :param W: weight matrix
        :param permutation: permutation order for the weights
        :param dim: 'row' or 'col'
        :param permute_bias: whether to permute the bias

        """
        _W = deepcopy(W)
        if permute_bias:
            _W.bias.data.copy_(_W.bias[permutation])

        if permute_weight:
            if dim == "col":
                _W.weight.data.copy_(_W.weight[:, permutation])
            elif dim == "row":
                _W.weight.data.copy_(_W.weight[permutation, :])
            else:
                raise NotImplementedError

        return _W

    with torch.no_grad():
        num_layers = config.num_hidden_layers
        emb_dim = config.hidden_size
        embeddings = model.bert.embeddings.word_embeddings

        assert model.bert.embeddings.word_embeddings.importance_order is not None

        weight_permutation_order = (
            model.bert.embeddings.word_embeddings.importance_order
        )

        _ = permute_linear(
            embeddings,
            weight_permutation_order,
            dim="col",
            permute_weight=True,
            permute_bias=False,
        )

        for i in range(num_layers):

            keys_to_permute = [
                (f"bert.encoder.layer.{i}.attention.self.query", "col"),
                (f"bert.encoder.layer.{i}.attention.self.key", "col"),
                (f"bert.encoder.layer.{i}.attention.self.value", "col"),
                (
                    f"bert.encoder.layer.{i}.attention.output.dense",
                    "row",
                ),
                (f"bert.encoder.layer.{i}.intermediate.dense", "col"),
                (f"bert.encoder.layer.{i}.output.dense", "row"),
            ]

            for key_mode in keys_to_permute:
                key, mode = key_mode

                module = getattr(model, key)

                if i == 0 and "input_bottleneck" in key:
                    continue
                if mode == "row":
                    weight_permutation_order = module.importance_order
                    permute_bias = True
                else:
                    permute_bias = False
                    # Column permutaion weights are not registered for backward hooks
                    # This is because, columns just use the permutation orders computed
                    # on their previous layers with embedding output dimension.
                    # Thus, they will not have their own imporatance order or
                    # inverse importance order
                    # So we use the prev importance order and assign it to them
                    # for ease of use in custom_bert script
                    setattr(module, "importance_order", weight_permutation_order)
                    setattr(
                        module,
                        "inv_importance_order",
                        inverse_permutation(weight_permutation_order),
                    )

                _ = permute_linear(
                    module,
                    weight_permutation_order,
                    dim=mode,
                    permute_bias=permute_bias,
                )
        # final importance order is stored
        setattr(
            model.bert,
            "inv_importance_order",
            inverse_permutation(weight_permutation_order),
        )


if __name__ == "__main__":

    max_seq_len = 64
    batch_size = 4

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

    global_config = get_supertransformer_config(
        "bert-base-cased", mixing="bert-bottleneck"
    )

    global_config.max_seq_length = max_seq_len
    global_config.rewire = True

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    model = custom_bert.BertForMaskedLM.from_pretrained(
        "bert-base-cased", config=global_config
    )
    model = custom_bert.BertForMaskedLM.from_pretrained(
        "bert-base-cased", config=global_config
    )

    identity = torch.eye(global_config.hidden_size)

    for key in model.state_dict().keys():
        if "input_bottleneck.weight" in key or "output_bottleneck.weight" in key:
            model.state_dict()[key].data.copy_(identity)
        elif "input_bottleneck.bias" in key or "output_bottleneck.bias" in key:
            model.state_dict()[key].data.zero_()

    print("BERT-Bottleneck Initiliazed with BERT-base")
    raw_datasets = load_dataset("wikitext", "wikitext-2-v1")
    padding = False
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_seq_len,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        # remove_columns=column_names,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_len) * max_seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4)
    model.set_sample_config(global_config)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
    )

    model.train()

    print("validating perplexity before rewiring")
    eval_metric = validate_subtransformer(
        model, eval_dataloader, accelerator, len(eval_dataset), batch_size, False
    )
    print(eval_metric["perplexity"])

    max_steps = 4
    nsteps = max_steps / (batch_size * torch.cuda.device_count())

    bhookfn = BackHook(max_steps=max_steps)

    model.bert.embeddings.word_embeddings.register_backward_hook(bhookfn)

    for i in range(global_config.num_hidden_layers):

        keys_to_hook = [
            f"bert.encoder.layer.{i}.attention.output.dense",
            f"bert.encoder.layer.{i}.output.dense",
        ]

        for key in keys_to_hook:
            module = getattr(model, key)
            module.register_backward_hook(bhookfn)
    print(f"Calculating gradients for {max_steps} with hooks: ")
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        # mul = torch.mean(
        #    model.bert.embeddings.word_embeddings.weight
        #    * model.bert.embeddings.word_embeddings.weight.grad,
        #    dim=0,
        # )

        # mul_sort = torch.sort(mul, descending=True)
        if step == nsteps:
            break

    rewire_model(model, global_config)
    eval_metric = validate_subtransformer(
        model, eval_dataloader, accelerator, len(eval_dataset), batch_size, False
    )
    print(eval_metric["perplexity"])
