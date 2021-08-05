import argparse
import logging
import math
import os
import random
import wandb


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


def rewire_model(model, config):
    def compute_permutation_order(value, gradient, reduce_dim=0):
        importance = torch.mean(torch.abs(gradient * value), dim=reduce_dim)
        permutation_indices = torch.argsort(importance, descending=True)
        return permutation_indices

    def permute_weights(weights, permutation_order, mode="col"):
        if mode == "col":
            permuted_weights = weights[:, permutation_order]
        elif mode == "row":
            permuted_weights = weights[permutation_order, :]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return permuted_weights

    with torch.no_grad():
        num_layers = config.num_hidden_layers
        emb_dim = config.hidden_size
        emb_weight = model.bert.embeddings.word_embeddings.weight
        emb_grad = model.bert.embeddings.word_embeddings.weight.grad

        weight_permutation_order = compute_permutation_order(
            emb_weight, emb_grad, reduce_dim=0
        )
        # we dont change bias permutation order here
        #bias_permutation_order = torch.arange(emb_dim)

        emb_weight = permute_weights(emb_weight, weight_permutation_order)

        model.state_dict()["bert.embeddings.word_embeddings.weight"].data.copy_(
            emb_weight
        )

        for i in range(num_layers):
            # recalculate permutation order for new layers from the input bottleneck
            if i:
                current_weight_permutation_order = compute_permutation_order(
                    model.bert.encoder.layer[i].input_bottleneck.weight,
                    model.bert.encoder.layer[i].input_bottleneck.weight.grad,
                    reduce_dim=0,
                )
                #current_bias_permutation_order = compute_permutation_order(
                #    model.bert.encoder.layer[i].input_bottleneck.bias,
                #    model.bert.encoder.layer[i].input_bottleneck.bias.grad,
                #    reduce_dim=-1,
                #)

                # compose both permutations to get one permutation order
                weight_permutation_order = current_weight_permutation_order[
                    weight_permutation_order
                ]
                #bias_permutation_order = current_bias_permutation_order[
                #    bias_permutation_order
                #]

            keys_to_permute = [
                (f"bert.encoder.layer.{i}.input_bottleneck.weight", "col"),
                #(f"bert.encoder.layer.{i}.input_bottleneck.bias", "bias"),
                (f"bert.encoder.layer.{i}.attention.self.query.weight", "row"),
                (f"bert.encoder.layer.{i}.attention.self.query.bias", "row"),
                (f"bert.encoder.layer.{i}.attention.self.key.weight", "row"),
                (f"bert.encoder.layer.{i}.attention.self.key.bias", "row"),
                (f"bert.encoder.layer.{i}.attention.self.value.weight", "row"),
                (f"bert.encoder.layer.{i}.attention.self.value.bias", "row"),
                (f"bert.encoder.layer.{i}.attention.output.dense.weight", "col"),
                #(f"bert.encoder.layer.{i}.attention.output.dense.bias", "bias"),
                (f"bert.encoder.layer.{i}.output.dense.weight", "row"),
                (f"bert.encoder.layer.{i}.output.dense.bias", "row"),
                (f"bert.encoder.layer.{i}.intermediate.dense.weight", "col"),
                #(f"bert.encoder.layer.{i}.intermediate.dense.bias", "bias"),
             ]

            for key_mode in keys_to_permute:
                key, mode = key_mode

                if i == 0 and 'input_bottleneck' in key:
                    continue

                if "weight" in key:
                    weight = permute_weights(
                        model.state_dict()[key], weight_permutation_order, mode=mode
                    )
                    model.state_dict()[key].data.copy_(weight)
                elif "bias" in key:
                    bias = permute_weights(
                        model.state_dict()[key], weight_permutation_order, mode=mode
                    )
                    model.state_dict()[key].data.copy_(bias)

        # TODO: check this modes
        keys_to_permute = [
            (f"cls.predictions.transform.dense.weight", "row"),
            (f"cls.predictions.transform.dense.bias", "bias"),
            (f"cls.predictions.decoder.weight", "col"),
            (f"cls.predictions.decoder.bias", "bias"),
        ]
        for key_mode in keys_to_permute:
            key, mode = key_mode
            if "weight" in key:
                weight = permute_weights(
                    model.state_dict()[key], weight_permutation_order, mode=mode
                )
                model.state_dict()[key].data.copy_(weight)
            elif "bias" in key:
                bias = permute_weights(
                    model.state_dict()[key], weight_permutation_order, mode=mode
                )
                model.state_dict()[key].data.copy_(bias)


if __name__ == "__main__":

    max_seq_len = 64
    batch_size = 4

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=True, kwargs_handlers=[param])


    global_config = get_supertransformer_config("bert-base-cased", mixing="bert-bottleneck")

    global_config.max_seq_length = max_seq_len

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    model = custom_bert.BertForMaskedLM.from_pretrained(
        "bert-base-cased", config=global_config
    )
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
    
    max_steps = 4
    nsteps = max_steps / (batch_size * torch.cuda.device_count())
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        #mul = torch.mean(
        #    model.bert.embeddings.word_embeddings.weight
        #    * model.bert.embeddings.word_embeddings.weight.grad,
        #    dim=0,
        #)

        #mul_sort = torch.sort(mul, descending=True)
        if step == nsteps: 
            break

    rewire_model(model, global_config)  
    eval_metric = validate_subtransformer(model, eval_dataloader, accelerator, len(eval_dataset), batch_size, False)
    print(eval_metric["perplexity"])

