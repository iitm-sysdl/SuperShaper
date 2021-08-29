#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

# taken and modified from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
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
from operator import attrgetter

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

logger = logging.getLogger(__name__)


def validate_subtransformer(
    model,
    eval_dataloader,
    accelerator,
    len_eval_dataset,
    per_device_eval_batch_size,
    pad_to_max_length,
):
    metric = load_metric("custom_metrics/mlm_accuracy.py")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if accelerator.device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [str(p) for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [str(l) for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]

        return true_predictions, true_labels

    losses = []
    progress_bar = tqdm(
        range(0, len(eval_dataloader)),
        disable=not accelerator.is_local_main_process,
    )
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(per_device_eval_batch_size)))

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if (
            not pad_to_max_length
        ):  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(
                predictions, dim=1, pad_index=-100
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        preds, refs = get_labels(predictions_gathered, labels_gathered)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids
        progress_bar.update(1)

    losses = torch.cat(losses)
    losses = losses[:len_eval_dataset]
    eval_metric = metric.compute()

    try:
        val_loss = torch.mean(losses)
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    eval_metric["val_loss"] = val_loss
    eval_metric["perplexity"] = perplexity

    return eval_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain/Finetune a transformers model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=7e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=True,
        help="""
        Whether distinct lines of text in the dataset are to be handled as
        distinct sequences. This is deafult for bert/electra models and should
        be set to False for gpt/gpt2 type models""",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    # args we add
    parser.add_argument(
        "--early_stopping_patience",
        default=5,
        type=int,
        help="Patience for early stopping to stop training if val_acc doesnt converge",
    )
    # parser.add_argument(
    #     "--limit_subtransformer_choices",
    #     default=0,
    #     type=int,
    #     help="If set to 1, it will limit the hidden_size and number of encoder layers of the subtransformer choices",
    # )
    parser.add_argument(
        "--eval_random_subtransformers",
        default=1,
        type=int,
        help="If set to 1, this will evaluate 25 random subtransformers after every training epoch when training a supertransformer",
    )
    parser.add_argument(
        "--train_subtransformers_from_scratch",
        default=0,
        type=int,
        help="""
        If set to 1, this will train 25 random subtransformers from scratch.
        By default, it is set to False (0) and we train a supertransformer and finetune subtransformers
        """,
    )
    parser.add_argument(
        "--fp16", type=int, default=1, help="If set to 1, will use FP16 training."
    )
    parser.add_argument(
        "--mixing",
        type=str,
        required=True,
        help=f"specifies how to mix the tokens in bertlayers",
        choices=["attention", "gmlp", "fnet", "mobilebert", "bert-bottleneck"],
    )
    parser.add_argument(
        "--resume_from_checkpoint_dir",
        type=str,
        default=None,
        help=f"directory that contains checkpoints, optimizer, scheduler to resume training",
    )
    parser.add_argument(
        "--tiny_attn",
        type=int,
        default=0,
        help=f"Choose this if you need Tiny Attention Module along-with gMLP dense block",
    )
    parser.add_argument(
        "--num_subtransformers_monitor",
        type=int,
        default=25,
        help=f"Choose the number of subtransformers whose performance you wish to monitor",
    )

    parser.add_argument(
        "--c4_dir",
        type=str,
        default=None,
        help=f"The directory path for C4",
    )

    parser.add_argument(
        "--tokenized_c4_dir",
        type=str,
        default=None,
        help=f"The directory path for tokenized C4",
    )

    parser.add_argument(
        "--sampling_type",
        type=str,
        default="random",
        help=f"The sampling type for super-transformer",
        choices=["none", "naive_params", "biased_params", "random"],
    )
    parser.add_argument(
        "--sampling_rule",
        type=str,
        default="none",
        help=f"The sampling rule for sampling super-transformers",
        choices=["none", "sandwich"],
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=1,
        help=f"Number of random subtransformers to sample at each step",
    )

    parser.add_argument(
        "--k_sampling",
        type=int,
        required=True,
        help=f"The step frequency of sampling a sub-transformers",
    )

    parser.add_argument(
        "--inplace_distillation",
        type=int,
        default=0,
        help=f"Whether to use inplace distillation",
    )

    parser.add_argument(
        "--kd_ratio",
        type=float,
        default=1,
        help=f"Sensitizes the amount of KD-loss that needs to be added with existing loss",
    )

    parser.add_argument(
        "--layerwise_distillation",
        type=int,
        default=0,
        help=f"Conditional layerwise attention and feature map transfer for in-place distillation",
    )

    parser.add_argument(
        "--alpha_divergence",
        type=int,
        default=0,
        help=f"Enable Alpha Divergence KL loss",
    )

    parser.add_argument(
        "--alpha_min",
        type=float,
        default=-1.0,
        help=f"Alpha min value",
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        help=f"Alpha max value",
    )

    parser.add_argument(
        "--beta_clip",
        type=float,
        default=5.0,
        help=f"The clip value for alpha divergence",
    )

    parser.add_argument(
        "--subtransformer_config_path",
        type=str,
        default=None,
        help=f"The path to a subtransformer configration",
    )

    parser.add_argument(
        "--rewire",
        type=int,
        default=0,
        help=f"Whether to rewire model",
    )

    parser.add_argument(
        "--rewired_model_checkpoint_dir",
        type=str,
        default=None,
        help=f"Path to rewired model checkpoint",
    )

    parser.add_argument(
        "--wandb_suffix",
        type=str,
        default=None,
        help=f"suffix for wandb",
    )


    # parser.add_argument(
    #     "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    # )

    args = parser.parse_args()

    if args.subtransformer_config_path is None:
        args.model_name_or_path = "bert-base-cased"

    # Sanity checks

    if args.layerwise_distillation:
        assert args.inplace_distillation

    if args.alpha_divergence:
        assert args.inplace_distillation

    if args.inplace_distillation == 1:
        assert (
            args.sampling_rule == "sandwich"
        ), "Sampling rule needs to be sandwich if using inplace distillation"

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.c4_dir is None
        and args.tokenized_c4_dir is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if args.tiny_attn == 1:
        assert args.mixing == "gmlp", "Tiny Attention can work only in GMLP setup"

    if args.sampling_type == "none":
        # if we are not sampling, dont test random subtransformers every n epochs
        args.eval_random_subtransformers = False

    if args.c4_dir is not None:
        check_path(args.c4_dir)
        # c4_train_dir = os.path.join(args.c4_dir, "train")
        # c4_val_dir = os.path.join(args.c4_dir, "val")
        # check_path(c4_train_dir)
        # check_path(c4_val_dir)

        args.dataset_name = "c4_realnews"
        # args.c4_train_dir = c4_train_dir
        # args.c4_val_dir = c4_val_dir

    if args.tokenized_c4_dir is not None:
        check_path(args.tokenized_c4_dir)
        args.dataset_name = "c4_realnews"

    if args.resume_from_checkpoint_dir is not None:

        args.optim_scheduler_states_path = os.path.join(
            args.resume_from_checkpoint_dir, "optimizer_scheduler.pt"
        )
        check_path(args.resume_from_checkpoint_dir)
        check_path(args.optim_scheduler_states_path)

        model_path = os.path.join(args.resume_from_checkpoint_dir, "pytorch_model.bin")
        check_path(model_path)
        # overwrite on the same directory
        args.output_dir = args.resume_from_checkpoint_dir

    assert args.k_sampling > 0

    if args.subtransformer_config_path:
        check_path(args.subtransformer_config_path)
        assert (
            args.sampling_type == "none"
        ), "sampling_type is not supported when providing custom_subtransformer_config"
        assert (
            args.eval_random_subtransformers == 0
        ), "no need to evaluate random subtransformers when a custom_subtransformer_config is provided"

    if args.rewire:
        assert (
            args.rewired_model_checkpoint_dir is not None
        ), "Rewired model checkpoint path cannot be None if rewire is set to True"
        check_path(args.rewired_model_checkpoint_dir)

        if args.resume_from_checkpoint_dir is not None:
            args.resume_from_checkpoint_dir = args.rewired_model_checkpoint_dir

        ## Temporary Assert until rewiring support is providied for all other BERT variants
        assert args.mixing == "bert-bottleneck"

    return args


def main():
    args = parse_args()

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[param])

    show_args(accelerator, args)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    str_name = (
        args.mixing + "_tiny_attn"
        if args.tiny_attn == 1
        else args.mixing + "_" + args.sampling_type + "_K=" + str(args.k_sampling)
    )
    str_name = "rewired_" + str_name if args.rewire else str_name
    if args.inplace_distillation:
        str_name += "_ip_distill"
        if args.layerwise_distillation:
            str_name += "_layerwise_distill"
        if args.alpha_divergence:
            str_name += "_alpha_div"
    else:
        str_name += "_pretraining"

    if args.wandb_suffix is not None:
        str_name += "_" + args.wandb_suffix

    if accelerator.is_main_process:
        wandb.init(
            project="super-pretraining",
            entity="efficient-hat",
            name=args.dataset_name.split("/")[-1].strip() + "_" + str_name,
        )

    if args.output_dir is not None and args.resume_from_checkpoint_dir is None:
        dataset_name = args.dataset_name.split("/")[-1].strip()
        args.output_dir += (
            "/" + dataset_name + "_" + str_name + "_" + get_current_datetime()
        )
        args.optim_scheduler_states_path = os.path.join(
            args.output_dir, "{}/optimizer_scheduler.pt"
        )
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.tokenized_c4_dir is not None:
        logger.info("Loading Tokenized C4 Dataset...")
        tokenized_datasets = datasets.load_from_disk(args.tokenized_c4_dir)
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "c4_realnews":
            logger.info("Loading C4 Dataset...")
            raw_datasets = datasets.load_from_disk(args.c4_dir)
            # train_files = [
            #    os.path.join(args.c4_train_dir, file)
            #    for file in os.listdir(args.c4_train_dir)
            #    if file.endswith("json.gz")
            # ]
            # val_files = [
            #    os.path.join(args.c4_val_dir, file)
            #    for file in os.listdir(args.c4_val_dir)
            #    if file.endswith("json.gz")
            # ]
            # train_files = sorted(train_files)
            # val_files = sorted(val_files)
            # raw_datasets = load_dataset(
            #    "json", data_files={"train": train_files, "validation": val_files}
            # )

        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
            # limiting dataset for testing
            # raw_datasets["train"] = raw_datasets["train"].select(range(100))
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.subtransformer_config_path is None:
        global_config = get_supertransformer_config(
            args.model_name_or_path, mixing=args.mixing
        )
    else:
        global_config = get_supertransformer_config(
            "bert-base-cased", mixing=args.mixing
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        if args.subtransformer_config_path is None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=not args.use_slow_tokenizer
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-cased", use_fast=not args.use_slow_tokenizer
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if args.model_name_or_path:
    #     model = AutoModelForMaskedLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForMaskedLM.from_config(config)

    # add max_seq_len or model_max_len to config
    if args.max_seq_length:
        global_config.max_seq_length = args.max_seq_length
    else:
        logger.warning(
            f"The max_seq_length is not defined!! Setting it to max length in tokenizer"
        )
        global_config.max_seq_length = tokenizer.model_max_length

    # overriding the hideen dropout inline with hyperparms in gmlp paper
    global_config.hidden_dropout_prob = 0

    if args.layerwise_distillation:
        # for attention transfer and feature transfer enable these.
        global_config.output_attentions = True
        global_config.output_hidden_states = True

    global_config.alpha_divergence = args.alpha_divergence

    if args.alpha_divergence:
        global_config.alpha_min = args.alpha_min
        global_config.alpha_max = args.alpha_max
        global_config.beta_clip = args.beta_clip

    global_config.rewire = args.rewire

    if args.subtransformer_config_path is not None:
        subtransformer_config = read_json(args.subtransformer_config_path)
        for key, value in subtransformer_config.items():
            # update global_config with attributes of subtransformer_config
            setattr(global_config, key, value)

        logger.info(
            "=================================================================="
        )
        logger.info(
            f"Number of parameters in custom config is {millify(calculate_params_from_config(global_config, scaling_laws=False, add_output_emb_layer=False))}"
        )
        logger.info(
            "=================================================================="
        )

    # if check_path(args.model_name_or_path):
    #    model = custom_bert.BertForMaskedLM.from_pretrained(
    #        args.model_name_or_path, config=global_config
    #    )
    # else:

    # TODO: decouple mixing and mobilebert model declarato
    if args.mixing == "mobilebert":

        states = OD()

        model = custom_mobile_bert.MobileBertForMaskedLM(config=global_config)
        model2 = BertForMaskedLM.from_pretrained("bert-base-cased")

        for key in model2.state_dict().keys():
            _key = key.replace("bert.", "mobilebert.")
            states[_key] = model2.state_dict()[key]

        del model2
        model.load_state_dict(states, strict=False)
        del states

        identity = torch.eye(global_config.true_hidden_size)
        for key in model.state_dict().keys():
            if (
                "bottleneck.input.dense.weight" in key
                or "output.bottleneck.dense.weight" in key
            ):
                model.state_dict()[key].data.copy_(identity)
            elif (
                "bottleneck.output.dense.bias" in key
                or "output.bottleneck.dense.bias" in key
            ):
                model.state_dict()[key].data.zero_()

        logger.info("MobileBert Initiliazed with bert-base")

    elif args.mixing == "bert-bottleneck":
        model = custom_bert.BertForMaskedLM.from_pretrained(
            "bert-base-cased", config=global_config
        )

        identity = torch.eye(global_config.hidden_size)

        for key in model.state_dict().keys():
            if "input_bottleneck.weight" in key or "output_bottleneck.weight" in key:
                model.state_dict()[key].data.copy_(identity)
            elif "input_bottleneck.bias" in key or "output_bottleneck.bias" in key:
                model.state_dict()[key].data.zero_()

        logger.info("BERT-Bottleneck Initiliazed with BERT-base")

    elif args.inplace_distillation or args.sampling_type == "none":
        # initialize with pretrained model if we are using inplace distillation or if we are using no sampling
        model = custom_bert.BertForMaskedLM.from_pretrained(
            args.model_name_or_path, config=global_config
        )
    else:
        model = custom_bert.BertForMaskedLM(global_config)

    # if rewiring and resuming from checkpoint, we will load the rewiring weights seperately
    if args.rewire:
        # load the rewiring weights
        checkpoints = torch.load(
            os.path.join(args.rewired_model_checkpoint_dir, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(checkpoints, strict=False)
        # hacky way to load the importance order without introducing these
        # into __init__ of all classes and introducing problems with other
        # checkpoints
        for key in checkpoints.keys():
            if "inv_importance_order" in key:
                module_str = key.split(".inv_importance_order")[0]
                module = attrgetter(module_str)(model)
                module.register_buffer("inv_importance_order", checkpoints[key])
            elif "importance_order" in key:
                module_str = key.split(".importance_order")[0]
                module = attrgetter(module_str)(model)
                module.register_buffer("importance_order", checkpoints[key])

    model.resize_token_embeddings(len(tokenizer))
    logger.info(summary(model, depth=4, verbose=0))

    # if args.rewire is True, we would load it differently
    if args.resume_from_checkpoint_dir is not None and not args.rewire:
        logger.info("Loading model weights from checkpoint ..")
        # we load the model before preparing
        # see this for details: https://github.com/huggingface/accelerate/issues/95
        checkpoints = torch.load(
            os.path.join(args.resume_from_checkpoint_dir, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(checkpoints)

    # checkpoints = torch.load(
    #     os.path.join(
    #         "../../efficient-hardware-aware-transformers/checkpoints/random_sampling_experiments/c4_realnews_bert-bottleneck_random_K=1_pretraining_22-07-2021-22-40-40/best_model",
    #         "pytorch_model.bin",
    #     ),
    #     map_location="cpu",
    # )
    # model.load_state_dict(checkpoints)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.tokenized_c4_dir is None:
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.tokenized_c4_dir is None:
        if args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [
                    line
                    for line in examples["text"]
                    if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                # remove_columns=column_names,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column_name], return_special_tokens_mask=True
                )

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
            )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + max_seq_length]
                        for i in range(0, total_length, max_seq_length)
                    ]
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
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
            )

        if args.c4_dir is not None:
            # tokenized_datasets.save_to_disk(os.path.join(args.c4_dir, "../c4-tokenized"))
            tokenized_datasets = tokenized_datasets.remove_columns(["url", "timestamp"])

    else:
        logger.info(
            f"Skipping tokenization! as we have the tokenized dataset is already loaded from {args.tokenized_c4_dir}"
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.resume_from_checkpoint_dir is not None:

        optim_scheduler_states = torch.load(
            args.optim_scheduler_states_path, map_location="cpu"
        )

        logger.info("Loading optimizer states from checkpoint dir ..")
        accelerator.scaler.load_state_dict(optim_scheduler_states["scaler"])
        optimizer.load_state_dict(optim_scheduler_states["optimizer"])

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    # if args.resume_from_checkpoint_dir is not None:
    #    unwrapped_model = accelerator.unwrap_model(model)
    #    unwrapped_model.from_pretrained(args.resume_from_checkpoint_dir)
    #
    if (
        accelerator.distributed_type == DistributedType.MULTI_GPU
        or accelerator.distributed_type == DistributedType.TPU
    ):
        # forward missing getattr and state_dict/load_state_dict to orig model
        model = ModuleProxyWrapper(model)
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    model.set_sample_config(global_config)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    logger.info(f"Number of steps/updates per epoch: {num_update_steps_per_epoch}")
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.resume_from_checkpoint_dir is not None:
        logger.info("Loading scheduler and scalar states from checkpoint dir ..")
        completed_epochs = optim_scheduler_states["epoch"]
        completed_steps = optim_scheduler_states["steps"]
        lr_scheduler.load_state_dict(optim_scheduler_states["scheduler"])

        logger.info(f"epochs: {completed_epochs}, completed_steps: {completed_steps}")

        assert (
            completed_steps < args.max_train_steps
        ), "model is already trained to specified number of epochs or max steps"

        if optim_scheduler_states.get("current_seed", None):
            seed = optim_scheduler_states["current_seed"]
        else:
            seed = -1

    else:
        completed_epochs = 0
        completed_steps = 0
        seed = -1

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total optimization steps = {args.max_train_steps}, {completed_steps} steps completed so far"
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(completed_steps, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        wandb.watch(model, log=None, log_freq=100)

    sampler = Sampler(
        args.sampling_type, args.sampling_rule, args.mixing, global_config
    )

    if args.eval_random_subtransformers:
        if args.mixing == "mobilebert":
            diverse_num_intra_subs = sampler.get_diverse_subtransformers(
                "sample_intra_bottleneck_size"
            )
            diverse_subtransformers = diverse_num_intra_subs
            marker_colors = ["black"] * len(diverse_num_intra_subs)
            sampling_dimensions = [
                "sample_hidden_size",
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_num_hidden_layers",
                "sample_intra_bottleneck_size",
            ]
        elif args.mixing == "bert-bottleneck":
            diverse_num_intra_subs = sampler.get_diverse_subtransformers(
                "sample_hidden_size"
            )
            diverse_subtransformers = diverse_num_intra_subs
            marker_colors = ["black"] * len(diverse_num_intra_subs)
            sampling_dimensions = [
                "sample_hidden_size",
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_num_hidden_layers",
            ]
        else:
            diverse_hidden_state_subs = sampler.get_diverse_subtransformers(
                "sample_hidden_size"
            )
            diverse_attention_subs = sampler.get_diverse_subtransformers(
                "sample_num_attention_heads"
            )
            diverse_intermediate_state_subs = sampler.get_diverse_subtransformers(
                "sample_intermediate_size"
            )
            diverse_num_hidden_subs = sampler.get_diverse_subtransformers(
                "sample_num_hidden_layers"
            )

            diverse_subtransformers = (
                diverse_hidden_state_subs
                + diverse_attention_subs
                + diverse_intermediate_state_subs
            )
            marker_colors = (
                ["yellow"] * len(diverse_hidden_state_subs)
                + ["green"] * len(diverse_attention_subs)
                + ["blue"] * len(diverse_intermediate_state_subs)
                + ["red"] * len(diverse_num_hidden_subs)
            )
            sampling_dimensions = [
                "sample_hidden_size",
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_num_hidden_layers",
            ]

    best_val_perplexity = 1000000
    logger.info("=============================")
    logger.info(f"Starting training from epoch {completed_epochs}")
    logger.info(f"Training till epoch  {args.num_train_epochs}")
    logger.info("=============================")

    eval_metric = validate_subtransformer(
        model,
        eval_dataloader,
        accelerator,
        len(eval_dataset),
        args.per_device_eval_batch_size,
        args.pad_to_max_length,
    )
    perpx = eval_metric["perplexity"]
    logger.info(f"perplexity before starting: {perpx:.2f} ")
    for epoch in range(completed_epochs, args.num_train_epochs):
        # first evaluate random subtransformers before starting training
        if args.eval_random_subtransformers and completed_epochs % 1 == 0:
            hover_templates = []
            label_perplex = []
            for i, config in enumerate(diverse_subtransformers):
                model.set_sample_config(config)

                eval_metric = validate_subtransformer(
                    model,
                    eval_dataloader,
                    accelerator,
                    len(eval_dataset),
                    args.per_device_eval_batch_size,
                    args.pad_to_max_length,
                )
                # eval_metric['validation_random_seed'] = random_seed
                # label_lst.append([eval_metric['accuracy'], random_seed])
                # label_lst.append([random_seed, eval_metric['accuracy']])
                hover_templates.append(
                    "<br>".join(
                        [
                            f"{key}: {getattr(config, key)}"
                            for key in sampling_dimensions
                        ]
                        # adding the evaluation metrics to print
                        + [f"{key}: {eval_metric[key]}" for key in eval_metric]
                    )
                )
                label_perplex.append(eval_metric["perplexity"])
                # label_seed.append(random_seed)
                # accelerator.print(eval_metric)
                # wandb.log(eval_metric)

            if accelerator.is_main_process:
                ## If plotting using Custom Plotly
                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=np.arange(len(diverse_subtransformers)),
                        y=label_perplex,
                        hovertext=hover_templates,
                        marker_color=marker_colors,
                    )
                )
                fig.update_layout(
                    title="Relative Performance Order",
                    xaxis_title="Random Seed",
                    yaxis_title="Perplexity",
                )
                wandb.log({"bar_chart": wandb.data_types.Plotly(fig)})

        model.train()
        k_count = args.k_sampling - 1

        for step, batch in enumerate(train_dataloader):
            seed += 1
            k_count += 1
            if k_count == args.k_sampling and args.sampling_type != "none":
                config_dict = sampler.sample_subtransformer(
                    randomize=True,
                    rand_seed=seed,
                    pop_size=args.pop_size,
                )
                k_count = 0

                super_config_small = config_dict["smallest_subtransformer"]
                # list of random subtransformers with len pop_size
                super_configs = config_dict["random_subtransformers"]

            track_loss = step % args.logging_steps == 0 and step > 0

            ### Applying Sandwich Rule ###
            if args.sampling_rule == "sandwich":

                ## Sample Supertransformer
                model.set_sample_config(global_config)
                outputs = model(**batch)
                loss = outputs.loss

                teacher_mlm_loss = loss
                teacher_mlm_loss = teacher_mlm_loss / args.gradient_accumulation_steps
                accelerator.backward(teacher_mlm_loss)

                if args.inplace_distillation:

                    teacher_hidden_states = outputs.hidden_states
                    teacher_attention_maps = outputs.attentions

                    # logits are of shape batch_size, sequence_length, config.vocab_size
                    soft_logits = outputs.logits.detach()

                    # replace the labels in our batch to soft_targets
                    batch["labels"] = soft_logits

                ## Sample Smallest Subtransformer
                model.set_sample_config(super_config_small)
                outputs = model(**batch, use_soft_loss=args.inplace_distillation)
                loss = outputs.loss

                if args.inplace_distillation:

                    (
                        smallest_student_loss,
                        smallest_student_losses_dict,
                    ) = compute_student_loss(
                        outputs,
                        teacher_hidden_states,
                        teacher_attention_maps,
                        args,
                        track_layerwise_loss=track_loss,
                    )
                else:
                    # TODO: Terminology consistency needs to be maintained - technically not a student!
                    smallest_student_loss = loss
                    smallest_student_loss = (
                        smallest_student_loss / args.gradient_accumulation_steps
                    )

                accelerator.backward(smallest_student_loss)

            ## Sample "n" subtransformers based on sampling_type: random, biased-params, etc.
            ## This happens regardless of sandwich rule is applied or not! Allows for Conventional Sampling

            for idx in range(args.pop_size):

                if args.sampling_type != "none":
                    super_config = super_configs[idx]
                    model.set_sample_config(super_config)

                outputs = model(**batch, use_soft_loss=args.inplace_distillation)
                loss = outputs.loss

                if args.inplace_distillation:

                    (
                        sampled_student_loss,
                        sampled_student_losses_dict,
                    ) = compute_student_loss(
                        outputs,
                        teacher_hidden_states,
                        teacher_attention_maps,
                        args,
                        track_layerwise_loss=track_loss,
                    )
                else:
                    sampled_student_loss = loss / args.gradient_accumulation_steps

                accelerator.backward(sampled_student_loss)

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress_bar.update(1)
                completed_steps += 1

                ### Plot the high-res step-loss ###
                if accelerator.is_main_process and track_loss:
                    if not args.inplace_distillation:
                        if args.sampling_rule == "sandwich":
                            wandb.log(
                                {
                                    "Supertransformer mlm loss": teacher_mlm_loss.item(),
                                    "Smallest mlm loss": smallest_student_loss.item(),
                                    "Subtransformer mlm loss": sampled_student_loss.item(),
                                }
                            )
                        else:

                            wandb.log(
                                {
                                    "Subtransformer mlm loss": sampled_student_loss.item(),
                                }
                            )
                    else:

                        if args.layerwise_distillation:
                            wandb.log(
                                {
                                    "Supertransformer Teacher mlm loss": teacher_mlm_loss.item(),
                                    "Smallest Student mlm loss": smallest_student_losses_dict[
                                        "student_mlm_loss"
                                    ].item(),
                                    "Smallest distill loss": smallest_student_losses_dict[
                                        "student_distill_loss"
                                    ].item(),
                                    "Smallest feature knowledge transfer loss": smallest_student_losses_dict[
                                        "student_feature_knowledge_transfer_loss"
                                    ].item(),
                                    "Smallest attention knowledge transfer loss": smallest_student_losses_dict[
                                        "student_attention_knowledge_transfer_loss"
                                    ].item(),
                                    "Subtransformer Student mlm loss": sampled_student_losses_dict[
                                        "student_mlm_loss"
                                    ].item(),
                                    "Subtransformer distill loss": sampled_student_losses_dict[
                                        "student_distill_loss"
                                    ].item(),
                                    "Subtransformer feature knowledge transfer loss": sampled_student_losses_dict[
                                        "student_feature_knowledge_transfer_loss"
                                    ].item(),
                                    "Subtransformer attention knowledge transfer loss": sampled_student_losses_dict[
                                        "student_attention_knowledge_transfer_loss"
                                    ].item(),
                                }
                            )
                            for idx in range(
                                len(smallest_student_losses_dict["layer_wise_akt"])
                            ):
                                wandb.log(
                                    {
                                        f"Smallest layer_wise_akt_{idx}": smallest_student_losses_dict[
                                            "layer_wise_akt"
                                        ][
                                            idx
                                        ].item(),
                                        f"Subtransformer layer_wise_akt_{idx}": sampled_student_losses_dict[
                                            "layer_wise_akt"
                                        ][
                                            idx
                                        ].item(),
                                        f"Smallest layer_wise_fkt_{idx}": smallest_student_losses_dict[
                                            "layer_wise_fkt"
                                        ][
                                            idx
                                        ].item(),
                                        f"Subtransformer layer_wise_fkt_{idx}": sampled_student_losses_dict[
                                            "layer_wise_fkt"
                                        ][
                                            idx
                                        ].item(),
                                    }
                                )
                        else:
                            wandb.log(
                                {
                                    "Supertransformer Teacher mlm loss": teacher_mlm_loss.item(),
                                    "Smallest Student mlm loss": smallest_student_losses_dict[
                                        "student_mlm_loss"
                                    ].item(),
                                    "Subtransformer Student mlm loss": sampled_student_losses_dict[
                                        "student_mlm_loss"
                                    ].item(),
                                }
                            )

            if accelerator.is_main_process:
                wandb.log({"epochs": epoch})

            if completed_steps >= args.max_train_steps:
                break

        # change to supertransformer config
        if args.sampling_type != "none":
            model.set_sample_config(global_config)

        eval_metric = validate_subtransformer(
            model,
            eval_dataloader,
            accelerator,
            len(eval_dataset),
            args.per_device_eval_batch_size,
            args.pad_to_max_length,
        )
        val_accuracy, val_loss, perplexity = (
            eval_metric["accuracy"] * 100,
            eval_metric["val_loss"],
            eval_metric["perplexity"],
        )

        if accelerator.is_main_process:
            wandb.log(
                {
                    "SuperTransformer Val Accuracy": val_accuracy,
                    "SuperTransformer Val loss": val_loss,
                    "SuperTransformer Perplexity": perplexity,
                }
            )
        logger.info(
            f"epoch {epoch}: val_perplexity: {perplexity:.2f}, val_loss: {val_loss:.2f}, val_accuracy:  {val_accuracy:.2f}"
        )
        completed_epochs += 1

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(args.output_dir, "best_model"),
                save_function=accelerator.save,
            )
            if (
                best_val_perplexity >= eval_metric["perplexity"]
            ):  ## Saving the best model
                best_val_perplexity = eval_metric["perplexity"]
                accelerator.save(
                    {
                        "epoch": completed_epochs,
                        "current_seed": seed,
                        "steps": completed_steps,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "scaler": accelerator.scaler.state_dict(),
                    },
                    args.optim_scheduler_states_path.format("best_model"),
                )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.output_dir, "last_model"), save_function=accelerator.save
        )
        accelerator.save(
            {
                "epoch": completed_epochs,
                "current_seed": seed,
                "steps": completed_steps,
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": accelerator.scaler.state_dict(),
            },
            args.optim_scheduler_states_path.format("last_model"),
        )


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    # main()
    main()
