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


import numpy as np
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

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

from engine import (
    sample_subtransformer,
    get_supertransformer_config,
    show_random_elements,
    show_args,
    get_diverse_subtransformers,
)
from custom_layers import custom_bert, custom_mobile_bert

import plotly.graph_objects as go
from utils import count_parameters, check_path, get_current_datetime, unique_everseen

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
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
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
        default=None,
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
        choices=["attention", "gmlp", "fnet", "mobilebert"],
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
        "--no_sampling",
        type=int,
        default=0,
        help=f"If set to 1, there will be no sampling. This is useful for training/testing pretrained or whole models",
    )

    parser.add_argument(
        "--sampling_type",
        type=str,
        default="random",
        help=f"The sampling type for super-transformer",
        choices=["naive_params", "biased_params", "random", "sandwich"],
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

    args = parser.parse_args()

    args.model_name_or_path = "bert-base-cased"
    # Sanity checks

    if args.inplace_distillation == 1:
        # hard setting this for now
        args.sampling_type = "sandwich"
        args.mixing = "attention"

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.c4_dir is None
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

    if args.no_sampling == 1:
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
    if args.inplace_distillation:
        str_name += "_ip_distill"
    else:
        str_name += "_pretraining"
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
    if args.dataset_name is not None:
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
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.config_name)
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    global_config = get_supertransformer_config(
        args.model_name_or_path, tiny_attn=args.tiny_attn, mixing=args.mixing
    )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
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

    # TODO: decouple mixing and mobilebert model declarato
    if args.mixing == "mobilebert":
        # config.sample_hidden_size
        # config.sample_intermediate_size
        # config.sample_intra_bottleneck_size
        # config.sample_num_attention_heads
        # config.sample_num_hidden_layers
        # config.sample_true_hidden_size

        # for mobilebert, dont use layernorm
        # global_config.normalization_type = "no_norm"
        global_config.normalization_type = "layer_norm"
        # number of ffn blocks
        global_config.num_feedforward_networks = 1
        global_config.use_bottleneck_attention = False

        # for attention transfer and feature transfer enable these.
        global_config.output_attentions = True
        global_config.output_hidden_layers = True

        states = OD()

        model = custom_mobile_bert.MobileBertForMaskedLM(
            "bert-base-cased", config=global_config
        )
        model2 = custom_bert.BertForMaskedLM.from_pretrained(
            "bert-base-cased", config=global_config
        )

        for key in model.state_dict().keys():
            _key = key.replace("mobilebert.", "bert.")
            states[_key] = model.state_dict()[_key]

        del model2
        model.load_state_dict(states, strict=False)
        del states

        identity = torch.eye(global_config.true_hidden_size)
        zero_bias = torch.zeros(global_config.true_hidden_size)
        for key in model.state_dict().keys():
            if (
                "bottleneck.input.dense.weight" in key
                or "output.bottleneck.dense.weight" in key
            ):
                model.state_dict()[key].data.copy_(identity)
            elif (
                "bottleneck.output.dense.bias" in key
                or "output.bottleneck.dense.weight" in key
            ):
                model.state_dict()[key].data.copy_(zero_bias)

    else:
        global_config.normalization_type = "layer_norm"
        global_config.num_feedforward_networks = 1

    if args.inplace_distillation or args.no_sampling:
        # initialize with pretrained model if we are using inplace distillation or if we are using no sampling
        model = custom_bert.BertForMaskedLM.from_pretrained(
            args.model_name_or_path, config=global_config
        )
    else:
        model = custom_bert.BertForMaskedLM(global_config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info(summary(model, depth=4, verbose=0))

    # maybe not required but doing it just to be sure
    model.set_sample_config(global_config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
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
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
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
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
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
        logger.info("Loading model weights from checkpoint ..")
        # we load the model before preparing
        # see this for details: https://github.com/huggingface/accelerate/issues/95
        model.load_state_dict(
            torch.load(
                os.path.join(args.resume_from_checkpoint_dir, "pytorch_model.bin")
            )
        )

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

    else:
        completed_epochs = 0
        completed_steps = 0

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
        wandb.watch(model)

    def get_diverse_seeds(num_subtransformers, config):
        diverse_seeds = []
        num_hidden_layers_seeds = defaultdict(list)
        for seed in range(num_subtransformers * 4):
            # num_hidden_layers = sample_subtransformer(
            #        True, seed, config=config
            # ).sample_num_hidden_layers
            super_config, _ = sample_subtransformer(
                True, seed, config=config, sampling_type=args.sampling_type
            )
            num_hidden_layers = super_config.sample_num_hidden_layers
            num_hidden_layers_seeds[num_hidden_layers].append(seed)
        uniq_num_hidden_layers = len(num_hidden_layers_seeds.keys())
        num_per_uniq_layer = (num_subtransformers // uniq_num_hidden_layers) + 1
        for k, v in num_hidden_layers_seeds.items():
            diverse_seeds.extend(v[:num_per_uniq_layer])
        return diverse_seeds[:num_subtransformers]

    # logger.info("Generating diverse random seeds..")
    # rand_seed_lst = get_diverse_seeds(args.num_subtransformers_monitor, global_config)
    # logger.info(len(rand_seed_lst))
    # logger.info("Random seeds generation done..")
    if args.eval_random_subtransformers:
        diverse_hidden_state_subs = get_diverse_subtransformers(
            "sample_hidden_size", global_config
        )
        diverse_attention_subs = get_diverse_subtransformers(
            "sample_num_attention_heads", global_config
        )
        diverse_intermediate_state_subs = get_diverse_subtransformers(
            "sample_intermediate_size", global_config
        )
        diverse_num_hidden_subs = get_diverse_subtransformers(
            "sample_num_hidden_layers", global_config
        )

        diverse_subtransformers = (
            diverse_hidden_state_subs
            + diverse_attention_subs
            + diverse_intermediate_state_subs
            + diverse_num_hidden_subs
        )
        # get unique subtransformers
        diverse_subtransformers = list(unique_everseen(diverse_subtransformers))

        # colors = px.colors.sequential.Viridis
        marker_colors = (
            ["yellow"] * len(diverse_hidden_state_subs)
            + ["green"] * len(diverse_attention_subs)
            + ["blue"] * len(diverse_intermediate_state_subs)
            + ["red"] * len(diverse_num_hidden_subs)
        )

    best_val_perplexity = 1000000
    seed = -1
    logger.info("=============================")
    logger.info(completed_epochs)
    logger.info(args.num_train_epochs)
    logger.info("=============================")
    for epoch in range(completed_epochs, args.num_train_epochs):
        # first evaluate random subtransformers before starting training
        if args.eval_random_subtransformers and completed_epochs % 3 == 0:
            hover_templates = []
            label_perplex = []
            sampling_dimensions = [
                "sample_hidden_size",
                "sample_num_attention_heads",
                "sample_intermediate_size",
                "sample_num_hidden_layers",
            ]
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
                        + [f"{key}: {getattr(config, key)}" for key in eval_metric]
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
        # seed = -1 ## Don't re-initialize the seed! Allow totally random subtransformers
        for step, batch in enumerate(train_dataloader):
            seed += 1
            k_count += 1
            if k_count == args.k_sampling and args.no_sampling != 1:
                super_config, super_config_small = sample_subtransformer(
                    randomize=True,
                    rand_seed=seed,
                    tiny_attn=args.tiny_attn,
                    config=global_config,
                    sampling_type=args.sampling_type,
                )
                k_count = 0

            if args.inplace_distillation:

                model.set_sample_config(global_config)
                outputs = model(**batch)
                loss = outputs.loss
                teacher_hidden_states = outputs.hidden_states
                teacher_attention_maps = outputs.attention_maps
                loss /= args.gradient_accumulation_steps
                accelerator.backward(loss)
                # logits are of shape batch_size, sequence_length, config.vocab_size
                # hence applying softmanx to last dim
                soft_targets = torch.nn.functional.softmax(
                    outputs.logits.detach(), dim=-1
                )

                # replace the labels in our batch to soft_targets
                batch["labels"] = soft_targets

                model.set_sample_config(super_config_small)
                outputs = model(**batch, use_soft_loss=True)
                loss = outputs.loss
                smallest_student_hidden_states = outputs.hidden_states
                smallest_student_attention_maps = outputs.attention_maps
                loss = loss
                loss /= args.gradient_accumulation_steps
                accelerator.backward(loss)

                model.set_sample_config(super_config)
                outputs = model(**batch, use_soft_loss=True)
                loss = outputs.loss
                student_hidden_states = outputs.hidden_states
                student_attention_maps = outputs.attention_maps
                loss = loss
                loss /= args.gradient_accumulation_steps
                accelerator.backward(loss)

                # loss = (loss_big + loss_small + loss_nl) / 3

            else:  # Other means of sampling
                # model.set_sample_config(super_config)

                # super_config.max_seq_length = config.max_seq_length
                # super_config.mixing = config.mixing
                # super_config.num_hidden_layers = super_config.sample_num_hidden_layers
                # subnet = model.get_active_subnet(super_config)

                # logger.info(subnet)
                if args.no_sampling != 1:
                    model.set_sample_config(super_config)

                outputs = model(**batch)
                loss = outputs.loss
                loss /= args.gradient_accumulation_steps
                accelerator.backward(loss)

            # loss = loss / args.gradient_accumulation_steps
            # accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                ### Plot the high-res step-loss ###
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "Supertransformer Train loss": loss,
                        }
                    )

            if accelerator.is_main_process:
                wandb.log({"epochs": epoch})

            if completed_steps >= args.max_train_steps:
                break

        # change to supertransformer config
        if args.no_sampling != 1:
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
                "steps": completed_steps,
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": accelerator.scaler.state_dict(),
            },
            args.optim_scheduler_states_path.format("last_model"),
        )


if __name__ == "__main__":
    main()
