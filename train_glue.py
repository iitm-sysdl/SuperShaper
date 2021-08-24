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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""

# taken and modified from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
# on 21-6-2021
import argparse
import logging
import math
import os
import random
import wandb
import torch

import numpy as np
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

from collections import defaultdict
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
from utils import (
    count_parameters,
    check_path,
    get_current_datetime,
    read_json,
    calculate_params_from_config,
    millify,
)

from torchinfo import summary

from utils.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
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
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
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
        default=0,
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

    # args we add
    parser.add_argument(
        "--early_stopping_patience",
        default=5,
        type=int,
        help="Patience for early stopping to stop training if val_acc doesnt converge",
    )
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
        "--rewire",
        type=int,
        default=0,
        help=f"Whether to rewire model",
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
        "--debug",
        action="store_true",
        help="If passed, use 100 samples of dataset to quickly run and check code.",
    )

    parser.add_argument(
        "--sampling_type",
        type=str,
        default="random",
        help=f"The sampling type for super-transformer",
        choices=["none", "naive_params", "biased_params", "random"],
    )
    parser.add_argument(
        "--subtransformer_config_path",
        type=str,
        default=None,
        help=f"The path to a subtransformer configration",
    )
    parser.add_argument(
        "--wandb_suffix",
        type=str,
        default=None,
        help=f"suffix for wandb",
    )

    parser.add_argument(
        "--mnli_checkpoint_path",
        type=str,
        default=None,
        help=f"path to mnli checkpoint",
    )

    args = parser.parse_args()

    # args.model_name_or_path = "bert-base-cased"
    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
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

    if args.sampling_type == "none":
        # if we are not sampling, dont test random subtransformers every n epochs
        args.eval_random_subtransformers = False

    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.tiny_attn == 1:
        assert args.mixing == "gmlp", "Tiny Attention can work only in GMLP setup"

    if args.mixing == "gmlp" and not args.pad_to_max_length:
        raise ValueError("Need to pad to max length when using gmlp")

    if args.output_dir is not None and args.resume_from_checkpoint_dir is None:
        task_name = args.task_name.split("/")[-1].strip()
        args.output_dir += (
            "/" + task_name + "_" + args.mixing + "_" + get_current_datetime()
        )
        args.optim_scheduler_states_path = os.path.join(
            args.output_dir, "optimizer_scheduler.pt"
        )
        os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_from_checkpoint_dir is not None:

        args.optim_scheduler_states_path = os.path.join(
            args.resume_from_checkpoint_dir,
            "optimizer_scheduler.pt",
        )
        check_path(args.resume_from_checkpoint_dir)
        check_path(args.optim_scheduler_states_path)

        model_path = os.path.join(args.resume_from_checkpoint_dir, "pytorch_model.bin")
        check_path(model_path)
        # overwrite on the same directory
        args.output_dir = args.resume_from_checkpoint_dir

    if args.subtransformer_config_path:
        check_path(args.subtransformer_config_path)
        assert (
            args.sampling_type == "none"
        ), "sampling_type is not supported when providing custom_subtransformer_config"
        assert (
            args.eval_random_subtransformers == 0
        ), "no need to evaluate random subtransformers when a custom_subtransformer_config is provided"

    if args.mnli_checkpoint_path:
        check_path(args.mnli_checkpoint_path)
        assert args.task_name in [
            "cola",
            "stsb",
            "rte",
        ], "mnli checkpoint can only be used for mnli"

    return args


def validate_subtransformer(model, task_name, eval_dataloader, accelerator):
    is_regression = task_name == "stsb"
    if task_name is not None:
        metric = load_metric("glue", task_name)
    else:
        metric = load_metric("accuracy")
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = (
            outputs.logits.argmax(dim=-1)
            if not is_regression
            else outputs.logits.squeeze()
        )
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    return eval_metric


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
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    str_name = (
        args.mixing + "_tiny_attn"
        if args.tiny_attn == 1
        else args.mixing + "_" + args.sampling_type
    )
    if args.subtransformer_config_path:
        str_name += "_custom_subtransformer"

    if args.wandb_suffix:
        str_name += "_" + args.wandb_suffix

    if args.debug:
        str_name = "debugging"

    if accelerator.is_main_process:
        wandb.init(
            project="Glue-Finetuning",
            entity="efficient-hat",
            name=args.task_name.split("/")[-1].strip() + "_" + str_name,
        )

    if args.output_dir is not None and args.resume_from_checkpoint_dir is None:
        dataset_name = args.task_name.split("/")[-1].strip()
        args.output_dir += (
            "/" + dataset_name + "_" + str_name + "_" + get_current_datetime()
        )
        args.optim_scheduler_states_path = os.path.join(
            args.output_dir, "{}/optimizer_scheduler.pt"
        )
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
        if args.debug:
            raw_datasets["train"] = raw_datasets["train"].select(range(100))
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (
            args.train_file if args.train_file is not None else args.valid_file
        ).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(
    #     args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name
    # )
    global_config = get_supertransformer_config("bert-base-cased", mixing=args.mixing)
    global_config.rewire = args.rewire

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )

    if args.max_length:
        global_config.max_seq_length = args.max_length
    else:
        logger.warning(
            f"The max_seq_length is not defined!! Setting it to max length in tokenizer"
        )
        global_config.max_seq_length = tokenizer.model_max_length

    global_config.num_labels = num_labels
    # global_config.hidden_dropout_prob = 0

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

    if args.mixing == "mobilebert":
        model = custom_mobile_bert.MobileBertForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=global_config
        )
    else:
        model = custom_bert.BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=global_config,
        )

    if args.mnli_checkpoint_path is not None:
        checkpoints = torch.load(
            os.path.join(args.mnli_checkpoint_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(checkpoints, strict=True)

    logger.info(summary(model, depth=4, verbose=0))

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in global_config.label2id.items()
        }

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

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
        model.from_pretrained(args.resume_from_checkpoint_dir)

        optim_scheduler_states = torch.load(args.optim_scheduler_states_path)

        logger.info("Loading optimizer states from checkpoint dir ..")
        accelerator.scaler.load_state_dict(optim_scheduler_states["scaler"])
        optimizer.load_state_dict(optim_scheduler_states["optimizer"])

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

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

        assert (completed_epochs < args.num_train_epochs) and (
            completed_steps < args.max_train_steps
        ), "model is already trained to specified number of epochs or max steps"

    else:
        completed_epochs = 0
        completed_steps = 0

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    if args.task_name == "stsb":
        metric_key = "pearson"
    elif args.task_name == "cola":
        metric_key = "matthews_correlation"
    else:
        metric_key = "accuracy"

    early_stopping = EarlyStopping(metric_key, patience=args.early_stopping_patience)

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
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    if accelerator.is_main_process:
        wandb.watch(model)

    sampler = Sampler(args.sampling_type, "none", args.mixing, global_config)

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

    logger.info("=============================")
    logger.info(f"Starting training from epoch {completed_epochs}")
    logger.info(f"Training till epoch  {args.num_train_epochs}")
    logger.info("=============================")
    best_val_acc = 0
    for epoch in range(completed_epochs, args.num_train_epochs):
        # first evaluate random subtransformers before starting training
        if args.eval_random_subtransformers and completed_epochs % 1 == 0:
            hover_templates = []
            label_perplex = []
            for i, config in enumerate(diverse_subtransformers):
                model.set_sample_config(config)

                eval_metric = validate_subtransformer(
                    model,
                    args.task_name,
                    eval_dataloader,
                    accelerator,
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
                label_perplex.append(eval_metric[metric_key])

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
                    yaxis_title="Accuracies",
                )
                wandb.log({"bar_chart": wandb.data_types.Plotly(fig)})
        model.train()
        seed = -1
        for step, batch in enumerate(train_dataloader):
            seed += 1
            if args.sampling_type != "none":
                super_config = sampler.sample_subtransformer(
                    randomize=True, rand_seed=seed, pop_size=1
                )["random_subtransformers"][0]

                model.set_sample_config(super_config)

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if accelerator.is_main_process:
                wandb.log({"epochs": epoch})

            if completed_steps >= args.max_train_steps:
                break

        eval_metric = validate_subtransformer(
            model, args.task_name, eval_dataloader, accelerator
        )
        logger.info(f"epoch {epoch}: {eval_metric}")

        ## Logging all the eval metrics + best accuracy for ease of tracking
        if accelerator.is_main_process:
            wandb.log(eval_metric)
            if best_val_acc <= eval_metric[metric_key]:
                best_val_acc = eval_metric[metric_key]
            wandb.log({f"Best {metric_key}": best_val_acc})

        completed_epochs += 1

        if args.output_dir is not None:

            early_stopping(eval_metric)

            if early_stopping.counter == 0:
                # if counter is 0, it means the metric has improved
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(args.output_dir, "best_model"),
                    save_function=accelerator.save,
                )
                accelerator.save(
                    {
                        "epoch": completed_epochs,
                        "steps": completed_steps,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "scaler": accelerator.scaler.state_dict(),
                        metric_key: early_stopping.best_score,
                    },
                    args.optim_scheduler_states_path.format("best_model"),
                )
            if early_stopping.early_stop:
                logger.info(
                    "==========================================================================="
                )
                logger.info(
                    f"Early Stopping !!! {metric_key} hasnt improved for {args.early_stopping_patience} epochs"
                )
                logger.info(
                    "==========================================================================="
                )
                break

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")


if __name__ == "__main__":
    main()