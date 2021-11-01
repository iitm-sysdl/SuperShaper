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
"""
Fine-tuning a 🤗 Transformers model on question answering.
"""
# taken and modified from https://github.com/huggingface/transformers/blob/f0dde6012784c79155f7c1bd721864ea8ba40821/examples/pytorch/question-answering/run_qa_no_trainer.py

import argparse
import logging
import math
import os
import random
import wandb

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils.utils_qa import postprocess_qa_predictions

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


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/question-answering/requirements.txt",
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Question Answering task"
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
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="To do prediction on the question answering model",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the Prediction data.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
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
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=False,
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
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

    args = parser.parse_args()

    # args.model_name_or_path = "bert-base-cased"
    # Sanity checks
    if (
        args.dataset_name is None
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
        args.dataset_name is None
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

    if args.tiny_attn == 1:
        assert args.mixing == "gmlp", "Tiny Attention can work only in GMLP setup"

    if args.mixing == "gmlp" and not args.pad_to_max_length:
        raise ValueError("Need to pad to max length when using gmlp")

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

    return args


# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


# Post-processing:
def post_processing_function(args, examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

    references = [
        {"id": ex["id"], "answers": ex[args.answer_column_name]} for ex in examples
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def validate_subtransformer(model, eval_dataset_tuple, accelerator, args):
    metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")
    eval_examples, eval_dataset, eval_dataloader = eval_dataset_tuple
    all_start_logits = []
    all_end_logits = []

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if (
                not args.pad_to_max_length
            ):  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(
                    start_logits, dim=1, pad_index=-100
                )
                end_logits = accelerator.pad_across_processes(
                    end_logits, dim=1, pad_index=-100
                )

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max(
        [x.shape[1] for x in all_start_logits]
    )  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(
        all_start_logits, eval_dataset, max_len
    )
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(
        args, eval_examples, eval_dataset, outputs_numpy
    )
    eval_metric = metric.compute(
        predictions=prediction.predictions, references=prediction.label_ids
    )

    return eval_metric, prediction


def main():
    args = parse_args()

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    show_args(accelerator, args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
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
            project="Squad-finetuning",
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
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    global_config = get_supertransformer_config("bert-base-cased", mixing=args.mixing)
    global_config.rewire = args.rewire
    global_config.layer_drop_prob = 0.0

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # this is needed in postprocessing function
    # hence we store it in args for convenience
    args.answer_column_name = answer_column_name

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length:
        global_config.max_seq_length = args.max_seq_length
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
    else:
        logger.warning(
            f"The max_seq_length is not defined!! Setting it to max length in tokenizer"
        )
        global_config.max_seq_length = tokenizer.model_max_length

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

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
        model = custom_mobile_bert.MobileBertModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path, config=global_config
        )
    else:
        model = custom_bert.BertForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            config=global_config,
        )

    logger.info(summary(model, depth=4, verbose=0))

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))
    # Create train feature from dataset
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_examples = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        # Predict Feature Creation
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
        if args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))

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

    eval_dataset_for_model = eval_dataset.remove_columns(
        ["example_id", "offset_mapping"]
    )
    eval_dataloader = DataLoader(
        eval_dataset_for_model,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(
            ["example_id", "offset_mapping"]
        )
        predict_dataloader = DataLoader(
            predict_dataset_for_model,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    early_stopping = EarlyStopping("f1", patience=args.early_stopping_patience)

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

    # needed for validate_subtransformer
    eval_dataset_tuple = (eval_examples, eval_dataset, eval_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

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
    best_f1 = 0

    for epoch in range(completed_epochs, args.num_train_epochs):
        # first evaluate random subtransformers before starting training
        if args.eval_random_subtransformers and completed_epochs % 1 == 0:
            hover_templates = []
            label_perplex = []
            for i, config in enumerate(diverse_subtransformers):
                model.set_sample_config(config)

                eval_metric, _ = validate_subtransformer(
                    model, eval_dataset_tuple, accelerator, args
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
                label_perplex.append(eval_metric["f1"])

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
                    yaxis_title="F1 scores",
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

        # Evaluation
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
        eval_metric, _ = validate_subtransformer(
            model, eval_dataset_tuple, accelerator, args
        )

        logger.info(f"epoch {epoch} Evaluation metrics: {eval_metric}")
        ## Logging all the eval metrics + best accuracy for ease of tracking
        if accelerator.is_main_process:
            wandb.log(eval_metric)
            if best_f1 <= eval_metric["f1"]:
                best_f1 = eval_metric["f1"]
            wandb.log({"Best F1": best_f1})

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
                        "accuracy": early_stopping.best_score,
                    },
                    args.optim_scheduler_states_path.format("best_model"),
                )
            if early_stopping.early_stop:
                logger.info(
                    "==========================================================================="
                )
                logger.info(
                    f"Early Stopping !!! F1 hasnt improved for {args.early_stopping_patience} epochs"
                )
                logger.info(
                    "==========================================================================="
                )
                break

    # Prediction
    if args.do_predict:
        logger.info("***** Loading best model before evaluating on test dataset *****")
        checkpoints = torch.load(
            os.path.join(args.output_dir, "best_model"),
            map_location="cpu",
        )
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoints)

        logger.info("***** Running Test Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        pred_dataset_tuple = (predict_examples, predict_dataset, predict_dataloader)

        predict_metric, _ = validate_subtransformer(
            model, pred_dataset_tuple, accelerator, args
        )

        logger.info(f"Predict metrics: {predict_metric}")


if __name__ == "__main__":
    main()