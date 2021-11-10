import argparse
import logging
import math
import os
import random
import wandb


import plotly.express as px
from datetime import datetime
from collections import defaultdict
import time
from statistics import mean
import pandas as pd

from utils import dropout_layers
import numpy as np
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from utils.module_proxy_wrapper import ModuleProxyWrapper
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType

from sampling import (
    Sampler,
    get_supertransformer_config,
    show_args,
)
from custom_layers import custom_bert

import plotly.graph_objects as go
from utils import check_path, read_json, millify, calculate_params_from_config
from tqdm import tqdm
from torchinfo import summary

logger = logging.getLogger(__name__)


def validate_subtransformer(
    model,
    eval_dataloader,
    accelerator,
    len_eval_dataset,
    per_device_eval_batch_size,
    pad_to_max_length,
    evaluate_latency=False,
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
    exec_time = []
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)

        if evaluate_latency:
            if accelerator.device == "cuda":
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
            else:
                start_time = time.monotonic()

        with torch.no_grad():
            outputs = model(**batch)

        if evaluate_latency:
            if accelerator.device == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                exec_time.append(
                    start_time.elapsed_time(end_time)
                )  ## Measured in seconds
            else:
                exec_time.append(
                    (time.monotonic() - start_time)
                )  ## Measured in seconds

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
        if evaluate_latency:
            execution_time = mean(exec_time)
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    eval_metric["val_loss"] = val_loss
    eval_metric["perplexity"] = perplexity

    if evaluate_latency:
        eval_metric["exec_time"] = execution_time

    return eval_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Test/validate a saved checkpoint ")
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
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
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
        "--num_subtransformers_eval",
        type=int,
        default=0,
        help=f"Number of random subtransformers to evaluate",
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
        "--tiny_attn",
        type=int,
        default=0,
        help=f"Choose this if you need Tiny Attention Module along-with gMLP dense block",
    )

    parser.add_argument(
        "--c4_dir",
        type=str,
        default=None,
        help=f"The directory path for C4",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=f"Directory storing the checkpoint",
    )
    parser.add_argument(
        "--subtransformer_config_path",
        type=str,
        default=None,
        help=f"The path to a subtransformer configration",
    )

    parser.add_argument(
        "--evaluate_latency",
        type=int,
        default=0,
        help=f"Evaluate Latency",
    )

    parser.add_argument(
        "--only_latency",
        type=int,
        required=True,
        help=f"Evaluate only Latency",
    )

    parser.add_argument(
        "--num_iteration",
        type=int,
        default=10,
        help=f"The number of iterations to evaluate latency",
    )

    parser.add_argument(
        "--tokenized_c4_dir",
        type=str,
        default=None,
        help=f"The directory path for tokenized C4",
    )
    parser.add_argument(
        "--layer_drop_prob",
        default=0.0,
        type=float,
        help="Probability to drop layers",
    )
    parser.add_argument(
        "--additional_random_softmaxing",
        action="store_true",
        help=f"if true then random softmax layers will be softmaxed in addition to the last layer, except that there will be a random walk when it comes to choosing the layer to softmax",
    )

    args = parser.parse_args()
    args.model_name_or_path = "bert-base-cased"

    if (
        args.dataset_name is None
        and args.validation_file is None
        and args.c4_dir is None
        and args.tokenized_c4_dir is None
    ):
        raise ValueError("Need either a dataset name or a validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if args.c4_dir is not None:
        check_path(args.c4_dir)
        args.dataset_name = "c4_realnews"

    if args.tokenized_c4_dir is not None:
        check_path(args.tokenized_c4_dir)
        args.dataset_name = "c4_realnews"

    return args


def main():
    args = parse_args()

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[param])

    if accelerator.device == "cuda":
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
        args.model_name_or_path, mixing=args.mixing
    )
    global_config.layer_drop_prob = args.layer_drop_prob
    # track the layers to drop with layerdrop
    # this is needed for predictor later
    global_config.depth_features = None

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
    global_config.mixing = args.mixing
    if args.mixing == "mobilebert":
        # for mobilebert, dont use layernorm
        global_config.normalization_type = "no_norm"
        # number of ffn blocks
        global_config.num_feedforward_networks = 4
    else:
        global_config.normalization_type = "layer_norm"
        global_config.num_feedforward_networks = 1

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

    model = custom_bert.BertForMaskedLM(global_config)

    model.resize_token_embeddings(len(tokenizer))
    # logger.info(summary(model, depth=4, verbose=0))

    # maybe not required but doing it just to be sure
    model.set_sample_config(global_config)

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

    eval_dataset = tokenized_datasets["validation"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    if not args.only_latency:

        logger.info("Loading model weights from checkpoint ..")
        # we load the model before preparing
        # see this for details: https://github.com/huggingface/accelerate/issues/95
        # model.from_pretrained(args.checkpoint_dir)
        if args.checkpoint_dir is not None:
            model.load_state_dict(
                torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin"))
            )

        # Prepare everything with our `accelerator`.
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        if (
            accelerator.distributed_type == DistributedType.MULTI_GPU
            or accelerator.distributed_type == DistributedType.TPU
        ):
            # forward missing getattr and state_dict/load_state_dict to orig model
            model = ModuleProxyWrapper(model)

        logger.info("***** Running Validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")

        logger.info(f"{global_config.depth_features}")
        # change to supertransformer config
        model.set_sample_config(global_config, drop_vector=global_config.depth_features)

        eval_metric = validate_subtransformer(
            model,
            eval_dataloader,
            accelerator,
            len(eval_dataset),
            args.per_device_eval_batch_size,
            args.pad_to_max_length,
            args.evaluate_latency,
        )
        val_accuracy, val_loss, perplexity = (
            eval_metric["accuracy"] * 100,
            eval_metric["val_loss"],
            eval_metric["perplexity"],
        )
        logger.info(
            "==============================================================================================\n"
        )
        logger.info(
            f"Supertransformer stats: val_perplexity: {perplexity:.2f}, val_loss: {val_loss:.2f}, val_accuracy:  {val_accuracy:.2f}"
        )
        logger.info(
            "\n=============================================================================================="
        )
    else:
        eval_dataloader = accelerator.prepare(eval_dataloader)
        eval_batches = []
        for idx, batch in enumerate(eval_dataloader):
            eval_batches.append(batch)
            if idx == args.num_iteration:
                break

    # this is already set by set_seed function which we call above
    # but doing this again just to be sure
    random.seed(args.seed)

    if args.subtransformer_config_path is not None:
        # some dummy seeed list of len 1
        random_seeds = [args.seed]
    else:
        # sample args.num_subtransformers_eval random seeds from 1e6
        random_seeds = random.choices(np.arange(1e6), k=args.num_subtransformers_eval)

    subtransformer_peplexities = []
    subtransformer_accuracies = []
    subtransformer_losses = []
    subtransformer_configs = []

    if args.evaluate_latency or args.only_latency:
        subtransformer_latencies = []

    sampler = Sampler("random", "sandwich", args.mixing, global_config)

    for idx, _seed in enumerate(tqdm(random_seeds)):
        if args.subtransformer_config_path is not None:
            subtransformer_config = global_config
            if args.layer_drop_prob > 0:
                ## Add layerdrop function ##
                to_drop = dropout_layers(
                    subtransformer_config.sample_num_hidden_layers, args.layer_drop_prob
                )

                subtransformer_config.depth_features = to_drop

        else:
            config_dict = sampler.sample_subtransformer(
                randomize=True,
                rand_seed=_seed,
            )
            super_config_small = config_dict["smallest_subtransformer"]
            subtransformer_config = config_dict["random_subtransformers"][0]

            if args.layer_drop_prob > 0:
                ## Add layerdrop function ##
                to_drop = dropout_layers(
                    subtransformer_config.sample_num_hidden_layers, args.layer_drop_prob
                )

                subtransformer_config.depth_features = to_drop

                to_drop = dropout_layers(
                    super_config_small.sample_num_hidden_layers, args.layer_drop_prob
                )

                super_config_small.depth_features = to_drop
            if args.additional_random_softmaxing:
                random_softmaxing_idx = random.randint(0, 12)
                depth_features = [0] * (random_softmaxing_idx + 1) + [1] * (
                    subtransformer_config.sample_num_hidden_layers
                    - (random_softmaxing_idx + 1)
                )
                if (
                    depth_features
                    == [1] * subtransformer_config.sample_num_hidden_layers
                ):
                    depth_features[0] = 0
                subtransformer_config.depth_features = depth_features

        if not args.only_latency:
            model.set_sample_config(
                subtransformer_config,
                drop_vector=subtransformer_config.depth_features,
            )

            if args.evaluate_latency:
                model = model.get_active_subnet(subtransformer_config)

            eval_metric = validate_subtransformer(
                model,
                eval_dataloader,
                accelerator,
                len(eval_dataset),
                args.per_device_eval_batch_size,
                args.pad_to_max_length,
                args.evaluate_latency,
            )

            val_accuracy, val_loss, perplexity = (
                eval_metric["accuracy"] * 100,
                eval_metric["val_loss"],
                eval_metric["perplexity"],
            )
            subtransformer_peplexities.append(perplexity)
            subtransformer_accuracies.append(val_accuracy)
            subtransformer_losses.append(val_loss.item())
            if subtransformer_config.depth_features is not None:
                if torch.is_tensor(subtransformer_config.depth_features):
                    subtransformer_config.depth_features = (
                        subtransformer_config.depth_features.cpu().numpy().tolist()
                    )
                # print(subtransformer_config)
            subtransformer_configs.append(subtransformer_config)

            if args.evaluate_latency:
                subtransformer_latencies.append(eval_metric["exec_time"])

        else:  ## This is used when we want to evaluate latency alone
            # assert args.per_device_eval_batch_size == 1
            if args.subtransformer_config_path is None:
                if idx == 0:
                    subtransformer_config = super_config_small
                elif idx == 1:
                    subtransformer_config = global_config

            model = custom_bert.BertForMaskedLM.from_pretrained(
                "bert-base-cased", config=subtransformer_config
            )
            model.set_sample_config(
                subtransformer_config, drop_vector=subtransformer_config.depth_features
            )
            model = model.get_active_subnet(subtransformer_config)

            model.to(accelerator.device)
            model.eval()

            exec_time = []
            for index, batch in enumerate(eval_batches):

                if accelerator.device == "cuda":
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                else:
                    start_time = time.monotonic()

                with torch.no_grad():
                    outputs = model(**batch)

                if accelerator.device == "cuda":
                    end_time.record()
                    torch.cuda.synchronize()
                    exec_time.append(
                        start_time.elapsed_time(end_time)
                    )  ## Measured in seconds
                else:
                    exec_time.append(
                        (time.monotonic() - start_time)
                    )  ## Measured in seconds

            execution_time = mean(exec_time)
            # print(execution_time)
            subtransformer_latencies.append(execution_time)
            if subtransformer_config.depth_features is not None:
                if torch.is_tensor(subtransformer_config.depth_features):
                    subtransformer_config.depth_features = (
                        subtransformer_config.depth_features.cpu().numpy().tolist()
                    )
                # print(subtransformer_config)
            subtransformer_configs.append(subtransformer_config)

    if not args.only_latency:
        logger.info(
            f"Subtransformer average stats: val_perplexity: {mean(subtransformer_peplexities):.2f}, val_loss: {mean(subtransformer_losses):.2f}, val_accuracy:  {mean(subtransformer_accuracies):.2f}"
        )

        # no need to save stats if we are evaluating one subtransformer config
        if args.subtransformer_config_path is None:
            # dictionary of lists
            _dict = {
                "config": subtransformer_configs,
                "loss": subtransformer_losses,
                "perplexity": subtransformer_peplexities,
                "accuracy": subtransformer_accuracies,
            }

            if args.evaluate_latency:
                _dict["latency"] = subtransformer_latencies

            df = pd.DataFrame(_dict)
            if args.checkpoint_dir is not None:
                csv_path = os.path.join(args.checkpoint_dir, "subtransformer_stats.csv")
            else:
                csv_path = os.path.join(
                    f"subtransformer_{args.model_name_or_path}_perplexity_stats.csv"
                )
            df.to_csv(csv_path, index=False)
    else:
        logger.info(
            f"Subtransformer Average Latency: {mean(subtransformer_latencies):.2f}"
        )

        if args.subtransformer_config_path is None:
            _dict = {
                "config": subtransformer_configs,
                "latency": subtransformer_latencies,
            }

            df = pd.DataFrame(_dict)
            csv_path = os.path.join(
                "subtransformer_latencies_" + str(args.seed) + ".csv",
            )
            df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
