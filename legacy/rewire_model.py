import argparse
import logging
import math
import os
import random
import wandb
from copy import deepcopy
from operator import attrgetter
import time

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
from utils import rsetattr

logger = logging.getLogger(__name__)


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
        self.steps += 1

        # mean along the seq_len dimension
        # different batches have diff seq_lens and hence it ll be easy to deal with them
        # like stack etc once they are uniform
        # for idx, out in enumerate(grad_out):
        #     # print(f"grad_outputs{idx}: {out.shape}")

        # for idx, out in enumerate(grad_in):
        #     print(f"grad_inputs{idx}: {out.shape}")

        grad_out = torch.mean(torch.abs(grad_out[0]), dim=1)
        if not hasattr(module, "name"):
            setattr(module, "name", self.layer_num)
            # setattr(module, "steps", 0)
            self.grad_output[self.layer_num] = grad_out
            self.layer_num += 1
        else:
            layer_num = getattr(module, "name")
            # steps = getattr(module, "steps")
            # print(f"computing mean for layer {layer_num}")
            self.grad_output[layer_num] = torch.mean(
                torch.stack([self.grad_output[layer_num], grad_out]), dim=0
            )
            # setattr(module, "steps", steps + 1)
        if self.steps >= self.max_steps:
            # logger.info("Max steps achieved")
            layer_num = getattr(module, "name")
            grad_output = self.grad_output[layer_num]
            # logger.info(grad_output.shape)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_output = torch.mean(grad_output, dim=0)
            importance_order = torch.argsort(grad_output, descending=True)
            # print(f"steps: {self.steps}, max_steps: {self.max_steps}")
            # print(
            #     f"module {module} layer num {layer_num}, imp order {importance_order[:10]}"
            # )
            module.register_buffer("importance_order", importance_order)
            module.register_buffer(
                "inv_importance_order", inverse_permutation(importance_order)
            )
            # logger.info(importance_order)
            # logger.info()


def permute_linear(W, permutation, dim="col", permute_weight=True, permute_bias=False):
    """
    Permute linear layer

    :param W: weight matrix
    :param permutation: permutation order for the weights
    :param dim: 'row' or 'col' or 'layernorm'
    :param permute_bias: whether to permute the bias

    """
    if permute_bias:
        W.bias.data.copy_(W.bias[permutation])

    if permute_weight:
        if dim == "col":
            W.weight.data.copy_(W.weight[:, permutation])
        elif dim == "row":
            W.weight.data.copy_(W.weight[permutation, :])
        elif dim == "layernorm":
            W.weight.data.copy_(W.weight[permutation])
        else:
            raise NotImplementedError

    return W


def rewire_model(
    model,
    config,
    layerwise_importance=False,
):
    with torch.no_grad():
        num_layers = config.num_hidden_layers
        embeddings = model.bert.embeddings.word_embeddings

        assert embeddings.importance_order is not None

        weight_permutation_order = embeddings.importance_order

        _ = permute_linear(
            model.bert.encoder.layer[0].input_bottleneck,
            weight_permutation_order,
            dim="row",
            permute_weight=True,
            permute_bias=False,
        )

        for i in range(num_layers):
            if layerwise_importance:
                keys_to_permute = [
                    (f"bert.encoder.layer.{i}.input_bottleneck", "row"),
                    (f"bert.encoder.layer.{i}.attention.self.query", "col"),
                    (f"bert.encoder.layer.{i}.attention.self.key", "col"),
                    (f"bert.encoder.layer.{i}.attention.self.value", "col"),
                    (
                        f"bert.encoder.layer.{i}.attention.output.dense",
                        "row",
                    ),
                    (f"bert.encoder.layer.{i}.attention.output.LayerNorm", "layernorm"),
                    (f"bert.encoder.layer.{i}.intermediate.dense", "col"),
                    (f"bert.encoder.layer.{i}.output.dense", "row"),
                    (f"bert.encoder.layer.{i}.output.LayerNorm", "layernorm"),
                    (f"bert.encoder.layer.{i}.output_bottleneck", "col"),
                ]
                # get the weight permutation order for that layer
                weight_permutation_order = attrgetter(
                    f"bert.encoder.layer.{i}.input_bottleneck"
                )(model).importance_order
            else:
                keys_to_permute = [
                    (f"bert.encoder.layer.{i}.attention.self.query", "col"),
                    (f"bert.encoder.layer.{i}.attention.self.key", "col"),
                    (f"bert.encoder.layer.{i}.attention.self.value", "col"),
                    (
                        f"bert.encoder.layer.{i}.attention.output.dense",
                        "row",
                    ),
                    (f"bert.encoder.layer.{i}.attention.output.LayerNorm", "layernorm"),
                    (f"bert.encoder.layer.{i}.intermediate.dense", "col"),
                    (f"bert.encoder.layer.{i}.output.dense", "row"),
                    (f"bert.encoder.layer.{i}.output.LayerNorm", "layernorm"),
                ]
            for key_mode in keys_to_permute:
                key, mode = key_mode
                module = attrgetter(key)(model)

                if i == 0 and "input_bottleneck" in key:
                    continue

                if layerwise_importance and "output_bottleneck" in key:
                    if i == num_layers - 1:
                        # don't permute the last layer
                        continue
                    else:
                        # # output bottlenecks should inv permute the input so that the input order is back to normal
                        # inv_importance_order = inverse_permutation(
                        #     weight_permutation_order
                        # )
                        # print(key, inv_importance_order[:10])
                        # permute the output layer
                        new_module = permute_linear(
                            module,
                            weight_permutation_order,
                            dim=mode,
                            permute_bias=False,
                        )
                        rsetattr(model, key, new_module)
                        continue

                if mode == "row":
                    weight_permutation_order = module.importance_order
                    permute_bias = True
                elif mode == "layernorm":
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
                    module.register_buffer("importance_order", weight_permutation_order)
                    module.register_buffer(
                        "inv_importance_order",
                        inverse_permutation(weight_permutation_order),
                    )

                print(key, weight_permutation_order[:10])

                new_module = permute_linear(
                    module,
                    weight_permutation_order,
                    dim=mode,
                    permute_bias=permute_bias,
                )
                rsetattr(model, key, new_module)
        # final importance order is stored
        model.bert.register_buffer(
            "inv_importance_order", inverse_permutation(weight_permutation_order)
        )


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
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=512,
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
        "--c4_dir",
        type=str,
        default=None,
        help=f"The directory path for C4",
    )
    parser.add_argument(
        "--num_sentences_for_rewiring",
        type=int,
        default=10000,
        help=f"Number of sentences to use for calculating importance values before rewiring",
    )
    parser.add_argument(
        "--rewire_outputs",
        type=int,
        default=1,
        help=f"Whether to rewire output grads or weights",
    )
    parser.add_argument(
        "--aggregate_imp_order",
        type=int,
        default=0,
        help=f"sum all imp order",
    )
    parser.add_argument(
        "--layerwise_importance",
        type=int,
        default=0,
        help=f"sum all imp order",
    )

    args = parser.parse_args()

    args.model_name_or_path = "bert-base-cased"
    # Sanity checks

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

    if args.c4_dir is not None:
        check_path(args.c4_dir)

        args.dataset_name = "c4_realnews"

    if args.layerwise_importance:
        assert (
            args.rewire_outputs == 1
        ), "layerwise importance only works with rewire_outputs=1"

    return args


if __name__ == "__main__":

    args = parse_args()

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[param])

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

    if args.output_dir is not None:
        dataset_name = args.dataset_name.split("/")[-1].strip()
        args.output_dir += (
            "/" + "rewired_" + dataset_name + "_" + get_current_datetime()
        )
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "c4_realnews":
            logger.info("Loading C4 Dataset...")
            raw_datasets = datasets.load_from_disk(args.c4_dir)

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

    global_config = get_supertransformer_config(
        args.model_name_or_path, mixing=args.mixing
    )
    global_config.rewire = False

    global_config.max_seq_length = args.max_seq_length

    # make all dropouts zero
    global_config.hidden_dropout_prob = 0.0
    global_config.sample_hidden_dropout_prob = 0.0
    global_config.attention_probs_dropout_prob = 0.0
    global_config.sample_attention_probs_dropout_prob = 0.0

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

    if args.max_seq_length:
        global_config.max_seq_length = args.max_seq_length
    else:
        logger.warning(
            f"The max_seq_length is not defined!! Setting it to max length in tokenizer"
        )
        global_config.max_seq_length = tokenizer.model_max_length

    if args.mixing == "bert-bottleneck":
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

    else:
        raise NotImplementedError

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

    if len(train_dataset) >= args.num_sentences_for_rewiring:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset = train_dataset.select(range(args.num_sentences_for_rewiring))
    else:
        raise ValueError(
            f"train dataset has less than {args.num_sentences_for_rewiring} sentences "
        )

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
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
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

    logger.info("validating perplexity before rewiring")
    eval_metric = validate_subtransformer(
        model,
        eval_dataloader,
        accelerator,
        len(eval_dataset),
        args.per_device_eval_batch_size,
        args.pad_to_max_length,
    )
    perplexity_before_rewiring = eval_metric["perplexity"]

    model.train()

    if args.rewire_outputs:
        # backhook runs for 2 modules every step. Thats why we run it for global_config.num_hidden_layers * 2
        # len(train_dataloader) - 1 is the last training step
        max_steps = (len(train_dataloader) - 1) * global_config.num_hidden_layers * 2
        # print(f"max_steps: {max_steps}")
        bhookfn = BackHook(max_steps=max_steps)
        model.bert.embeddings.word_embeddings.register_backward_hook(bhookfn)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # The following keys need to be backhooked:
                # "bert.encoder.layer.*.attention.output.dense",
                # "bert.encoder.layer.*.output.dense",
                if "output.dense" in name:
                    module.register_backward_hook(bhookfn)

    logger.info(f"Calculating gradients with hooks: ")
    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

    if args.layerwise_importance and args.rewire_outputs:
        emb_importance_order = model.bert.embeddings.word_embeddings.importance_order

        for i in range(global_config.num_hidden_layers):
            keys = [
                (f"bert.encoder.layer.{i}.input_bottleneck", "row"),
                (f"bert.encoder.layer.{i}.attention.self.query", "col"),
                (f"bert.encoder.layer.{i}.attention.self.key", "col"),
                (f"bert.encoder.layer.{i}.attention.self.value", "col"),
                (
                    f"bert.encoder.layer.{i}.attention.output.dense",
                    "row",
                ),
                (f"bert.encoder.layer.{i}.attention.output.LayerNorm", "layernorm"),
                (f"bert.encoder.layer.{i}.intermediate.dense", "col"),
                (f"bert.encoder.layer.{i}.output.dense", "row"),
                (f"bert.encoder.layer.{i}.output.LayerNorm", "layernorm"),
                (f"bert.encoder.layer.{i}.output_bottleneck", "col"),
            ]

            if i == 0:
                importance_order = emb_importance_order
            else:
                importance_order = attrgetter(f"bert.encoder.layer.{i}.output.dense")(
                    model
                ).importance_order

            inv_importance_order = inverse_permutation(importance_order)

            for key_mode in keys:
                key, mode = key_mode
                module = attrgetter(key)(model)
                if "input_bottleneck" in key:
                    module.register_buffer("importance_order", importance_order)
                    module.register_buffer("inv_importance_order", inv_importance_order)
                elif mode == "row":
                    module.importance_order = importance_order
                    module.inv_importance_order = inv_importance_order

    # weight rewiring
    if not args.rewire_outputs:

        imp_orders = []

        word_embeddings = model.bert.embeddings.word_embeddings
        grad_embeddings = word_embeddings.weight.grad
        grad_output = torch.abs(grad_embeddings * word_embeddings.weight).mean(dim=0)
        imp_order = grad_output.sort(descending=True)[1]
        if not args.aggregate_imp_order:
            word_embeddings.register_buffer("importance_order", imp_order)
            word_embeddings.register_buffer("grad_output", grad_output)
            word_embeddings.register_buffer(
                "inv_importance_order", inverse_permutation(imp_order)
            )
        else:
            imp_orders.append(grad_output)

        for i in range(global_config.num_hidden_layers):

            keys = [
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
            for key_mode in keys:
                key, mode = key_mode
                module = attrgetter(key)(model)
                grad = module.weight.grad
                if mode == "row":
                    grad_output = torch.abs(grad * module.weight).mean(dim=1)
                    imp_order = grad_output.sort(descending=True)[1]
                    if not args.aggregate_imp_order:
                        module.register_buffer("importance_order", imp_order)
                        module.register_buffer("grad_output", grad_output)
                        module.register_buffer(
                            "inv_importance_order", inverse_permutation(imp_order)
                        )
                    else:
                        imp_orders.append(grad_output)

                elif mode == "col" and args.aggregate_imp_order:
                    grad_output = torch.abs(grad * module.weight).mean(dim=0)
                    imp_orders.append(grad_output)

        if args.aggregate_imp_order:
            imp_order = torch.stack(imp_orders, dim=0)
            imp_order = imp_order.sum(dim=0).sort(descending=True)[1]
            word_embeddings.register_buffer("importance_order", imp_order)
            word_embeddings.register_buffer(
                "inv_importance_order", inverse_permutation(imp_order)
            )
            for i in range(global_config.num_hidden_layers):

                keys = [
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
                for key_mode in keys:
                    key, mode = key_mode
                    module = attrgetter(key)(model)
                    if mode == "row":
                        module.register_buffer("importance_order", imp_order)
                        module.register_buffer(
                            "inv_importance_order", inverse_permutation(imp_order)
                        )

    logger.info("Rewiring model: ")
    model(**batch)
    rewire_model(model, global_config, args.layerwise_importance)

    global_config.rewire = True
    # to set sample_hidden_size
    model.set_sample_config(global_config)
    model.bert.encoder.layer[0].print_outputs = True
    model(**batch)

    model.eval()
    eval_metric = validate_subtransformer(
        model,
        eval_dataloader,
        accelerator,
        len(eval_dataset),
        args.per_device_eval_batch_size,
        args.pad_to_max_length,
    )
    perplexity_after_rewiring = eval_metric["perplexity"]

    logger.info(f"Perplexity before rewiring : {perplexity_before_rewiring}")
    logger.info(f"Perplexity after rewiring : {perplexity_after_rewiring}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            save_function=accelerator.save,
        )
        logger.info(f"Saved Rewired model to {args.output_dir}")
