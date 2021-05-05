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
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig,
)
from custom_bert import BertForSequenceClassification
import random
import os
from prepare_task import GlueTask
from module_proxy_wrapper import ModuleProxyWrapper

from pprint import pprint


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64


def get_supertransformer_config():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.sample_hidden_size = config.hidden_size
    config.sample_num_hidden_layers = config.num_hidden_layers
    config.sample_num_attention_heads = [
        config.num_attention_heads
    ] * config.sample_num_hidden_layers
    config.sample_intermediate_size = [
        config.intermediate_size
    ] * config.sample_num_hidden_layers

    return config


def get_choices():
    # choices = {
    #     "sample_hidden_size": [360, 480, 540, 600, 768],
    #     "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
    #     "sample_intermediate_size": [512, 1024, 2048, 3072],
    #     "sample_num_hidden_layers": [6, 8, 10, 12],
    # }
    choices = {
        "sample_hidden_size": [768],
        "sample_num_attention_heads": [12],
        "sample_intermediate_size": [3072],
        "sample_num_hidden_layers": [12],
    }
    return choices


def print_subtransformer_config(config):
    print("===========================================================")
    print("hidden size: ", config.sample_hidden_size)
    print("num attention heads: ", config.sample_num_attention_heads)
    print("intermediate sizes: ", config.sample_intermediate_size)
    print("num hidden layers: ", config.sample_num_hidden_layers)
    print("===========================================================")


def sample_subtransformer(randomize=False, rand_seed=0):
    if randomize:
        random.seed(rand_seed)
    choices = get_choices()
    config = get_supertransformer_config()

    ### Figuring the number of hidden layers
    hidden_layers_list = choices["sample_num_hidden_layers"]
    num_hidden_layers = random.choice(hidden_layers_list)
    setattr(config, "sample_num_hidden_layers", num_hidden_layers)

    ## Figuring the hidden size for BERT embeddings
    hidden_size_embeddings_list = choices["sample_hidden_size"]
    num_hidden_size = random.choice(hidden_size_embeddings_list)
    setattr(config, "sample_hidden_size", num_hidden_size)

    config_dict = {
        "sample_num_attention_heads": [],
        "sample_intermediate_size": [],
    }

    for i in range(num_hidden_layers):
        while True:
            for key in config_dict.keys():

                choice_list = choices[key]
                choice = random.choice(choice_list)
                config_dict[key].append(choice)

            if config.sample_hidden_size % config_dict["sample_num_attention_heads"][i]:
                for key in config_dict.keys():
                    config_dict[key] = config_dict[key][:-1]
                continue
            else:
                break

    for key in config_dict.keys():
        setattr(config, key, config_dict[key])

    return config


def validate_subtransformer(model, config, eval_dataloader, accelerator, metric):
    model.set_sample_config(config=config)
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    # Use accelerator.print to print only on the main process.
    # accelerator.print(eval_metric)
    return eval_metric


def training_function(args):

    print("===================================================================")
    print("Training Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===================================================================")
    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    print("Running on: ", accelerator.device)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = args.learning_rate
    num_epochs = int(args.num_epochs)
    # for now correcting adam bias is hardcoded to True
    correct_bias = True
    seed = int(args.seed)

    # note your effective batch size while training would be:
    # per_gpu_train_batch_size * gradient_accumulation_steps * num_gpus
    # so for instance, if you train with:
    # per_gpu_train_batch_size = 32
    # gradient_accumulation_steps = 4
    # num_gpus (number of gpus) = 2
    # then your effective training batch size is (32 * 4 * 2) = 256
    per_gpu_train_batch_size = int(args.per_gpu_train_batch_size)
    per_gpu_eval_batch_size = int(args.per_gpu_eval_batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)

    set_seed(seed)
    config = get_supertransformer_config()
    # print(f"Batch Size: {batch_size}")
    task = args.task
    use_pretained = args.use_pretrained_supertransformer
    model_checkpoint = args.model_name_or_path

    # if this is not a path to a saved model checkpoint, ensure that we are using a bert-base model
    if not os.path.exists(model_checkpoint):
        assert (
            model_checkpoint == "bert-base-cased"
            or model_checkpoint == "bert-base-uncased"
        ), f"HF model {model_checkpoint} is not supported, pls use bert-base"

    glue_task = GlueTask(
        task, model_checkpoint, config, initialize_pretrained_model=use_pretained
    )

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            return glue_task.tokenizer.pad(
                examples,
                padding="max_length",
                max_length=args.max_seq_length,
                return_tensors="pt",
            )
        return glue_task.tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        glue_task.train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=per_gpu_train_batch_size,
    )
    eval_dataloader = DataLoader(
        glue_task.eval_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=per_gpu_eval_batch_size,
    )

    model = glue_task.model
    metric = glue_task.metric

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    if (
        accelerator.distributed_type == DistributedType.MULTI_GPU
        or accelerator.distributed_type == DistributedType.TPU
    ):
        # forward missing getattr and state_dict/load_state_dict to orig model
        model = ModuleProxyWrapper(model)

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):

            random_number = step + int(10000 * time.perf_counter())
            super_config = sample_subtransformer(
                randomize=True, rand_seed=random_number
            )
            try:
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                model.set_sample_config(super_config)
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            except RuntimeError as e:
                print(e)
                print_subtransformer_config(config)
                print("please recheck the above config")
        print(f"Epoch {epoch + 1}:", end=" ")
        # resetting to supertransformer before validation
        config = get_supertransformer_config()
        eval_metric = validate_subtransformer(
            model, config, eval_dataloader, accelerator, metric
        )
        accelerator.print(eval_metric)

        model.save_pretrained(args.output_dir)
        # sub_transformer_configs_metrics = []
        # for i in range(10):
        #     config = sample_subtransformer(randomize=True, rand_seed=i)
        #     print_subtransformer_config(config)
        #     metrics = validate_subtransformer(
        #         model, config, eval_dataloader, accelerator, metric
        #     )
        #     print(metrics)
        #     sub_transformer_configs_metrics.append((metrics["accuracy"], config, i))

        # for acc, f1, config, idx in sorted(
        #     sub_transformer_configs_metrics, key=lambda tup: tup[1]
        # ):

        #     print(f"{acc:.2f} ({idx})", end=", ")
        #     # print_subtransformer_config(config)
        #     # print(f1)
        # print()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--task", type=str, default="mrpc", help="The Glue task you want to run"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to model checkpoint or name of hf pretrained model",
    )
    parser.add_argument(
        "--use_pretrained_supertransformer",
        type=bool,
        default=True,
        help="If passed and set to True, will use pretrained bert-uncased model. If set to False, it will initialize a random model and train from scratch",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints",
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_epochs",
        default=5.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--fp16", type=bool, default=True, help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--cpu", type=bool, default=False, help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()
    # if the mentioned output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    training_function(args)


if __name__ == "__main__":
    main()
