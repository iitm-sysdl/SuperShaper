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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig,
)
from custom_bert import BertForSequenceClassification
import random


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


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def add_sampling_params():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.sample_hidden_size = config.hidden_size
    config.sample_num_attention_heads = config.num_attention_heads
    config.sample_intermediate_size = config.intermediate_size
    config.sample_num_hidden_layers = config.num_hidden_layers
    return config


def get_supertransformer_config():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.sample_hidden_size = config.hidden_size
    config.sample_num_attention_heads = config.num_attention_heads
    config.sample_intermediate_size = config.intermediate_size
    config.sample_num_hidden_layers = config.num_hidden_layers
    return config


def get_choices():
    choices = {
        "sample_hidden_size": [360, 480, 540, 600, 768],
        "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
        "sample_intermediate_size": [512, 1024, 2048, 3072],
        # "sample_num_hidden_layers": [4, 6, 8, 10, 12],
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
    for key in choices.keys():
        choice_list = choices[key]
        choice = random.choice(choice_list)
        setattr(config, key, choice)
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
    accelerator.print(eval_metric)


def training_function(config, args):
    print(args)
    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    print(accelerator.device)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    correct_bias = config["correct_bias"]
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    metric = load_metric("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets.rename_column_("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(
                examples, padding="max_length", max_length=128, return_tensors="pt"
            )
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
    )

    set_seed(seed)
    config = get_supertransformer_config()
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    # uncomment this if you want to start the supertransformer from pretrained
    # bert weights
    # model = BertForSequenceClassification.from_pretrained("bert-base-cased")
    # use this from randomly initialized supertransformer
    model = BertForSequenceClassification(config=config)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # config.sample_hidden_size = 756  # 768 -> 756
    # config.sample_intermediate_size = 3000  # 3072 -> 756
    # config.sample_num_hidden_layers = 6  # 12 -> 6
    # config.sample_num_attention_heads = 6
    # model.set_sample_config(config)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

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
            while True:
                random_number = step + int(time.perf_counter())
                config = sample_subtransformer(randomize=True, rand_seed=random_number)
                # make sure hidden_size is divisible by number of heads
                if config.sample_hidden_size % config.sample_num_attention_heads:
                    # print_subtransformer_config(config)
                    # print(
                    #     f"Error with config generated with random number {random_number}.... regenerating"
                    # )
                    continue
                else:
                    break
            try:
                # We could avoid this line since we set the accelerator with `device_placement=True`.

                model.set_sample_config(config)
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
        validate_subtransformer(model, config, eval_dataloader, accelerator, metric)

        model.save_pretrained("checkpoints")

    for i in range(5):
        config = sample_subtransformer(randomize=True, rand_seed=i)
        print_subtransformer_config(config)
        validate_subtransformer(model, config, eval_dataloader, accelerator, metric)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--fp16", type=bool, default=False, help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--cpu", type=bool, default=False, help="If passed, will train on the CPU."
    )
    args = parser.parse_args()
    config = {
        "lr": 2e-5,
        "num_epochs": 10,
        "correct_bias": True,
        "seed": 42,
        "batch_size": 16,
    }
    training_function(config, args)


if __name__ == "__main__":
    main()