import argparse
import os
import random
from pprint import pprint

import datasets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import wandb
from custom_layers.custom_bert import BertForSequenceClassification
from custom_layers.custom_bert import BertForMaskedLM
from tasks.glue.prepare_task import GlueTask
from utils.module_proxy_wrapper import ModuleProxyWrapper
from utils.wipe_memory import free_memory, get_gpu_memory
from utils.early_stopping import EarlyStopping
from utils import count_parameters
import copy

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

# SUPPORTED_MODELS = ['bert-base-cased', 'bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased', 'bert-large-cased']


def get_task(task_name):
    if task_name in GLUE_TASKS:
        return GlueTask


def calc_probs(choices_dictionary):
    normalized_probs = {}
    for choice, v in choices_dictionary.items():
        _v = []
        _sum = sum(v)
        for i in v:
            _v.append(i / _sum)
        normalized_probs[choice] = _v
    return normalized_probs


def show_random_elements(dataset, accelerator, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    accelerator.print(df)


def get_supertransformer_config(
    model_name_or_path="bert-base-cased", tiny_attn=False, mixing="attention"
):
    config = AutoConfig.from_pretrained(model_name_or_path)

    if mixing == "gmlp":
        # gmlp needs twice the encoder layers to match bert param size
        config.num_hidden_layers = 36
        config.hidden_size = 512

    config.sample_hidden_size = config.hidden_size
    config.sample_num_hidden_layers = config.num_hidden_layers

    if not tiny_attn:
        config.sample_num_attention_heads = [
            config.num_attention_heads
        ] * config.sample_num_hidden_layers
    else:
        config.sample_num_attention_heads = [1] * config.sample_num_hidden_layers

    config.num_attention_heads = 1 if tiny_attn else config.num_attention_heads

    config.sample_intermediate_size = [
        config.intermediate_size
    ] * config.sample_num_hidden_layers

    if mixing == "mobilebert":
        config.normalization_type = "no_norm"
        config.num_feedforward_networks = 4

        config.sample_embedding_size = config.embedding_size
        config.sample_num_hidden_layers = config.num_hidden_layers
        config.sample_intra_bottleneck_size = config.intra_bottleneck_size
        config.sample_true_hidden_size = config.true_hidden_size
    else:
        config.embedding_size = 768
        config.intra_bottleneck_size = 768
        config.true_hidden_size = 768
        config.normalization_type = "no_norm"
        config.num_feedforward_networks = 4

        config.use_bottleneck = True
        config.use_bottleneck_attention = True
        config.key_query_shared_bottleneck = False

        config.sample_embedding_size = config.embedding_size
        config.sample_num_hidden_layers = config.num_hidden_layers
        config.sample_intra_bottleneck_size = config.intra_bottleneck_size
        config.sample_true_hidden_size = config.true_hidden_size

    config.mixing = mixing
    config.tiny_attn = tiny_attn
    return config


def show_args(accelerator, args):
    accelerator.print(
        f"Free gpu Memory ( in MBs) on each gpus before starting training: {get_gpu_memory()}"
    )
    accelerator.print(
        "==================================================================="
    )
    accelerator.print("Training Arguments:")
    for arg in vars(args):
        accelerator.print(f"{arg}: {getattr(args, arg)}")
    accelerator.print(
        "==================================================================="
    )


def get_choices(num_hidden_layers=12, mixing="attention"):
    choices = {
        "sample_hidden_size": [360, 480, 540, 600, 768],
        "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
        "sample_intermediate_size": [512, 1024, 2048, 3072],
        "sample_num_hidden_layers": list(range(6, num_hidden_layers, 2))
        + [num_hidden_layers],
    }
    choices["sample_hidden_size"] = (
        [120, 240, 360, 480, 512] if mixing == "gmlp" else choices["sample_hidden_size"]
    )
    if mixing == "mobilebert":
        choices["sample_hidden_size"] = [768]
        choices["sample_intra_bottleneck_size"] = [360, 480, 540, 600, 768]

    return choices


def get_diverse_subtransformers(elastic_variable, config):
    diverse_configs = []
    all_choices = get_choices(config.num_hidden_layers, config.mixing)

    num_hidden_layers = int(config.num_hidden_layers)

    elastic_variable_choices = all_choices[elastic_variable]

    diverse_config = copy.deepcopy(config)
    choices1_keys = ["sample_hidden_size", "sample_num_hidden_layers"]
    choices2_keys = ["sample_num_attention_heads", "sample_intermediate_size"]

    for key in choices1_keys:
        if key == elastic_variable:
            continue
        value = max(all_choices[key])
        setattr(diverse_config, key, value)

    for key in choices2_keys:
        if key == elastic_variable:
            continue
        value = [max(all_choices[key])] * num_hidden_layers
        setattr(diverse_config, key, value)

    for choice in elastic_variable_choices:
        if elastic_variable in choices1_keys:
            value = choice
            setattr(diverse_config, elastic_variable, value)
        else:
            value = [choice] * num_hidden_layers
            setattr(diverse_config, elastic_variable, value)

        if (
            getattr(diverse_config, "sample_hidden_size")
            % getattr(diverse_config, "sample_num_attention_heads")[0]
        ):
            continue
        diverse_configs.append(copy.deepcopy(diverse_config))

    def sorter(x):
        value = getattr(x, elastic_variable)
        if isinstance(value, list):
            return value[0]
        else:
            return value

    diverse_configs = sorted(diverse_configs, key=sorter)

    return diverse_configs


def random_sampling(config, tiny_attn=False):
    choices = get_choices(config.num_hidden_layers, mixing=config.mixing)

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

    if tiny_attn:
        setattr(config, "sample_num_attention_heads", 1)

    return config


def naive_params_sampling(config, tiny_attn=False, population_size=30):
    choices = get_choices(config.num_hidden_layers, mixing=config.mixing)

    max_params = 0
    best_config = None

    assert population_size > 0

    ## We can replace this with a simple mathematical function to compute params given a config and a maximizing
    ## function to give precedence for params! That might be faster
    ## For now implemented as using the best params from a randomly sampled population!

    for i in range(population_size):
        config = random_sampling(config, tiny_attn)
        model = BertForMaskedLM(config)
        params = count_parameters(model)

        if max_params < params:
            max_params = params
            best_config = config

    return best_config


def biased_params_sampling(config, tiny_attn=False):
    choices = get_choices(config.num_hidden_layers, mixing=config.mixing)
    normalized_probs = calc_probs(choices)

    ### Figuring the number of hidden layers
    hidden_layers_list = choices["sample_num_hidden_layers"]
    num_hidden_layers = random.choices(
        hidden_layers_list, k=1, weights=normalized_probs["sample_num_hidden_layers"]
    )[0]
    setattr(config, "sample_num_hidden_layers", num_hidden_layers)

    ## Figuring the hidden size for BERT embeddings
    hidden_size_embeddings_list = choices["sample_hidden_size"]
    num_hidden_size = random.choices(
        hidden_size_embeddings_list, k=1, weights=normalized_probs["sample_hidden_size"]
    )[0]
    setattr(config, "sample_hidden_size", num_hidden_size)

    config_dict = {
        "sample_num_attention_heads": [],
        "sample_intermediate_size": [],
    }

    for i in range(num_hidden_layers):
        while True:
            for key in config_dict.keys():

                choice_list = choices[key]
                choice = random.choices(
                    choice_list, k=1, weights=normalized_probs[key]
                )[0]
                config_dict[key].append(choice)

            if config.sample_hidden_size % config_dict["sample_num_attention_heads"][i]:
                for key in config_dict.keys():
                    config_dict[key] = config_dict[key][:-1]
                continue
            else:
                break

    for key in config_dict.keys():
        setattr(config, key, config_dict[key])

    if tiny_attn:
        setattr(config, "sample_num_attention_heads", 1)

    return config, None


def get_small_config(config):
    choices = get_choices(config.num_hidden_layers, mixing=config.mixing)

    hidden_layers_list = choices["sample_num_hidden_layers"]
    hidden_size_embeddings_list = choices["sample_hidden_size"]

    ## Choosing the small network
    num_hidden_layers = hidden_layers_list[0]
    setattr(config, "sample_num_hidden_layers", num_hidden_layers)
    hidden_size = hidden_size_embeddings_list[0]
    setattr(config, "sample_hidden_size", hidden_size)

    config_dict = {
        "sample_num_attention_heads": [],
        "sample_intermediate_size": [],
    }

    for i in range(num_hidden_layers):
        while True:
            for key in config_dict.keys():

                choice_list = choices[key]
                choice = choice_list[0]
                config_dict[key].append(choice)

            if config.sample_hidden_size % config_dict["sample_num_attention_heads"][i]:
                for key in config_dict.keys():
                    config_dict[key] = config_dict[key][:-1]
                continue
            else:
                break

    return config


## Population size will be implemented later
def sandwich_sampling(config, tiny_attn=False, pop_size=1):
    small_config = get_small_config(config)
    random_config, _ = biased_params_sampling(config, tiny_attn)

    return random_config, small_config


def sample_subtransformer(
    randomize=True, rand_seed=0, tiny_attn=False, config=None, sampling_type="random"
):
    if randomize:
        random.seed(rand_seed)
    if config is None:
        config = get_supertransformer_config(mixing=config.mixing)
    config = copy.deepcopy(config)

    config_big = None
    config_small = None

    if sampling_type == "random":
        config = random_sampling(config, tiny_attn)
    elif sampling_type == "naive_params":
        config = naive_params_sampling(config, tiny_attn)
    elif sampling_type == "sandwich":
        config, config_small = sandwich_sampling(config, False, 1)
        assert config_small is not None
    elif sampling_type == "biased_params":
        config, config_small = biased_params_sampling(config, tiny_attn=tiny_attn)
    else:
        raise NotImplementedError

    return config, config_small


def validate_subtransformer(
    model,
    eval_dataloader,
    accelerator,
    metric,
    task="mrpc",
):
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        if task == "stsb":
            predictions = predictions[:, 0]
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    return eval_metric


def train_transformer_one_epoch(
    model,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_dataloader,
    accelerator,
    train_subtransformer=False,
):
    optimizer.zero_grad()

    model.train()
    seed = -1
    for step, batch in enumerate(
        tqdm(train_dataloader, disable=not accelerator.is_local_main_process),
    ):
        if not train_subtransformer:
            # if we are training a supertransformer, then we need to change the
            # seed in each step
            seed += 1
            super_config = sample_subtransformer(
                limit_subtransformer_choices=False, randomize=True, rand_seed=seed
            )
            model.set_sample_config(super_config)

        batch.to(accelerator.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            # print(super_config)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if accelerator.is_main_process:
            wandb.log({"random-subtransformer-loss": loss.item(), "rand-seed": seed})


class Engine:
    def __init__(self, args):

        self.args = args
        if hasattr(args, "fp16") and hasattr(args, "cpu"):
            self.accelerator = self.setup_accelerator(
                fp16=self.args.fp16, cpu=self.args.cpu
            )
            show_args(self.accelerator, args)

        if hasattr(args, "task"):
            if args.task in GLUE_TASKS:
                self.num_labels = (
                    3
                    if args.task.startswith("mnli")
                    else 1
                    if args.task == "stsb"
                    else 2
                )
            if (
                hasattr(args, "model_name_or_path")
                and hasattr(args, "max_seq_length")
                and hasattr(args, "use_pretrained_supertransformer")
            ):

                self.model, self.tokenizer = self.setup_model_and_tokenizer(
                    args.model_name_or_path,
                    num_labels=self.num_labels,
                    initialize_pretrained_model=args.use_pretrained_supertransformer,
                )

                self.task = self.setup_task(
                    args.task, self.tokenizer, args.max_seq_length
                )
                self.metric = self.task.metric
                self.train_dataloader = self.setup_dataloader(
                    self.task.train_dataset,
                    args.per_gpu_train_batch_size,
                    self.tokenizer,
                    args.max_seq_length,
                    shuffle_dataset=True,
                )
                self.eval_dataloader = self.setup_dataloader(
                    self.task.eval_dataset,
                    args.per_gpu_eval_batch_size,
                    self.tokenizer,
                    args.max_seq_length,
                    shuffle_dataset=False,
                )

                num_training_steps = len(self.train_dataloader) * args.num_epochs

                self.optimizer, self.lr_scheduler = self.setup_optimizer_and_scheduler(
                    self.model.parameters(),
                    args.learning_rate,
                    num_training_steps,
                )

        if hasattr(args, "wandb_run_name"):
            # project and entity set to defaults
            self.setup_wandb(name=args.wandb_run_name)

    def setup_accelerator(self, fp16=True, cpu=False):
        param = DistributedDataParallelKwargs(
            find_unused_parameters=True, check_reduction=False
        )
        return Accelerator(fp16=fp16, cpu=cpu, kwargs_handlers=[param])

    def setup_wandb(self, project="eHAT-warmups", entity="efficient-hat", name=None):
        wandb.init(project=project, entity=entity, name=name)
        # if accelerator and model are already defined, track it with wandb
        if hasattr(self, "accelerator") and hasattr(self, "model"):
            if self.accelerator.is_main_process:
                wandb.watch(self.model)

    def setup_dataloader(
        self, dataset, batch_size, tokenizer, max_seq_length, shuffle_dataset=True
    ):
        accelerator = self.accelerator

        def collate_fn(examples):
            # On TPU it's best to pad everything to the same length or training will be very slow.
            if accelerator.distributed_type == DistributedType.TPU:
                return tokenizer.pad(
                    examples,
                    padding="max_length",
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )
        return self.accelerator.prepare(dataloader)

    def setup_model_and_tokenizer(
        self,
        model_name_or_path,
        num_labels=None,
        initialize_pretrained_model=True,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if num_labels is None and self.num_labels is not None:
            num_labels = self.num_labels
        elif num_labels is not None:
            self.num_labels = num_labels
        else:
            raise ValueError(
                "num_labels cant be none, it is not defined in Engine.__init__ nor passed to setup_model_and_tokenizer"
            )
        if initialize_pretrained_model or os.path.exists(model_name_or_path):
            model = BertForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=self.num_labels
            )
        else:
            model_config = get_supertransformer_config()
            model_config.num_labels = num_labels
            model = BertForSequenceClassification(config=model_config)

        # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
        # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
        model = model.to(self.accelerator.device)

        model = self.accelerator.prepare(model)
        if (
            self.accelerator.distributed_type == DistributedType.MULTI_GPU
            or self.accelerator.distributed_type == DistributedType.TPU
        ):
            # forward missing getattr and state_dict/load_state_dict to orig model
            model = ModuleProxyWrapper(model)
            # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
            # may change its length.

        return model, tokenizer

    def setup_optimizer_and_scheduler(
        self,
        model_parameters,
        lr,
        num_training_steps,
        num_warmup_steps=100,
        correct_bias=True,
    ):

        optimizer = AdamW(params=model_parameters, lr=lr, correct_bias=correct_bias)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        optimizer = self.accelerator.prepare(optimizer)
        return optimizer, lr_scheduler

    def setup_task(self, task_name, tokenizer, max_seq_len=128):
        task_fn = get_task(task_name)
        task = task_fn(task_name, tokenizer, max_seq_len)
        return task

    def train(self, train_subtransformer=False, subtransformer_config=None):
        if train_subtransformer:
            assert (
                subtransformer_config is not None
            ), "Config cant be None when training subtransformer"
            assert (
                self.args.eval_random_subtransformers is False
            ), "We can only evaluate random subtransformers during supertransformer training"
            self.model.set_sample_config(subtransformer_config)

            metric_to_track = "subtransformer_accuracy"
        else:
            metric_to_track = "supertransformer_accuracy"

        early_stopping = EarlyStopping(
            metric_to_track, patience=self.args.early_stopping_patience
        )
        for epoch in range(self.args.num_epochs):

            train_transformer_one_epoch(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.args.gradient_accumulation_steps,
                self.train_dataloader,
                self.accelerator,
                train_subtransformer=train_subtransformer,  # first we will train the supertransformer
            )

            self.accelerator.print(f"Epoch {epoch + 1}:", end=" ")
            if self.accelerator.is_main_process:
                wandb.log({"epochs": epoch})

            if not train_subtransformer:
                # resetting to supertransformer before validation
                config = get_supertransformer_config()
                self.model.set_sample_config(config)

            eval_metric = validate_subtransformer(
                self.model,
                self.eval_dataloader,
                self.accelerator,
                self.metric,
                self.task,
            )

            _dict = {}
            for key in eval_metric:
                if train_subtransformer:
                    _key = "subtransformer_" + key
                else:
                    _key = "supertransformer_" + key
                _dict[_key] = eval_metric[key]

            self.accelerator.print(_dict)
            if self.accelerator.is_main_process:
                wandb.log(_dict)

            if self.args.eval_random_subtransformers:
                label_seed = []
                label_acc = []
                hover_templates = []
                sampling_dimensions = [
                    "sample_hidden_size",
                    "sample_num_attention_heads",
                    "sample_intermediate_size",
                    "sample_num_hidden_layers",
                ]
                # Sampling 25 random sub-transformers and evaluate them to understand the relative performance order
                for i in range(25):
                    random_seed = i * 1000
                    config = sample_subtransformer(
                        limit_subtransformer_choices=False,
                        randomize=True,
                        rand_seed=random_seed,
                    )
                    eval_metric = validate_subtransformer(
                        self.model,
                        self.eval_dataloader,
                        self.accelerator,
                        self.metric,
                        self.task,
                    )

                    hover_templates.append(
                        "<br>".join(
                            [
                                f"{key}: {getattr(config, key)}"
                                for key in sampling_dimensions
                            ]
                        )
                    )

                    label_acc.append(eval_metric["accuracy"])
                    label_seed.append(random_seed)

                if self.accelerator.is_main_process:
                    ## If plotting using Custom Plotly
                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(x=label_seed, y=label_acc, hovertext=hover_templates)
                    )
                    fig.update_layout(
                        title="Relative Performance Order",
                        xaxis_title="Random Seed",
                        yaxis_title="Accuracy",
                    )
                    wandb.log({"bar_chart": wandb.data_types.Plotly(fig)})

            if self.accelerator.is_main_process:
                early_stopping(_dict)
                # if counter is 0, it means the metric has improved
                if early_stopping.counter == 0:
                    self.save_model(self.args.output_dir)

                if early_stopping.early_stop:
                    self.accelerator.print(
                        f"Early Stopping !!! {metric_to_track} hasnt improved for {self.args.early_stopping_patience} epochs"
                    )
                    return

    def save_model(self, save_dir):
        # How to save your 🤗 Transformer?
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_dir,
            save_function=self.accelerator.save,
            state_dict=self.model.state_dict(),
        )

    def load_model(self, save_dir):
        if hasattr(self, "model") or hasattr(self, "optimizer"):
            del self.train_dataloader
            del self.eval_dataloader
            # free up memory before loading a new model
            free_memory(self.accelerator, self.model, self.optimizer)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.from_pretrained(save_dir)


def main():
    parser = argparse.ArgumentParser(description="Script to train efficient HAT models")
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
        type=int,
        default=1,
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
        "--early_stopping_patience",
        default=5,
        type=int,
        help="Patience for early stopping to stop training if val_acc doesnt converge",
    )
    parser.add_argument(
        "--limit_subtransformer_choices",
        default=0,
        type=int,
        help="If set to 1, it will limit the hidden_size and number of encoder layers of the subtransformer choices",
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
        default=5,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--fp16", type=int, default=1, help="If set to 1, will use FP16 training."
    )
    parser.add_argument(
        "--cpu", type=int, default=0, help="If set to 1, will train on the CPU."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.task)
    # if the mentioned output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.wandb_run_name = f"{args.task}_supertransformer_training"
    engine = Engine(args)
    engine.train()
    model = engine.load_model(args.output_dir)


if __name__ == "__main__":
    main()
