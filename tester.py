import random
from tqdm import tqdm
from hflat import LatencyPredictor
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tasks.glue.prepare_task import GlueTask
import argparse
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
from custom_layers.custom_bert import BertForSequenceClassification
import os
from pprint import pprint
import pandas as pd
import time
import psutil
    

class Tester:
    # def init(self, ckpt_path, task):
    #     pass
    def __init__(
        self,
        ckpt_path=None,
        task="sst2",
        model_name_or_path="bert-base-uncased",
        use_pretrained_supertransformer=False,
        max_seq_length=128,
        per_gpu_eval_batch_size=64,
        fp16=True,
        cpu=False,
        seed=42,
        accel=None
    ):

        # Initialize accelerator
        self.cpu = cpu
        self.accel = accel
        accelerator = accel
        if accel is None:
            accelerator = Accelerator(fp16=fp16, cpu=cpu)
            self.accel = accelerator

        print("Running on: ", accelerator.device)

        # for now correcting adam bias is hardcoded to True
        correct_bias = True
        seed = int(seed)
        per_gpu_eval_batch_size = int(per_gpu_eval_batch_size)

        set_seed(seed)
        config = self.get_supertransformer_config()
        use_pretained = use_pretrained_supertransformer
        model_checkpoint = model_name_or_path

        # if this is not a path to a saved model checkpoint, ensure that we are using a bert-base model
        if not os.path.exists(model_checkpoint):
            assert (
                model_checkpoint == "bert-base-cased"
                or model_checkpoint == "bert-base-uncased"
            ), f"HF model {model_checkpoint} is not supported, pls use bert-base"

        glue_task = GlueTask(
            task,
            model_checkpoint,
            config,
            max_seq_length,
            accelerator,
            initialize_pretrained_model=use_pretained,
        )

        def collate_fn(examples):
            # On TPU it's best to pad everything to the same length or training will be very slow.
            if accelerator.distributed_type == DistributedType.TPU:
                return glue_task.tokenizer.pad(
                    examples,
                    padding="max_length",
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            return glue_task.tokenizer.pad(
                examples, padding="longest", return_tensors="pt"
            )

        # Instantiate dataloader
        eval_dataloader = None
        if task == 'qqp' or task == 'qnli':
            eval_dataloader = DataLoader(
                glue_task.eval_dataset.shuffle(seed=seed).select(range(1000)),
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=per_gpu_eval_batch_size,
            )
        else:
            eval_dataloader = DataLoader(
                glue_task.eval_dataset,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=per_gpu_eval_batch_size,
            )

        model = glue_task.model
        metric = glue_task.metric
        model = model.to(accelerator.device)

        # load the saved checkpoint
        if ckpt_path is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(ckpt_path))

        self.model = model
        self.accelerator = accelerator
        self.eval_dataloader = eval_dataloader
        self.metric = metric
        self.task = task

        # FINISH USING ckpt:
        self.ckpt_path = ckpt_path

    def get_gpu_temperature(self):
        if self.cpu:
            return max([x[1] for x in psutil.sensors_temperatures()['coretemp'] if x[0][:4] == 'Core'])
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            raise ValueError("Explicitly run script with cuda visible devices using: CUDA_VISIBLE_DEICES=<> python <> ...")
        tempstr = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read().strip()
        templst = [float(temp) for temp in tempstr.split('\n')]
        devices = [int(deviceNo) for deviceNo in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        avg_temp = 0
        for device in devices:
            avg_temp += templst[device]
        avg_temp /= len(devices)
        return avg_temp

    # Returns the search space from which
    # sub-transformers are sampled:
    def get_choices(self):
        choices = {
            "sample_hidden_size": [360, 480, 540, 600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            "sample_num_hidden_layers": [6, 8, 10, 12],
        }
        return choices

    # Returns the Bert-Supertransformer Configuration:
    def get_supertransformer_config(self):
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

    # Samples a subtransformer using given random seed:
    def sample_subtransformer(self, randomize=False, rand_seed=0):
        if randomize:
            random.seed(rand_seed)
        choices = self.get_choices()
        config = self.get_supertransformer_config()

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

                if (
                    config.sample_hidden_size
                    % config_dict["sample_num_attention_heads"][i]
                ):
                    for key in config_dict.keys():
                        config_dict[key] = config_dict[key][:-1]
                    continue
                else:
                    break

        for key in config_dict.keys():
            setattr(config, key, config_dict[key])

        return config

    # Function to get Latency and Eval metrics:
    def get_latency_eval_subtransformer(
        self, model, config, eval_dataloader, accelerator, metric, task="sst2", check_temp=1
    ):
        # Check temperature. Data points generated within [45, 70] C range
        # Data points will include temperature as a feature
        if check_temp and self.get_gpu_temperature() >= 70:
            while self.get_gpu_temperature() >= 55:
                time.sleep(60)
        temp = self.get_gpu_temperature()

        # Set the subtransformer configuration:
        model.set_sample_config(config=config)

        # Start measuring the Latency:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Inference of subtransformer:
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

        # End measuring the Latency:
        end_time.record()
        torch.cuda.synchronize()
        latency = start_time.elapsed_time(end_time) / 1000

        return eval_metric, latency, temp

    def inference(self, config):
        eval_metric, latency, _ = self.get_latency_eval_subtransformer(
            self.model,
            config,
            self.eval_dataloader,
            self.accelerator,
            self.metric,
            task=self.task,
            check_temp=0
        )
        return eval_metric["accuracy"]

    def lat_data_point(self, sampled_config):
        _, latency, temp = self.get_latency_eval_subtransformer(
            self.model, sampled_config, self.eval_dataloader, self.accelerator, self.metric, task=self.task
        )
        return latency, temp