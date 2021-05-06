import numpy as np
import matplotlib.pyplot as plt
import random
import time
from prepare_task import GlueTask
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
from custom_bert import BertForSequenceClassification
import os
from pprint import pprint
import pandas as pd

class LatencyDatasetGenerator():
    def __init__(
            self,
            datasize=2000,
            task = "sst2",
            model_name_or_path = "bert-base-uncased",
            use_pretrained_supertransformer = False,  
            output_dir = "checkpoints",
            max_seq_length = 128,
            per_gpu_eval_batch_size = 64,
            gradient_accumulation_steps = 1,
            fp16 = True,
            cpu = False,
            seed = 42
        ):

        # Initialize accelerator
        accelerator = Accelerator(fp16=fp16, cpu=cpu)

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
            task, model_checkpoint, config, initialize_pretrained_model=use_pretained
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
            return glue_task.tokenizer.pad(examples, padding="longest", return_tensors="pt")

        # Instantiate dataloader
        eval_dataloader = DataLoader(
            glue_task.eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=per_gpu_eval_batch_size,
        )

        model = glue_task.model
        metric = glue_task.metric
        model = model.to(accelerator.device)

        self.model = model
        self.accelerator = accelerator
        self.eval_dataloader = eval_dataloader
        self.metric = metric
        self.task = task

        # The dataset to be stored:
        self.num_sub = datasize
        self.dataset = []


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

                if config.sample_hidden_size % config_dict["sample_num_attention_heads"][i]:
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
        self, model, config, eval_dataloader, accelerator, metric, task="sst2"
    ):
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
        latency = start_time.elapsed_time(end_time)/1000

        return eval_metric, latency
    
    # Function to generate a latency dataset:
    def generate(self):
        # Measure Average Latency for different sub-transformers:
        avg_iter = 30
        for subNo in tqdm(range(self.num_sub)):
            lat_list = []

            # Sample subtransformer to measure latency for:
            random_number = int(10000 * time.perf_counter())
            sampled_config = self.sample_subtransformer(
                randomize=True,
                rand_seed=random_number
            )

            for _ in range(avg_iter):
                eval_metric, latency = self.get_latency_eval_subtransformer(
                    self.model, sampled_config, self.eval_dataloader, self.accelerator, self.metric, task=self.task
                )
                lat_list.append(latency)

            avg_latency = np.mean(np.array(lat_list))
            std_latency = np.std(np.array(lat_list))

            # Create a datapoint in dataset
            # The first four columns correspond to config:
            # encoder_embed_dim, encoder_layer_num, encoder_ffn_embed_dim, encoder_self_attention_heads
            # The last two columns correspond to the avg latency and std dev
            row = []
            row.append(sampled_config.sample_hidden_size)
            row.append(sampled_config.sample_num_hidden_layers)
            row.append(np.mean(np.array(sampled_config.sample_intermediate_size)))
            row.append(np.mean(np.array(sampled_config.sample_num_attention_heads)))
            row.append(avg_latency)
            row.append(std_latency)
            self.dataset.append(row)
    
    # Function to write dataset to a csv file:
    def writeData(self, path='./latency_dataset/sst2_gpu_gtx1080.csv'):
        data = np.array(self.dataset)
        columns = [
            'encoder_embed_dim',
            'encoder_layer_num',
            'encoder_ffn_embed_dim_avg',
            'encoder_self_attention_heads_avg',
            'latency_mean_encoder',
            'latency_std_encoder'
        ]
        pd.DataFrame(data=data, columns=columns).to_csv(path, index=False)


        

ldg = LatencyDatasetGenerator(datasize=1000)
ldg.generate()
ldg.writeData('./latency_dataset/sst2_gpu_gtx1080_part_2.csv')