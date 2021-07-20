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
from tester import Tester

class LatencyDatasetGenerator():
    def __init__(self, path='./latency_dataset/sst2_gpu_gtx1080_tmp.csv', data_size=1000, cpu=False, fp16=True):
        self.tester = Tester(cpu=cpu, fp16=fp16)
        self.num_sub = data_size
        self.num_supertransformer_layers = self.tester.get_supertransformer_config().sample_num_hidden_layers

        # make columns of csv:
        self.columns = [
            'encoder_embed_dim',
            'encoder_layer_num',
        ]
        for i in range(self.num_supertransformer_layers):
            self.columns.append(f'encoder_ffn_embed_dim_{i}')
        for i in range(self.num_supertransformer_layers):
            self.columns.append(f'encoder_self_attention_heads_{i}')
        self.columns += [
            'latency',
            'temp',
        ]
        # self.columns += [
        #     'latency_mean_encoder',
        #     'latency_std_encoder',
        #     'temp_mean_encoder',
        #     'temp_std_encoder',
        # ]
        self.path = path
    
    # Function to generate a latency dataset:
    def generate(self):
        # Measure Average Latency for different sub-transformers:
        avg_iter = 5
        for subNo in range(self.num_sub):
            print(f'Evaluating subtransformer number {subNo}')
            lat_list = []
            temp_list = []

            # Sample subtransformer to measure latency for:
            random_number = int(10000 * time.perf_counter())
            sampled_config = self.tester.sample_subtransformer(
                randomize=True,
                rand_seed=random_number
            )

            for iter in tqdm(range(avg_iter)):
                latency, temp = self.tester.lat_data_point(sampled_config)
                lat_list.append(latency)
                temp_list.append(temp)

                avg_latency = np.mean(np.array(lat_list))
                std_latency = np.std(np.array(lat_list))
                avg_temp = np.mean(np.array(temp_list))
                std_temp = np.std(np.array(temp_list))

                # Create a datapoint in dataset
                # The first four columns correspond to config:
                # encoder_embed_dim, encoder_layer_num, encoder_ffn_embed_dim, encoder_self_attention_heads
                # The last two columns correspond to the avg latency and std dev
                row = []
                row.append(sampled_config.sample_hidden_size)
                row.append(sampled_config.sample_num_hidden_layers)
                row += self.encode(sampled_config.sample_intermediate_size)
                row += self.encode(sampled_config.sample_num_attention_heads)
                row.append(latency)
                row.append(temp)
                # row.append(avg_latency)
                # row.append(std_latency)
                # row.append(avg_temp)
                # row.append(std_temp)
                self.writeRow(row, first_line_flag = (subNo > 0 or iter > 0))
    
    # Function to write dataset to a csv file:
    def writeRow(self, row, first_line_flag = 0):
        data = np.array([row])
        if first_line_flag == 0:
            pd.DataFrame(data=data, columns=self.columns).to_csv(self.path, index=False)
        else:
            pd.DataFrame(data=data, columns=self.columns).to_csv(self.path, index=False, header=False, mode='a')
    
    def encode(self, lst):
        extra = self.num_supertransformer_layers-len(lst)
        return lst + [-1]*extra


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"]='false'
    ldg = LatencyDatasetGenerator(path = './latency_dataset/sst2_cpu.csv', data_size=1000, cpu=True, fp16=False)
    ldg.generate()