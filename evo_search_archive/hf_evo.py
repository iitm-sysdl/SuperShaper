import random
from tqdm import tqdm


# from hflat import LatencyPredictor
from hf_lat_lgbm import LatencyPredictor

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
from tester import Tester


class Evosearch:
    def __init__(
        self,
        super_num_layers,
        population_size,
        parent_size,
        mutation_size,
        crossover_size,
        search_space,
        latency_cap,
        task,
        mutation_prob,
        evo_iter,
        ckpt_path=None,  # Trained supertransformer model
        accel=None
        # Figure this out as we go
    ):
        self.super_num_layers = super_num_layers
        self.gene_len = 2 + 2 * self.super_num_layers
        self.population_size = population_size
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.crossover_size = crossover_size
        self.mutation_prob = mutation_prob
        self.evo_iter = evo_iter

        self.gene_choice = []
        self.gene_choice.append(search_space["encoder_embed_dim"])
        self.gene_choice.append(search_space["encoder_layer_num"])
        for _ in range(self.super_num_layers):
            self.gene_choice.append(search_space["encoder_ffn_embed_dim"])
        for _ in range(self.super_num_layers):
            self.gene_choice.append(search_space["encoder_self_attention_heads"])

        self.latency_cap = latency_cap
        self.predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_cpu_985.txt')
        # self.predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_724.txt')
        self.predictor.load_ckpt()

        self.tester = Tester(ckpt_path=ckpt_path, task=task, accel=accel)
        self.accelerator = self.tester.accel

    def config2gene(self, config):
        gene = []

        gene.append(config["encoder"]["encoder_embed_dim"])
        gene.append(config["encoder"]["encoder_layer_num"])

        for i in range(self.super_num_layers):
            if i < config["encoder"]["encoder_layer_num"]:
                gene.append(config["encoder"]["encoder_ffn_embed_dim"][i])
            else:
                gene.append(0)

        for i in range(self.super_num_layers):
            if i < config["encoder"]["encoder_layer_num"]:
                gene.append(config["encoder"]["encoder_self_attention_heads"][i])
            else:
                gene.append(0)

        return gene

    def gene2config(self, gene):
        config = {
            "encoder": {
                "encoder_embed_dim": None,
                "encoder_layer_num": None,
                "encoder_ffn_embed_dim": [],
                "encoder_self_attention_heads": [],
            }
        }
        index = 0

        config["encoder"]["encoder_embed_dim"] = gene[index]
        index += 1

        config["encoder"]["encoder_layer_num"] = gene[index]
        index += 1

        for i in range(config["encoder"]["encoder_layer_num"]):
            config["encoder"]["encoder_ffn_embed_dim"].append(gene[index + i])
        index += self.super_num_layers

        for i in range(config["encoder"]["encoder_layer_num"]):
            config["encoder"]["encoder_self_attention_heads"].append(gene[index + i])
        index += self.super_num_layers

        return config

    def random_sample(self):
        population = []
        cnt = 0
        total = 0
        self.accelerator.print(f"Random sampling beginning...")
        while cnt < self.population_size:
            candidate_gene = []
            for i in range(self.gene_len):
                candidate_gene.append(random.choice(self.gene_choice[i]))
            if self.satisfy_constraints(candidate_gene):
                population.append(candidate_gene)
                cnt += 1
            total += 1
        self.accelerator.print(
            f"Only {cnt} out of {total} total generated samples were under latency cap."
        )
        return population

    def satisfy_constraints(self, gene):
        latency_pred = self.predictor.predict_lat(self.gene2config(gene))
        if latency_pred > self.latency_cap:
            return False
        return True

    def convert_to_right_config(self, config):
        final_config_dict = {
            "sample_hidden_size": config["encoder"]["encoder_embed_dim"],
            "sample_num_attention_heads": config["encoder"][
                "encoder_self_attention_heads"
            ],
            "sample_intermediate_size": config["encoder"]["encoder_ffn_embed_dim"],
            "sample_num_hidden_layers": config["encoder"]["encoder_layer_num"],
        }
        final_config = self.tester.get_supertransformer_config()
        for key in final_config_dict.keys():
            setattr(final_config, key, final_config_dict[key])
        return final_config

    def get_scores(self, genes):
        accuracies = []
        for gene in genes:
            config = self.convert_to_right_config(self.gene2config(gene))
            accuracies.append(self.tester.inference(config))
        return accuracies

    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choice(self.gene_choice[i]))
            else:
                mutated_gene.append(gene[i])

        return mutated_gene

    def run_evo_search(self):
        popu = self.random_sample()
        all_scores_list = []

        for i in range(self.evo_iter):
            self.accelerator.print(f"| Start Iteration {i}:")
            popu_scores = self.get_scores(popu)
            self.max_acc = max(popu_scores)
            self.accelerator.print(f"| Iteration {i}, Highest Accuracy: {self.max_acc}")

            sorted_ind = np.array(popu_scores).argsort()[::-1][: self.parent_size]

            self.best_config = self.gene2config(popu[sorted_ind[0]])
            self.accelerator.print(f"| Config for highest accuracy model: {self.best_config}")
            self.config_latency = self.predictor.predict_lat(self.gene2config(popu[sorted_ind[0]]))
            self.accelerator.print(
                f"| Predicted latency for highest accuracy model: {self.config_latency}"
            )

            parents_popu = [popu[m] for m in sorted_ind]

            parents_score = [popu_scores[m] for m in sorted_ind]
            all_scores_list.append(parents_score)

            mutate_popu = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_popu)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutate_popu.append(mutated_gene)
                    k += 1

            crossover_popu = []

            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_popu, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_popu.append(crossovered_gene)
                    k += 1

            popu = parents_popu + mutate_popu + crossover_popu
        return self.best_config, self.max_acc, self.config_latency


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"]='false'
    search_space_example = {
        "encoder_embed_dim": [360, 480, 540, 600, 768],
        "encoder_layer_num": [2, 4, 6, 8, 10, 12],
        "encoder_ffn_embed_dim": [512, 1024, 2048, 3072],
        "encoder_self_attention_heads": [6, 8, 10, 12],
    }
    
    # for 
    runner = Evosearch(
        12,
        10,
        3,
        4,
        3,
        search_space_example,
        2.5,
        "mrpc",
        0.4,
        1,
        "checkpoints/qqp/pytorch_model.bin",
    )
    best_config, _, _ = runner.run_evo_search()

