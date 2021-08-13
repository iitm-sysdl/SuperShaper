import random
from tqdm import tqdm

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
from custom_layers import custom_bert, custom_mobile_bert
import os
from pprint import pprint
import pandas as pd
import time
from tester import Tester
from transformers.models.bert.modeling_bert import BertForMaskedLM
from utils import calculate_params_from_config


class EvolSearch:
    def __init__(
        self,
        population_size, 
        parent_size,
        mutation_size, 
        crossover_size, 
        task,
        mutation_prob, 
        time_budget,
        search_space_config,
        bert_config=None,
        constraints_set=None,
        latency_predictor = None,
        fitness_set = None,
        ckpt_path = None, 
        accelerator = None, 
    ):

    self.search_space_config = search_space_config
    self.config = bert_config
    self.features = None
    self.constraints_set = constraints_set
    self.latency_predictor = latency_predictor
    self.gene_len = None
    self.time_budget = time_budget

    self.parent_size = parent_size
    self.mutation_size = mutation_size
    self.crossover_size = crossover_size

    self.mutation_prob = mutation_prob

    def get_search_space(self):
        space = {
            "sample_hidden_size": [120, 240, 360, 480, 540, 600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            "sample_num_hidden_layers": list(range(6, self.config.num_hidden_layers+1, 2))
        } 

        if self.search_space_config == 'bert-bottleneck':
            space["sample_hidden_size"] = [120, 240, 360, 480, 540, 600, 768] 
            space["sample_num_attention_heads"] = [12]
            space["sample_intermediate_size"] = [3072]
            space["sample_num_hidden_layers"] = [12]
        elif self.search_space_config is not 'bert-bottleneck' and self.search_space_config is not 'attention':
            raise NotImplementedError

        gene_len = 0
        num_hidden_layers = space["sample_num_hidden_layers"]
        for key in space:
            gene_len += len(space[key])*num_hidden_layers

        self.gene_len = gene_len

        return space

    def arch2feature(self, config=None):
        config = config or self.config
        features = []

        for key in config:
            if 'sample_' in key:
                if isinstance(config[key], list):
                    features += config[key]
                else:
                    features.append(config[key])

        self.features = features
        return features

    def feature2arch(self, feature_in):
        features = feature_in or self.features
        feature_cnt = 0 
        space = self.get_search_space()
        num_hidden_layers = space["sample_num_hidden_layers"]

        for key in space:
            if 'sample_num_hidden_layers' in key:
                continue

            for i in range(num_hidden_layers):
                space[key][i] = features[feature_cnt]
                feature_cnt += 1

        return space

    def satisfy_constraints(self, feature_in): 
        satisfy = None
        feature = feature_in or self.features

        ## This simple composition of multiple-objectives is a very slow process -> TODO: Use something like NSGA-II
        for constraints in self.constraints_set:
            if 'latency' in constraints: 
                lat = self.latency_predictor.predict(feature)
                if lat <= self.constraints_set[constraints]:
                    satisfy = True
                else:
                    return False
            elif 'perplexity' in constraints: 
                perp = self.perplexity_predictor.predict(feature)
                if perp <= self.constraints_set[constraints]:
                    satisfy = True
                else:
                    return False
            elif 'params' in constraints:
                params = calculate_params_from_config(self.feature2arch(feature))
                if params <= self.constraints_set[constraints]:
                    satisfy = True
                else:
                    return False

        if satisfy is None:
            raise NotImplementedError

        return satisfy
    
    def random_sample_arch(self):
        space = self.get_search_space()
        num_hidden_layers = space["num_hidden_layers"]

        return {
            "sample_hidden_size": random.choices(space["sample_hidden_size"], k=num_hidden_layers),
            "sample_num_attention_heads": random.choices(space["sample_num_attention_heads"], k=num_hidden_layers),
            "sample_intermediate_size": random.choices(space["sample_intermediate_size"], k=num_hidden_layers),
            "sample_num_hidden_layers": random.choices(space["sample_num_hidden_layers"], k=num_hidden_layers),
        }

    def random_sample(self):
        population = []
        cnt = 0
        total = 0
        self.accelerator.print(f"Randomly sampling architectures")
        
        space = self.get_search_space()

        while cnt < self.population_size:
            arch = self.random_sample_arch()
            candidate_gene = self.arch2feature(arch)

            if self.satisfy_constraints(candidate_gene):
                population.append(candidate_gene)
                cnt += 1
            total += 1
        self.accelerator.print(
            f"Only {cnt} out of {total} total generated samples were under given constraints."
        )
        return population

    def evaluate_fitness(self, population):
        eval_archs = []
        fitness_fn = self.fitness_fn

        for p in population: 
            score = self.tester.get_fitness(self.feature2arch(p), fitness_fn)
            eval_archs.append(score)

        return eval_archs

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

    ## Implementation is a bit suboptimal, where fitness evaluation happens all the times for the architectures that are
    ## selected in each iteration. We can memoize their evaluations to save time.
    def run_evo_search(self):
        population = self.random_sample()

        for i in range(self.time_budget):
            self.accelerator.print(f"| Start Iteration {i}:")
            fitness_scores = self.get_fitness(population)

            sorted_ind = np.array(fitness_scores).argsort()[::-1][: self.parent_size]

            self.best_config = self.feature2arch(population[sorted_ind[0])
            self.accelerator.print(f"| Config for highest accuracy model: {self.best_config}")

            parents_next_iter  =  [population[m] for m in sorted_ind]
            parents_next_score =  [fitness_scores[m] for m in sorted_ind]

            mutations = [] 
            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_next_iter)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutations.append(mutated_gene)
                    k += 1

            
            crossovers = []
            k=0
            while k < self.crossover_size:
                crossedover_gene = self.crossover(random.sample(parents_next_iter, 2))
                if self.satisfy_constraints(crossedover_gene):
                    crossovers.append(crossedover_gene)
                    k += 1

            population = parents_next_iter + mutations + crossovers

        return self.best_config, self.max_acc, self.config_latency


if __name__ == "__main__":
    pass
