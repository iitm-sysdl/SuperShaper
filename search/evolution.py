import random
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
import torch
from tqdm import tqdm
import copy
from custom_layers import custom_bert, custom_mobile_bert
from pprint import pprint
import pandas as pd
import time
from transformers.models.bert.modeling_bert import BertForMaskedLM
from utils import calculate_params_from_config
from predictor import Predictor
import pandas

# Add sampling folder to the PYTHONPATH
from sampling import (
    Sampler,
    get_supertransformer_config,
    show_random_elements,
    show_args,
)

parent_dir = "./"
mode = 0o777


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
        perplexity_predictor = None,
        fitness_set = None,
        ckpt_path = None, 
        accelerator = None, 
    ):

        self.search_space_config = search_space_config
        self.config = bert_config
        self.features = None
        self.constraints_set = constraints_set
        self.latency_predictor = latency_predictor
        self.perplexity_predictor = perplexity_predictor
        self.gene_len = None
        self.time_budget = time_budget
        self.population_size = population_size

        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.crossover_size = crossover_size

        self.mutation_prob = mutation_prob
        self.keys = ["sample_hidden_size", "sample_num_attention_heads", "sample_intermediate_size", "sample_num_hidden_layers"]

    def get_search_space(self):
        space = {
            "sample_hidden_size": [120, 240, 360, 480, 540, 600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            #"sample_num_hidden_layers": list(range(6, self.config.num_hidden_layers+1, 2))
            "sample_num_hidden_layers": [12]
        } 

        if self.search_space_config == 'bert-bottleneck':
            space["sample_hidden_size"] = [120, 240, 360, 480, 540, 600, 768] 
            #space["sample_num_attention_heads"] = [6,8,12]
            space["sample_num_attention_heads"] = [12]
            #space["sample_intermediate_size"] = [1024,2048,3072]
            space["sample_intermediate_size"] = [3072]
            space["sample_num_hidden_layers"] = [12]
        elif self.search_space_config != 'bert-bottleneck' and self.search_space_config != 'attention':
            raise NotImplementedError

        gene_len = 0
        num_hidden_layers = space["sample_num_hidden_layers"][0]

        ### TODO: This logic breaks down if depth is elastic in the search_space - Diagnose
        gene_len = (len(space.keys()) - 1) * num_hidden_layers + 1
        self.gene_len = gene_len
        
        self.gene_choice = []
        for key in self.keys:
            if key == "sample_num_hidden_layers":
                break

            for i in range(num_hidden_layers):
                self.gene_choice.append(space[key])
                
        
        self.gene_choice.append(space["sample_num_hidden_layers"])
            

        return space
    
    def fitness_fn(self, feature): ## This is just temporary, we can remove this for a a more general implementation 
        feature = feature or self.features 

        feature = np.array(feature)
        feature = np.reshape(feature, (1, feature.shape[0]))

        score = self.perplexity_predictor.predict(feature)

        return score[0]


    def arch2feature(self, config=None):
        config = config or self.config
        features = []

        ## Have the features arranged layerwise
        for key in self.keys:
            attr = getattr(config, key)
            if isinstance(attr, list):
                features += attr
            else:
                features.append(attr)

        self.features = features

        return features

    def feature2arch(self, feature_in):
        features = feature_in or self.features
        feature_cnt = 0 
        num_hidden_layers = getattr(self.config, "sample_num_hidden_layers")

        arch_config = copy.deepcopy(self.config)

        ## Change config here
        for key in self.keys:
            if 'sample_num_hidden_layers' in key:
                continue

            arch_conf_lst = []
            for i in range(num_hidden_layers):
                arch_conf_lst.append(features[feature_cnt])
                feature_cnt += 1
            
            setattr(arch_config, key, arch_conf_lst)

        return arch_config

    def satisfy_constraints(self, feature_in): 
        satisfy = None
        feature = feature_in or self.features
        
        feature = np.array(feature)
        feature = np.reshape(feature, (1,feature.shape[0])) ## Weird but seems necessary

        ## This simple composition of multiple-objectives is a very slow process -> TODO: Use something like NSGA-II
        for constraints in self.constraints_set:
            if 'latency' in constraints: 
                assert self.latency_predictor is not None

                lat = self.latency_predictor.predict(feature)
                if lat <= self.constraints_set[constraints] or self.constraints_set[constraints] == -1:
                    satisfy = True
                else:
                    return False
            elif 'perplexity' in constraints: 
                assert self.perplexity_predictor is not None

                perp = self.perplexity_predictor.predict(feature)
                if perp <= self.constraints_set[constraints] or self.constraints_set[constraints] == -1:
                    satisfy = True
                else:
                    return False
            elif 'params' in constraints:
                params = calculate_params_from_config(self.feature2arch(feature))
                if params <= self.constraints_set[constraints]:
                    satisfy = True
                else:
                    return False
            elif 'none' in constraints: ## Trivially True
                return True

        if satisfy is None:
            raise NotImplementedError

        return satisfy
    
    def random_sample_arch(self, config=None):
        space = self.get_search_space()
        
        config = config or copy.deepcopy(self.config) 

        num_hidden_layers = space["sample_num_hidden_layers"][0]
        
        tmp_dict = {
            "sample_hidden_size": random.choices(space["sample_hidden_size"], k=num_hidden_layers),
            "sample_num_attention_heads": random.choices(space["sample_num_attention_heads"], k=num_hidden_layers),
            "sample_intermediate_size": random.choices(space["sample_intermediate_size"], k=num_hidden_layers),
            "sample_num_hidden_layers": random.choices(space["sample_num_hidden_layers"], k=1),
        }

        for keys in tmp_dict.keys():
            setattr(config, keys, tmp_dict[keys])

        return config

    def random_sample(self):
        population = []
        cnt = 0
        total = 0
        print(f"Randomly sampling architectures")
        
        space = self.get_search_space()

        while cnt < self.population_size:
            arch = self.random_sample_arch()
            candidate_gene = self.arch2feature(arch)

            if self.satisfy_constraints(candidate_gene):
                population.append(candidate_gene)
                cnt += 1
                print("Adding gene no. %d to the population"%(cnt))
            total += 1
        print(
            f"Only {cnt} out of {total} total generated samples were under given constraints."
        )
        return population

    def evaluate_fitness(self, population):
        assert self.fitness_fn is not None

        eval_archs = []
        fitness_fn = self.fitness_fn

        for p in population: 
            #score = fitness_fn(self.feature2arch(p))
            score = fitness_fn(p)
            #score = self.tester.get_fitness(self.feature2arch(p), fitness_fn)
            eval_archs.append(score)

        return eval_archs

    def crossover(self, genes):
        crossedover_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossedover_gene.append(genes[0][i])
            else:
                crossedover_gene.append(genes[1][i])

        return crossedover_gene


    def mutate(self, feature_in):
        feature = feature_in or self.features
        search_space = self.get_search_space() # Just to ensure self.gene_choice is updated 

        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choice(self.gene_choice[i]))
            else:
                mutated_gene.append(feature[i])

        return mutated_gene

    ## Implementation is a bit suboptimal, where fitness evaluation happens all the times for the architectures that are
    ## selected in each iteration. We can memoize their evaluations to save time.
    def run_evo_search(self):
        population = self.random_sample()

        directory = 'perpx_'+str(self.constraints_set['perplexity'])
        path = os.path.join(parent_dir, directory)
        os.makedirs(path, mode, exist_ok=True)

        for i in range(self.time_budget):
            self.best_config_lst = []
            print(f"| Start Iteration {i}:")
            fitness_scores = self.evaluate_fitness(population)

            new_f = []
            new_population = []
            cnt = 0
            for f in fitness_scores:
                if f not in new_f:
                    new_f.append(f)
                    new_population.append(population[cnt])
                cnt+=1
                    
            fitness_scores = new_f
            population = new_population

            
            sorted_ind = np.array(fitness_scores).argsort()[::-1][: self.parent_size]

            fitness_scores_top = np.array(fitness_scores)[sorted_ind]

            self.best_config = self.feature2arch(population[sorted_ind[0].item()])
            #print(f"| Config for highest accuracy model: {self.best_config}")

            for j in range(self.parent_size):
                self.best_config_lst.append(self.feature2arch(population[sorted_ind[j].item()]))
            
            df = pd.DataFrame(list(zip(self.best_config_lst, fitness_scores_top)), columns=['configs', 'predicted_perpx'])
           
            df.to_csv(path+'/best_configs_iter_'+str(i)+'.csv', index=False)
    
            parents_next_iter  =  [population[m.item()] for m in sorted_ind]
            parents_next_score =  [fitness_scores[m.item()] for m in sorted_ind]

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

        return self.best_config


def search(): 
    population_size = 100
    parent_size = 10
    mutation_size = 100
    crossover_size = 100
    task = 'mlm'
    mutation_prob = 0.4 
    time_budget = 300
    search_space_config = 'bert-bottleneck'

    bert_config=get_supertransformer_config("bert-base-cased", mixing=search_space_config)

    latency_predictor = None,
    fitness_set = None,
    ckpt_path = None, 
    accelerator = None,     

    constraints_set = { 'perplexity' : 5.65 } ## Just specifying an unbounded value to begin with
    perplexity_predictor = Predictor(ckpt='./outputs/perplexity_predictor.xgb', pred_type='perplexity', model_type='xgb')
    perplexity_predictor.load_ckpt()
    
    evolution = EvolSearch(population_size, 
                           parent_size, 
                           mutation_size, 
                           crossover_size, 
                           task, 
                           mutation_prob, 
                           time_budget, 
                           search_space_config, 
                           bert_config,
                           constraints_set = constraints_set, 
                           perplexity_predictor = perplexity_predictor
                           )

    ### Testing Get Search Space ### 
    #space = evolution.get_search_space()
    #print(space)
    #print(evolution.gene_len)
    #
    #print("--------------------------------------")

    #### Testing arch2feature and feature2arch ### 
    #gene = evolution.arch2feature(bert_config)
    #print(gene)
    #
    #print("--------------------------------------")
    #    
    #gene[0] = gene[1] = gene[-2] = gene[-3] = 256
    #feature = evolution.feature2arch(gene)
    #print(feature)
    #
    #print("--------------------------------------")

    #### Testing Mutation and Crossover ###
    #mutated_gene = evolution.mutate(gene)
    #print(mutated_gene)

    #print("--------------------------------------")

    #crossedover_gene = evolution.crossover([gene, mutated_gene])
    #print(gene)
    #print(mutated_gene)
    #print(crossedover_gene)

    #print("--------------------------------------")

    ### Testing Random Sampling and Evolutionary Search ### 
    #sampled_population = evolution.random_sample()
    #print(sampled_population)

    print(evolution.run_evo_search())

if __name__ == "__main__":
    search()
