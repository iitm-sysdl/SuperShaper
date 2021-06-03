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


def get_gpu_temperature():
    return float(
        os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")
        .read()
        .strip()
    )


class Tester:
    # def init(self, ckpt_path, task):
    #     pass
    def __init__(
        self,
        ckpt_path,
        task="sst2",
        model_name_or_path="bert-base-uncased",
        use_pretrained_supertransformer=False,
        max_seq_length=128,
        per_gpu_eval_batch_size=64,
        fp16=True,
        cpu=False,
        seed=42,
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
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(ckpt_path))

        self.model = model
        self.accelerator = accelerator
        self.eval_dataloader = eval_dataloader
        self.metric = metric
        self.task = task

        # FINISH USING ckpt:
        self.ckpt_path = ckpt_path

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
        latency = start_time.elapsed_time(end_time) / 1000

        return eval_metric, latency

    def inference(self, config):
        eval_metric, latency = self.get_latency_eval_subtransformer(
            self.model,
            config,
            self.eval_dataloader,
            self.accelerator,
            self.metric,
            task=self.task,
        )
        return eval_metric["accuracy"]


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
        ckpt_path,  # Trained supertransformer model
        task,
        mutation_prob,
        evo_iter,
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
        self.predictor = LatencyPredictor()
        self.predictor.load_ckpt()

        self.tester = Tester(ckpt_path=ckpt_path, task=task)

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
        while cnt < self.population_size:
            candidate_gene = []
            for i in range(self.gene_len):
                candidate_gene.append(random.choice(self.gene_choice[i]))
            if self.satisfy_constraints(candidate_gene):
                population.append(candidate_gene)
                cnt += 1
            total += 1
        print(
            f"Only {cnt} out of {total} total generated samples were under latency cap."
        )
        return population

    def satisfy_constraints(self, gene):
        latency_pred = self.predictor.predict_lat(self.gene2config(gene))
        if latency_pred > self.latency_cap:
            return False
        return True

        # choices = {
        #     "sample_hidden_size": [600, 768],
        #     "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
        #     "sample_intermediate_size": [512, 1024, 2048, 3072],
        #     "sample_num_hidden_layers": [8, 10, 12],
        # }

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
            print(f"| Start Iteration {i}:")
            popu_scores = self.get_scores(popu)
            print(f"| Iteration {i}, Highest Accuracy: {max(popu_scores)}")

            sorted_ind = np.array(popu_scores).argsort()[::-1][: self.parent_size]

            self.best_config = self.gene2config(popu[sorted_ind[0]])
            print(f"| Config for highest accuracy model: {self.best_config}")
            print(
                f"| Predicted latency for highest accuracy model: {self.predictor.predict_lat(self.gene2config(popu[sorted_ind[0]]))}"
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

        return self.best_config


if __name__ == "__main__":
    search_space_example = {
        "encoder_embed_dim": [360, 480, 540, 600, 768],
        "encoder_layer_num": [2, 4, 6, 8, 10, 12],
        "encoder_ffn_embed_dim": [512, 1024, 2048, 3072],
        "encoder_self_attention_heads": [6, 8, 10, 12],
    }
    runner = Evosearch(
        12,
        6,
        2,
        2,
        2,
        search_space_example,
        6.2,
        "checkpoints/mrpc/pytorch_model.bin",
        "mrpc",
        0.5,
        3,
    )
    best_config = runner.run_evo_search()
