import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from custom_layers.custom_bert import BertForSequenceClassification
import random
import numpy as np
import os
from tasks.glue.prepare_task import GlueTask
from utils.module_proxy_wrapper import ModuleProxyWrapper

from pprint import pprint
import wandb

import plotly.graph_objects as go
from utils.wipe_memory import wipe_memory


def seed_everything(accelerator, seed=1234, randomize_across_diff_devices=False):
    if randomize_across_diff_devices:
        # following sgugger's comment here https://github.com/huggingface/accelerate/issues/90
        random.seed(seed + accelerator.process_index)
        np.random.seed(seed + accelerator.process_index)
        torch.manual_seed(seed + accelerator.process_index)
        torch.cuda.manual_seed_all(seed + accelerator.process_index)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def get_choices(limit_subtransformer_choices=False):
    if limit_subtransformer_choices:
        choices = {
            "sample_hidden_size": [600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            "sample_num_hidden_layers": [8, 10, 12],
        }
    else:
        choices = {
            "sample_hidden_size": [360, 480, 540, 600, 768],
            "sample_num_attention_heads": [2, 4, 6, 8, 10, 12],
            "sample_intermediate_size": [512, 1024, 2048, 3072],
            "sample_num_hidden_layers": [6, 8, 10, 12],
        }
    return choices


def print_subtransformer_config(config, accelerator):
    accelerator.print("===========================================================")
    accelerator.print("hidden size: ", config.sample_hidden_size)
    accelerator.print("num attention heads: ", config.sample_num_attention_heads)
    accelerator.print("intermediate sizes: ", config.sample_intermediate_size)
    accelerator.print("num hidden layers: ", config.sample_num_hidden_layers)
    accelerator.print("===========================================================")


def sample_subtransformer(
    limit_subtransformer_choices=False, randomize=False, rand_seed=0
):
    if randomize:
        random.seed(rand_seed)
    choices = get_choices(limit_subtransformer_choices)
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


def validate_subtransformer(
    model, config, eval_dataloader, accelerator, metric, task="mrpc", sample=True
):
    if sample:
        model.set_sample_config(config=config)
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
    # Use accelerator.print to print only on the main process.
    # accelerator.print(eval_metric)
    return eval_metric


def train_transformer_one_epoch(
    args,
    model,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_dataloader,
    accelerator,
    train_subtransformer=False,
    subtransformer_seed=42,
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
                args.limit_subtransformer_choices, randomize=True, rand_seed=seed
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


def training_function(args):
    # path to save the optimizer and scheduler states
    optim_scheduler_states_path = os.path.join(args.output_dir, "optim_scheduler.pt")

    param = DistributedDataParallelKwargs(
        find_unused_parameters=True, check_reduction=False
    )
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, kwargs_handlers=[param])
    seed_everything(
        accelerator=accelerator,
        seed=args.seed,
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
    # Initialize accelerator

    accelerator.print("Running on: ", accelerator.device)

    if accelerator.is_main_process:
        if not args.train_subtransformers_from_scratch:
            # TODO: change this to a better name for trianing + finetuning
            wandb.init(
                project="eHAT-warmups",
                entity="efficient-hat",
                name=args.task + "_train_scratch",
            )
        else:
            wandb.init(
                project="eHAT-warmups",
                entity="efficient-hat",
                name=args.task + "subtransformers_train_scratch",
            )

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
        task,
        model_checkpoint,
        config,
        args.max_seq_length,
        accelerator,
        initialize_pretrained_model=use_pretained,
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
        model = ModuleProxyWrapper(
            model
        )  # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    if accelerator.is_main_process:
        wandb.watch(model)

    if not args.train_subtransformers_from_scratch:
        ## train and finetune the supertransformer

        best_val_accuracy = 0
        metric_not_improved_count = 0
        metric_to_track = "supertransformer_accuracy"

        # Now we train the model
        for epoch in range(num_epochs):

            train_transformer_one_epoch(
                args,
                model,
                optimizer,
                lr_scheduler,
                gradient_accumulation_steps,
                train_dataloader,
                accelerator,
                train_subtransformer=False,  # first we will train the supertransformer
            )

            accelerator.print(f"Epoch {epoch + 1}:", end=" ")
            if accelerator.is_main_process:
                wandb.log({"epochs": epoch})
            # resetting to supertransformer before validation
            config = get_supertransformer_config()
            eval_metric = validate_subtransformer(
                model, config, eval_dataloader, accelerator, metric, task
            )
            super_dict = {}
            for key in eval_metric:
                super_key = "supertransformer_" + key
                super_dict[super_key] = eval_metric[key]

            accelerator.print(super_dict)
            if accelerator.is_main_process:
                wandb.log(super_dict)
            if args.eval_random_subtransformers:
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
                        args.limit_subtransformer_choices,
                        randomize=True,
                        rand_seed=random_seed,
                    )
                    eval_metric = validate_subtransformer(
                        model, config, eval_dataloader, accelerator, metric, task
                    )
                    # eval_metric['validation_random_seed'] = random_seed
                    # label_lst.append([eval_metric['accuracy'], random_seed])
                    # label_lst.append([random_seed, eval_metric['accuracy']])
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
                    # accelerator.print(eval_metric)
                    # wandb.log(eval_metric)

                if accelerator.is_main_process:
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

            # early stopping
            if super_dict[metric_to_track] > best_val_accuracy:
                metric_not_improved_count = 0
                best_val_accuracy = super_dict[metric_to_track]
                # unwrap and save best model so far
                accelerator.wait_for_everyone()

                unwrapped_model = accelerator.unwrap_model(model)
                # accelerator.unwrap_model(model).save_pretrained(args.output_dir)
                accelerator.save(
                    unwrapped_model.state_dict(), args.output_dir + "/pytorch_model.bin"
                )
                # accelerator.save(
                #    {
                #        "epoch": epoch + 1,
                #        "optimizer": optimizer.state_dict(),
                #        "scheduler": lr_scheduler.state_dict(),
                #    },
                #    optim_scheduler_states_path,
                # )
            else:
                metric_not_improved_count += 1
                if metric_not_improved_count >= args.early_stopping_patience:
                    break
        # accelerator.print()
        # accelerator.print("Evaluating subtransformer training")
        # accelerator.print()

        ## for finetuning, we load the best model, optimizer and scheduler
        # states. Naively loading them is causing OOM issue. Hence we first
        # clear torch cuda memory


        print(
            "===========================Wiping memory================================================="
        )
        # GR: suspecting that memory is not fully cleared here
        # based on some testing, the supertrasnformer training on mrpc with batchsize of 32
        # used around 3921 MB on gpu. After wiping mem, we are still left with aronud 1.6 GB
        # Have to check if there is a leak
        # TODO: revisit this and modify utils.wipe_memory

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(
            torch.load(args.output_dir + "/pytorch_model.bin")
        )

        # wipe_memory(optimizer)

        ## we will finetune 3 random subtransformers
        num_subtransformers_for_finetuning = 10
        fine_tuning_epochs = 10

        ## initialize the model to the best pretrained checkpoint
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        #
        ##model = accelerator.unwrap_model(model)
        # model.load_state_dict(torch.load(args.output_dir+'/pytorch_model.bin'))
        #
        ## it is important to send the model to device before sampling.
        ## Else we would get an error that weights are in cpu (not fullly sure why)
        # model = model.to(accelerator.device)

        # optimizer = AdamW(
        #    params=model.parameters(), lr=args.learning_rate, correct_bias=correct_bias
        # )

        # lr_scheduler = get_linear_schedule_with_warmup(
        #    optimizer=optimizer,
        #    num_warmup_steps=100,
        #    num_training_steps=len(train_dataloader)
        #    * fine_tuning_epochs,
        # )
        # model, optimizer = accelerator.prepare(model, optimizer)

        # model = ModuleProxyWrapper(model)

        for idx in range(num_subtransformers_for_finetuning):

            metric_to_track = "finetuned_subtransformer_" + str(idx) + "_accuracy"

            subtransformer_output_dir = os.path.join(
                args.output_dir, f"finetune_subtransformer_{str(idx)}"
            )

            best_val_accuracy = 0
            # sample one random subtransformer
            random_subtransformer_seed = idx * 1000

            ## initialize subtransformer
            super_config = sample_subtransformer(
                args.limit_subtransformer_choices,
                randomize=True,
                rand_seed=random_subtransformer_seed,
            )
            model.set_sample_config(super_config)

            # uncomment this block to use subtransformer from get_active_subnet
            # super_config.num_hidden_layers = super_config.sample_num_hidden_layers
            # model = model.get_active_subnet(super_config)
            # model = model.to(accelerator.device)

            accelerator.print(
                "Finetuning subtransformer with config: ",
                print_subtransformer_config(super_config, accelerator),
            )

            optim_scheduler_states = torch.load(optim_scheduler_states_path)
            optimizer.load_state_dict(optim_scheduler_states["optimizer"])
            lr_scheduler.load_state_dict(optim_scheduler_states["scheduler"])

            for epoch in range(fine_tuning_epochs):

                train_transformer_one_epoch(
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    gradient_accumulation_steps,
                    train_dataloader,
                    accelerator,
                    train_subtransformer=True,
                    subtransformer_seed=random_subtransformer_seed,
                )
                # no need to sample config while validating in this case
                # hence setting the sample to False
                eval_metric = validate_subtransformer(
                    model,
                    super_config,
                    eval_dataloader,
                    accelerator,
                    metric,
                    task,
                    sample=False,
                )

                accelerator.print(f"Epoch {epoch + 1}:", end=" ")

                accelerator.print(eval_metric)

                if accelerator.is_main_process:
                    wandb.log({"finetuning_epochs": epoch})

                sub_dict = {}
                for key in eval_metric:
                    sub_key = "finetuned_subtransformer_" + str(idx) + "_" + key
                    sub_dict[sub_key] = eval_metric[key]

                accelerator.print(sub_dict)

                if accelerator.is_main_process:
                    wandb.log(sub_dict)

                # early stopping
                if eval_metric[metric_to_track] > best_val_accuracy:
                    metric_not_improved_count = 0
                    best_val_accuracy = eval_metric[metric_to_track]

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(
                        unwrapped_model.state_dict(), subtransformer_output_dir
                    )
                    # accelerator.unwrap_model(model).save_pretrained(
                    #    subtransformer_output_dir
                    # )
                else:
                    metric_not_improved_count += 1
                    if metric_not_improved_count >= args.early_stopping_patience:
                        break
        # print(
        #     "===========================Wiping memory================================================="
        # )
        # # GR: suspecting that memory is not fully cleared here
        # # based on some testing, the supertrasnformer training on mrpc with batchsize of 32
        # # used around 3921 MB on gpu. After wiping mem, we are still left with aronud 1.6 GB
        # # Have to check if there is a leak
        # # TODO: revisit this and modify utils.wipe_memory

        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.load_state_dict(torch.load(args.output_dir+'/pytorch_model.bin'))

        # #wipe_memory(optimizer)

        # ## we will finetune 3 random subtransformers
        # num_subtransformers_for_finetuning = 10
        # fine_tuning_epochs = 10

        # ## initialize the model to the best pretrained checkpoint
        # #model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # #
        # ##model = accelerator.unwrap_model(model)
        # #model.load_state_dict(torch.load(args.output_dir+'/pytorch_model.bin'))
        # #
        # ## it is important to send the model to device before sampling.
        # ## Else we would get an error that weights are in cpu (not fullly sure why)
        # #model = model.to(accelerator.device)

        # #optimizer = AdamW(
        # #    params=model.parameters(), lr=args.learning_rate, correct_bias=correct_bias
        # #)


        # #lr_scheduler = get_linear_schedule_with_warmup(
        # #    optimizer=optimizer,
        # #    num_warmup_steps=100,
        # #    num_training_steps=len(train_dataloader)
        # #    * fine_tuning_epochs,
        # #)
        # #model, optimizer = accelerator.prepare(model, optimizer)

        # #model = ModuleProxyWrapper(model)

        # for idx in range(num_subtransformers_for_finetuning):

        #     metric_to_track = "finetuned_subtransformer_" + str(idx) + "_accuracy"

        #     subtransformer_output_dir = os.path.join(
        #         args.output_dir, f"finetune_subtransformer_{str(idx)}"
        #     )


        #     best_val_accuracy = 0
        #     # sample one random subtransformer
        #     random_subtransformer_seed = idx * 1000

        #     ## initialize subtransformer
        #     super_config = sample_subtransformer(
        #         args.limit_subtransformer_choices,
        #         randomize=True,
        #         rand_seed=random_subtransformer_seed,
        #     )
        #     model.set_sample_config(super_config)

        #     # uncomment this block to use subtransformer from get_active_subnet
        #     # super_config.num_hidden_layers = super_config.sample_num_hidden_layers
        #     # model = model.get_active_subnet(super_config)
        #     # model = model.to(accelerator.device)

        #     accelerator.print(
        #         "Finetuning subtransformer with config: ",
        #         print_subtransformer_config(super_config, accelerator),
        #     )

        #     optim_scheduler_states = torch.load(optim_scheduler_states_path)
        #     optimizer.load_state_dict(optim_scheduler_states["optimizer"])
        #     lr_scheduler.load_state_dict(optim_scheduler_states["scheduler"])

        #     for epoch in range(fine_tuning_epochs):

        #         train_transformer_one_epoch(
        #             args,
        #             model,
        #             optimizer,
        #             lr_scheduler,
        #             gradient_accumulation_steps,
        #             train_dataloader,
        #             accelerator,
        #             train_subtransformer=True,
        #             subtransformer_seed=random_subtransformer_seed,
        #         )
        #         # no need to sample config while validating in this case
        #         # hence setting the sample to False
        #         eval_metric = validate_subtransformer(
        #             model,
        #             super_config,
        #             eval_dataloader,
        #             accelerator,
        #             metric,
        #             task,
        #             sample=False,
        #         )

        #         accelerator.print(f"Epoch {epoch + 1}:", end=" ")

        #         accelerator.print(eval_metric)

        #         if accelerator.is_main_process:
        #             wandb.log({"finetuning_epochs": epoch})

        #         sub_dict = {}
        #         for key in eval_metric:
        #             sub_key = "finetuned_subtransformer_" + str(idx) + "_" + key
        #             sub_dict[sub_key] = eval_metric[key]

        #         accelerator.print(sub_dict)

        #         if accelerator.is_main_process:
        #             wandb.log(sub_dict)

        #        # early stopping
        #         if eval_metric[metric_to_track] > best_val_accuracy:
        #             metric_not_improved_count = 0
        #             best_val_accuracy = eval_metric[metric_to_track]

        #             accelerator.wait_for_everyone()
        #             unwrapped_model = accelerator.unwrap_model(model)
        #             accelerator.save(unwrapped_model.state_dict(), subtransformer_output_dir)
        #             #accelerator.unwrap_model(model).save_pretrained(
        #             #    subtransformer_output_dir
        #             #)
        #         else:
        #             metric_not_improved_count += 1
        #             if metric_not_improved_count >= args.early_stopping_patience:
        #                 break

            model = BertForSequenceClassification.from_pretrained(
                args.model_name_or_path
            )

            model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)

            model = model.to(accelerator.device)
            super_config.num_hidden_layers = super_config.sample_num_hidden_layers
            model.set_sample_config(super_config)

            model = model.get_active_subnet(super_config)

            optimizer = AdamW(
                params=model.parameters(),
                lr=args.learning_rate,
                correct_bias=correct_bias,
            )

            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=100,
                num_training_steps=len(train_dataloader) * num_epochs,
            )

            model, optimizer = accelerator.prepare(model, optimizer)

            accelerator.print(
                "Training subtransformer from scratch with config: ",
                print_subtransformer_config(super_config, accelerator),
            )

            os.makedirs(subtransformer_output_dir, exist_ok=True)

            for epoch in range(num_epochs):

                train_transformer_one_epoch(
                    args,
                    model,
                    optimizer,
                    lr_scheduler,
                    gradient_accumulation_steps,
                    train_dataloader,
                    accelerator,
                    train_subtransformer=True,  # first we will train the supertransformer
                )

                accelerator.print(f"Epoch {epoch + 1}:", end=" ")
                if accelerator.is_main_process:
                    wandb.log({"epochs": epoch})

                eval_metric = validate_subtransformer(
                    model,
                    config,
                    eval_dataloader,
                    accelerator,
                    metric,
                    task,
                    sample=False,
                )

                sub_dict = {}
                for key in eval_metric:
                    sub_key = "subtransformer_" + str(idx) + "_" + key
                    sub_dict[sub_key] = eval_metric[key]

                accelerator.print(sub_dict)

                if accelerator.is_main_process:
                    wandb.log(sub_dict)

                # early stopping
                if sub_dict[metric_to_track] > best_val_accuracy:
                    metric_not_improved_count = 0
                    best_val_accuracy = sub_dict[metric_to_track]
                    # unwrap and save best model so far
                    accelerator.unwrap_model(model).save_pretrained(
                        subtransformer_output_dir
                    )
                else:
                    metric_not_improved_count += 1
                    if metric_not_improved_count >= args.early_stopping_patience:
                        break


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
        default=0,
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
        default=5.0,
        type=float,
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

    args.output_dir = args.output_dir + "/" + args.task
    # if the mentioned output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    training_function(args)


if __name__ == "__main__":
    main()
