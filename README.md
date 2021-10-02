# SuperShaper

This repository contains our PyTorch training code, evaluation code and pre-trained models for SuperShaper.

If you find this repo useful in your work, please consider citing our work:
TODO: Add citation when ready.

## Quick Start
```
pip install -r requirements.txt
```
Install the required packages.

## Usage

We use [accelerate](https://huggingface.co/docs/accelerate/index.html) to train the transformers with no code changes on different setups (multi-gpu, TPU, etc)

### Configure your training setup
```bash
accelerate config                                       # answer questions wrt your training setup (multi-gpu, tpu, fp16 etc)

accelerate config  --config_file <path to config>       # to create custom training setup for different tasks
```

### Run the code with accelerate launch

```bash
accelerate launch train_mlm.py <args>
```

### List of arguments for `train_mlm.py`:

```doc
usage: train_mlm.py [-h] [--dataset_name DATASET_NAME]
                    [--dataset_config_name DATASET_CONFIG_NAME]
                    [--train_file TRAIN_FILE]
                    [--validation_file VALIDATION_FILE]
                    [--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE]
                    [--pad_to_max_length]
                    [--model_name_or_path MODEL_NAME_OR_PATH]
                    [--config_name CONFIG_NAME]
                    [--tokenizer_name TOKENIZER_NAME] [--use_slow_tokenizer]
                    [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                    [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                    [--learning_rate LEARNING_RATE]
                    [--weight_decay WEIGHT_DECAY]
                    [--num_train_epochs NUM_TRAIN_EPOCHS]
                    [--max_train_steps MAX_TRAIN_STEPS]
                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                    [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                    [--num_warmup_steps NUM_WARMUP_STEPS]
                    [--output_dir OUTPUT_DIR] [--seed SEED]
                    [--model_type MODEL_TYPE] [--logging_steps LOGGING_STEPS]
                    [--max_seq_length MAX_SEQ_LENGTH]
                    [--line_by_line LINE_BY_LINE]
                    [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                    [--overwrite_cache OVERWRITE_CACHE]
                    [--mlm_probability MLM_PROBABILITY]
                    [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                    [--layer_drop_prob LAYER_DROP_PROB]
                    [--eval_random_subtransformers EVAL_RANDOM_SUBTRANSFORMERS]
                    [--train_subtransformers_from_scratch TRAIN_SUBTRANSFORMERS_FROM_SCRATCH]
                    [--fp16 FP16] --mixing
                    {attention,gmlp,fnet,mobilebert,bert-bottleneck}
                    [--resume_from_checkpoint_dir RESUME_FROM_CHECKPOINT_DIR]
                    [--tiny_attn TINY_ATTN]
                    [--num_subtransformers_monitor NUM_SUBTRANSFORMERS_MONITOR]
                    [--c4_dir C4_DIR] [--tokenized_c4_dir TOKENIZED_C4_DIR]
                    [--sampling_type {none,naive_params,biased_params,random}]
                    [--sampling_rule {none,sandwich}] [--pop_size POP_SIZE]
                    --k_sampling K_SAMPLING
                    [--inplace_distillation INPLACE_DISTILLATION]
                    [--kd_ratio KD_RATIO]
                    [--layerwise_distillation LAYERWISE_DISTILLATION]
                    [--alpha_divergence ALPHA_DIVERGENCE]
                    [--alpha_min ALPHA_MIN] [--alpha_max ALPHA_MAX]
                    [--beta_clip BETA_CLIP]
                    [--subtransformer_config_path SUBTRANSFORMER_CONFIG_PATH]
                    [--rewire REWIRE]
                    [--rewired_model_checkpoint_dir REWIRED_MODEL_CHECKPOINT_DIR]
                    [--wandb_suffix WANDB_SUFFIX]
                    [--target_perplexity TARGET_PERPLEXITY]

Pretrain/Finetune a transformers model on a Masked Language Modeling task

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets
                        library).
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the
                        datasets library).
  --train_file TRAIN_FILE
                        A csv or a json file containing the training data.
  --validation_file VALIDATION_FILE
                        A csv or a json file containing the validation data.
  --validation_split_percentage VALIDATION_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set
                        in case there's no validation split
  --pad_to_max_length   If passed, pad all samples to `max_length`. Otherwise,
                        dynamic padding is used.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models.
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name
  --use_slow_tokenizer  If passed, will use a slow tokenizer (not backed by
                        the 🤗 Tokenizers library).
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size (per device) for the evaluation dataloader.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup
                        period) to use.
  --weight_decay WEIGHT_DECAY
                        Weight decay to use.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If
                        provided, overrides num_train_epochs.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.
  --num_warmup_steps NUM_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --output_dir OUTPUT_DIR
                        Where to store the final model.
  --seed SEED           A seed for reproducible training.
  --model_type MODEL_TYPE
                        Model type to use if training from scratch.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated.
  --line_by_line LINE_BY_LINE
                        Whether distinct lines of text in the dataset are to
                        be handled as distinct sequences. This is deafult for
                        bert/electra models and should be set to False for
                        gpt/gpt2 type models
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
  --overwrite_cache OVERWRITE_CACHE
                        Overwrite the cached training and evaluation sets
  --mlm_probability MLM_PROBABILITY
                        Ratio of tokens to mask for masked language modeling
                        loss
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Patience for early stopping to stop training if
                        val_acc doesnt converge
  --layer_drop_prob LAYER_DROP_PROB
                        Probability to drop layers
  --eval_random_subtransformers EVAL_RANDOM_SUBTRANSFORMERS
                        If set to 1, this will evaluate 25 random
                        subtransformers after every training epoch when
                        training a supertransformer
  --train_subtransformers_from_scratch TRAIN_SUBTRANSFORMERS_FROM_SCRATCH
                        If set to 1, this will train 25 random subtransformers
                        from scratch. By default, it is set to False (0) and
                        we train a supertransformer and finetune
                        subtransformers
  --fp16 FP16           If set to 1, will use FP16 training.
  --mixing {attention,gmlp,fnet,mobilebert,bert-bottleneck}
                        specifies how to mix the tokens in bertlayers
  --resume_from_checkpoint_dir RESUME_FROM_CHECKPOINT_DIR
                        directory that contains checkpoints, optimizer,
                        scheduler to resume training
  --tiny_attn TINY_ATTN
                        Choose this if you need Tiny Attention Module along-
                        with gMLP dense block
  --num_subtransformers_monitor NUM_SUBTRANSFORMERS_MONITOR
                        Choose the number of subtransformers whose performance
                        you wish to monitor
  --c4_dir C4_DIR       The directory path for C4
  --tokenized_c4_dir TOKENIZED_C4_DIR
                        The directory path for tokenized C4
  --sampling_type {none,naive_params,biased_params,random}
                        The sampling type for super-transformer
  --sampling_rule {none,sandwich}
                        The sampling rule for sampling super-transformers
  --pop_size POP_SIZE   Number of random subtransformers to sample at each
                        step
  --k_sampling K_SAMPLING
                        The step frequency of sampling a sub-transformers
  --inplace_distillation INPLACE_DISTILLATION
                        Whether to use inplace distillation
  --kd_ratio KD_RATIO   Sensitizes the amount of KD-loss that needs to be
                        added with existing loss
  --layerwise_distillation LAYERWISE_DISTILLATION
                        Conditional layerwise attention and feature map
                        transfer for in-place distillation
  --alpha_divergence ALPHA_DIVERGENCE
                        Enable Alpha Divergence KL loss
  --alpha_min ALPHA_MIN
                        Alpha min value
  --alpha_max ALPHA_MAX
                        Alpha max value
  --beta_clip BETA_CLIP
                        The clip value for alpha divergence
  --subtransformer_config_path SUBTRANSFORMER_CONFIG_PATH
                        The path to a subtransformer configration
  --rewire REWIRE       Whether to rewire model
  --rewired_model_checkpoint_dir REWIRED_MODEL_CHECKPOINT_DIR
                        Path to rewired model checkpoint
  --wandb_suffix WANDB_SUFFIX
                        suffix for wandb
  --target_perplexity TARGET_PERPLEXITY
                        perplexity to stop further pretraining
```
