# Super Pre-training

This repository contains our PyTorch training code, evaluation code and pre-trained models for Super-Pretraining. 

If you find this repo useful in your work, please consider citing our work: 
TODO: Add citation when ready. 


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
usage: train_mlm.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                [--dataset_name DATASET_NAME]
                [--dataset_config_name DATASET_CONFIG_NAME] 
                [--train_file TRAIN_FILE]
                [--validation_file VAL_FILE] 
                [--validation_split_percentage VAL_SPLIT]
                [--pad_to_max_length]
                [--config_name CONFIG_NAME]
                [--per_device_train_batch_size PER_DEV_TRAIN_BATCH_SIZE]
                [--per_device_eval_batch_size PER_DEV_EVAL_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] 
                [--num_train_epochs NUM_TRAIN_EPOCHS][--max_train_steps MAX_TRAIN_STEPS]
                [--gradient_accumulation_steps GRAD_ACC_STEPS]
                [--lr_scheduler_type LR_SCHED_TYPE]
                [--num_warmup_steps NUM_WARMUP_STEPS]
                [--output_dir OUT_DIR] [--max_seq_length MAX_SEQ_LENGTH]
                [--seed SEED] [--logging_steps LOG] [--preprocessing_num_workers PRE]
                [--overwrite_cache OVERWRITE] [--mlm_probability MLM_PROB]
                [--early_stopping_patience PATIENCE] [--eval_random_subtransformers EVAL]
                [--num_epochs NUM_EPOCHS] [--fp16 FP16] [--cpu CPU]
                [--mixing MIXING] [--resume_from_checkpoint_dir RESUME] 
                [--num_subtransformers_monitor MONITOR] [--c4_dir C4_DIR]
                [--sampling_type SAMPLING_TYPE] [--sampling_rule SAMPLING_RULE]
                [--pop_size POP_SIZE] [--k_sampling KVAL] [--inplace_distillation IKD]
                [--kd_ratio KD_RATIO] [--layerwise_distillation LAYERWISE]
                [--alpha_divergence ADIV] [--alpha_min MIN] [--alpha_max MAX] [--beta_clip CLIP]

Script to train Super-Pretraining Models

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to model checkpoint or name of hf pretrained
                        model
  --output_dir OUTPUT_DIR
                        The output directory where the model checkpoints and
                        predictions will be written.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --per_device_train_batch_size PER_device_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU/CPU for training.
  --per_device_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU/CPU for evaluation.
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Patience for early stopping to stop training if
                        val_acc doesnt converge
  --limit_subtransformer_choices LIMIT_SUBTRANSFORMER_CHOICES
                        If set to 1, it will limit the hidden_size and number
                        of encoder layers of the subtransformer choices
  --eval_random_subtransformers EVAL_RANDOM_SUBTRANSFORMERS
                        If set to 1, this will evaluate 25 random
                        subtransformers after every training epoch when
                        training a supertransformer
  --train_subtransformers_from_scratch TRAIN_SUBTRANSFORMERS_FROM_SCRATCH
                        If set to 1, this will train 25 random subtransformers
                        from scratch. By default, it is set to False (0) and
                        we train a supertransformer and finetune
                        subtransformers
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --num_epochs NUM_EPOCHS
                        Total number of training epochs to perform.
  --fp16 FP16           If set to 1, will use FP16 training.
  --cpu CPU             If set to 1, will train on the CPU.
  --seed SEED           random seed for initialization
```
