# eHAT: Efficient-Hardware-Aware-Transformers




## Usage

### Configure your training setup
```
accelerate config         # answer questions wrt your training setup (multi-gpu, tpu, fp16 etc)

accelerate config  --config_file <path to config>       # to create custom training setup for differnet tasks
```

### Run the code with accelerate launch

```
accelerate launch train.py <args>
```

### List of arguments for `train.py`:

```
usage: train.py [-h] [--task TASK] [--model_name_or_path MODEL_NAME_OR_PATH]
                [--use_pretrained_supertransformer USE_PRETRAINED_SUPERTRANSFORMER]
                [--output_dir OUTPUT_DIR] [--max_seq_length MAX_SEQ_LENGTH]
                [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                [--limit_subtransformer_choices LIMIT_SUBTRANSFORMER_CHOICES]
                [--eval_random_subtransformers EVAL_RANDOM_SUBTRANSFORMERS]
                [--train_subtransformers_from_scratch TRAIN_SUBTRANSFORMERS_FROM_SCRATCH]
                [--learning_rate LEARNING_RATE]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--num_epochs NUM_EPOCHS] [--fp16 FP16] [--cpu CPU]
                [--seed SEED]

Script to train efficient HAT models

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           The Glue task you want to run
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to model checkpoint or name of hf pretrained
                        model
  --use_pretrained_supertransformer USE_PRETRAINED_SUPERTRANSFORMER
                        If passed and set to True, will use pretrained bert-
                        uncased model. If set to False, it will initialize a
                        random model and train from scratch
  --output_dir OUTPUT_DIR
                        The output directory where the model checkpoints and
                        predictions will be written.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
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