from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
from transformers import AutoTokenizer
from custom_layers.custom_bert import BertForSequenceClassification
import numpy as np

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

GLUE_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GlueTask:
    def __init__(
        self,
        task,
        tokenizer,
        max_seq_len,
    ):
        assert task in GLUE_TASKS, f"Task must be one of these: {GLUE_TASKS} "
        self.actual_task = "mnli" if task == "mnli-mm" else task
        dataset = load_dataset("glue", self.actual_task)
        self.metric = load_metric("glue", self.actual_task)
        self.max_seq_len = max_seq_len
        validation_key = (
            "validation_mismatched"
            if task == "mnli-mm"
            else "validation_matched"
            if task == "mnli"
            else "validation"
        )
        # metric_name = (
        #     "pearson"
        #     if task == "stsb"
        #     else "matthews_correlation"
        #     if task == "cola"
        #     else "accuracy"
        # )

        self.sentence1_key, self.sentence2_key = GLUE_task_to_keys[task]
        self.tokenizer = tokenizer

        encoded_dataset = dataset.map(self.preprocess_function, batched=True)
        encoded_dataset.rename_column_("label", "labels")
        # lets just return this.
        # setting columns to return is easier to setting cols to remove as
        # each task has diff col names
        columns_to_return = ["input_ids", "labels", "attention_mask"]
        encoded_dataset.set_format(type="torch", columns=columns_to_return)
        self.train_dataset = encoded_dataset["train"]
        self.eval_dataset = encoded_dataset[validation_key]

    def preprocess_function(self, examples):
        if self.sentence2_key is None:
            return self.tokenizer(
                examples[self.sentence1_key],
                truncation=True,
                max_length=self.max_seq_len,
            )
        return self.tokenizer(
            examples[self.sentence1_key],
            examples[self.sentence2_key],
            truncation=True,
            max_length=self.max_seq_len,
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if self.task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return self.metric.compute(predictions=predictions, references=labels)
