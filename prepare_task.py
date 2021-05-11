from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
from transformers import AutoTokenizer
from custom_bert import BertForSequenceClassification
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


def show_random_elements(dataset, accelerator, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    accelerator.print(df)


class GlueTask:
    def __init__(
        self, task, model_checkpoint, model_config, accelerator, initialize_pretrained_model=True
    ):
        assert task in GLUE_TASKS, f"Task must be one of these: {GLUE_TASKS} "
        self.actual_task = "mnli" if task == "mnli-mm" else task
        dataset = load_dataset("glue", self.actual_task)
        accelerator.print(f"{task} dataset statistics:")
        accelerator.print(dataset)
        accelerator.print(f"Random samples from {task} dataset")
        show_random_elements(dataset["train"], accelerator, 5)
        self.metric = load_metric("glue", self.actual_task)
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
        self.config = model_config

        self.sentence1_key, self.sentence2_key = GLUE_task_to_keys[task]
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        encoded_dataset = dataset.map(self.preprocess_function, batched=True)
        encoded_dataset.rename_column_("label", "labels")
        # lets just return this.
        # setting columns to return is easier to setting cols to remove as
        # each task has diff col names
        columns_to_return = ["input_ids", "labels", "attention_mask"]
        encoded_dataset.set_format(type="torch", columns=columns_to_return)
        self.train_dataset = encoded_dataset["train"]
        self.eval_dataset = encoded_dataset[validation_key]
        self.num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

        if initialize_pretrained_model:
            self.model = BertForSequenceClassification.from_pretrained(
                model_checkpoint, num_labels=self.num_labels
            )
        else:
            model_config.num_labels = self.num_labels
            self.model = BertForSequenceClassification(config=model_config)

    def preprocess_function(self, examples):
        if self.sentence2_key is None:
            return self.tokenizer(
                examples[self.sentence1_key], truncation=True, max_length=None
            )
        return self.tokenizer(
            examples[self.sentence1_key],
            examples[self.sentence2_key],
            truncation=True,
            max_length=None,
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if self.task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return self.metric.compute(predictions=predictions, references=labels)
