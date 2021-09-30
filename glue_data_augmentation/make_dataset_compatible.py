import pandas as pd
import csv
import argparse

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# this label list is inline with hf datasets https://huggingface.co/datasets/viewer/?dataset=glue
label_list_for_aug_data = {
    "cola": {"unacceptable": 0, "acceptable": 1},
    "sst2": {"negative": 0, "positive": 1},
    "mrpc": {"not_equivalent": 0, "equivalent": 1},
    "qqp": {"not_duplicate": 0, "duplicate": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "rte": {"entailment": 0, "not_entailment": 1},
    "wnli": {"not_entailment": 0, "entailment": 1},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to a TSV file.")
    parser.add_argument("--path", help="Path to folder")
    parser.add_argument("--task", help="Glue task name")
    args = parser.parse_args()

    df = pd.read_csv(
        args.path + "/" + args.task + "/train_aug.tsv",
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    keys = task_to_keys[args.task.lower().replace("-", "")]
    labels_map = label_list_for_aug_data[args.task.lower().replace("-", "")]
    print(df.head())
    if args.task != "STS-B" and list(df["label"].unique())[0] in labels_map.keys():
        print("mapping labels ...")
        df.replace({"label": labels_map}, inplace=True)
    print(df.head())
    rows = []
    rows.append(keys[0])
    if keys[1] is not None:
        rows.append(keys[1])
    rows.append("label")
    print("Shape before dropping nans:", df.shape)
    df.dropna(subset=rows, inplace=True)
    print("Shape after dropping nans:", df.shape)
    # drop duplicates in subset
    #df.drop_duplicates(subset=rows, inplace=True)
    #print("Shape after dropping duplicates:", df.shape)
    print("Saving final dataset .....")
    df.to_csv(args.path + "/" + args.task + f"/{args.task}_train_aug_mod.tsv", sep="\t", index=False)
