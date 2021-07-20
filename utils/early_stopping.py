import numpy as np
import torch

# taken and modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        metric_to_track,
        higher_is_better=True,
        patience=5,
        delta=0,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.metric_to_track = metric_to_track
        self.higher_is_better = higher_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metric_dict):
        metric = metric_dict[self.metric_to_track]

        score = metric
        if not self.higher_is_better:
            score = -metric

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0