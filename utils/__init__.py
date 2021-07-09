# taken from https://github.com/deepset-ai/FARM/blob/master/farm/utils.py
import os
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn as nn



def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_path(path, error_message_template="Specified path - {} does not exist"):
    assert os.path.exists(path), error_message_template.format(path)


def get_current_datetime():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

def ce_soft(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

