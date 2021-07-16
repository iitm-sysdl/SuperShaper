# taken from https://github.com/deepset-ai/FARM/blob/master/farm/utils.py
import os
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from collections import OrderedDict


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


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_path(path, error_message_template="Specified path - {} does not exist"):
    assert os.path.exists(path), error_message_template.format(path)


def get_current_datetime():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


def ce_soft(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


#  https://github.com/pytorch/pytorch/issues/11959
class CrossEntropyLossSoft(_Loss):
    def forward(self, preds, target, reduction="mean"):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(
            preds.view(preds.shape[0], -1), dim=1
        )
        batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == "none":
            return batchloss
        elif reduction == "mean":
            return torch.mean(batchloss)
        elif reduction == "sum":
            return torch.sum(batchloss)
        else:
            raise NotImplementedError("Unsupported reduction mode.")
