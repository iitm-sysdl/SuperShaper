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


# def unique_everseen(iterable, key=None):
#     "List unique elements, preserving order. Remember all elements ever seen."
#     # unique_everseen('AAAABBBCCDAABBB') --> A B C D
#     # unique_everseen('ABBCcAD', str.lower) --> A B C D
#     seen = set()
#     seen_add = seen.add
#     if key is None:
#         for element in filterfalse(seen.__contains__, iterable):
#             seen_add(element)
#             yield element
#     else:
#         for element in iterable:
#             k = key(element)
#             if k not in seen:
#                 seen_add(k)
#                 yield element


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# verify with this https://github.com/google-research/bert/issues/656
def calculate_params(
    emb_dim,
    num_attention_heads,
    d_ff_list,
    num_enc,
    vocab_size=28996,
    add_output_emb_layer=True,
    add_embs_dim=True,
):

    layer_norm_params = 2 * emb_dim

    if add_embs_dim:
        # vocab_size + position_embeddings + token_type_embeddings
        emb_params = (vocab_size + 512 + 2) * emb_dim + layer_norm_params
    else:
        # for scaling laws dont add emb_dim and output emb dimension
        emb_params = 0
        add_output_emb_layer = False

    assert len(d_ff_list) == num_enc
    per_layer_params = 0

    for d_ff in d_ff_list:

        per_layer_params += (
            4
            * (
                (emb_dim * emb_dim) + emb_dim
            )  # q, k,v and fc layer and their biases in bertattention
            + 2 * (emb_dim * d_ff)  # intermediate and bertouput
            + (d_ff + emb_dim)  # intermediate and bertoutput bias terms
            + 2 * layer_norm_params  # layernorms
        )
    # BertPredictionHeadTransform parameters
    output_emb_params = (emb_dim * emb_dim) + emb_dim + layer_norm_params

    if add_output_emb_layer:
        output_emb_layer = vocab_size * emb_dim + vocab_size
    else:
        output_emb_layer = 0

    return emb_params + per_layer_params + output_emb_params + output_emb_layer


def calculate_params_from_config(config, scaling_laws=False):
    add_embs_dim = scaling_laws != True

    return calculate_params(
        config.sample_hidden_size,
        config.sample_num_attention_heads,
        config.sample_intermediate_size,
        config.sample_num_hidden_layers,
        config.vocab_size,
        add_output_emb_layer=True,
        add_embs_dim=add_embs_dim,
    )


def check_path(path, error_message_template="Specified path - {} does not exist"):
    assert os.path.exists(path), error_message_template.format(path)


def get_current_datetime():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
