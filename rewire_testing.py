import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from transformers import (
    AdamW,
)

from torch.nn import MSELoss as mse
from pprint import pprint


class BackHook:
    def __init__(self):
        self.grad_input = {}
        self.grad_output = {}
        self.layer_num = 0

    def __call__(self, module, grad_in, grad_out):

        print(module)

        print(grad_out)

        grad_out = torch.abs(grad_out[0])
        if not hasattr(module, "name"):
            setattr(module, "name", self.layer_num)
            print("updating")
            self.grad_output[self.layer_num] = grad_out
            self.layer_num += 1
        else:
            print("calculating mean")
            # take mean along batch dimension
            layer_num = getattr(module, "name")
            self.grad_output[layer_num] = torch.mean(
                torch.stack([self.grad_output[layer_num], grad_out]), dim=0
            )

    def print(self):
        print(len(self.grad_output))
        pprint(self.grad_output)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


bhookfn = BackHook()
seed_everything(101)


def assert_array_equal(tens_a, tens_b):
    # check if they are fully equal
    print("Fully Equal: ", torch.all(tens_a.eq(tens_b)))
    print("Outputs without permuting weights: ")
    print(tens_a.detach().numpy())
    print("Outputs with permuting weights:")
    print(tens_b.detach().numpy())
    # check if they are close to each other upto tolerance of 1e-08
    assert torch.all(torch.isclose(tens_a, tens_b))


def inverse_permutation(permutation_order):
    inv = torch.empty_like(permutation_order)
    inv[permutation_order] = torch.arange(
        permutation_order.size(0), device=permutation_order.device
    )
    return inv


def permute_linear(W, permutation, dim="col", permute_weight=False, permute_bias=False):
    """
    Permute linear layer

    :param W: weight matrix
    :param permutation: permutation order for the weights
    :param dim: 'row' or 'col'
    :param permute_bias: whether to permute the bias

    """
    _W = deepcopy(W)
    if permute_bias:
        # print("Bias: ")
        # print(_W.bias.shape)
        # print(permutation.shape)
        # print(dim)
        _W.bias.data.copy_(_W.bias[permutation])

    if permute_weight:
        # print("Weights: ")
        # print(_W.weight.shape)
        # print(permutation.shape)
        # print(dim)
        if dim == "col":
            # permute columns
            _W.weight.data.copy_(_W.weight[:, permutation])
        elif dim == "row":
            # permute rows
            _W.weight.data.copy_(_W.weight[permutation, :])
        else:
            raise NotImplementedError

    return _W


def get_permutation_order(dim):
    # get random permutation order
    # this is used for testing
    return torch.randperm(dim)


def transformer_layer(
    _input, W_att, W_o, W_1, W_2, permute=False, residual=True, return_all_outputs=True
):
    """
    W_* -> linear layer with weights and bias

    :param _input: input of size batch, emb_dim
    :param W_att: linear layer of weight: (emb_dim, attn_dim), bias: (attn_dim,)
    :param W_o: linear layer of weight: (attn_dim, emb_dim), bias: (emb_dim,)
    :param W_1: linear layer of weight: (emb_dim, intermediate_dim), bias: (intermediate_dim,)
    :param W_2: linear layer of weight: (intermediate_dim, emb_dim), bias: (emb_dim,)
    :param permute: Whether to permute weights
    :param residual: whether to use residual connection

    """

    final_outputs = {
        "output": None,
        "per_layer_outputs": None,
        "permutation_orders": None,
    }

    if permute:
        _, emb_dim = _input.size()

        permutation_order_1 = get_permutation_order(emb_dim)
        # permuting according to embedding dim
        # in our code with embeddings matrix, we will permute the emb matrix and not the actual inputs (like we do here)
        _input = _input[:, permutation_order_1]

        # print("permuting W_att")
        W_att = permute_linear(
            W_att,
            permutation_order_1,
            dim="col",
            permute_weight=True,
            permute_bias=False,
        )

        # since emb_dim is destroyed in prev layer, we recompute the permutation order here
        permutation_order_2 = get_permutation_order(emb_dim)
        # print("permuting W_o")
        W_o = permute_linear(
            W_o, permutation_order_2, dim="row", permute_weight=True, permute_bias=True
        )
        # print("permuting W_1")
        W_1 = permute_linear(
            W_1, permutation_order_2, dim="col", permute_weight=True, permute_bias=False
        )

        # since emb_dim is destroyed in prev layer, we recompute the permutation order here
        permutation_order_3 = get_permutation_order(emb_dim)
        # print("permuting W_2")
        W_2 = permute_linear(
            W_2, permutation_order_3, dim="row", permute_weight=True, permute_bias=True
        )

    outputs = []

    out1 = W_att(_input)
    out2 = W_o(out1)

    outputs.append(out1)
    outputs.append(out2)

    if residual:
        if permute:
            inv_permutation_order_1 = inverse_permutation(permutation_order_1)
            _input = _input[:, inv_permutation_order_1]
            _input = _input[:, permutation_order_2]
        out2 = _input + out2
        outputs.append(out2)

    out3 = W_1(out2)
    out4 = W_2(out3)

    outputs.append(out3)
    outputs.append(out4)

    if residual:
        if permute:
            inv_permutation_order_2 = inverse_permutation(permutation_order_2)
            out2 = out2[:, inv_permutation_order_2]
            out2 = out2[:, permutation_order_3]
        out4 = out4 + out2
        outputs.append(out4)

    if permute:
        inv_permutation_order_3 = inverse_permutation(permutation_order_3)
        out4 = out4[:, inv_permutation_order_3]

    if return_all_outputs:
        final_outputs["per_layer_outputs"] = outputs
        if permute:
            permutation_orders = [
                permutation_order_1,
                permutation_order_2,
                None,
                permutation_order_3,
            ]
            if residual:
                permutation_orders = [
                    permutation_order_1,
                    permutation_order_2,
                    None,
                    None,
                    permutation_order_3,
                    None,
                ]

            final_outputs["permutation_orders"] = permutation_orders

    final_outputs["output"] = out4

    return final_outputs


if __name__ == "__main__":
    lr = 5e-5
    batch_size = 1
    emb_dim = 5

    attn_dim = 6
    intermediate_dim = 7

    output_size = 10

    n_layers = 1
    n_epochs = 2

    loss = mse(reduction="mean")

    parameters = []
    W_atts = []
    W_os = []
    W_1s = []
    W_2s = []

    for _ in range(n_layers):
        W_att = nn.Linear(emb_dim, attn_dim)
        W_o = nn.Linear(attn_dim, emb_dim)
        W_1 = nn.Linear(emb_dim, intermediate_dim)
        W_2 = nn.Linear(intermediate_dim, emb_dim)

        W_att.register_backward_hook(bhookfn)
        W_o.register_backward_hook(bhookfn)
        W_1.register_backward_hook(bhookfn)
        W_2.register_backward_hook(bhookfn)

        W_atts.append(W_att)
        W_os.append(W_o)
        W_1s.append(W_1)
        W_2s.append(W_2)

        parameters += W_att.parameters()
        parameters += W_o.parameters()
        parameters += W_1.parameters()
        parameters += W_2.parameters()

    W_output = nn.Linear(emb_dim, output_size)
    W_output.register_backward_hook(bhookfn)
    parameters += W_output.parameters()

    optimizer = AdamW(parameters, lr=lr)

    for epoch in range(n_epochs):
        _input = torch.randn(batch_size, emb_dim, requires_grad=False)
        labels = torch.randn((batch_size, output_size))
        prev_layer_output1 = _input
        prev_layer_output2 = _input
        for idx in range(n_layers):

            W_att, W_o, W_1, W_2 = W_atts[idx], W_os[idx], W_1s[idx], W_2s[idx]

            # W_att.zero_grad()
            # W_o.zero_grad()
            # W_1.zero_grad()
            # W_2.zero_grad()

            # exp1 = transformer_layer(
            #     prev_layer_output1, W_att, W_o, W_1, W_2, permute=False, residual=False
            # )
            exp2 = transformer_layer(
                prev_layer_output2,
                W_att,
                W_o,
                W_1,
                W_2,
                permute=False,
                residual=False,
            )

            # prev_layer_output1 = exp1["output"]
            prev_layer_output2 = exp2["output"]

        logits = W_output(prev_layer_output2)
        loss_val = loss(logits, labels)
        loss_val.backward()

        bhookfn.print()

    strings = ["W_att_output: ", "W_o_output: ", "W_1_output: ", "W_2_output: "]

    # for _string, exp1_out, exp2_out, permutation_order in zip(
    #     strings,
    #     exp1["per_layer_outputs"],
    #     exp2["per_layer_outputs"],
    #     exp2["permutation_orders"],
    # ):

    #     print(_string)
    #     print(exp1_out)
    #     print(exp2_out)
    #     print(permutation_order)
    #     print()
    print("Testing without residual connections: ")
    # assert_array_equal(exp1["output"], exp2["output"])

    # print(_input)

    # prev_layer_output1 = _input
    # prev_layer_output2 = _input
    # for _ in range(n_layers):
    #     W_att = nn.Linear(emb_dim, attn_dim)
    #     W_o = nn.Linear(attn_dim, emb_dim)
    #     W_1 = nn.Linear(emb_dim, intermediate_dim)
    #     W_2 = nn.Linear(intermediate_dim, emb_dim)
    #     with torch.no_grad():
    #         exp1 = transformer_layer(
    #             _input, W_att, W_o, W_1, W_2, permute=False, residual=True
    #         )
    #     exp2 = transformer_layer(
    #         _input, W_att, W_o, W_1, W_2, permute=True, residual=True
    #     )
    #     logits = W_output(exp1["output"])

    #     prev_layer_output1 = exp1["output"]
    #     prev_layer_output2 = exp2["output"]

    # # strings = [
    # #     "W_att_output: ",
    # #     "W_o_output: ",
    # #     "Residual1: ",
    # #     "W_1_output: ",
    # #     "W_2_output: ",
    # #     "Residual2: ",
    # # ]
    # # for _string, exp1_out, exp2_out, permutation_order in zip(
    # #     strings,
    # #     exp1["per_layer_outputs"],
    # #     exp2["per_layer_outputs"],
    # #     exp2["permutation_orders"],
    # # ):

    # #     print(_string)
    # #     print(exp1_out)
    # #     print(exp2_out)
    # #     print(permutation_order)
    # #     print()
    # print()
    # print("Testing with residual connections: ")
    # assert_array_equal(exp1["output"], exp2["output"])