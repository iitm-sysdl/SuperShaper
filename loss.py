import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def compute_layerwise_distillation(
    args,
    teacher_hidden_states,
    student_hidden_states,
    teacher_attention_maps,
    student_attention_maps,
    track_layerwise_loss=False,
):

    student_fkt = 0.0
    student_akt = 0.0
    if track_layerwise_loss:
        layer_wise_fkt = []
        layer_wise_akt = []
    else:
        layer_wise_fkt = None
        layer_wise_akt = None

    non_trainable_layernorm = nn.LayerNorm(
        teacher_hidden_states[-1].shape[1:], elementwise_affine=False
    )
    for teacher_hidden, student_hidden in zip(
        teacher_hidden_states, student_hidden_states
    ):
        teacher_hidden = non_trainable_layernorm(teacher_hidden.detach())
        student_hidden = non_trainable_layernorm(student_hidden)
        fkt = nn.MSELoss()(teacher_hidden, student_hidden)
        student_fkt = student_fkt + fkt
        if track_layerwise_loss:
            layer_wise_fkt.append(fkt)
    # the attention maps already have softmax applied, hence we pass logits = False
    loss_alpha_div = AdaptiveLossSoft(
        args.alpha_min, args.alpha_max, args.beta_clip, logits=False
    )

    for (teacher_attention, student_attention) in zip(
        teacher_attention_maps, student_attention_maps
    ):
        # attentions are already in probabilities, hence no softmax
        if args.alpha_divergence:
            # TODO - Check if the reduction is mean or sum
            student_akt = loss_alpha_div(teacher_attention, student_attention)
        else:
            student_attention = student_attention.clamp(min=1e-4).log()
            student_kl = -(teacher_attention.detach() * student_attention)
            akt = torch.mean(torch.sum(student_kl, axis=-1))
            student_akt = student_akt + akt

        if track_layerwise_loss:
            layer_wise_akt.append(akt)

    return student_akt, student_fkt, layer_wise_akt, layer_wise_fkt


def compute_student_loss(
    outputs,
    teacher_hidden_states,
    teacher_attention_maps,
    args,
    track_layerwise_loss=False,
):

    # outputs = model(**batch, use_soft_loss=True)
    loss = outputs.loss
    student_hidden_states = outputs.hidden_states
    student_attention_maps = outputs.attentions

    student_mlm_loss = loss
    student_mlm_loss = student_mlm_loss / args.gradient_accumulation_steps

    overall_loss = student_mlm_loss

    losses = {
        "overall_loss": overall_loss,
        "student_distill_loss": 0,
        "student_mlm_loss": student_mlm_loss,
        "student_feature_knowledge_transfer_loss": 0,
        "student_attention_knowledge_transfer_loss": 0,
        "layer_wise_akt": [],
        "layer_wise_fkt": [],
    }

    if args.layerwise_distillation:
        (
            student_akt,
            student_fkt,
            layer_wise_akt,
            layer_wise_fkt,
        ) = compute_layerwise_distillation(
            # the official mobilbeBert repo skips the first layer
            # teacher_hidden_states[1:],
            # student_hidden_states[1:],
            # teacher_attention_maps[1:],
            # student_attention_maps[1:],
            args,
            teacher_hidden_states,
            student_hidden_states,
            teacher_attention_maps,
            student_attention_maps,
            track_layerwise_loss=track_layerwise_loss,
        )

        student_distill_loss = 0.5 * student_fkt + 0.5 * student_akt
        student_distill_loss = student_distill_loss / args.gradient_accumulation_steps

        overall_loss = overall_loss + student_distill_loss

        losses["overall_loss"] = overall_loss
        losses["student_distill_loss"] = student_distill_loss
        losses["student_feature_knowledge_transfer_loss"] = student_fkt
        losses["student_attention_knowledge_transfer_loss"] = student_akt
        losses["layer_wise_akt"] = layer_wise_akt
        losses["layer_wise_fkt"] = layer_wise_fkt

    return overall_loss, losses


## Alpha Divergence loss codes adapted from https://github.com/facebookresearch/AlphaNet ##
def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3, logits=True):
    assert isinstance(alpha, float)
    if logits:
        q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
        p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
        q_log_prob = torch.nn.functional.log_softmax(
            q_logits, dim=1
        )  # gradient is only backpropagated here
    else:
        q_prob = q_logits.detach()
        p_prob = p_logits.detach()
        p_prob = p_prob.view(p_prob.shape[0], -1)  ### Getting the correct view
        q_log_prob = q_logits.log()

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


"""
It's often necessary to clip the maximum
gradient value (e.g., 1.0) when using this adaptive KD loss
"""


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0, logits=True):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip
        self.logits = logits

    def forward(self, output, target, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(
            output, target, alpha_min, iw_clip=self.iw_clip, logits=self.logits
        )
        loss_right, grad_loss_right = f_divergence(
            output, target, alpha_max, iw_clip=self.iw_clip, logits=self.logits
        )

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        # reduction is mean by default https://pytorch-enhance.readthedocs.io/en/latest/_modules/torch/nn/modules/loss.html
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def ce_soft(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


#  https://github.com/pytorch/pytorch/issues/11959
class CrossEntropyLossSoft(_Loss):
    def forward(self, preds, target_logits, reduction="mean"):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(
            preds.view(preds.shape[0], -1), dim=1
        )
        target = torch.nn.functional.softmax(
            target_logits.view(target_logits.shape[0], -1).detach(), dim=1
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
