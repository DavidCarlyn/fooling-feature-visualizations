import math
import numbers
from itertools import product
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from adv_lib.utils.projections import clamp_
from adv_lib.utils.visdom_logger import VisdomLogger

# Modified from https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/attacks/decoupled_direction_norm.py
def ddn(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        targeted: bool = False,
        steps: int = 100,
        γ: float = 0.05,
        init_norm: float = 1.,
        levels: Optional[int] = 256,
        callback: Optional[VisdomLogger] = None,
        reset_module_counts = None) -> Tensor:
    """
    Decoupled Direction and Norm attack from https://arxiv.org/abs/1811.09600.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    targeted : bool
        Whether to perform a targeted attack or not.
    steps : int
        Number of optimization steps.
    γ : float
        Factor by which the norm will be modified. new_norm = norm * (1 + or - γ).
    init_norm : float
        Initial value for the norm of the attack.
    levels : int
        If not None, the returned adversarials will have quantized values to the specified number of levels.
    callback : Optional

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

    # Init variables
    multiplier = -1 if targeted else 1
    δ = torch.zeros_like(inputs, requires_grad=True)
    ε = torch.full((batch_size,), init_norm, device=device, dtype=torch.float)
    worst_norm = torch.max(inputs, 1 - inputs).flatten(1).norm(p=2, dim=1)

    # Init trackers
    best_l2 = worst_norm.clone()
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(steps):
        l2 = δ.data.flatten(1).norm(p=2, dim=1)
        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        reset_module_counts()
        pred_labels = logits.argmax(1)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        loss = multiplier * ce_loss

        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = l2 < best_l2
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_both, l2, best_l2)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]
        # renorming gradient
        grad_norms = δ_grad.flatten(1).norm(p=2, dim=1)
        δ_grad.div_(batch_view(grad_norms))
        # avoid nan or inf if gradient is 0
        if (zero_grad := (grad_norms < 1e-12)).any():
            δ_grad[zero_grad] = torch.randn_like(δ_grad[zero_grad])

        α = 0.01 + (1 - 0.01) * (1 + math.cos(math.pi * i / steps)) / 2

        if callback is not None:
            cosine = F.cosine_similarity(δ_grad.flatten(1), δ.data.flatten(1), dim=1).mean()
            callback.accumulate_line('ce', i, ce_loss.mean())
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['ε', 'l2', 'best_l2'], i, [ε.mean(), l2.mean(), callback_best])
            callback.accumulate_line(['cosine', 'α', 'success'], i,
                                     [cosine, torch.tensor(α, device=device), adv_found.float().mean()])

            if (i + 1) % (steps // 20) == 0 or (i + 1) == steps:
                callback.update_lines()

        δ.data.add_(δ_grad, alpha=α)

        ε = torch.where(is_adv, (1 - γ) * ε, (1 + γ) * ε)
        ε = torch.minimum(ε, worst_norm)

        δ.data.mul_(batch_view(ε / δ.data.flatten(1).norm(p=2, dim=1)))
        δ.data.add_(inputs).clamp_(min=0, max=1)
        if levels is not None:
            δ.data.mul_(levels - 1).round_().div_(levels - 1)
        δ.data.sub_(inputs)

    return best_adv


def pgd_linf(model: nn.Module,
             inputs: Tensor,
             labels: Tensor,
             ε: Union[float, Tensor],
             targeted: bool = False,
             steps: int = 40,
             random_init: bool = True,
             restarts: int = 1,
             loss_function: str = 'ce',
             relative_step_size: float = 0.01 / 0.3,
             absolute_step_size: Optional[float] = None,
             callback: Optional[VisdomLogger] = None,
             reset_module_counts = None) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if isinstance(ε, numbers.Real):
        ε = torch.full_like(adv_found, ε, dtype=inputs.dtype)

    pgd_attack = partial(_pgd_linf, model=model, targeted=targeted, steps=steps, random_init=random_init,
                         loss_function=loss_function, relative_step_size=relative_step_size,
                         absolute_step_size=absolute_step_size, reset_module_counts=reset_module_counts)

    for i in range(restarts):

        adv_found_run, adv_inputs_run = pgd_attack(inputs=inputs[~adv_found], labels=labels[~adv_found],
                                                   ε=ε[~adv_found])
        adv_inputs[~adv_found] = adv_inputs_run
        adv_found[~adv_found] = adv_found_run

        if callback:
            callback.line('success', i + 1, adv_found.float().mean())

        if adv_found.all():
            break

    return adv_inputs


def _pgd_linf(model: nn.Module,
              inputs: Tensor,
              labels: Tensor,
              ε: Tensor,
              targeted: bool = False,
              steps: int = 40,
              random_init: bool = True,
              loss_function: str = 'ce',
              relative_step_size: float = 0.01 / 0.3,
              absolute_step_size: Optional[float] = None,
             reset_module_counts = None) -> Tuple[Tensor, Tensor]:
    _loss_functions = {
        'ce': (partial(F.cross_entropy, reduction='none'), 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
    }

    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    lower, upper = torch.maximum(-inputs, -batch_view(ε)), torch.minimum(1 - inputs, batch_view(ε))

    loss_func, multiplier = _loss_functions[loss_function.lower()]

    if absolute_step_size is not None:
        step_size = absolute_step_size
    else:
        step_size = ε * relative_step_size

    if targeted:
        step_size *= -1

    δ = torch.zeros_like(inputs, requires_grad=True)
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if random_init:
        δ.data.uniform_(-1, 1).mul_(batch_view(ε))
        clamp_(δ, lower=lower, upper=upper)

    for i in range(steps):
        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        reset_module_counts()

        if i == 0 and loss_function.lower() in ['dl', 'dlr']:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, labels)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0].sign_().mul_(batch_view(step_size))

        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        best_adv = torch.where(batch_view(is_adv), adv_inputs.detach(), best_adv)
        adv_found.logical_or_(is_adv)

        δ.data.add_(δ_grad)
        clamp_(δ, lower=lower, upper=upper)

    return adv_found, best_adv

def str_attack(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               targeted: bool = False,
               confidence: float = 0,
               initial_const: float = 1,
               binary_search_steps: int = 6,
               max_iterations: int = 2000,
               ρ: float = 1,
               α: float = 5,
               τ: float = 2,
               γ: float = 1,
               group_size: int = 2,
               stride: int = 2,
               retrain: bool = True,
               σ: float = 3,
               fix_y_step: bool = False,
               callback: Optional[VisdomLogger] = None,
               reset_module_counts = None) -> Tensor:
    """
    StrAttack from https://arxiv.org/abs/1808.01664.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    targeted : bool
        Whether to perform a targeted attack or not.
    confidence : float
        Confidence of adversarial examples: higher produces examples that are farther away, but more strongly classified
        as adversarial.
    initial_const : float
        The initial tradeoff-constant to use to tune the relative importance of distance and confidence. If
        binary_search_steps is large, the initial constant is not important.
    binary_search_steps : int
        The number of times we perform binary search to find the optimal tradeoff-constant between distance and
        confidence.
    max_iterations : int
        The maximum number of iterations. Larger values are more accurate; setting too small will require a large
        learning rate and will produce poor results.
    ρ : float
        Penalty parameter adjusting the trade-off between convergence speed and value. Larger ρ leads to faster
        convergence but larger perturbations.
    α : float
        Initial learning rate (η_1 in section F of the paper).
    τ : float
        Weight of the group sparsity penalty.
    γ : float
        Weight of the l2-norm penalty.
    group_size : int
        Size of the groups for the sparsity penalty.
    stride : int
        Stride of the groups for the sparsity penalty. If stride < group_size, then the groups are overlapping.
    retrain : bool
        If True, refines the perturbation by constraining it to σ% of the pixels.
    σ : float
        Percentage of pixels allowed to be perturbed in the refinement procedure.
    fix_y_step : bool
        Fix the group folding of the original implementation by correctly summing over groups for pixels in the
        overlapping regions. Typically results in smaller perturbations and is faster.
    callback : Optional

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    device = inputs.device
    batch_size, C, H, W = inputs.shape
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    zeros = torch.zeros_like(inputs)

    # set the lower and upper bounds accordingly
    const = torch.full((batch_size,), initial_const, device=device, dtype=torch.float)
    lower_bound = torch.zeros_like(const)
    upper_bound = torch.full_like(const, 1e10)

    # bounds for the perturbations to get valid inputs
    sup_bound = 1 - inputs
    inf_bound = -inputs

    # number of groups per row and column
    P, Q = math.floor((W - group_size) / stride) + 1, math.floor((H - group_size) / stride) + 1
    overlap = group_size > stride

    z = torch.zeros_like(inputs)
    v = torch.zeros_like(inputs)
    u = torch.zeros_like(inputs)
    s = torch.zeros_like(inputs)

    o_best_l2 = torch.full_like(const, float('inf'))
    o_best_adv = inputs.clone()
    o_adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    i_total = 0
    for outer_step in range(binary_search_steps):

        best_l2 = torch.full_like(const, float('inf'))
        adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

        # The last iteration (if we run many steps) repeat the search once.
        if (binary_search_steps >= 10) and outer_step == (binary_search_steps - 1):
            const = upper_bound

        for i in range(max_iterations):  # max_iterations + outer_step * 1000 in the original implementation

            z.requires_grad_(True)
            adv_inputs = z + inputs
            logits = model(adv_inputs)
            reset_module_counts()

            if outer_step == 0 and i == 0:
                # setup the target variable, we need it to be in one-hot form for the loss function
                labels_onehot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
                labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

            logit_dists = multiplier * difference_of_logits(logits, labels, labels_infhot=labels_infhot)
            loss = const * (logit_dists + confidence).clamp(min=0)
            z_grad = grad(loss.sum(), inputs=z, only_inputs=True)[0]
            z.detach_()

            # δ step (equation 10)
            a = z - u
            δ = ρ / (ρ + 2 * γ) * a

            # w step (equation 11)
            b = z - s
            w = torch.minimum(torch.maximum(b, inf_bound), sup_bound)

            # y step (equation 17)
            c = z - v
            groups = F.unfold(c, kernel_size=group_size, stride=stride)
            group_norms = groups.norm(dim=1, p=2, keepdim=True)
            temp = torch.where(group_norms != 0, 1 - τ / (ρ * group_norms), torch.zeros_like(group_norms))
            temp_ = groups * temp.clamp(min=0)

            if overlap and not fix_y_step:  # match original implementation when overlapping
                y = c
                for i, (p, q) in enumerate(product(range(P), range(Q))):
                    p_start, p_end = p * stride, p * stride + group_size
                    q_start, q_end = q * stride, q * stride + group_size
                    y[:, :, p_start:p_end, q_start:q_end] = temp_[:, :, i].view(-1, C, group_size, group_size)
            else:  # faster folding (matches original implementation when groups are not overlapping)
                y = F.fold(temp_, output_size=(H, W), kernel_size=group_size, stride=stride)

            # MODIFIED: add projection for valid perturbation
            y = torch.minimum(torch.maximum(y, inf_bound), sup_bound)

            # z step (equation 18)
            a_prime = δ + u
            b_prime = w + s
            c_prime = y + v
            η = α * (i + 1) ** 0.5
            z = (z * η + ρ * (2 * a_prime + b_prime + c_prime) - z_grad) / (η + 4 * ρ)

            # MODIFIED: add projection for valid perturbation
            z = torch.minimum(torch.maximum(z, inf_bound), sup_bound)

            # update steps
            u.add_(δ - z)
            v.add_(y - z)
            s.add_(w - z)

            # new predictions
            adv_inputs = y + inputs
            l2 = (adv_inputs - inputs).flatten(1).norm(p=2, dim=1)
            logits = model(adv_inputs)
            reset_module_counts()
            logit_dists = multiplier * difference_of_logits(logits, labels, labels_infhot=labels_infhot)

            # adjust the best result found so far
            predicted_classes = (logits - labels_onehot * confidence).argmax(1) if targeted else \
                (logits + labels_onehot * confidence).argmax(1)

            is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
            is_smaller = l2 < best_l2
            o_is_smaller = l2 < o_best_l2
            is_both = is_adv & is_smaller
            o_is_both = is_adv & o_is_smaller

            best_l2 = torch.where(is_both, l2, best_l2)
            adv_found.logical_or_(is_both)
            o_best_l2 = torch.where(o_is_both, l2, o_best_l2)
            o_adv_found.logical_or_(is_both)
            o_best_adv = torch.where(batch_view(o_is_both), adv_inputs.detach(), o_best_adv)

            if callback:
                i_total += 1
                callback.accumulate_line('logit_dist', i_total, logit_dists.mean())
                callback.accumulate_line('l2_norm', i_total, l2.mean())
                if i_total % (max_iterations // 20) == 0:
                    callback.update_lines()

        if callback:
            best_l2 = o_best_l2[o_adv_found].mean() if o_adv_found.any() else torch.tensor(float('nan'), device=device)
            callback.line(['success', 'best_l2', 'c'], outer_step, [o_adv_found.float().mean(), best_l2, c.mean()])

        # adjust the constant as needed
        upper_bound[adv_found] = torch.min(upper_bound[adv_found], const[adv_found])
        adv_not_found = ~adv_found
        lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], const[adv_not_found])
        is_smaller = upper_bound < 1e9
        const[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
        const[(~is_smaller) & adv_not_found] *= 5

    if retrain:
        lower_bound = torch.zeros_like(const)
        const = torch.full_like(const, initial_const)
        upper_bound = torch.full_like(const, 1e10)

        for i in range(8):  # taken from the original implementation

            best_l2 = torch.full_like(const, float('inf'))
            adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

            o_best_y = o_best_adv - inputs
            np_o_best_y = o_best_y.cpu().numpy()
            Nz = np.abs(np_o_best_y[np.nonzero(np_o_best_y)])
            threshold = np.percentile(Nz, σ)

            S_σ = o_best_y.abs() <= threshold
            z = o_best_y.clone()
            u = torch.zeros_like(inputs)
            tmpC = ρ / (ρ + γ / 100)

            for outer_step in range(400):  # taken from the original implementation
                # δ step (equation 21)
                temp_a = (z - u) * tmpC
                δ = torch.where(S_σ, zeros, torch.minimum(torch.maximum(temp_a, inf_bound), sup_bound))

                # new predictions
                δ.requires_grad_(True)
                adv_inputs = δ + inputs
                logits = model(adv_inputs)
                reset_module_counts()
                l2 = (adv_inputs.detach() - inputs).flatten(1).norm(p=2, dim=1)
                logit_dists = multiplier * difference_of_logits(logits, labels, labels_infhot=labels_infhot)
                loss = const * (logit_dists + confidence).clamp(min=0)
                z_grad = grad(loss.sum(), inputs=δ, only_inputs=True)[0]
                δ.detach_()

                # z step (equation 22)
                a_prime = δ + u
                z = torch.where(S_σ, zeros, (α * z + ρ * a_prime - z_grad) / (α + 2 * ρ))
                u.add_(δ - z)

                # adjust the best result found so far
                predicted_classes = (logits - labels_onehot * confidence).argmax(1) if targeted else \
                    (logits + labels_onehot * confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv & is_smaller
                o_is_both = is_adv & o_is_smaller

                best_l2 = torch.where(is_both, l2, best_l2)
                adv_found.logical_or_(is_both)
                o_best_l2 = torch.where(o_is_both, l2, o_best_l2)
                o_adv_found.logical_or_(is_both)
                o_best_adv = torch.where(batch_view(o_is_both), adv_inputs.detach(), o_best_adv)

            # adjust the constant as needed
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], const[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], const[adv_not_found])
            is_smaller = upper_bound < 1e9
            const[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            const[(~is_smaller) & adv_not_found] *= 5

    # return the best solution found
    return o_best_adv