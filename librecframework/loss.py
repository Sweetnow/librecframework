#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Optional, cast
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=W0221

__all__ = ['L2Loss', 'BPRLoss', 'MaskedMSELoss', 'TrueFalsesCrossEntropy']


class _Loss(nn.Module):
    """
    The base class for loss function
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
        - reduction: Specifies the reduction to apply to the output: `none` | `mean` (default) | `sum`. 
            - `none`: no reduction will be applied.
            - `mean`: the sum of the output will be divided by the number of elements in the output. 
            - `sum`: the output will be summed.
        """
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction


class L2Loss(_Loss):
    """
    L2 loss function for normalization
    `loss = weight * pool(tensors^2) / batch_size`
    """
    def __init__(self, weight: float):
        """
        Args:
        - weight: the `weight` in `loss = weight * pool(tensors^2) / batch_size`
        """
        super().__init__()
        self.weight = weight

    def forward(
            self,
            *tensors: torch.Tensor,
            batch_size: Optional[int] = None) -> torch.Tensor:
        loss = cast(torch.Tensor, 0)
        for tensor in tensors:
            loss = loss + (tensor ** 2).sum()
        loss = loss * self.weight
        if not batch_size is None:
            loss = loss / batch_size
        return loss


class BPRLoss(_Loss):
    """
    BPR loss function.
    Proposed in Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence, UAI 2009, 452–461.
    """
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        `model_output` (tensor) - column 0 must be the scores of positive pois, others must be the negative.
        '''
        if inputs.shape[1] > 2:
            wstr = f'In BPRLoss, model_output/pred shape is {inputs.shape}, the second dim > 2'
            warnings.warn(wstr, RuntimeWarning)
        loss: torch.Tensor = -F.logsigmoid(inputs[:, 0:1] - inputs[:, 1:])
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss


class MaskedMSELoss(_Loss):
    """
    Square error loss function with masks,
    the mask will be multiplied into square error
    `loss = masks * (inputs - labels)^2`.
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        loss = (inputs - labels) ** 2
        loss = loss * masks
        if self.reduction == 'mean':
            loss = loss.sum() / masks.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss


class TrueFalsesCrossEntropy(_Loss):
    """
    Binary cross entropy loss function, the targets of the `inputs[:, 0]` are 1 and others are 0.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
        - reduction: Specifies the reduction to apply to the output: `none` | `mean` (default) | `sum`. 
            - `none`: no reduction will be applied.
            - `mean`: the sum of the output will be divided by the number of elements in the output. 
            - `sum`: the output will be summed.
        """
        super().__init__(reduction)
        self._loss_func = nn.BCELoss(reduction=reduction)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(inputs)
        targets[:, 0] = 1
        loss = self._loss_func(inputs, targets)
        return loss
