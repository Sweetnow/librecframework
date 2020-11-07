#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, cast
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=W0221

__all__ = ['L2Loss', 'BPRLoss', 'MaskedMSELoss', 'TrueFalsesCrossEntropy']


class _Loss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `mean`
        '''
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction


class L2Loss(_Loss):
    def __init__(self, weight: float):
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
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        if self.reduction in ['sum', 'mean']:
            self._loss_func = nn.BCELoss(reduction='sum')
        else:
            self._loss_func = nn.BCELoss(reduction='none')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        true_part, false_part = inputs[:, 0:1], inputs[:, 1:]
        true_part = self._loss_func(true_part, torch.ones_like(true_part))
        false_part = self._loss_func(false_part, torch.zeros_like(false_part))

        if self.reduction == 'mean':
            loss = true_part + false_part
            loss = loss / inputs.numel()
        elif self.reduction == 'sum':
            loss = true_part + false_part
        elif self.reduction == 'none':
            loss = torch.cat((true_part, false_part), 1)
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss
