#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time
from typing import List, Callable
import logging
import torch
from torch.utils.data import DataLoader
from .metric import Metric
from .model import Model


__all__ = ['test', 'fully_ranking_test', 'leave_one_out_test']

_MAX = 1e8
_MODE_NAME = ['fully-ranking', 'leave-one-out']


def test(model: Model,
         loader: DataLoader,
         metrics: List[Metric],
         mode: str) -> None:
    """
    Evaluate model

    Args:
    - model: the model for evaluation
    - loader: the corresponding dataloader for evaluation
    - metrics: the list of metrics
    - mode: the evaluation model, `fully-ranking` or `leave-one-out`

    Exception:
    - ValueError: the mode flag is invalid
    """
    if mode not in _MODE_NAME:
        raise ValueError(f'mode {mode} should be in {_MODE_NAME}.')

    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        # model defined
        if mode == 'fully-ranking':
            before = model.before_evaluate()
            for data, other in loader:
                train_mask = other['train_mask'].cuda()
                ground_truth = other['ground_truth'].cuda()
                for k, v in data.items():
                    data[k] = v.cuda()
                pred = model.evaluate(before, **data)
                pred -= _MAX * train_mask
                for metric in metrics:
                    metric(pred, ground_truth)
        else:
            for data, other in loader:
                ground_truth = other['ground_truth'].cuda()
                for k, v in data.items():
                    data[k] = v.cuda()
                modelout = model(**data)
                if isinstance(modelout, tuple):
                    pred = modelout[0]
                else:
                    pred = modelout
                for metric in metrics:
                    metric(pred, ground_truth)
    logging.debug(f'Test: time={int(time()-start):d}s')
    for metric in metrics:
        metric.stop()
        logging.info(f'{metric}:{metric.metric}')
    print('')


<<<<<<< Updated upstream
fully_ranking_test: Callable[[Model, DataLoader, List[Metric]], None] = partial(
    test, mode='fully-ranking')
leave_one_out_test: Callable[[Model, DataLoader, List[Metric]], None] = partial(
    test, mode='leave-one-out')
=======
def fully_ranking_test(model: Model,
                       loader: DataLoader,
                       metrics: List[_Metric]
                       ) -> None:
    """
    Evaluate model by fully ranking

    Args:
    - model: the model for evaluation
    - loader: the corresponding dataloader for evaluation
    - metrics: the list of metrics

    Exception:
    - ValueError: the mode flag is invalid
    """
    return test(model, loader, metrics, 'fully-ranking')


def leave_one_out_test(model: Model,
                       loader: DataLoader,
                       metrics: List[_Metric]
                       ) -> None:
    """
    Evaluate model by leave-one-out

    Args:
    - model: the model for evaluation
    - loader: the corresponding dataloader for evaluation
    - metrics: the list of metrics

    Exception:
    - ValueError: the mode flag is invalid
    """
    return test(model, loader, metrics, 'leave-one-out')
>>>>>>> Stashed changes
