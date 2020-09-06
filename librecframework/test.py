#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time
from typing import List, Callable
import logging
from functools import partial
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
    '''
    test model
    '''
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
                pred = modelout[0] if isinstance(modelout, tuple) else modelout
                for metric in metrics:
                    metric(pred, ground_truth)
    logging.debug(f'Test: time={int(time()-start):d}s')
    for metric in metrics:
        metric.stop()
        logging.info(f'{metric}:{metric.metric}')
    print('')


fully_ranking_test: Callable[[Model, DataLoader, List[Metric]], None] = partial(
    test, mode='fully-ranking')
leave_one_out_test: Callable[[Model, DataLoader, List[Metric]], None] = partial(
    test, mode='leave-one-out')
