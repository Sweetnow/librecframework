#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict, cast
from math import ceil
from time import time
import logging
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from .model import Model
from .trainhook import TrainHook
from .data.dataset import TrainDataset

_line = 10

__all__ = ['train']


def _log_interval_infer(loader: DataLoader) -> int:
    """Infer the log interval to show 10 lines log each epoch"""
    batch_cnt = ceil(len(cast(TrainDataset, loader.dataset)) / cast(int, loader.batch_size))
    if batch_cnt < _line:
        return 1
    else:
        return round(batch_cnt/_line)


def train(model: Model,
          epoch: int,
          loader: DataLoader,
          op: Optimizer,
          trainhooks: Dict[str, TrainHook],
          log_interval: Optional[int] = None) -> None:
    """
    Train model

    Args:
    - model: the model for training
    - epoch: the epoch ID
    - loader: the corresponding dataloader for training
    - op: the optimizer for training
    - trainhooks: 
    if `log_interval`==`None`, log interval will be automatically set
    to ensure that 10 lines are shown per epoch.
    """
    if log_interval is None:
        log_interval = _log_interval_infer(loader)

    model.train()
    model.register_trainhooks(trainhooks)
    for v in trainhooks.values():
        v.start()
    start = time()
    for i, data in enumerate(loader):
        batch_size = cast(int, loader.batch_size)
        op.zero_grad()
        for k, v in data.items():
            data[k] = v.cuda()
        modelout = model(**data)
        loss = model.calculate_loss(modelout, batch_size=batch_size)
        if '__loss__' in trainhooks:
            trainhooks['__loss__'](loss.item())
        loss.backward()
        op.step()
        if i % log_interval == 0:
            now_cnt = (i + 1) * batch_size
            all_cnt = len(cast(TrainDataset, loader.dataset))
            prt = 100 * (i + 1) / len(loader)
            logging.info(
                f'Train Epoch: {epoch} [{now_cnt}/{all_cnt} ({prt:.0f}%)]\tLoss: {loss.item():.6f}')
    logging.debug(f'Train Epoch: {epoch}: time = {int(time() - start):d}s')
    for v in trainhooks.values():
        v.stop()
        logging.info(f'{v.title}:{v.value}')
    cast(TrainDataset, loader.dataset).next_epoch()
