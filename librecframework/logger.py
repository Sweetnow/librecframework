#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import string
import random
from json import dump
from pathlib import Path
from typing import Any, List, Dict, NamedTuple
from collections import defaultdict
import torch
import torch.nn as nn
from .utils.training import metrics_sliding_max
from .metric import Metric
from .trainhook import TrainHook

# pylint: disable=W0201

__all__ = ['Logger']

_MAX: int = 2**31


def _hash_model(modelinfo: Any) -> str:
    """Obtain unique ID to identify model"""
    return hex(hash(modelinfo))[-6:]


class Logger():
    """
    Utility that save models with metrics by specific policy and record all values during training in `log_path`.

    Used as CONTEXT MANAGER

    Supported policies:
    - `none`: record nothing
    - `always`: record the last one
    - `best`: record the best one accroding to model-specific target (default)
    """

    # policies for model saving
    CHECKPOINT_POLICIES = ['none', 'always', 'best']

    def __init__(self,
                 log_path: Path,
                 policy: str = 'best',
                 interval: int = _MAX) -> None:
        """
        Args:
        - log_path: the dir of every model's log dir
        - policy: when to save model. [ `none` | `always` | `best` (default)]
            - `none`: never save model
            - `always`: save model every `interval` epochs
            - `best`: save the best model by model-specific target
        - interval: int, for `always`
        """
        assert policy in Logger.CHECKPOINT_POLICIES
        if policy == 'always':
            assert _MAX > interval > 0

        self.policy = policy
        self.interval = interval
        self.epoch = 0

        self.random = ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(4))
        self.log_path = log_path
        self.time_path = time.strftime(
            '%m-%d-%H-%M-%S-', time.localtime(time.time()))+self.random

        self.root_path = self.log_path / self.time_path
        if not self.root_path.exists():
            os.makedirs(self.root_path)
        else:
            raise FileExistsError(f'{self.root_path} exists')

        self.log = self.root_path / 'model.json'
        self.__all_infos: List[Dict[str, Any]] = []
        self.cnt: int = 0

    @property
    def all_infos(self):
        """The content of `model.json`"""
        return self.__all_infos

    @all_infos.setter
    def all_infos(self, value: List[Dict[str, Any]]):
        self.__all_infos = value
        with open(self.log, 'w') as f:
            dump(self.__all_infos, f, indent=4)

    def get_model_id(self, modelinfo: Any) -> str:
        return f'{self.cnt}_{_hash_model(modelinfo)}'

    def __enter__(self):
        return self

    def __exit__(self, ty, value, trace):
        pass

    def update_modelinfo(
            self,
            modelname: str,
            modelinfo: NamedTuple,
            envinfo: Dict[str, Any],
            target: str,
            window_size: int = 10) -> 'Logger':
        """
        Append the information of the next model into `model.json` and reset metric and trainhook logs

        Args:
        - modelinfo:  model hyperparameters
        - envinfo:    other hyperparameters like `lr`
        """
        # set key variables
        self.target = target
        self.window_size = window_size
        self.modelinfo = modelinfo
        self.env = envinfo

        self._metrics_log: Dict[str, List[float]] = defaultdict(list)
        self._trainhooks_log: Dict[str, List[float]] = defaultdict(list)
        self.cnt += 1

        # update model json
        info = {
            'model': modelname,
            'id': self.get_model_id(self.modelinfo),
            'info': self.modelinfo._asdict(),
            'env': self.env
        }
        all_infos = self.all_infos
        all_infos.append(info)
        self.all_infos = all_infos
        return self

    def _update_log(self):
        """Update the model's log file `${model_id}.json`"""
        with open(self.root_path / f'{self.get_model_id(self.modelinfo)}.json', 'w') as f:
            dump({
                'metrics': self._metrics_log,
                'trainhooks': self._trainhooks_log
            }, f)

    def update_trainhooks(self, trainhooks: Dict[str, TrainHook]):
        """Add the value of each trainhook into the corresponding list and save"""
        for v in trainhooks.values():
            self._trainhooks_log[v.title] += [v.value]
        self._update_log()

    def update_metrics_and_model(self, metrics: List[Metric], model: nn.Module):
        """Add the value of each metric into the corresponding list and save the model aaccording to the policy"""
        # save metrics
        for metric in metrics:
            self._metrics_log[str(metric)] += [metric.metric]
        self._update_log()

        # save model
        tag = None
        if self.policy == 'always':
            self.epoch += 1
            if self.epoch % self.interval == 0:
                tag = 'always'
        elif self.policy == 'best':
            if self.metrics_log[self.target][-1] == max(self.metrics_log[self.target]):
                tag = self.target
        if tag is not None:
            model_path = self.root_path / \
                f'{self.get_model_id(self.modelinfo)}_{tag}.pth'
            torch.save(model.state_dict(), model_path)

        # update model json
        best = metrics_sliding_max(
            self.metrics_log,
            window_size=1,
            target=self.target)
        ave = metrics_sliding_max(
            self.metrics_log,
            window_size=self.window_size,
            target=self.target
        )
        all_infos = self.all_infos
        all_infos[-1]['metrics'] = {'best': best,
                                    f'ave@{self.window_size}': ave}
        self.all_infos = all_infos

    @property
    def metrics_log(self):
        """The log of metric history"""
        return self._metrics_log

    @property
    def trainhooks_log(self):
        """The log of trainhook history"""
        return self._trainhooks_log
