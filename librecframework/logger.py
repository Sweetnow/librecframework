#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import string
import random
from json import dump
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from collections import defaultdict
import torch
from .utils.training import metrics_sliding_max
from .metric import _Metric
from .trainhook import TrainHook

# pylint: disable=W0201

__all__ = ['Logger']


def _hash_model(modelinfo):
    return hex(hash(modelinfo))[-6:]


class Logger():
    '''
    Utility that save models and records
    '''
    # policies for model saving
    CHECKPOINT_POLICIES = ['none', 'always', 'best']

    def __init__(self,
                 log_path: Union[Path, str],
                 checkpoint_policy: str = 'best',
                 checkpoint_interval: Optional[int] = None,
                 checkpoint_target: Optional[Union[str, Tuple[str], List[str]]] = None) -> None:
        '''
        Args:
        - log_path: the dir of every model's log dir
        - checkpoint_policy: when to save model. [ `none` | `always` | `best` (default)]
            - `none`: never save model
            - `always`: save model every `checkpoint_interval` epochs
            - `best`: save the best(`checkpoint_target`) model
        - checkpoint_interval: int, for `always`
        - checkpoint_target: str or list of str, for `best`
        '''
        assert checkpoint_policy in Logger.CHECKPOINT_POLICIES
        if checkpoint_policy == 'always':
            assert checkpoint_interval > 0 and isinstance(
                checkpoint_interval, int)
        elif checkpoint_policy == 'best':
            if isinstance(checkpoint_target, (list, tuple)):
                for target in checkpoint_target:
                    assert isinstance(target, str)
            else:
                checkpoint_target = [checkpoint_target]

        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_epoch = 0
        self.checkpoint_target = checkpoint_target

        self.random = ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(4))
        self.log_path = Path(log_path)
        self.time_path = time.strftime(
            '%m-%d-%H-%M-%S-', time.localtime(time.time()))+self.random

        self.root_path = self.log_path / self.time_path
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        else:
            raise FileExistsError(f'{self.root_path} exists')

        self.log = self.root_path / 'model.json'
        self.__all_infos = []
        self.cnt = 0

    @property
    def all_infos(self):
        return self.__all_infos

    @all_infos.setter
    def all_infos(self, value: List):
        self.__all_infos = value
        with open(self.log, 'w') as f:
            dump(self.__all_infos, f, indent=4)

    def get_model_id(self, modelinfo) -> str:
        return f'{self.cnt}_{_hash_model(modelinfo)}'

    def __enter__(self):
        return self

    def __exit__(self, ty, value, trace):
        pass

    def update_modelinfo(
            self,
            modelname: str,
            modelinfo,
            envinfo: dict,
            target: str,
            window_size: int = 10):
        '''
        args:
        - `modelinfo`:  model hyperparameters (`ModelInfo`)
        - `envinfo`:    other hyperparameters like (`lr`) (`Dict`)
        '''
        # set key variables
        self.target = target
        self.window_size = window_size
        self.modelinfo = modelinfo
        self.env = envinfo
        self._metrics_log = defaultdict(list)
        self._trainhooks_log = defaultdict(list)
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
        with open(self.root_path / f'{self.get_model_id(self.modelinfo)}.json', 'w') as f:
            dump({
                'metrics': self._metrics_log,
                'trainhooks': self._trainhooks_log
            }, f)

    def update_trainhooks(self, trainhooks: Dict[str, TrainHook]):
        for v in trainhooks.values():
            self._trainhooks_log[v.title] += [v.value]
        self._update_log()

    def update_metrics_and_model(self, metrics: List[_Metric], model: torch.nn.Module):
        # save metrics
        for metric in metrics:
            metric_str = str(metric)
            self._metrics_log[metric_str] += [metric.metric]
        self._update_log()
        # save model
        if self.checkpoint_policy == 'always':
            self.checkpoint_epoch += 1
            if self.checkpoint_epoch % self.checkpoint_interval == 0:
                model_path = self.root_path / \
                    f'{self.get_model_id(self.modelinfo)}.pth'
                torch.save(model.state_dict(), model_path)
        elif self.checkpoint_policy == 'best':
            for target in self.checkpoint_target:
                if self.metrics_log[target][-1] == max(self.metrics_log[target]):
                    model_path = self.root_path / \
                        f'{self.get_model_id(self.modelinfo)}_{target}.pth'
                    torch.save(model.state_dict(), model_path)

        # update model json
        best = metrics_sliding_max(self.metrics_log,
                                   window_size=1, target=self.target)
        ave = metrics_sliding_max(
            self.metrics_log,
            window_size=self.window_size, target=self.target
        )
        all_infos = self.all_infos
        all_infos[-1]['metrics'] = {'best': best,
                                    f'ave@{self.window_size}': ave}
        self.all_infos = all_infos

    @property
    def metrics_log(self):
        return self._metrics_log

    @property
    def trainhooks_log(self):
        return self._trainhooks_log
