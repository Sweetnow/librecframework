#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from typing import Tuple, Dict, Any
import torch
from librecframework.argument.manager import HyperparamManager
from librecframework.pipeline import DefaultLeaveOneOutPipeline
from librecframework.data import DatasetFuncs
from librecframework.data.dataset import TrainDataset
import librecframework.data.functional as fdf
from librecframework.model import DotBasedModel
from librecframework.loss import BPRLoss
from librecframework.trainhook import ValueMeanHook


class MF(DotBasedModel):
    def __init__(self, info, dataset: TrainDataset):
        super().__init__(info, dataset, create_embeddings=True)
        self._bpr_loss = BPRLoss('mean')

    def load_pretrain(self, pretrain_info: Dict[str, Any]) -> None:
        path = pretrain_info['MF']
        pretrain = torch.load(path, map_location='cpu')
        self.ps_feature.data = pretrain['ps_feature']
        self.qs_feature.data = pretrain['qs_feature']

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ps_feature, self.qs_feature

    def calculate_loss(
            self,
            modelout: Tuple[torch.Tensor, torch.Tensor],
            batch_size: int) -> torch.Tensor:
        pred, tensors = modelout
        L2 = self._L2(*tensors, batch_size=batch_size)
        loss = self._bpr_loss(pred) + L2
        return loss


MODEL = MF


def hyperparameter() -> HyperparamManager:
    hpm = HyperparamManager('Hyperparameter Arguments',
                            '', f'{MODEL.__name__}Info')
    hpm.register(
        'embedding_size',
        ['-EB', '--embedding-size'],
        dtype=int,
        validator=lambda x: x > 0,
        helpstr='model embedding size',
        default=32
    )
    hpm.register(
        'lr',
        multi=True,
        dtype=float,
        validator=lambda x: x > 0,
        helpstr='learning rate'
    )
    hpm.register(
        'L2',
        ['--L2'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0,
        helpstr='model L2 normalization'
    )
    hpm.register(
        'pretrain',
        dtype=bool,
        default=False,
        helpstr='pretrain'
    )
    return hpm


def test_all():
    pipeline = DefaultLeaveOneOutPipeline(
        description=MODEL.__name__,
        supported_datasets=['demo'],
        train_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=fdf.do_nothing,
            sample=fdf.itemrec_sample,
            getitem=fdf.default_train_getitem,
            length=fdf.default_train_length
        ),
        test_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=fdf.set_num_negtive_qs(9),
            sample=None,
            getitem=fdf.default_leave_one_out_test_getitem,
            length=fdf.default_leave_one_out_test_length
        ),
        hyperparam_manager=hyperparameter(),
        other_arg_path='tests/config/config.json',
        pretrain_path='tests/config/pretrain.json',
        sample_tag='default',
        pin_memory=True,
        min_memory=1)
    pipeline.parse_args('train -T pytest -SEP 2 -SW 2 -EP 3 -BS 10 -TBS 10 --lr 1e-3 --L2 1e-4'.split(' '))
    pipeline.before_running()
    
    pipeline.during_running(MODEL, {}, {'L2Loss': ValueMeanHook('L2Loss')})
    pipeline.after_running()
