#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
from .data.dataset import DatasetBase
from .loss import L2Loss
from .trainhook import TrainHook

# pylint: disable=W0221

__all__ = ['Model', 'EmbeddingBasedModel', 'DotBasedModel']


class Model(nn.Module, ABC):
    '''
    abstract class for model which contains two pipelines
    - train: forward -> calculate_loss
    - test: before_evaluate -> evaluate
    '''

    @abstractmethod
    def load_pretrain(self, pretrain_info: Dict[str, Any]):
        return

    @abstractmethod
    def forward(self, *args, **kwargs):
        '''
        input args should match dataset `__getitem__` return values
        '''
        return

    @abstractmethod
    def calculate_loss(self, modelout: tuple, batch_size: int):
        '''
        `modelout` will be `forward` return values
        '''
        return

    @abstractmethod
    def before_evaluate(self):
        return

    @abstractmethod
    def evaluate(self, before, *args, **kwargs):
        '''
        `before` will be `before_evaluate` return values
        '''
        return

    def register_trainhooks(self, trainhooks: Dict[str, TrainHook]):
        self._trainhooks = trainhooks

    @property
    def trainhooks(self) -> Dict[str, TrainHook]:
        if self.training:
            return self._trainhooks
        else:
            return defaultdict(lambda: lambda x: None)


class EmbeddingBasedModel(Model):
    # pylint: disable=W0223
    def __init__(
            self,
            info,
            dataset: DatasetBase,
            create_embeddings: bool,
            hasL2: bool = True):
        super().__init__()
        self.info = info
        self.embedding_size = info.embedding_size
        if hasL2:
            self._L2 = L2Loss(info.L2)
        self.num_ps = dataset.num_ps
        self.num_qs = dataset.num_qs
        if create_embeddings:
            # embeddings
            self.ps_feature = nn.Parameter(
                torch.FloatTensor(self.num_ps, self.embedding_size))
            nn.init.xavier_normal_(self.ps_feature)
            self.qs_feature = nn.Parameter(
                torch.FloatTensor(self.num_qs, self.embedding_size))
            nn.init.xavier_normal_(self.qs_feature)


class DotBasedModel(EmbeddingBasedModel):
    @abstractmethod
    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return

    def forward(self, ps: torch.Tensor, qs: torch.Tensor):
        # FIXME: by torch.matmul([B,?,D], [B,?,D].t())
        ps_feature, qs_feature = self.propagate()
        qs_embedding = qs_feature[qs]
        ps_embedding = ps_feature[ps].expand(
            -1, qs.shape[1], -1)
        pred = torch.sum(ps_embedding * qs_embedding, dim=-1)
        return pred, (ps_embedding, qs_embedding)

    def before_evaluate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.propagate()

    def evaluate(
            self,
            before: Tuple[torch.Tensor, torch.Tensor],
            ps: torch.Tensor) -> torch.Tensor:
        ps_feature, qs_feature = before
        ps_feature = ps_feature[ps]
        scores = torch.mm(ps_feature, qs_feature.t())
        return scores
