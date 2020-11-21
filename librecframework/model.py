#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from librecframework import data
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
from .data.dataset import DatasetBase
from .loss import L2Loss
from .trainhook import TrainHook, IgnoredHook

# pylint: disable=W0221

__all__ = ['Model', 'EmbeddingBasedModel', 'DotBasedModel']


class Model(nn.Module, ABC):
    """
    Abstract class for model which contains two pipelines:
    - train: `forward` -> `calculate_loss`
    - test: `before_evaluate` -> `evaluate`
    """

    def __init__(self,
                 info,
                 dataset: DatasetBase):
        super().__init__()
        self.info = info
        self.dataset = dataset

    @abstractmethod
    def load_pretrain(self, pretrain_info: Dict[str, Any]) -> None:
        """
        Select and load pretrain model

        Args:
        - pretrain_info: the content loaded from `pretrain_path` and selected by dataset name
        """
        return

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forwarding propagation

        Args:
        - kwargs: the train dataset `__getitem__` return values
        """
        return

    @abstractmethod
    def calculate_loss(self, modelout: tuple, batch_size: int) -> torch.Tensor:
        """
        Calculate loss in forwarding propagation

        Args:
        - modelout: `forward` return value(s)
        - batch_size: the batch size of the forwarding phase
        """
        return

    @abstractmethod
    def before_evaluate(self) -> Any:
        """Pre-generate data in evaluation"""
        return

    @abstractmethod
    def evaluate(self, before, *args, **kwargs) -> torch.Tensor:
        """
        Evaluation

        Args:
        - before: the `before_evaluate` return value(s)
        - kwargs: the test dataset `__getitem__` return values
        """
        return

    def register_trainhooks(self, trainhooks: Dict[str, TrainHook]) -> None:
        """Assign trainhooks into the model"""
        self._trainhooks = trainhooks

    @property
    def trainhooks(self) -> Dict[str, TrainHook]:
        if self.training:
            return self._trainhooks
        else:
            return defaultdict(lambda: IgnoredHook())


class EmbeddingBasedModel(Model):
    """
    Abstract class for embedding based model which added two features based on `Model`:
    - Automatically create embeddings for two kinds of entities by `num_ps`, `num_qs` and `info.embedding_size`
    - Add L2 loss function as `self._L2` with `info.L2` weight
    """
    # pylint: disable=W0223

    def __init__(
            self,
            info,
            dataset: DatasetBase,
            create_embeddings: bool,
            hasL2: bool = True):
        """
        Args:
        - info: the CLI input args
        - dataset: the train dataset
        - create_embeddings: whether to auto-create embeddings
        - hasL2: whether to add L2 loss function
        """
        super().__init__(info, dataset)
        self.embedding_size = info.embedding_size
        if hasL2:
            self._L2 = L2Loss(info.L2)
        self.num_ps = self.dataset.num_ps
        self.num_qs = self.dataset.num_qs
        if create_embeddings:
            # embeddings
            self.ps_feature: nn.Parameter = nn.Parameter(
                torch.FloatTensor(self.num_ps, self.embedding_size))
            nn.init.xavier_normal_(self.ps_feature)
            self.qs_feature: nn.Parameter = nn.Parameter(
                torch.FloatTensor(self.num_qs, self.embedding_size))
            nn.init.xavier_normal_(self.qs_feature)


class DotBasedModel(EmbeddingBasedModel):
    """
    Abstract class for dot based model which added one feature based on `EmbeddingBasedModel`:
    - Simplify `forward`, `before_evaluate` and `evaluate` interfaces to `propagate` interface, which aims to generate final embeddings
    """
    @abstractmethod
    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate embeddings for dot-based prediction"""
        return

    def forward(self, ps: torch.Tensor, qs: torch.Tensor):
        ps_feature, qs_feature = self.propagate()
        qs_embedding = qs_feature[qs]
        ps_embedding = ps_feature[ps].expand(
            -1, qs.shape[1], -1)
        pred = torch.sum(ps_embedding * qs_embedding, dim=-1)
        return pred, (ps_embedding, qs_embedding)

    def before_evaluate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-generate data in evaluation"""
        return self.propagate()

    def evaluate(
            self,
            before: Tuple[torch.Tensor, torch.Tensor],
            ps: torch.Tensor) -> torch.Tensor:
        """
        Evaluation (for fully-ranking evaluation)

        Args:
        - before: the `before_evaluate` return value(s)
        - ps: the list of ps for evaluation
        """
        ps_feature, qs_feature = before
        ps_feature = ps_feature[ps]
        scores = torch.mm(ps_feature, qs_feature.t())
        return scores
