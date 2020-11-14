#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=W0603,W0201

from ast import Num
from typing import Dict, Any, Union, cast
from abc import ABC, abstractmethod
import torch

__all__ = ['Precision', 'Recall', 'NDCG', 'MRR',
           'LeaveOneHR', 'LeaveOneNDCG', 'LeaveOneMRR', 'Metric']

is_hit_cache: Dict[int, Dict[str, Any]] = {}

Number = Union[int, float]


def _get_is_hit(scores: torch.Tensor, ground_truth: torch.Tensor, topk: int) -> torch.Tensor:
    # cache
    global is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in is_hit_cache and is_hit_cache[topk]['id'] == cacheid:
        return is_hit_cache[topk]['is_hit']
    else:
        device = scores.device
        # indice tensor generate
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.arange(
            scores.shape[0],
            device=device,
            dtype=torch.long
        ).unsqueeze(1).expand_as(col_indice)
        is_hit = ground_truth[row_indice.reshape(-1),
                              col_indice.reshape(-1)].view(-1, topk)
        # cache
        is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit


class Metric(ABC):
    """
    Base class for metrics in evaluation stage

    Property & Methods:
    - metric: the meaning metric value
    - start(): reset the metric
    - stop(): close the metric after one epoch
    - __call__(): send a value into the metric
    """

    def __init__(self) -> None:
        self.start()

    @property
    def metric(self) -> float:
        if self._stopped:
            return self._metric
        else:
            raise RuntimeError(
                f'{self} is not stopped. Call stop() before reading metric')

    @abstractmethod
    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """
        Args:
        - scores: model output
        - ground_truth: one-hot test dataset shape as `scores`.
        """
        pass

    def start(self) -> None:
        """Initialize the metric"""
        self._stopped = False

        global is_hit_cache
        is_hit_cache = {}
        self._cnt: float = 0
        self._metric: float = 0
        self._sum: float = 0

    def stop(self) -> None:
        """Compute metric value to close the metric"""
        global is_hit_cache
        is_hit_cache = {}
        self._metric = self._sum / self._cnt
        self._stopped = True


class TopkMetric(Metric):
    """Base class for top-K based metric"""

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        self.eps = 1e-8

    def __str__(self) -> str:
        return f'{self.__class__.__name__}@{self.topk}'


class Precision(TopkMetric):
    """Precision = TP / (TP + FP)"""

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        self._cnt += scores.shape[0]
        self._sum += cast(Number, is_hit.mean(dim=1).sum().item())


class Recall(TopkMetric):
    """Recall = TP / (TP + FN)"""

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        # ignore row without positive result
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, (is_hit / (num_pos + self.eps)).sum().item())


class NDCG(TopkMetric):
    """Normalized Discounted Cumulative Gain"""

    def DCG(self, hit: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        hit = hit / torch.log2(torch.arange(
            2, self.topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def _IDCG(self, num_pos: int) -> Number:
        if num_pos == 0:
            return 1
        else:
            hit = torch.zeros(self.topk, dtype=torch.float)
            hit[:num_pos] = 1
            return self.DCG(hit).item()

    def __init__(self, topk: int) -> None:
        super().__init__(topk)
        self.IDCGs: torch.Tensor = torch.FloatTensor(
            [self._IDCG(i) for i in range(0, self.topk + 1)])

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).long()
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg / idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, ndcg.sum().item())


class MRR(TopkMetric):
    """Mean Reciprocal Rank = 1 / position(1st hit)"""

    def __init__(self, topk: int):
        super().__init__(topk)
        self.denominator = torch.arange(1, self.topk + 1, dtype=torch.float)
        self.denominator.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        is_hit /= self.denominator.to(device)
        first_hit_rr = is_hit.max(dim=1)[0]
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, first_hit_rr.sum().item())


class LeaveOneHR(TopkMetric):
    """Leave-one-out Hit Ratio = 1 if hit else 0"""

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        self._cnt += cast(int, ground_truth.sum().item())
        self._sum += cast(Number, is_hit.sum().item())


class LeaveOneNDCG(TopkMetric):
    """
    Leave-one-out Normalized Discounted Cumulative Gain
    NDCG = log(2) / log(1 + position(hit))
    """

    def __init__(self, topk: int) -> None:
        super().__init__(topk)
        self.NDCGs = 1 / \
            torch.log2(torch.arange(2, self.topk + 2,
                                    dtype=torch.float))
        self.NDCGs.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        ndcg = is_hit * self.NDCGs.to(device)
        self._cnt += cast(int, ground_truth.sum().item())
        self._sum += cast(Number, ndcg.sum().item())


class LeaveOneMRR(TopkMetric):
    """Leave-one-out Mean Reciprocal Rank = 1 / position(hit)"""

    def __init__(self, topk: int):
        super().__init__(topk)
        self.denominator = torch.arange(1, self.topk + 1, dtype=torch.float)
        self.denominator.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        mrr = is_hit / self.denominator.to(device)
        self._cnt += cast(int, ground_truth.sum().item())
        self._sum += cast(Number, mrr.sum().item())


_ALL_METRICS = {cls.__name__: cls for cls in (
    Precision, Recall, NDCG, MRR, LeaveOneHR, LeaveOneNDCG
)}
# TODO: one singleton named `metric_manager` which has function `register` for user-defined metric
