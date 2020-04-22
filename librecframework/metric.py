#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=W0603,W0201

from abc import ABC, abstractmethod
import torch

__all__ = ['Recall', 'NDCG', 'MRR',
           'LeaveOneHR', 'LeaveOneNDCG', 'LeaveOneMRR']

_is_hit_cache = {}


def _get_is_hit(scores: torch.Tensor, ground_truth: torch.Tensor, topk: int) -> torch.Tensor:
    # cache
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']
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
        _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit


class _Metric(ABC):
    '''
    base class of metrics like HR@k NDCG@k
    '''

    def __init__(self) -> None:
        self.start()

    @property
    def metric(self) -> float:
        return self._metric

    @abstractmethod
    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        '''
        - scores: model output
        - ground_truth: one-hot test dataset shape=(ps, all_qs).
        '''
        pass

    def start(self) -> None:
        '''
        clear all
        '''
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self) -> None:
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum / self._cnt


class TopkMetric(_Metric):
    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        self.eps = 1e-8

    def __str__(self) -> str:
        return f'{self.__class__.__name__}@{self.topk}'


class Recall(TopkMetric):
    '''
    Recall in top-k samples
    '''

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        # ignore row without positive result
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit / (num_pos + self.eps)).sum().item()


class NDCG(TopkMetric):
    '''
    NDCG in top-k samples
    '''

    def DCG(self, hit: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        hit = hit / torch.log2(torch.arange(2, self.topk+2,
                                            device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(self, num_pos: int) -> torch.Tensor:
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk: int) -> None:
        super().__init__(topk)
        self.IDCGs = torch.FloatTensor(
            [1] + [self.IDCG(i) for i in range(1, self.topk + 1)])

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).long()
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg / idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()


class MRR(TopkMetric):
    '''
    Mean reciprocal rank in top-k samples
    '''

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
        self._sum += first_hit_rr.sum().item()


class LeaveOneHR(TopkMetric):
    '''
    Leave-one-out Hit Ratio in top-k samples
    '''

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        self._cnt += ground_truth.sum().item()
        self._sum += is_hit.sum().item()


class LeaveOneNDCG(TopkMetric):
    '''
    Leave-one-out NDCG in top-k samples
    NDCG = log(2) / log(1+hit_positions)
    '''

    def __init__(self, topk: int) -> None:
        super().__init__(topk)
        self.NDCGs = 1 / \
            torch.log2(torch.arange(2, self.topk + 2, dtype=torch.float))
        self.NDCGs.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        ndcg = is_hit * self.NDCGs.to(device)
        self._cnt += ground_truth.sum().item()
        self._sum += ndcg.sum().item()


class LeaveOneMRR(TopkMetric):
    '''
    Leave-one-out Mean reciprocal rank in top-k samples
    '''

    def __init__(self, topk: int):
        super().__init__(topk)
        self.denominator = torch.arange(1, self.topk + 1, dtype=torch.float)
        self.denominator.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        mrr = is_hit / self.denominator.to(device)
        self._cnt += ground_truth.sum().item()
        self._sum += mrr.sum().item()


_ALL_METRICS = {cls.__name__: cls for cls in (
    Recall, NDCG, MRR, LeaveOneHR, LeaveOneNDCG
)}
# TODO: one singleton named `metric_manager` which has function `register` for user-defined metric
