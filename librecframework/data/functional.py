#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, List, Union, Any, Dict, Tuple
import os
import numpy as np
import torch
from .dataset import DatasetBase, TrainDataset, FullyRankingTestDataset, LeaveOneOutTestDataset

__all__ = [
    'RecordFuncCascade', 'modify_nothing', 'reverse_iu', 'Filter',
    'PostinitFuncSum', 'do_nothing', 'KhopFriends',
    'itemrec_sample',
    'default_train_getitem', 'default_fully_ranking_test_getitem',
    'default_train_length', 'default_fully_ranking_test_length',
    'default_leave_one_out_test_length', 'default_leave_one_out_test_getitem'
]

# record_funcs


class RecordFuncCascade():
    def __init__(self, *record_funcs: Callable):
        self._funcs = record_funcs

    def __call__(self, records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
        for f in self._funcs:
            records = f(records)
        return records


def modify_nothing(
        records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
    return records


def reverse_iu(
        records: List[List[int]]) -> List[List[int]]:
    records = [[record[1], record[0], *record[2:]] for record in records]
    return records


class Filter():
    def __init__(self, function: Callable[[Any], bool]):
        self._function = function

    def __call__(self, records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
        records = list(filter(self._function, records))
        return records

# postinit_funcs


class PostinitFuncSum():
    def __init__(self, *postinit_funcs: Callable):
        self._funcs = postinit_funcs

    def __call__(self, *args, **kwargs):
        for f in self._funcs:
            f(*args, **kwargs)


def do_nothing(_: Any) -> None:
    pass


class set_num_negtive_qs():
    def __init__(self, num: int):
        self._num = num

    def __call__(self, dataset: LeaveOneOutTestDataset):
        dataset.num_neg_qs = self._num


# sample_funcs


def itemrec_sample(dataset: TrainDataset, index: int) -> int:
    p, q_pos = dataset.pos_pairs[index]
    while True:
        i = np.random.randint(dataset.num_qs)
        if dataset.ground_truth[p, i] == 0 and i != q_pos:
            return i

# getitem_funcs

# for training, return values should be dict
# whose keys is the same as model's forward.


def default_train_getitem(self: TrainDataset, index: int) -> Dict[str, torch.Tensor]:
    p, q_pos = self.pos_pairs[index]
    neg_q = self.neg_qs[index][self.epoch]
    # dict -> model.forward
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor([q_pos, neg_q])
    }

# for testing, return values should be dict
# whose keys is the same as model's evaluate
# followed by ground truth and train mask.


def default_fully_ranking_test_getitem(self: FullyRankingTestDataset, index: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    ground_truth = torch.from_numpy(
        self.ground_truth[index].toarray()).view(-1)
    train_mask = torch.from_numpy(self.train_mask[index].toarray()).view(-1)
    # dict1 -> model.evaluate dict2 -> test
    return {'ps': index}, {'train_mask': train_mask, 'ground_truth': ground_truth}


def default_leave_one_out_test_getitem(self: LeaveOneOutTestDataset, index: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    p, q_pos = self.pos_pairs[index]
    qs_neg = self.neg_qs[index]
    gt = torch.zeros(len(qs_neg)+1, dtype=torch.float)
    gt[-1] = 1
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor(np.r_[qs_neg, q_pos])
    }, {'train_mask': 0, 'ground_truth': gt}


# length_funcs


def default_train_length(self: TrainDataset) -> int:
    return len(self.pos_pairs)


def default_fully_ranking_test_length(self: FullyRankingTestDataset) -> int:
    return self.ground_truth.shape[0]


def default_leave_one_out_test_length(self: LeaveOneOutTestDataset) -> int:
    return len(self.pos_pairs)
