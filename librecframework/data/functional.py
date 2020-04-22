#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union
import os
import numpy as np
import torch
from . import save_pyobj, load_pyobj
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
    def __init__(self, *record_funcs):
        self._funcs = record_funcs

    def __call__(self, records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
        for f in self._funcs:
            records = f(records)
        return records


def modify_nothing(
        records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
    return records


def reverse_iu(
        records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
    records = [(record[1], record[0], *record[2:]) for record in records]
    return records


class Filter():
    def __init__(self, function):
        self._function = function

    def __call__(self, records: List[Union[list, tuple]]) -> List[Union[list, tuple]]:
        records = list(filter(self._function, records))
        return records

# postinit_funcs


class PostinitFuncSum():
    def __init__(self, *postinit_funcs):
        self._funcs = postinit_funcs

    def __call__(self, *args, **kwargs):
        for f in self._funcs:
            f(*args, **kwargs)


def do_nothing(_) -> None:
    pass


class set_num_negtive_qs:
    def __init__(self, num: int):
        self._num = num

    def __call__(self, dataset):
        dataset.num_neg_qs = self._num


class KhopFriends():
    def __init__(
            self,
            k: int,
            tag: str,
            has_subgraph: bool,
            use_backup: bool = True):
        self.k = k
        self.tag = tag
        self.has_subgraph = has_subgraph
        self.use_backup = use_backup

    def __call__(self, dataset: DatasetBase):
        '''
        Find k-hop subgraph in social graph for each user

        In subgraphs[?], the FIRST of `ids` is central user and `ids` is `graph`'s indice.
        '''
        k_hop_file = dataset.path/dataset.name / \
            f'{dataset.name}-{self.k}-hop-{self.tag}.pkl'
        if self.use_backup and os.path.exists(k_hop_file):
            dataset.subgraphs = load_pyobj(k_hop_file)
        else:
            subgraphs = {}
            for start_u in range(dataset.num_users):
                all_friends = {start_u}
                last_hop = {start_u}
                for _ in range(self.k):
                    last_hop_next = set()
                    for u in last_hop:
                        last_hop_next.update(dataset.friend_dict[u])
                    all_friends.update(last_hop_next)
                    last_hop = last_hop_next
                all_friends.remove(start_u)
                all_friends = np.array(
                    [start_u] + list(all_friends), dtype=np.long)
                if self.has_subgraph:
                    subgraphs[start_u] = {
                        'ids': all_friends,
                        'graph': dataset.social_graph[all_friends].tocsc()[:, all_friends]
                    }
                else:
                    subgraphs[start_u] = {'ids': all_friends}
            dataset.subgraphs = subgraphs
            if self.use_backup:
                save_pyobj(k_hop_file, dataset.subgraphs)

# sample_funcs


def itemrec_sample(dataset: TrainDataset, index: int):
    p, q_pos = dataset.pos_pairs[index]
    while True:
        i = np.random.randint(dataset.num_qs)
        if dataset.ground_truth[p, i] == 0 and i != q_pos:
            return i

# getitem_funcs

# for training, return values should be dict
# whose keys is the same as model's forward.


def default_train_getitem(self: TrainDataset, index: int):
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


def default_fully_ranking_test_getitem(self: FullyRankingTestDataset, index: int):
    ground_truth = torch.from_numpy(
        self.ground_truth[index].toarray()).view(-1)
    train_mask = torch.from_numpy(self.train_mask[index].toarray()).view(-1)
    # dict1 -> model.evaluate dict2 -> test
    return {'ps': index}, {'train_mask': train_mask, 'ground_truth': ground_truth}


def default_leave_one_out_test_getitem(self: LeaveOneOutTestDataset, index: int):
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
