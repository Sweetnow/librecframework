#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import sys
import os
from pathlib import Path
from time import time
from typing import Union, Optional
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import logging
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from . import DatasetFuncs, save_pyobj, load_pyobj

__all__ = ['DatasetBase', 'TrainDataset',
           'FullyRankingTestDataset', 'LeaveOneOutTestDataset']


def _sampler(args):
    seed, index, dataset, onesampler, other_args = args
    cnt = other_args['cnt']
    np.random.seed(seed)
    rs = [onesampler(dataset, index) for _ in range(cnt)]
    return index, rs


class DatasetBase(Dataset):
    '''
    generate dataset from raw *.txt

    contains:
        tensors like (`p`, `q_p`, `q_n`) for BPR

    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `seed`: seed of `np.random`
    '''

    def __init__(
            self,
            path: Union[str, Path],
            name: str,
            task: str,
            funcs: DatasetFuncs) -> None:
        self.path, self.name, self.task = Path(path), name, task
        self.funcs = funcs
        self.num_ps, self.num_qs = self._load_data_size()
        self.num_users, self.num_items = self.num_ps, self.num_qs
        self.records, self.pos_pairs, self.ground_truth = self._load_records(
            self.funcs.record)
        self.friend_dict, self.social_graph = self._load_social_relation()
        self._post_init()

    def _post_init(self):
        self.funcs.postinit(self)

    def __getitem__(self, index):
        return self.funcs.getitem(self, index)

    def __len__(self):
        return self.funcs.length(self)

    def _load_data_size(self):
        with open(self.path / self.name / 'data_size.txt', 'r') as f:
            num_users, num_items = [
                int(s) for s in f.readline().strip().split('\t')][:2]
            return num_users, num_items

    def _load_records(self, record_func) -> Tuple[list, list, sp.csr_matrix]:
        '''
        one record: p, q, ...
        '''
        with open(self.path / self.name / f'{self.task}.txt', 'r') as f:
            records = [[int(one) for one in line.strip().split('\t')]
                       for line in f]
        records = record_func(records)
        pos_pairs = [x[0:2] for x in records]
        indice = np.array(pos_pairs, dtype=np.int32)
        values = np.ones(len(pos_pairs), dtype=np.float32)
        # FIXME: if the same index pair appears many times, the value will be not equal to one
        ground_truth = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_ps, self.num_qs)).tocsr()
        return records, pos_pairs, ground_truth

    def _load_social_relation(self) -> Tuple[dict, sp.csr_matrix]:
        try:
            with open(self.path / self.name / 'social_relation.txt', 'r') as f:
                friends = [[int(one) for one in line.strip().split('\t')]
                           for line in f]
            indice = []
            friend_dict = defaultdict(set)
            for u1, u2 in friends:
                indice += [(u1, u2), (u2, u1)]
                friend_dict[u1].add(u2)
                friend_dict[u2].add(u1)
            indice_sp = np.array(indice, dtype=np.int32)
            values_sp = np.ones(len(indice), dtype=np.float32)
            social_graph = sp.coo_matrix(
                (values_sp, (indice_sp[:, 0], indice_sp[:, 1])),
                shape=(self.num_users, self.num_users)).tocsr()
        except FileNotFoundError:
            return None, None
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        else:
            return friend_dict, social_graph


class TrainDataset(DatasetBase):
    '''
    do negative sample with multi-processes when __init__ TO AVOID 'SAME RNG' BUG
    contains:
        self.neg_qs: negative qs for each positive pair
    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `num_worker`: #worker for sampling
    - `max_epoch`: the number of negative qs (when runing `e` epoch, use `e`-th neg-q)
    - `seed`: seed of `np.random`
    - `use_backup`: save/load negative qs to/from specific file
    '''

    def __init__(
            self,
            path: Union[str, Path],
            name: str,
            num_worker: int,
            max_epoch: int,
            *,
            funcs: DatasetFuncs,
            sample_tag: str,
            seed: Optional[int] = None,
            use_backup: bool = True):
        super().__init__(path, name, 'train', funcs)

        # prepare negative sample
        self.init_epoch()
        self.max_epoch, self.num_worker = max_epoch, num_worker
        self.tag = sample_tag
        self.neg_qs = None
        self.use_backup = use_backup
        self.sample(funcs.sample, self.tag, seed, self.use_backup)
        logging.debug('finish loading neg sample')

    def init_epoch(self) -> None:
        '''
        Must be CALLED before training
        For multi model in one task
        '''
        self.epoch = 0

    def next_epoch(self) -> None:
        '''
        Must be CALLED after one epoch
        '''
        self.epoch += 1
        if not 0 <= self.epoch < self.max_epoch:
            logging.critical(
                'SAMPLE LIST REUSED! PLEASE ENSURE CALL init_epoch FIRST')
            self.epoch %= self.max_epoch

    def resample(self, onesampler) -> None:
        '''
        refresh negative qs list
        '''
        assert self.max_epoch > 0

        def myiter(self, onesampler, other_args):
            for i in range(len(self)):
                yield (np.random.randint(2 ** 31 - 1), i, self, onesampler, other_args)

        start = time()
        logging.info(
            f'Start sample max_epoch={self.max_epoch} #Core={self.num_worker}')
        with Pool(self.num_worker) as p:
            neg_qs = dict(p.map(
                _sampler,
                myiter(self, onesampler, {'cnt': self.max_epoch})
            ))
        self.neg_qs = []
        for i in range(len(neg_qs)):
            self.neg_qs.append(np.array(neg_qs[i], dtype=np.int32))
        self.neg_qs = np.vstack(self.neg_qs)
        logging.info(
            f'Neg-sample with #Core={self.num_worker} Time: {time()-start}s')

    def sample(self, onesampler, tag: str, seed: Optional[int], use_backup: bool) -> None:
        np.random.seed(seed)
        if self.max_epoch > 0:
            sample_file = self.path/self.name / \
                f'{self.name}-neg-{self.max_epoch}-{seed}-{tag}.pkl'
            if use_backup and os.path.exists(sample_file):
                self.neg_qs = load_pyobj(sample_file)
            else:
                self.resample(onesampler)
                if use_backup:
                    save_pyobj(sample_file, self.neg_qs)
        else:
            self.neg_qs = np.empty([len(self), 0], dtype=np.int32)


class FullyRankingTestDataset(DatasetBase):
    '''
    For ranking-all test
    '''

    def __init__(
            self,
            path: Union[str, Path],
            name: str,
            task: str,
            train_dataset: DatasetBase,
            *,
            funcs: DatasetFuncs) -> None:
        super().__init__(path, name, task, funcs)
        self.train_mask = train_dataset.ground_truth
        assert self.train_mask.shape == self.ground_truth.shape


def _leave_one_out_sampler(args):
    seed, index, dataset, ignore_sets, cnt = args
    np.random.seed(seed)
    p, q = dataset.pos_pairs[index]
    sample_list = list(set(range(dataset.num_qs)) - ignore_sets[p] - {q})
    rs = np.random.choice(sample_list, dataset.num_neg_qs, replace=False)
    return index, rs


class LeaveOneOutTestDataset(DatasetBase):
    def __init__(
            self,
            path: Union[str, Path],
            name: str,
            task: str,
            train_dataset: DatasetBase,
            *,
            funcs: DatasetFuncs) -> None:
        self.num_neg_qs = 999
        super().__init__(path, name, task, funcs)
        self._train_pos_pair = train_dataset.pos_pairs
        self.neg_qs = self.load_negative_qs()

    def generate_negative_qs(self):
        def myiter(self, ignore_sets):
            for i in range(len(self)):
                yield (np.random.randint(2 ** 31 - 1), i, self, ignore_sets, self.num_neg_qs)
        pos_dict = defaultdict(set)
        for p, q in self._train_pos_pair:
            pos_dict[p].add(q)
        neg_qs = []
        with Pool(int(cpu_count() / 2)) as p:
            neg_qs_dict = dict(p.map(
                _leave_one_out_sampler,
                myiter(self, pos_dict)
            ))
        for i in range(len(self.pos_pairs)):
            neg_qs.append(neg_qs_dict[i])
        return neg_qs

    def load_negative_qs(self):
        path = self.path / self.name / f'{self.task}.negative.txt'
        if os.path.exists(path):
            neg_qs = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    line = [int(one) for one in line]
                    if len(line) != self.num_neg_qs:
                        raise ValueError(f'Wrong negtive qs length')
                    neg_qs.append(line)
        else:
            neg_qs = self.generate_negative_qs()
            with open(path, 'w') as f:
                lines = ['\t'.join(map(str, line)) + '\n' for line in neg_qs]
                f.writelines(lines)
        neg_qs = np.array(neg_qs, dtype=np.int32)
        return neg_qs
