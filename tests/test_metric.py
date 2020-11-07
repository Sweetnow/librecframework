#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
from librecframework.metric import _ALL_METRICS
from librecframework.metric import *
from librecframework.utils.convert import name_to_metric

topk = 3
eps = 1e-6


def test_convert():
    for k, v in _ALL_METRICS.items():
        m = v(topk)
        r = name_to_metric(k, topk)
        assert (str(m) == str(r))


def test_precision():
    m = Precision(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = (1 + 0 + 1) / 3 / topk
    assert (abs(m.metric - t) < eps)

def test_recall():
    m = Recall(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    assert (m.metric == 0.75)


def test_ndcg():
    m = NDCG(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = (1 / math.log2(3) + 1 / (1 / math.log2(3) + 1)) / 2
    assert (abs(m.metric - t) < eps)


def test_mrr():
    m = MRR(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = (1 + 0.5) / 2
    assert (abs(m.metric - t) < eps)


def test_leaveonehr():
    m = LeaveOneHR(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = 2 / 3
    assert (abs(m.metric - t) < eps)


def test_leaveonendcg():
    m = LeaveOneNDCG(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = (1 + 1 / math.log2(3)) / 3
    assert (abs(m.metric - t) < eps)


def test_leaveonemrr():
    m = LeaveOneMRR(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3]
    ])
    m(score, gt)
    m.stop()
    t = (1 + 0.5) / 3
    assert (abs(m.metric - t) < eps)


def test_square():
    m = MRR(topk)
    m.start()
    gt = torch.FloatTensor([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1]
    ])
    score = torch.FloatTensor([
        [10, 4, 3, 2, 9],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 4, 3],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 1, 2]
    ])
    m(score, gt)
    m.stop()
    t = (1 + 0.5+1 + 1 / 3) / 4
    assert (abs(m.metric - t) < eps)
