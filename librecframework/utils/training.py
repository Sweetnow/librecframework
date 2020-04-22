#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List
import numpy as np

__all__ = ['metrics_sliding_max', 'check_overfitting', 'early_stop']


def metrics_sliding_max(
        metrics_log: Dict[str, List[float]],
        window_size: int,
        target: str) -> Dict[str, List[float]]:
    # max
    maxs = {title: 0 for title in metrics_log.keys()}
    assert target in maxs
    length = len(metrics_log[target])
    for v in metrics_log.values():
        assert length == len(v)
    if window_size >= length:
        for k, v in metrics_log.items():
            maxs[k] = np.mean(v)
    else:
        for i in range(length - window_size):
            now = np.mean(metrics_log[target][i:i + window_size])
            if now > maxs[target]:
                for k, v in metrics_log.items():
                    maxs[k] = np.mean(v[i:i + window_size])
    return maxs


def check_overfitting(
        metrics_log: Dict[str, List[float]],
        target: str,
        threshold: float) -> bool:
    maxs = metrics_sliding_max(metrics_log, 1, target)
    assert target in maxs
    overfit = (maxs[target] - metrics_log[target][-1]) > threshold
    return overfit


def early_stop(
        metric_log: Dict[str, List[float]],
        early: int,
        threshold: float = 0) -> int:
    if len(metric_log) >= 2 and metric_log[-1] < metric_log[-2] and metric_log[-1] > threshold:
        return early - 1
    else:
        return early
