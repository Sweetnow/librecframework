#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The module organizes (best) models' metrics with table (csv)
# The input should be json (key: model name, value: path of the best model)
# The output will be an archive (tar) which contains
# metrics table and best hyperparameters or two files

from typing import Dict, Union
import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from ..utils.convert import path_to_saved_log

__all__ = ['multimodel_performance_reporter']

def multimodel_performance_reporter(
        config: Dict[str, str],
        output: Union[str, Path],
        metric_tag: str = 'best') -> None:
    metrics = {}
    hyperparameters = {}
    metrics_name = []
    metrics_name_ignore = set()
    hyperparameters_name = []
    hyperparameters_name_ignore = set()
    models = list(config.keys())
    for k, v in config.items():
        metadatas, _ = path_to_saved_log(v)
        if len(metadatas) > 1:
            raise ValueError(f'the number of metadata is more than one {v}')
        metadata = metadatas[0]
        if k != metadata['model']:
            raise ValueError(
                f"model names are not the same {k} - {metadata['model']}")
        metrics[k] = metadata['metrics'][metric_tag]
        hyperparameters[k] = metadata['info']._asdict()
        for metric in metrics[k].keys():
            if metric not in metrics_name_ignore:
                metrics_name.append(metric)
                metrics_name_ignore.add(metric)
        for hyperparameter in hyperparameters[k].keys():
            if hyperparameter not in hyperparameters_name_ignore:
                hyperparameters_name.append(hyperparameter)
                hyperparameters_name_ignore.add(hyperparameter)
    df = defaultdict(list)
    for model in models:
        for metric in metrics_name:
            df[metric].append(str(metrics[model].get(metric, '')))
        for hyperparameter in hyperparameters_name:
            df[hyperparameter].append(str(hyperparameters[model].get(hyperparameter, '')))
    df = pd.DataFrame({k: pd.Series(v, name=k, index=models) for k, v in df.items()})
    df.to_csv(output)
