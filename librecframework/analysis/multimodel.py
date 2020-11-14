#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Union
from pathlib import Path
from collections import defaultdict
import pandas as pd
from ..utils.convert import path_to_saved_log

__all__ = ['multimodel_performance_reporter']


def multimodel_performance_reporter(
        config: Dict[str, str],
        output: Union[str, Path],
        metric_tag: str = 'best',
        check_name: bool = True) -> None:
    """
    Organize models' metrics and report them as a csv table

    Args:
    - config: the paths of model log file `model.json`, e.g. `{ modelname: path_to_file }`
    - output: the output csv path
    - metric_tag: the tag of used metrics, e.g. `best`
    - check_name: whether to check the consistency between `modelname` in `config` and `model` in log file

    Exception:
    - ValueError: find more than 1 model log in the path or find inconsistent modelname if `check_name` is `True`

    """
    metrics = {}
    hyperparameters = {}
    metrics_name = []
    metrics_name_ignore = set()
    hyperparameters_name = []
    hyperparameters_name_ignore = set()
    models = list(config.keys())
    for k, v in config.items():
        metadatas, _ = path_to_saved_log(Path(v))
        if len(metadatas) > 1:
            raise ValueError(f'the number of metadata is more than one {v}')
        metadata = metadatas[0]
        if check_name:
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
            df[hyperparameter].append(
                str(hyperparameters[model].get(hyperparameter, '')))
    df = pd.DataFrame({k: pd.Series(v, name=k, index=models)
                       for k, v in df.items()})
    df.to_csv(output)
