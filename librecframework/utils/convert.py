#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict
import os
import json
import re
from pathlib import Path
from collections import namedtuple
from fnmatch import fnmatch
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..metric import Metric, _ALL_METRICS

__all__ = ['name_to_metric', 'name_to_activation',
           'path_to_saved_model', 'path_to_saved_log',
           'scisp_to_torch', 'torch_to_scisp']


def name_to_metric(name: str, topk: int) -> Metric:
    return _ALL_METRICS[name](topk)


def name_to_activation(name: str) -> nn.Module:
    name = name.lower()
    tables = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'elu': nn.ELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    if name in tables:
        return tables[name]()
    else:
        raise ValueError(
            f'Improper activation function name {name}. Existing activations are {tables.keys()}')


def _path_to_saved_thing(path: Path, suffix: str) -> Tuple[list, str]:
    # pylint: disable=R1705
    '''
    get infos and saved model or log paths by related file path
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is not found')
    path = path.resolve()
    if os.path.isdir(path):
        return _path_to_saved_thing(path / 'model.json', suffix)
    else:

        LEGAL_SUFFIX = ['json', 'pth']
        if suffix not in LEGAL_SUFFIX:
            raise ValueError(f'No such suffix {suffix} in {LEGAL_SUFFIX}')
        group_infos_path = path.parent / 'model.json'
        with open(group_infos_path, 'r') as f:
            group_infos = json.load(f)
            group_infos_dct = {
                info['id']: info for info in group_infos
            }
            dataset_name = group_infos[0]['env']['dataset']
            for info in group_infos:
                assert info['env']['dataset'] == dataset_name
        basename = os.path.basename(path)
        if basename == 'model.json':
            rawinfos = group_infos
        elif basename.split('.')[-1] in LEGAL_SUFFIX:
            infoid = '_'.join(basename.split('.')[0].split('_')[:2])
            rawinfos = [group_infos_dct[infoid]]
        else:
            raise ValueError(f'Unknown file {path}')

        # info -> namedtuple / saved model path
        metadatas = []
        for rawinfo in rawinfos:
            infoid = rawinfo['id']
            info = rawinfo['info']
            infotype = namedtuple('info', info.keys())
            info = infotype(**info)
            if path.is_dir():
                save_dir = path
            else:
                save_dir = path.parent
            target_paths = []
            for file in save_dir.iterdir():
                if fnmatch(str(file), f"{infoid}*.{suffix}"):
                    target_paths.append(file)
            if len(target_paths) == 0:
                raise FileNotFoundError(f"Cannot find {infoid} in {save_dir}")
            elif len(target_paths) > 1:
                print('==================================')
                for i, file in enumerate(target_paths):
                    print(f'[{i}]: {file}')
                num = input('Enter pth number:')
                target_path = target_paths[int(num)]
            else:
                target_path = target_paths[0]
            target_path = Path(save_dir) / target_path
            metadatas.append({
                'id': infoid,
                'info': info,
                'path': target_path,
                'env': rawinfo['env'],
                'metrics': rawinfo['metrics'],
                'model': rawinfo['model']
            })
        return metadatas, dataset_name


def path_to_saved_model(path: Path) -> Tuple[list, str]:
    return _path_to_saved_thing(path, 'pth')


def path_to_saved_log(path: Path) -> Tuple[list, str]:
    return _path_to_saved_thing(path, 'json')


def scisp_to_torch(m: Union[sp.bsr_matrix, sp.coo_matrix,
                            sp.csc_matrix, sp.csr_matrix,
                            sp.dok_matrix, sp.dia_matrix,
                            sp.lil_matrix]) -> torch.Tensor:
    m = m.tocoo()
    t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((m.row, m.col))).long(),
        torch.from_numpy(m.data),
        size=m.shape)
    return t


def torch_to_scisp(t: torch.Tensor) -> sp.coo_matrix:
    if t.is_sparse:
        indice = t._indices().numpy()
        values = t._values().numpy()
        m = sp.coo_matrix(
            (values, (indice[0, :], indice[1, :])), shape=t.shape)
        return m
    else:
        raise TypeError(f'{type(t)} is not torch.sparse')


def split_log_path(path: Path) -> Dict[str, str]:
    pathstr = str(path.resolve(strict=False))
    dirs = pathstr.split(os.sep)
    dirs.reverse()
    last = None
    for i, d in enumerate(dirs):
        rs = re.match(r'\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[A-Z0-9]{4}', d)
        last = i
        if rs is not None:
            break
    return {
        'model': dirs[last + 3],
        'tag': dirs[last + 1],
        'type': dirs[last + 2][-4:]  # test or tune
    }
