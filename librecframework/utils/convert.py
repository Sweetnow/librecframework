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
from ..metric import _Metric, _ALL_METRICS

__all__ = ['name_to_metric', 'name_to_activation',
           'path_to_saved_model', 'path_to_saved_log', 'scisp_to_torch']


def name_to_metric(name: str, topk: int) -> _Metric:
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


def _path_to_saved_thing(path: Union[str, Path], suffix: str) -> Tuple[list, str]:
    # pylint: disable=R1705
    '''
    get infos and saved model or log paths by related file path
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is not found')
    path = Path(os.path.abspath(path))
    if os.path.isdir(path):
        return _path_to_saved_thing(path / 'model.json', suffix)
    else:

        LEGAL_SUFFIX = ['json', 'pth']
        if suffix not in LEGAL_SUFFIX:
            raise ValueError(f'No such suffix {suffix} in {LEGAL_SUFFIX}')

        group_infos_path = Path(os.path.dirname(path)) / 'model.json'
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
            if os.path.isdir(path):
                save_dir = path
            else:
                save_dir = os.path.dirname(path)
            target_paths = []
            for file in os.listdir(save_dir):
                if fnmatch(file, f"{infoid}*.{suffix}"):
                    target_paths.append(file)
            if len(target_paths) == 0:
                raise FileNotFoundError(f"Cannot find {infoid} in {save_dir}")
            elif len(target_paths) > 1:
                print('==================================')
                for i, file in enumerate(target_paths):
                    print(f'[{i}]: {file}')
                num = input('Enter pth number:')
                target_path = target_path[int(num)]
            else:
                target_path = target_paths[0]
            target_path = Path(save_dir) / target_path
            metadatas.append({
                'id': infoid,
                'info': info,
                'path': target_path,
                'env': rawinfo['env']
            })
        return metadatas, dataset_name


def path_to_saved_model(path: Union[str, Path]) -> Tuple[list, str]:
    return _path_to_saved_thing(path, 'pth')


def path_to_saved_log(path: Union[str, Path]) -> Tuple[list, str]:
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


def _activation_rebuild(name: str) -> nn.Module:
    # pylint: disable=W1401,W0123
    _, func, arg = re.match('(.+)\((.*)\)', name)
    if arg != '':
        arg = arg.split('=')
    tables = {
        # name: (class, arg lambda)
        'ReLU': (nn.ReLU, None),
        'ReLU6': (nn.ReLU6, None),
        'LeakyReLU': (nn.LeakyReLU, float),
        'PReLU': (nn.PReLU, int),
        'ELU': (nn.ELU, float),
        'GLU': (nn.GLU, int),
        'CELU': (nn.CELU, float),
        'SELU': (nn.SELU, None),
        'Sigmoid': (nn.Sigmoid, None),
        'Softmax': (nn.Softmax, lambda x: None if x == 'None' else int(x)),
        'Tanh': (nn.Tanh, None)
    }
    if func in tables:
        cls, arg_lambda = tables[func]
        if arg_lambda is None:
            return cls()
        else:
            arg = {arg[0]: arg_lambda(arg[1])}
            return cls(**arg)
    else:
        raise ValueError(f'Unknown activation function {name}')


def split_log_path(path: Union[str, Path]) -> Dict[str, str]:
    path = os.path.normpath(path)
    dirs = path.split(os.sep)
    dirs.reverse()
    for i, d in enumerate(dirs):
        rs = re.match(r'\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[A-Z0-9]{4}', d)
        if rs is not None:
            break
    return {
        'model': dirs[i + 3],
        'tag': dirs[i + 1],
        'type': dirs[i + 2][-4:] # test or tune
    }
