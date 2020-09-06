#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['dataset', 'functional', 'DatasetFuncs', 'save_pyobj', 'load_pyobj']

from collections import namedtuple
from typing import Union, Any, cast, IO
import pickle
import gzip
import logging
from pathlib import Path

DatasetFuncs = namedtuple(
    'DatasetFuncs',
    ('record', 'postinit', 'sample', 'getitem', 'length')
)


def save_pyobj(path: Union[str, Path], obj: Any, compress_level=2) -> None:
    logging.debug(f'Save {path} with GZIP level {compress_level}')
    with gzip.open(path, 'wb', compress_level) as f:
        pickle.dump(obj, cast(IO[bytes], f))


def load_pyobj(path: Union[str, Path]) -> Any:
    logging.debug(f'Load {path}')
    with gzip.open(path, 'rb') as f:
        return pickle.load(cast(IO[bytes], f))
