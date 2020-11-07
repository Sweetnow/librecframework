#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union, Optional
from pathlib import Path
import json
from visdom import Visdom
from .convert import path_to_saved_log, split_log_path

__all__ = ['VisShow', 'replay']


class VisShow(object):
    '''
    Quick utility that wraps Visdom by `name`.

    Provide low-level interface `get_window` and `set_window`

    Provide high-level interface `update`, `xlabel` and `ylabel`

    `update`: append data to an existed window or create a new window with data

    `xlabel` `ylabel`: change the xlabel/ylabel of an existed window
    '''

    def __init__(self, server: str, port: int, envdir: str, subenv: str) -> None:
        self.vis = Visdom(server, port=port, env=f'{envdir}_{subenv}')

    def get_window(self, target: str) -> Optional[str]:
        attrname = f'__{target}'
        if hasattr(self, attrname):
            return getattr(self, attrname)
        else:
            return None

    def set_window(self, name: str, win: str) -> None:
        attrname = f'__{name}'
        if hasattr(self, attrname):
            raise ValueError(f'{name} has existed')
        else:
            setattr(self, attrname, win)

    def update(self, target: str, X: List[Union[int, float]], Y: List[Union[int, float]]) -> None:
        win = self.get_window(target)
        if not win is None:
            self.vis.line(Y, X, win=win, update='append')
        else:
            self.set_window(target, self.vis.line(
                Y, X, opts={'title': target}))

    def xlabel(self, target: str, label: str) -> None:
        win = self.get_window(target)
        if not win is None:
            self.vis.update_window_opts(win, {'xlabel': label})
        else:
            raise ValueError(f'{target} has not existed')

    def ylabel(self, target: str, label: str) -> None:
        win = self.get_window(target)
        if not win is None:
            self.vis.update_window_opts(win, {'ylabel': label})
        else:
            raise ValueError(f'{target} has not existed')


def replay(path: Union[str, Path], config_path: Union[str, Path]):
    with open(config_path, 'r') as f:
        config = json.load(f)
    server = config['visdom']['server']

    metadatas, dataset_name = path_to_saved_log(path)
    port = config['visdom']['port'][dataset_name]
    path_info = split_log_path(path)

    for metadata in metadatas:
        test_interval = metadata['env']['training']['test_interval']
        with open(metadata['path'], 'r') as f:
            data = json.load(f)
        envdir = path_info['model']
        sub_env = f"{path_info['tag']}-{'-'.join(map(str,metadata['info']._asdict().values()))}"
        vis = VisShow(
            server=server,
            port=port,
            envdir=envdir,
            subenv=sub_env
        )
        for k, v in data['metrics'].items():
            vis.update(
                k, list(range(0, test_interval * len(v), test_interval)), v)
        for k, v in data['trainhooks'].items():
            vis.update(k, list(range(0, len(v))), v)
