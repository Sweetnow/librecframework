#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import namedtuple, OrderedDict
from typing import List, Any, Optional, Union
from argparse import ArgumentParser, Action
from itertools import product
from multiprocessing import cpu_count
import torch
from .functional import add_bool_argument, add_int_argument, add_str_argument, add_float_argument
from . import Argument

__all__ = ['ArgumentManager', 'HyperparamManager',
           'default_loader_argument_manager',
           'default_env_argument_manager']


def _validate(arg: Argument, value):
    if arg.validator is not None:
        if arg.multi:
            for one in value:
                is_ok = arg.validator(one)
                if not is_ok:
                    raise ValueError(f'{one} did not pass the validator of {arg}')
        else:
            if not arg.validator(value):
                raise ValueError(f'{value} did not pass the validator of {arg}')


class ArgumentManager():
    '''
    manage argument pipeline in Recframework
    - register arguments
    - get arguments input in cli
    - divide arguments into environment part and model part
    - create info class and its instance (See `HyperparamManager`)
    '''

    def __init__(self, title: str, description: str):
        self._args = OrderedDict()
        self._parsed_args = OrderedDict()
        self._is_parsed = False
        self._add_arg_functions = {
            float: add_float_argument,
            int: add_int_argument,
            str: add_str_argument,
            bool: add_bool_argument
        }
        self._actions = {
            float: self._callback_action('other'),
            int: self._callback_action('other'),
            str: self._callback_action('other'),
            bool: {
                'true': self._callback_action('true'),
                'false': self._callback_action('false')
            }
        }
        self.title = title
        self.description = description

    def _callback_action(self, tag: str) -> type:
        tags = ('true', 'false', 'other')
        if not tag in tags:
            raise ValueError(f'{tag} should be in {tags}')

        class CallbackAction(Action):
            # pylint: disable=W0622
            def __init__(
                    self,
                    option_strings,
                    dest,
                    nargs=None,
                    const=None,
                    default=None,
                    type=None,
                    choices=None,
                    required=False,
                    help=None,
                    metavar=None):
                super().__init__(
                    option_strings=option_strings,
                    dest=dest,
                    nargs=nargs,
                    const=const,
                    default=default,
                    type=type,
                    choices=choices,
                    required=required,
                    help=help,
                    metavar=metavar)

            def __call__(self_action, parser, namespace, values, option_string=None):
                # pylint: disable=E0213
                pname = self_action.dest
                if tag == 'true':
                    values = True
                elif tag == 'false':
                    values = False
                setattr(namespace, pname, values)

                if not pname in self._args:
                    raise RuntimeError(f'{pname} not in {self._args}')

                arg = self._args[pname]
                _validate(arg, values)
                self._parsed_args[pname] = values
                if self._args.keys() == self._parsed_args.keys():
                    self._is_parsed = True

        return CallbackAction

    def keys(self):
        if not self._is_parsed:
            return self._args.keys()
        else:
            return self._parsed_args.keys()

    def values(self):
        if not self._is_parsed:
            return self._args.values()
        else:
            return self._parsed_args.values()

    def items(self):
        if not self._is_parsed:
            return self._args.items()
        else:
            return self._parsed_args.items()

    def __contains__(self, index):
        if not self._is_parsed:
            return index in self._args
        else:
            return index in self._parsed_args

    def __getitem__(self, index):
        if not self._is_parsed:
            return self._args[index]
        else:
            return self._parsed_args[index]

    def __repr__(self):
        if not self._is_parsed:
            return f'{self.title}({self._args})'
        else:
            return f'{self.title}({self._parsed_args})'

    def __add__(self, other):
        o = _ArgumentManagerSum()
        o._sub_managers.append(self)

        if isinstance(other, ArgumentManager):
            o._sub_managers.append(other)
        elif isinstance(other, _ArgumentManagerSum):
            o._sub_managers += other._sub_managers
        else:
            raise TypeError(f'Unknown type {type(other)}')

        return o

    def register_add_arg_function(self, dtype: type, func) -> None:
        '''
        user-defined add_argument function for `dtype`

        `func` must have TWO argument:
        - parser: Union[ArgumentParser, _ArgumentGroup],
        - arg: Argument
        '''
        if callable(func):
            self._add_arg_functions[dtype] = func
        else:
            raise ValueError(f'{func} is not callable')

    def register_action(self, dtype: type, action: type) -> None:
        '''
        user-defined action for `dtype`

        `action` must be subclass of argparse.Action
        '''
        self._actions[dtype] = action

    def register(
            self,
            pname_or_argument: Union[str, Argument],
            cli_aliases: Optional[List[str]] = None,
            *,
            multi: bool = False,
            dtype: type = float,
            validator: Optional = None,
            helpstr: Optional[str] = None,
            default: Optional[Any] = None) -> None:
        if isinstance(pname_or_argument, Argument):
            pname = pname_or_argument.pname
            cli_aliases = pname_or_argument.cli_aliases
            multi = pname_or_argument.multi
            dtype = pname_or_argument.dtype
            validator = pname_or_argument.validator
            helpstr = pname_or_argument.helpstr
            default = pname_or_argument.default
        else:
            pname = pname_or_argument

        if cli_aliases is None:
            cli_aliases = [f'--{pname.replace("_","-")}']

        if default is None:
            help_prefix = '<Required> '
        else:
            help_prefix = f'<Optional: {default}> '

        if helpstr is None:
            helpstr = help_prefix + pname
        else:
            helpstr = help_prefix + helpstr

        arg = Argument(
            pname=pname,
            cli_aliases=cli_aliases,
            multi=multi,
            dtype=dtype,
            validator=validator,
            helpstr=helpstr,
            default=default)

        if not default is None:
            _validate(arg, default)
            self._parsed_args[pname] = default

        self._args[pname] = arg
        if self._args.keys() == self._parsed_args.keys():
            self._is_parsed = True

    def add_args(self, parser: ArgumentParser) -> None:
        group = parser.add_argument_group(self.title, self.description)
        for arg in self._args.values():
            self._add_arg_functions[arg.dtype](
                group, arg, self._actions[arg.dtype])

    def parse_args(self) -> None:
        parser = ArgumentParser(prog=self.title, description=self.description)
        self.add_args(parser)
        parser.parse_args()


class HyperparamManager(ArgumentManager):
    def __init__(self, title: str, description: str, classname: str):
        super().__init__(title, description)
        self.classname = classname

    def info_class(self):
        Info = namedtuple(self.classname, self._args.keys())
        return Info

    def to_infos(self):
        cls = self.info_class()
        if self._parsed_args is None:
            raise RuntimeError('run `parse_args` before `to_infos`')
        arg_list = []
        pnames = []
        for pname, arg in self._args.items():
            pnames.append(pname)
            parsed_arg = self._parsed_args[pname]
            if arg.multi:
                arg_list.append(parsed_arg)
            else:
                arg_list.append([parsed_arg])
        for args in product(*arg_list):
            d = dict(zip(pnames, args))
            info = cls(**d)
            yield info


class _ArgumentManagerSum():
    def __init__(self):
        self._sub_managers = []

    def __add__(self, other):
        o = _ArgumentManagerSum()
        o._sub_managers = self._sub_managers

        if isinstance(other, ArgumentManager):
            o._sub_managers.append(other)
        elif isinstance(other, _ArgumentManagerSum):
            o._sub_managers += other._sub_managers
        else:
            raise TypeError(f'Unknown type {type(other)}')

        return o

    def add_args(self, parser: ArgumentParser) -> None:
        for manager in self._sub_managers:
            manager.add_args(parser)

    def parse_args(self) -> None:
        parser = ArgumentParser()
        self.add_args(parser)
        parser.parse_args()


def default_loader_argument_manager(train_or_test: str) -> ArgumentManager:
    train_or_test = train_or_test.lower()
    tags = ('train', 'test', 'both')
    if not train_or_test in tags:
        raise ValueError(
            f'{train_or_test} is not supported. Choose one from {tags}')

    manager = ArgumentManager('Dataloader Arguments', None)
    if train_or_test in ('train', 'both'):
        manager.register(
            'batch_size',
            ['-BS', '--batch-size'],
            dtype=int,
            validator=lambda x: x > 0,
            helpstr='train loader batch size',
            default=4096
        )
        manager.register(
            'batch_worker',
            ['-BW', '--batch-worker'],
            dtype=int,
            validator=lambda x: cpu_count() >= x >= 0,
            helpstr='train loader batch woker',
            default=8
        )

    if train_or_test in ('test', 'both'):
        manager.register(
            'test_batch_size',
            ['-TBS', '--test-batch-size'],
            dtype=int,
            validator=lambda x: x > 0,
            helpstr='test loader batch size',
            default=4096
        )
        manager.register(
            'test_batch_worker',
            ['-TBW', '--test-batch-worker'],
            dtype=int,
            validator=lambda x: cpu_count() >= x >= 0,
            helpstr='test loader batch woker',
            default=4
        )
    return manager


def default_env_argument_manager(
        train_or_test: str,
        supported_datasets: Optional[List[str]] = None) -> ArgumentManager:
    train_or_test = train_or_test.lower()

    tags = ('train', 'test', 'both')
    if not train_or_test in tags:
        raise ValueError(
            f'{train_or_test} is not supported. Choose one from {tags}')

    if train_or_test == 'both':
        train_or_test = 'train'

    manager = ArgumentManager('Environment Arguments', None)

    if train_or_test == 'train':
        if supported_datasets is None or len(supported_datasets) == 0:
            raise ValueError('No supported datasets in train mode')
        if len(supported_datasets) == 1:
            manager.register(
                'dataset',
                ['-DS', '--dataset'],
                validator=lambda x: x in supported_datasets,
                dtype=str,
                helpstr=f'Which dataset {supported_datasets}',
                default=supported_datasets[0]
            )
        else:
            manager.register(
                'dataset',
                ['-DS', '--dataset'],
                validator=lambda x: x in supported_datasets,
                dtype=str,
                helpstr=f'Which dataset {supported_datasets}'
            )

    manager.register(
        'device',
        ['-D', '--device'],
        dtype=int,
        validator=lambda x: torch.cuda.device_count() > x >= 0,
        helpstr=f'Which GPU 0-{torch.cuda.device_count() - 1}'
    )
    manager.register(
        'tag',
        ['-T', '--tag'],
        dtype=str,
        helpstr='Just a tag'
    )

    if train_or_test == 'train':
        manager.register(
            'sample_epoch',
            ['-SEP', '--sample-epoch'],
            dtype=int,
            validator=lambda x: x > 0,
            helpstr='the maximum number of samples(epochs)',
            default=500
        )
        manager.register(
            'sample_worker',
            ['-SW', '--sample-worker'],
            dtype=int,
            validator=lambda x: x > 0,
            helpstr='the number of cores for sampling',
            default=16
        )
        manager.register(
            'epoch',
            ['-EP', '--epoch'],
            dtype=int,
            validator=lambda x: x > 0,
            helpstr='the number of epochs',
            default=500
        )

    if train_or_test == 'test':
        manager.register(
            'file',
            ['-F', '--file'],
            dtype=str,
            validator=os.path.exists,
            helpstr='file or dir for testing'
        )

    return manager
