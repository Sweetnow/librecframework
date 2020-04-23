#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Union
from argparse import ArgumentParser, _ArgumentGroup
import re
from . import Argument

__all__ = ['add_float_argument', 'add_int_argument',
           'add_str_argument', 'add_bool_argument']


def _add_argument_factory(dtype: type):
    def _add_argument(
            parser: Union[ArgumentParser, _ArgumentGroup],
            arg: Argument,
            action: Union[type, str] = 'default') -> None:
        assert arg.dtype is dtype
        if action == 'default':
            kwargs = {
                'help': arg.helpstr,
                'dest': arg.pname,
                'type': arg.dtype
            }
        else:
            kwargs = {
                'help': arg.helpstr,
                'dest': arg.pname,
                'type': arg.dtype,
                'action': action
            }

        if arg.multi:
            kwargs['nargs'] = '+'

        if not arg.default is None:
            kwargs['default'] = arg.default
        else:
            kwargs['required'] = True

        parser.add_argument(*arg.cli_aliases, **kwargs)

    supported_types = (float, str, int)
    if dtype in supported_types:
        return _add_argument
    else:
        raise ValueError(
            f'Unsupported type {dtype}. Supported types are {supported_types}')


add_float_argument = _add_argument_factory(float)
add_int_argument = _add_argument_factory(int)
add_str_argument = _add_argument_factory(str)


def add_bool_argument(
        parser: Union[ArgumentParser, _ArgumentGroup],
        arg: Argument,
        actions: Union[Dict[str, type], str] = 'default') -> None:
    assert arg.dtype is bool

    group = parser.add_mutually_exclusive_group()
    if actions == 'default':
        true_kwargs = {
            'help': arg.helpstr,
            'dest': arg.pname,
            'nargs': 0,
            'action': 'store_true'

        }
        false_kwargs = {
            'help': arg.helpstr,
            'dest': arg.pname,
            'nargs': 0,
            'action': 'store_false'
        }
    else:
        true_kwargs = {
            'help': arg.helpstr,
            'dest': arg.pname,
            'nargs': 0,
            'action': actions['true']
        }
        false_kwargs = {
            'help': arg.helpstr,
            'dest': arg.pname,
            'nargs': 0,
            'action': actions['false']
        }

    if arg.multi:
        raise ValueError('arg.multi is not supported by bool argument')

    if not arg.default is None:
        true_kwargs['default'] = arg.default
        false_kwargs['default'] = arg.default

    group.add_argument(*arg.cli_aliases, **true_kwargs)

    false_aliases = []
    for alias in arg.cli_aliases:
        rmatch = re.match('(-*)(.*)', alias)
        prefix, body = rmatch[1], rmatch[2]
        false_alias = prefix + 'no-' + body
        false_aliases.append(false_alias)
    group.add_argument(*false_aliases, **false_kwargs)
