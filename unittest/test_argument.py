#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import argparse
from librecframework.argument.manager import ArgumentManager


def test_argument_manager_int():
    am = ArgumentManager('test', 'pytest')
    am.register(
        'test_int',
        ['-I'],
        dtype=int,
        validator=lambda x: x > 0
    )
    parser = argparse.ArgumentParser()
    am.add_args(parser)
    parser.parse_args(['-I', '1'])
    assert am['test_int'] == 1


def test_argument_manager_float():
    am = ArgumentManager('test', 'pytest')
    am.register(
        'test_float_multi',
        ['-F'],
        dtype=float,
        validator=lambda x: x > 0
    )
    parser = argparse.ArgumentParser()
    am.add_args(parser)
    parser.parse_args(['-F', '2.5'])
    assert am['test_float_multi'] == 2.5


def test_argument_manager_multi():
    am = ArgumentManager('test', 'pytest')
    am.register(
        'test_float_multi',
        ['-F'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0
    )
    parser = argparse.ArgumentParser()
    am.add_args(parser)
    parser.parse_args(['-F', '2', '0'])
    assert tuple(am['test_float_multi']) == (2, 0)


def test_argument_manager_bool():
    am = ArgumentManager('test', 'pytest')
    am.register(
        'test_bool',
        ['-B'],
        dtype=bool,
        default=True
    )
    parser = argparse.ArgumentParser()
    am.add_args(parser)
    parser.parse_args(['-no-B'])
    assert am['test_bool'] == False


def test_argument_manager_sum():
    b = ArgumentManager('test', 'pytest')
    b.register(
        'test_bool',
        ['-B'],
        dtype=bool,
        default=True
    )
    m = ArgumentManager('test', 'pytest')
    m.register(
        'test_float_multi',
        ['-F'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0
    )
    i = ArgumentManager('test', 'pytest')
    i.register(
        'test_int',
        ['-I'],
        dtype=int,
        validator=lambda x: x > 0
    )
    parser = argparse.ArgumentParser()
    (i + b + m).add_args(parser)
    parser.parse_args('-I 100 -B -F 1.5 1e-6'.split(' '))
    assert b['test_bool'] == True
    assert tuple(m['test_float_multi']) == (1.5, 1e-6)
    assert i['test_int'] == 100


def test_argument_manager_default():
    am = ArgumentManager('test', 'pytest')
    am.register(
        'test_default',
        ['-S'],
        dtype=str,
        default='AreYouOK'
    )
    am.register(
        'test_int',
        ['-I'],
        dtype=int,
        validator=lambda x: x > 0
    )
    parser = argparse.ArgumentParser()
    am.add_args(parser)
    parser.parse_args(['-I', '10'])
    assert am['test_default'] == 'AreYouOK'
