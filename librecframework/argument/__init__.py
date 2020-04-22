#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['functional', 'manager', 'Argument']

from collections import namedtuple

Argument = namedtuple('Argument', [
    'pname',
    'cli_aliases',
    'multi',
    'dtype',
    'validator',
    'helpstr',
    'default'])
