#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['functional', 'manager', 'Argument']

from typing import Callable, Generic, NamedTuple, List, Optional, Type, TypeVar

T = TypeVar('T')


class _Argument(NamedTuple):
    pname: str
    cli_aliases: List[str]
    multi: bool
    dtype: Type[T]
    validator: Optional[Callable[[T], bool]]
    helpstr: str
    default: Optional[T]

class Argument(_Argument, Generic[T]):
    pass