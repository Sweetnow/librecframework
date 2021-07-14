#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['functional', 'manager', 'Argument']

from typing import Any, Callable, NamedTuple, List, Optional, Type


class Argument(NamedTuple):
    pname: str
    cli_aliases: List[str]
    multi: bool
    dtype: Type
    validator: Optional[Callable[[Any], bool]]   # Any = dtype
    helpstr: str
    default: Optional[Any]  # Any = dtype
