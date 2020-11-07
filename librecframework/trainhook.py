#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Union
from abc import ABC, abstractmethod

__all__ = ['TrainHook', 'ValueMeanHook']


class TrainHook(ABC):
    '''
    base class of train hook like loss
    '''

    def __init__(self, title: str) -> None:
        self.start()
        self.title = title

    def __str__(self):
        return f'{self.__class__.__name__}(title={self.title})'

    @property
    def value(self) -> float:
        return self._value

    @abstractmethod
    def __call__(self, value: Any) -> None:
        pass

    def start(self) -> None:
        '''
        clear all
        '''
        self._cnt = 0
        self._value = 0
        self._sum = 0

    def stop(self) -> None:
        if self._cnt == 0 and self._sum == 0:
            self._value = 0
        elif self._cnt == 0:
            raise RuntimeError('motify _sum but not _cnt')
        else:
            self._value = self._sum / self._cnt


class ValueMeanHook(TrainHook):
    def __call__(self, value: Union[float, int]) -> None:
        self._sum += value
        self._cnt += 1
