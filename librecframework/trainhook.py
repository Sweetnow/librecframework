#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Union
from abc import ABC, abstractmethod

__all__ = ['TrainHook', 'ValueMeanHook']


class TrainHook(ABC):
    """
    Base class for hook in training stage like loss

    Property & Methods:
    - title: the name of trainhook
    - start(): reset the trainhook
    - stop(): close the trainhook after one epoch
    - __call__(): send a value into the trainhook
    """

    def __init__(self, title: str) -> None:
        self._title = title
        self._value = 0
        self._stopped = False

    @property
    def title(self):
        return self._title

    def __str__(self):
        return f'{self.__class__.__name__}@{self.title}'

    @property
    def value(self) -> float:
        if self._stopped:
            return self._value
        else:
            raise RuntimeError(
                f'{self} is not stopped. Call stop() before reading value')

    @abstractmethod
    def __call__(self, value: Any) -> None:
        pass

    @abstractmethod
    def _start(self) -> None:
        pass

    @abstractmethod
    def _stop(self) -> None:
        pass

    def start(self) -> None:
        """Initialize the trainhook"""
        self._stopped = False
        self._start()

    def stop(self) -> None:
        """Compute value to close the trainhook"""
        self._stop()
        self._stopped = True


class IgnoredHook(TrainHook):
    """Trainhook which ignore the input"""
    def __init__(self) -> None:
        super().__init__('#')

    def __call__(self, value: Any) -> None:
        pass

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

class ValueMeanHook(TrainHook):
    """
    Trainhook which calculate the meaning of each value
    """
    def __call__(self, value: Union[float, int]) -> None:
        self._sum += value
        self._cnt += 1

    def _start(self) -> None:
        self._cnt = 0
        self._value = 0
        self._sum = 0

    def _stop(self) -> None:
        if self._cnt == 0 and self._sum == 0:
            self._value = 0
        elif self._cnt == 0:
            raise RuntimeError('Call start() before calling stop()')
        else:
            self._value = self._sum / self._cnt
