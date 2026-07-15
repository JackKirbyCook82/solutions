# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Finance Solutions
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable, Optional

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LocalizingComputation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class LocalizingCalculators:
    options: pd.DataFrame; surface: Callable; partitions: Callable; proximity: Callable; standardization: Optional[Callable]
    method: str; smoothing: float; weights: Optional[NDArray[np.floating]] = None

    def __getitem__(self, spreads):
        assert isinstance(spreads, pd.DataFrame)
        arguments = [self.options, spreads]
        parameters = dict(method=self.method, smoothing=self.smoothing, weights=self.weights)
        localized = self.proximity(*arguments)
        surface = self.surface(localized, **parameters)
        localized = self.standardization(localized, surface)
        return localized

    def __iter__(self):
        arguments = [self.options]
        parameters = dict(method=self.method, smoothing=self.smoothing, weights=self.weights)
        generator = self.partitions(*arguments)
        for localized in generator:
            surface = self.surface(localized, **parameters)
            localized = self.standardization(localized, surface)
            yield localized


class LocalizingComputation(object):
    def __init__(self, *args, surface, partitions, proximity, standardization, **kwargs):
        assert isinstance(surface, Callable)
        assert isinstance(partitions, Callable)
        assert isinstance(proximity, Callable)
        assert isinstance(standardization, Callable)
        self.__standardization = standardization
        self.__partitions = partitions
        self.__proximity = proximity
        self.__surface = surface

    def __call__(self, options, /, method="regression", smoothing=1/10, weights=None, **kwargs):
        calculators = dict(surface=self.surface, partitions=self.partitions, proximity=self.proximity, standardization=self.standardization)
        parameters = dict(method=method, smoothing=smoothing, weights=weights)
        localizing = LocalizingCalculators(options=options, **calculators, **parameters)
        return localizing

    @property
    def standardization(self): return self.__standardization
    @property
    def partitions(self): return self.__partitions
    @property
    def proximity(self): return self.__proximity
    @property
    def surface(self): return self.__surface



