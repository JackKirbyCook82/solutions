# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Finance Solutions
@author: Jack Kirby Cook

"""

import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PricingCalculators", "ImpliedCalculators", "FinanceComputation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class PricingCalculators: valuation: Optional[Callable]; greeks: Optional[Callable]
@dataclass(frozen=True)
class ImpliedCalculators: forward: Optional[Callable]; volatility: Optional[Callable]; variance: Optional[Callable]


class FinanceComputation(object):
    def __init__(self, *args, pricing, implied, **kwargs):
        super().__init__(*args, **kwargs)
        self.__implied = implied
        self.__pricing = pricing

    def __call__(self, options, /, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.implied.forward(options, interest=interest, dividends=dividends)
        options = self.pricing.valuation(options, interest=interest, dividends=dividends)
        options = self.implied.volatility(options, interest=interest, dividends=dividends)
        options = self.implied.variance(options)
        if self.pricing.greeks is not None:
            options = self.pricing.greeks(options, interest=interest, dividends=dividends)
        return options

    @property
    def implied(self): return self.__implied
    @property
    def pricing(self): return self.__pricing


