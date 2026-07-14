# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Market Solutions
@author: Jack Kirby Cook

"""

import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Downloaders", "Calculators", "Filters", "Computations", "Localizers", "StocksDownloader", "OptionsDownloader", "MarketComputation"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Computations: forward: Optional[Callable]; valuation: Optional[Callable]; volatility: Optional[Callable]; variance: Optional[Callable]; standardization: Callable; greeks: Optional[Callable]
@dataclass(frozen=True)
class Downloaders: bars: Callable; stocks: Callable; options: Callable; contracts: Callable
@dataclass(frozen=True)
class Calculators: technicals: Callable; stocks: Callable; options: Callable
@dataclass(frozen=True)
class Filters: sanity: Callable; viability: Callable
@dataclass(frozen=True)
class Localizers: surface: Callable; partitions: Callable; proximity: Callable


class StocksDownloader(object):
    def __init__(self, *args, downloaders, calculators, **kwargs):
        super().__init__(*args, **kwargs)
        self.__downloaders = downloaders
        self.__calculators = calculators

    def __call__(self, symbol, *args, history, period, **kwargs):
        bars = self.downloaders.bars([symbol], history=history)
        technicals = self.calculators.technicals(bars, period=period)
        technicals = technicals[technicals["date"] <= pd.Timestamp.today()]
        technicals = technicals.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).last()
        stocks = self.downloaders.stocks([symbol])
        stocks = stocks.merge(technicals[["ticker", "volatility", "trend"]], on="ticker", how="left", validate="many_to_one", sort=False)
        stocks = self.calculators.stocks(stocks)
        return stocks.squeeze()

    @property
    def downloaders(self): return self.__downloaders
    @property
    def calculators(self): return self.__calculators


class OptionsDownloader(object):
    def __init__(self, *args, downloaders, calculators, filters, **kwargs):
        super().__init__(*args, **kwargs)
        self.__downloaders = downloaders
        self.__calculators = calculators
        self.__filters = filters

    def __call__(self, stock, *args, expires, strikes, **kwargs):
        assert isinstance(stock, pd.Series)
        strikes = NumRange.create([stock["last"] * strikes.minimum, stock["last"] * strikes.maximum])
        contracts = self.downloaders.contracts([stock.ticker], expires=expires, strikes=strikes)
        options = self.downloaders.options(contracts)
        options["volatility"] = stock["volatility"]
        options["trend"] = stock["trend"]
        options["underlying"] = stock["median"]
        options = self.filters.sanity(options)
        options = self.calculators.options(options)
        options = self.filters.viability(options)
        return options

    @property
    def downloaders(self): return self.__downloaders
    @property
    def calculators(self): return self.__calculators
    @property
    def filters(self): return self.__filters


class MarketComputation(object):
    def __init__(self, *args, computations, **kwargs):
        super().__init__(*args, **kwargs)
        self.__computations = computations

    def __call__(self, options, *args, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.computations.forward(options, interest=interest, dividends=dividends)
        options = self.computations.valuation(options, interest=interest, dividends=dividends)
        options = self.computations.volatility(options, interest=interest, dividends=dividends)
        options = self.computations.variance(options)
        if self.computations.greeks is not None:
            options = self.computations.greeks(options, interest=interest, dividends=dividends)
        return options

    @property
    def computations(self): return self.__computations





