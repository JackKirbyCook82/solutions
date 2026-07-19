# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Trading Solutions
@author: Jack Kirby Cook

"""

import pandas as pd

from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionDownloading", "OptionFiltering", "OptionCalculating"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionDownloading(object):
    def __init__(self, *args, stocks, contracts, options, **kwargs):
        super().__init__(*args, **kwargs)
        self.__contracts = contracts
        self.__options = options
        self.__stocks = stocks

    def __call__(self, symbol, /, expires, strikes, **kwargs):
        stock = self.stocks([symbol]).squeeze()
        strikes = NumRange.create([stock["last"] * strikes.minimum, stock["last"] * strikes.maximum])
        contracts = self.contracts([stock.ticker], expires=expires, strikes=strikes)
        options = self.options(contracts)
        options["underlying"] = stock["last"]
        return options

    @property
    def contracts(self): return self.__contracts
    @property
    def options(self): return self.__options
    @property
    def stocks(self): return self.__stocks


class OptionFiltering(object):
    def __init__(self, *args, options, sanity, viability, **kwargs):
        self.__viability = viability
        self.__options = options
        self.__sanity = sanity

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.sanity(options)
        options = self.options(options)
        options = self.viability(options)
        return options

    @property
    def viability(self): return self.__viability
    @property
    def options(self): return self.__options
    @property
    def sanity(self): return self.__sanity


class OptionCalculating(object):
    def __init__(self, *args, forward, volatility, variance, screener, greeks, **kwargs):
        super().__init__(*args, **kwargs)
        self.__volatility = volatility
        self.__variance = variance
        self.__screener = screener
        self.__forward = forward
        self.__greeks = greeks

    def __call__(self, options, /, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.forward(options, interest=interest, dividends=dividends)
        options = self.volatility(options, interest=interest, dividends=dividends)
        options = self.variance(options)
        options = self.screener(options)
        options = self.greeks(options, interest=interest, dividends=dividends)
        return options

    @property
    def volatility(self): return self.__volatility
    @property
    def variance(self): return self.__variance
    @property
    def screener(self): return self.__screener
    @property
    def forward(self): return self.__forward
    @property
    def greeks(self): return self.__greeks



