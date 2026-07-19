# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Trading Option Solutions
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionDownloading", "OptionFiltering", "OptionMarketing", "OptionForecasting"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionDownloading(object):
    def __init__(self, *args, stocks, contracts, options, **kwargs):
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


class OptionMarketing(object):
    def __init__(self, *args, volatility, greeks, forward, variance, screener, **kwargs):
        self.__volatility = volatility
        self.__greeks = greeks
        self.__forward = forward
        self.__variance = variance
        self.__screener = screener

    def __call__(self, options, /, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.forward(options, interest=interest, dividends=dividends)
        options = self.volatility(options, interest=interest, dividends=dividends, signature="median->implied")
        options = self.variance(options)
        options = self.screener(options)
        options = self.greeks(options, interest=interest, dividends=dividends, signature="implied->", delimiter=None)
        return options

    @property
    def volatility(self): return self.__volatility
    @property
    def greeks(self): return self.__greeks
    @property
    def forward(self): return self.__forward
    @property
    def variance(self): return self.__variance
    @property
    def screener(self): return self.__screener


class OptionForecasting(object):
    def __init__(self, *args, surface, standardize, valuation, **kwargs):
        self.__standardize = standardize
        self.__valuation = valuation
        self.__surface = surface

    def __call__(self, options, /, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.surface(options, method="regression", smoothing=1 / 10, weights=None)
        options = self.standardize(options, surface)
        options["tsv"] = surface(options["tau"], options["mae"])
        options["surfaced"] = np.sqrt(options["tsv"] / options["tau"])
        options = self.valuation(options, interest=interest, dividends=dividends, signature="surfaced->forecast")
        return options

    @property
    def standardize(self): return self.__standardize
    @property
    def valuation(self): return self.__valuation
    @property
    def surface(self): return self.__surface




