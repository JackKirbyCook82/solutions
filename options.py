# -*- coding: utf-8 -*-
"""
Created on Tues Jul 14 2026
@name:   Trading Option Solutions
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from types import SimpleNamespace
from datetime import date as Date
from datetime import timedelta as Timedelta

from support.custom import DateRange, NumberRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionDownloading", "OptionFiltering", "OptionPricing", "OptionValuing"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionExpireError(Exception): pass
class OptionStrikeError(Exception): pass
class OptionDownloading(object):
    def __init__(self, *args, stocks, contracts, options, **kwargs):
        self.__contracts = contracts
        self.__options = options
        self.__stocks = stocks

    def __call__(self, symbol, /, expires, strikes, **kwargs):
        stock = self.stocks([symbol]).squeeze()
        tomorrow = Date.today() + Timedelta(days=1)
        underlying = stock["last"]
        if callable(expires): expires = expires(tomorrow=tomorrow, **kwargs)
        if callable(strikes): strikes = strikes(underlying=underlying, **kwargs)
        assert isinstance(expires, DateRange) and isinstance(strikes, NumberRange)
        expires = DateRange(max(expires.minimum, tomorrow), expires.maximum)
        strikes = NumberRange(max(strikes.minimum, 0), strikes.maximum)
        contracts = self.contracts([symbol], expires=expires, strikes=strikes)
        options = self.options(contracts)
        options["underlying"] = underlying
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


class OptionPricing(object):
    def __init__(self, *args, volatility, greeks, forward, variance, **kwargs):
        self.__volatility = volatility
        self.__greeks = greeks
        self.__forward = forward
        self.__variance = variance

    def __call__(self, options, /, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        assert len(options["ticker"].unique()) == 1
        options = self.forward(options, interest=interest, dividends=dividends)
        options = self.volatility(options, interest=interest, dividends=dividends, signature="market->implied")
        options = self.greeks(options, interest=interest, dividends=dividends, signature="implied->", delimiter=None)
        options = self.variance(options)
        volatility = list(self.interpolate(options, signature="implied->volatility"))
        options["volatility"] = float(np.median(volatility))
        return options

    @staticmethod
    def interpolate(dataframes, signature):
        assert isinstance(dataframes, pd.DataFrame)
        inlet, outlet = str(signature).split("->")
        dataframes = dataframes[["expire", "strike", "underlying", inlet]].rename(columns={inlet: outlet})
        for expire, dataframe in dataframes.groupby("expire"):
            underlying = float(dataframe["underlying"].median())
            dataframe = dataframe.groupby("strike", as_index=False)[outlet].median().sort_values("strike")
            lower = dataframe[dataframe["strike"] <= underlying].tail(1)
            upper = dataframe[dataframe["strike"] >= underlying].head(1)
            if lower.empty and upper.empty: continue
            if lower.empty: yield float(upper[outlet].iloc[0])
            elif upper.empty: yield float(lower[outlet].iloc[0])
            else:
                strikes = SimpleNamespace(lower=float(lower["strike"].iloc[0]), upper=float(upper["strike"].iloc[0]))
                outlets = SimpleNamespace(lower=float(lower[outlet].iloc[0]), upper=float(upper[outlet].iloc[0]))
                if strikes.lower == strikes.upper: yield outlets.lower
                else:
                    weight = ((underlying - strikes.lower) / (strikes.upper - strikes.lower))
                    yield outlets.lower + weight * (outlets.upper - outlets.lower)

    @property
    def volatility(self): return self.__volatility
    @property
    def greeks(self): return self.__greeks
    @property
    def forward(self): return self.__forward
    @property
    def variance(self): return self.__variance


class OptionValuing(object):
    def __init__(self, *args, screen, surface, standardize, valuation, **kwargs):
        self.__standardize = standardize
        self.__valuation = valuation
        self.__surface = surface
        self.__screen = screen

    def __call__(self, options, /, interest, dividends, method="regression", smoothing=1/10, weights=None, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.screen(options)
        surface = self.surface(options, method=method, smoothing=smoothing, weights=weights)
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
    @property
    def screen(self): return self.__screen



