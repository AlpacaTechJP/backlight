# Define how to create Portfolio
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from backlight.portfolio.portfolio import Portfolio, create_portfolio
from backlight.positions.positions import Positions
from backlight.trades.trades import Trades
from backlight.signal.signal import Signal
from backlight.datasource.marketdata import MarketData
from backlight.datasource.utils import get_forex_ratio, get_forex_ratios
from backlight import strategies
from backlight.asset.currency import Currency

from joblib import Parallel, delayed


def create_simple_trades(
    mkts: List[MarketData], sig: List[Signal], strategy_name: str, strategy_params: dict
) -> List[Trades]:
    """
    Create a list of trades from a list of signals of each asset

    Args:
        mkts: list of marketdata of each asset
        sig: list of signals to be used to create the porfolio
        strategy_name: a simple strategy from module strategies

    return:
        List of Trades
    """

    # Load strategy
    strategy = getattr(strategies, strategy_name)

    # check markets and signals given in order
    for (m, s) in zip(mkts, sig):
        assert m.symbol == s.symbol

    # Apply strategy on each asset and get list of trades
    trades = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(strategy)(market, asset, **strategy_params)
            for (asset, market) in zip(sig, mkts)
        ]
    )

    return trades


def equally_weighted_portfolio(
    trades: List[Trades],
    mkts: List[MarketData],
    principal: float,
    max_amount: float,
    currency_unit: Currency = Currency.USD,
) -> Portfolio:
    """
    Create a Portfolio from trades and mkts, given a principal which will be divided equally between the
    different currencies.
    args :
        - trades : a list of trades for each currencies
        - mkts : the market datas for at least each trades currencies
        - principal : the total amount allocated to the Portfolio
        - max_amount : the max amount
        - currency_unit : the unit type of the future Portfolio
    """
    principals, lot_sizes = _calculate_principals_lot_sizes(
        trades, mkts, principal, max_amount, currency_unit=currency_unit
    )

    return create_portfolio(trades, mkts, principals, lot_sizes, currency_unit)


def _calculate_principals_lot_sizes(
    trades: List[Trades],
    mkts: List[MarketData],
    principal: float,
    max_amount: float,
    currency_unit: Currency = Currency.USD,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    nb_trades = len(trades)
    symbol2mkt = {mkt.symbol: mkt for mkt in mkts}

    principals = {}
    lot_sizes = {}
    for trade in trades:
        symbol = trade.symbol
        trade_currency = trade.currency_unit

        mkt = symbol2mkt[symbol]
        current_price = mkt.mid[trade.index[0]]
        ratio = get_forex_ratio(trade.index[0], mkts, trade_currency, currency_unit)

        principals[symbol] = principal / (nb_trades * ratio)
        lot_sizes[symbol] = principal / (nb_trades * max_amount * ratio * current_price)

    return principals, lot_sizes
