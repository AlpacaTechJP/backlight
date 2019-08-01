# Define how to create Portfolio
from typing import List
from backlight.portfolio.portfolio import Portfolio, create_portfolio
from backlight.positions.positions import Positions
from backlight.trades.trades import Trades
from backlight.signal.signal import Signal
from backlight.datasource.marketdata import MarketData
from backlight.datasource.utils import get_forex_ratios
from backlight import strategies
from backlight.asset.currency import Currency

from joblib import Parallel, delayed


def create_simple_trades(
    mkt: List[MarketData], sig: List[Signal], strategy_name: str, strategy_params: dict
) -> List[Trades]:
    """
    Create a list of trades from a list of signals of each asset

    Args:
        mkt: list of marketdata of each asset
        sig: list of signals to be used to create the porfolio
        strategy_name: a simple strategy from module strategies

    return:
        List of Trades
    """

    # Load strategy
    strategy = getattr(strategies, strategy_name)

    # check markets and signals given in order
    for (m, s) in zip(mkt, sig):
        assert m.symbol == s.symbol

    # Apply strategy on each asset and get list of trades
    trades = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(strategy)(market, asset, **strategy_params)
            for (asset, market) in zip(sig, mkt)
        ]
    )

    return trades


def equally_weighted_portfolio(
    trades: List[Trades],
    mkt: List[MarketData],
    principal: float,
    max_amount: float,
    currency_unit: Currency = Currency.USD,
) -> Portfolio:
    """
    Create a Portfolio from trades and mkt, given a principal which will be divided equally between the 
    different currencies.
    args :
        - trades : a list of trades for each currencies
        - mkt : the market datas for at least each trades currencies
        - principal : the total amount allocated to the Portfolio
        - max_amount : the max amount
        - currency_unit : the unit type of the future Portfolio
    """
    nb_currencies = len(trades)

    symbols = [t.symbol for t in trades]

    principals = {}
    lts = {}
    for trade in trades:
        symbol = trade.symbol
        trade_currency = trade.currency_unit

        ratio = 1
        if trade_currency != currency_unit:
            ratios = get_forex_ratios(mkt, trade_currency, currency_unit)
            ratio = ratios.iloc[ratios.index.get_loc(trade.index[0]) - 1]

        count_symbol = symbols.count(symbol)
        principals[symbol] = principal / (nb_currencies * ratio * count_symbol)
        lts[symbol] = int(principal / (max_amount))

    return create_portfolio(trades, mkt, principals, lts, currency_unit)
