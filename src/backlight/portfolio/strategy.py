# Define how to create Portfolio
from typing import List
from backlight.portfolio.portfolio import Portfolio
from backlight.positions.positions import Positions
from backlight.trades.trades import Trades
from backlight.signal.signal import Signal
from backlight.datasource.marketdata import MarketData
from backlight import strategies

from joblib import Parallel, delayed


def generate_simple_trades(
    mkt: List[MarketData], sig: List[Signal], strategy_name: str, strategy_params: dict
) -> List[Trades]:
    """
    Create a list of trades from a list of signals of each asset

    Args:
        mkt: list of marketdata of each asset
        sig: list of signals to be used to construct the porfolio
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
