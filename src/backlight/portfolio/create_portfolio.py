# Define how to create Portfolio

from backlight.positions.positions import Positions
from backlight.trades.trades import Trades
from backlight.signal.signal import Signal
from backlight.datasource.marketdata import MarketData
from backlight import strategies
from backlight.positions import calc_positions

from joblib import Parallel, delayed


def create_simple_portfolio(
    mkt: List[MarketData],
    sig: List[Signal],
    strategy_name: str,
    strategy_params: dict,
    principal: float,
):
    """
    Create portfolio (as a list of positions) from a list of signals of each asset

    Args:
        mkt: list of marketdata of each asset
        sig: list of signals to be used to construct the porfolio
        strategy_name: a simple strategy from module strategies
        principal: initial principal value available for the portfolio

    return:
        Portfolio
    """

    # Load strategy
    strategy = getattr(strategies, strategy_name)
    principal_per_asset = principal / len(mkt)

    # check markets and signals given in order
    for (m, s) in zip(mkt, sig):
        assert m.symbol == s.symbol

    # Apply strategy on each asset and get list of trades

    trades = Parallel(n_jobs=-1)(
        [
            delayed(strategy)(market, asset, **strategy_params)
            for (asset, market) in zip(sig, mkt)
        ]
    )

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1)(
        [
            delayed(calc_positions)(trade, market, principal=principal_per_asset)
            for (trade, market) in zip(trades, mkt)
        ]
    )

    return Portfolio(positions)
