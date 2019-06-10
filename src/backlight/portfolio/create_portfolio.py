# Define how to create Portfolio

from backlight.positions.positions import Positions
from backlight.trades.trades import Trades
from backlight.signal.signal import Signal
from backlight.datasource.marketdata import MarketData
from backlight.portfolio.portfolio import Portfolio


def create_simple_portfolio(
    mkt: List[MarketData], sig: List[Signals], principal: float
):
    """
    Create portfolio (as a list of positions) from a list of signals of each asset

    Args:
        mkt: list of marketdata of each asset
        sig: list of signals to be used to construct the porfolio
        principal: initial principal value available for the portfolio

    return:
        Portfolio 
    """
