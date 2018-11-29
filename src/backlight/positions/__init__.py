from typing import Callable

from backlight.datasource.marketdata import MarketData
from backlight.positions.positions import Positions
from backlight.trades import Trades


def _mid_trader(trades: Trades, mkt: MarketData) -> Positions:
    positions = mkt.copy()
    positions.loc[:, "amount"] = trades.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid
    pos = Positions(positions)
    pos.symbol = trades.symbol
    return pos


def calc_positions(
    trades: Trades,
    mkt: MarketData,
    trader: Callable[[Trades, MarketData], Positions] = _mid_trader,
) -> Positions:
    assert trades.symbol == mkt.symbol
    positions = trader(trades, mkt)
    return positions
