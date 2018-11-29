from typing import Callable, List

from backlight.datasource.marketdata import MarketData
from backlight.positions.positions import Positions
from backlight.trades import Trade, flatten


def _mid_trader(trade: Trade, mkt: MarketData) -> Positions:
    positions = mkt.copy()
    positions.loc[:, "amount"] = trade.amount.cumsum()
    positions.loc[:, "price"] = mkt.mid
    pos = Positions(positions)
    pos.symbol = trade.symbol
    return pos


def calc_positions(
    trades: List[Trade],
    mkt: MarketData,
    trader: Callable[[Trade, MarketData], Positions] = _mid_trader,
) -> Positions:
    trade = flatten(trades)
    assert trade.symbol == mkt.symbol
    positions = trader(trade, mkt)
    return positions
