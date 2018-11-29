from typing import Callable

from backlight.positions.positions import Positions
from backlight.trades import Trades


def _mid_trader(trades: Trades) -> Positions:
    positions = trades.copy()
    positions.loc[:, "amount"] = trades.amount.cumsum()
    positions.loc[:, "price"] = trades.mid
    pos = Positions(positions)
    pos.symbol = trades.symbol
    return pos


def calc_positions(
    trades: Trades, trader: Callable[[Trades], Positions] = _mid_trader
) -> Positions:
    positions = trader(trades)
    return positions
