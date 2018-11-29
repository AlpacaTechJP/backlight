from typing import List

from backlight.datasource.marketdata import MarketData
from backlight.trades.trades import Trade, Trades, Transaction  # noqa


def make_trades(trades: List[Trade], mkt: MarketData) -> Trades:
    t = Trades(mkt)
    t._trades = trades
    t.reset()

    assert t.symbol == mkt.symbol

    return t
