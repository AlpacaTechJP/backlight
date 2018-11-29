import pandas as pd
from typing import List, Tuple

from backlight import datasource
from backlight.datasource.marketdata import MarketData
from backlight.trades.trades import Trade, Transaction


def _sum(a: list) -> int:
    return sum(a) if len(a) != 0 else 0


def _make_trade(sr: pd.Series, symbol: str) -> Trade:
    t = Trade(sr)
    t.symbol = symbol
    return t


def make_trade(symbol: str) -> Trade:
    t = Trade()
    t.symbol = symbol
    return t


def evaluate_pl(trade: Trade, mkt: MarketData) -> float:
    amount = trade.amount.cumsum()
    mkt, amount = mkt.align(amount, axis=0, join="inner")
    # TODO: ask bid pricing
    next_price = mkt["mid"].shift(periods=-1)
    price_diff = next_price - mkt["mid"]
    pl = (price_diff * amount).shift(periods=1)[1:]  # drop first nan
    return _sum(pl)


def count(trades: List[Trade], mkt: MarketData) -> Tuple[int, int, int]:
    pls = [evaluate_pl(t, mkt) for t in trades]
    total = len(pls)
    win = _sum([pl > 0.0 for pl in pls])
    lose = _sum([pl < 0.0 for pl in pls])
    return total, win, lose


def flatten(trades: List[Trade]) -> Trade:
    symbol = trades[0].symbol
    assert all([t.symbol == symbol for t in trades])

    # compute amount
    amounts = [t.amount for t in trades]
    amount = pd.Series()
    for a in amounts:
        amount = amount.add(a, fill_value=0.0)

    return _make_trade(amount, symbol)
